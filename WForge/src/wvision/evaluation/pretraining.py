"""Compare random and self-supervised initialization under a fixed detection protocol."""

import time
from pathlib import Path
from typing import Any

from wvision.adapters.openmmlab.backend import interpreter
from wvision.api import Vision
from wvision.core.contracts import TransferComparisonResult
from wvision.core.model_config import comparison_model
from wvision.datasets.budgets import load_budget_metadata
from wvision.datasets.detection import load_dataset
from wvision.experiments.artifacts import operation, write_json
from wvision.models.mmdetection.faster_rcnn_r50.config import MODEL_ID, MMDetConfig
from wvision.models.mmpretrain.simsiam_r50.adapter import load_pretraining


def compare_pretraining(
    data: str | Path,
    *,
    pretraining_run: str | Path,
    backend_python: str | Path | None = None,
    model: str | Path = MODEL_ID,
    output: str | Path = "runs",
    **overrides: Any,
) -> TransferComparisonResult:
    """Compare one seed with equal labeled exposure; SSL has additional pretraining compute."""
    values, definition = comparison_model(model, MODEL_ID)
    settings = MMDetConfig.resolve({**values, **overrides})
    if (
        settings.learning != "supervised"
        or settings.backend_options.get("backbone_style") != "pytorch"
    ):
        raise ValueError(
            "Transfer comparison requires supervised learning and backbone_style=pytorch"
        )
    dataset = load_dataset(data, ("train_labeled", "train_unlabeled", "val", "test"))
    source = load_pretraining(pretraining_run, dataset)
    executable = interpreter(backend_python)
    budget = load_budget_metadata(dataset)
    reference: str | Path = definition.path if definition else MODEL_ID
    with operation(
        output,
        "compare-pretraining",
        {
            "data": str(dataset.manifest),
            "pretraining": source,
            **settings.to_dict(),
        },
    ) as directory:
        write_json(directory / "dataset.json", dataset.snapshot())
        if definition:
            definition.assert_unchanged()
            write_json(directory / "model-definition.json", definition.snapshot())
        models = {
            name: Vision(reference, backend_python=executable) for name in ("random", "pretrained")
        }
        runs = {}
        seconds = {}
        for name, detector in models.items():
            started = time.monotonic()
            runs[name] = detector.train(
                dataset.manifest,
                output=directory,
                pretraining_run=pretraining_run if name == "pretrained" else None,
                **settings.to_dict(),
            )
            seconds[name] = time.monotonic() - started
        # Both models are selected exclusively on validation before either sees test labels.
        validation = {name: model.val(output=directory).metrics for name, model in models.items()}
        test = {name: model.test(output=directory).metrics for name, model in models.items()}
        report = {
            "schema_version": 1,
            "dataset_fingerprint": dataset.fingerprint,
            "model": MODEL_ID,
            "pretraining": source,
            "seed": settings.seed,
            "seed_count": 1,
            "label_budget": budget
            or {
                "policy": "user-defined pools",
                "labeled_count": len(dataset.splits["train_labeled"]),
                "training_pool_size": sum(
                    len(dataset.splits[n]) for n in ("train_labeled", "train_unlabeled")
                ),
            },
            "supervision": {
                "pretraining": "train_unlabeled images only; no target labels or external weights",
                "downstream_training": "train_labeled boxes and class IDs",
                "selection": "validation COCO AP; test after both checkpoints selected",
                "fully_unsupervised_detection": False,
            },
            "runs": {name: str(run.artifact_dir) for name, run in runs.items()},
            "compute": {
                "downstream_config": settings.to_dict(),
                "train_seconds": seconds,
                "walltime_scope": "preflight, startup, training, validation, checkpoint I/O",
                "additional_ssl_compute": source["config"],
            },
            "limitations": [
                "Single-seed comparison; no statistical significance or generalization claim",
                "Equal downstream steps and labeled exposure; additional SSL pretraining compute",
                "Same seed does not guarantee identical head tensors across initialization paths",
                "PyTorch-style RGB backbone in both arms; differs from the M2 Caffe-style baseline",
                "Pretraining consumes only train_unlabeled; train_labeled images are excluded",
            ],
            "validation": validation,
            "test": test,
            "test_delta": {
                key: test["pretrained"][key] - test["random"][key]
                if min(test["pretrained"][key], test["random"][key]) >= 0
                else None
                for key in sorted(test["pretrained"].keys() & test["random"].keys())
            },
        }
        target = directory / "comparison.json"
        write_json(target, report)
    return TransferComparisonResult(
        directory, runs["random"].artifact_dir, runs["pretrained"].artifact_dir, target
    )
