"""Supervised/Soft Teacher comparison with explicit label and compute budgets."""

import time
from pathlib import Path
from typing import Any

from wvision.adapters.openmmlab.backend import interpreter
from wvision.api import Vision
from wvision.core.contracts import ComparisonResult
from wvision.core.model_config import comparison_model
from wvision.datasets.budgets import load_budget_metadata
from wvision.datasets.detection import load_dataset
from wvision.experiments.artifacts import operation, write_json
from wvision.models.mmdetection.faster_rcnn_r50.config import MODEL_ID, MMDetConfig


def compare_learning(
    data: str | Path,
    *,
    backend_python: str | Path | None = None,
    model: str | Path = MODEL_ID,
    output: str | Path = "runs",
    **overrides: Any,
) -> ComparisonResult:
    """Run one seed on the same detector, labeled subset, image size, and iteration budget.

    Equal iterations are not equal compute or labeled exposure. Both checkpoints are
    selected on validation data before either test evaluation begins.
    """
    values, definition = comparison_model(model, MODEL_ID)
    if values.get("learning", "semi_supervised") != "semi_supervised":
        raise ValueError("Comparison configuration must use learning=semi_supervised")
    settings = MMDetConfig.resolve({**values, "learning": "semi_supervised", **overrides})
    if settings.learning != "semi_supervised":
        raise ValueError("Comparison requires learning=semi_supervised")
    dataset = load_dataset(data, ("train_labeled", "train_unlabeled", "val", "test"))
    executable = interpreter(backend_python)
    budget = load_budget_metadata(dataset)
    reference: str | Path = definition.path if definition else MODEL_ID
    with operation(
        output, "compare", {"data": str(dataset.manifest), **settings.to_dict()}
    ) as directory:
        write_json(directory / "dataset.json", dataset.snapshot())
        if definition:
            definition.assert_unchanged()
            write_json(directory / "model-definition.json", definition.snapshot())
        shared = settings.to_dict()
        shared["learning"] = "supervised"
        shared["backend_options"] = {
            k: v for k, v in settings.backend_options.items() if k in {"lr", "backbone_style"}
        }
        start = time.monotonic()
        baseline = Vision(reference, backend_python=executable)
        supervised_run = baseline.train(dataset.manifest, output=directory, **shared)
        baseline_seconds = time.monotonic() - start
        start = time.monotonic()
        semi = Vision(reference, backend_python=executable)
        semi_run = semi.train(dataset.manifest, output=directory, **settings.to_dict())
        semi_seconds = time.monotonic() - start
        validation = {
            "supervised": baseline.val(output=directory).metrics,
            "semi_supervised": semi.val(output=directory).metrics,
        }
        supervised_test = baseline.test(output=directory)
        semi_test = semi.test(output=directory)
        metrics = supervised_test.metrics.keys() & semi_test.metrics.keys()
        report = {
            "schema_version": 1,
            "dataset_fingerprint": dataset.fingerprint,
            "model": MODEL_ID,
            "pretraining": None,
            "initialization_policy": "random, same seed",
            "seed": settings.seed,
            "seed_count": 1,
            "limitations": [
                "Single-seed execution comparison; no quality or significance claim",
                "Random initialization and short schedules differ from the published recipe",
                "Equal iterations are not equal compute or labeled-image exposure",
                "Teacher construction consumes RNG; initial tensors need not be identical",
            ],
            "label_budget": budget
            or {
                "policy": "user-defined training pools",
                "labeled_count": len(dataset.splits["train_labeled"]),
                "training_pool_size": len(dataset.splits["train_labeled"])
                + len(dataset.splits["train_unlabeled"]),
            },
            "supervised_run": str(supervised_run.artifact_dir),
            "semi_supervised_run": str(semi_run.artifact_dir),
            "compute": {
                "device": settings.device,
                "max_iters": settings.max_iters,
                "imgsz": settings.imgsz,
                "batch": settings.batch,
                "supervised_labeled_per_iter": settings.batch,
                "semi_supervised_labeled_per_iter": settings.batch // 2,
                "semi_supervised_unlabeled_per_iter": settings.batch // 2,
                "supervised_train_seconds": baseline_seconds,
                "semi_supervised_train_seconds": semi_seconds,
                "teacher_and_box_jitter_compute": "additional in Soft Teacher",
            },
            "validation": validation,
            "test": {"supervised": supervised_test.metrics, "semi_supervised": semi_test.metrics},
            "test_delta": {
                key: semi_test.metrics[key] - supervised_test.metrics[key]
                if min(semi_test.metrics[key], supervised_test.metrics[key]) >= 0
                else None
                for key in sorted(metrics)
            },
        }
        target = directory / "comparison.json"
        write_json(target, report)
    return ComparisonResult(directory, supervised_run.artifact_dir, semi_run.artifact_dir, target)
