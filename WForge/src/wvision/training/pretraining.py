"""Image-only pretraining and validated transfer provenance; no ML dependencies."""

from pathlib import Path
from typing import Any

from wvision.adapters.openmmlab.backend import interpreter
from wvision.core.contracts import PretrainResult
from wvision.core.model_config import model_reference, training_values
from wvision.core.registry import get_model
from wvision.datasets.detection import load_dataset, sha256, stage_sample
from wvision.experiments.artifacts import operation, write_json
from wvision.models.mmpretrain.simsiam_r50.adapter import invoke
from wvision.models.mmpretrain.simsiam_r50.config import MODEL_ID, PretrainConfig
from wvision.training.recovery import resolve_resume, stage_resume


def pretrain(
    data: str | Path,
    *,
    model: str | Path = MODEL_ID,
    backend_python: str | Path | None = None,
    resume_run: str | Path | None = None,
    output: str | Path = "runs",
    **overrides: Any,
) -> PretrainResult:
    """Train upstream SimSiam on train_unlabeled; always select the final checkpoint."""
    model_id, definition = model_reference(model)
    if model_id != MODEL_ID:
        raise ValueError(f"Unsupported pretraining model; use {MODEL_ID}")
    settings = PretrainConfig.resolve(training_values(definition, overrides))
    get_model(model_id).require("self_supervised", "pretrain", settings.device)
    dataset = load_dataset(data, ("train_unlabeled",))
    samples = dataset.splits["train_unlabeled"]
    if len(samples) < settings.batch:
        raise ValueError("train_unlabeled must contain at least batch images")
    executable = interpreter(backend_python)
    source = (
        resolve_resume(resume_run, model_id, dataset.fingerprint, settings.to_dict())
        if resume_run
        else None
    )
    with operation(
        output,
        "pretrain",
        {
            "model": model_id,
            "data": str(dataset.manifest),
            **settings.to_dict(),
            "backend_python": str(executable),
        },
    ) as directory:
        write_json(
            directory / "training.json",
            {
                "schema_version": 1,
                "model": model_id,
                "data": str(dataset.manifest),
                "dataset_fingerprint": dataset.fingerprint,
                "config": settings.to_dict(),
                "backend_python": str(executable),
            },
        )
        write_json(directory / "dataset.json", dataset.snapshot())
        if definition is not None:
            definition.assert_unchanged()
            write_json(directory / "model-definition.json", definition.snapshot())
        staged = []
        for index, sample in enumerate(samples):
            image = directory / "data/train_unlabeled/images" / f"{index}{sample.path.suffix}"
            stage_sample(sample, image, include_label=False)
            staged.append(image.name)
        listing = directory / "images.txt"
        listing.write_text("\n".join(staged) + "\n", encoding="utf-8")
        result = invoke(
            executable,
            directory,
            {
                "operation": "pretrain",
                "config": settings.to_dict(),
                "images": str(listing),
                **({"resume": stage_resume(source, directory)} if source else {}),
            },
        )
        checkpoint = Path(result["checkpoint"]).resolve(strict=True)
        state = {
            "schema_version": 1,
            "run_kind": "self_supervised_pretrain",
            "model": model_id,
            "model_definition": definition.snapshot() if definition else None,
            "dataset_fingerprint": dataset.fingerprint,
            "config": settings.to_dict(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "backend_python": str(executable),
            "selection": "final_iteration",
            "training_images": [s.pixel_hash for s in samples],
            "supervision": "train_unlabeled images only; downstream labels not used",
        }
        write_json(directory / "run.json", state)
        write_json(directory / "metrics.json", result["metrics"])
    return PretrainResult(directory, checkpoint, dataset.fingerprint, result["metrics"])
