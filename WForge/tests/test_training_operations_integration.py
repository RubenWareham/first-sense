"""Real upstream recovery, mixed precision, relocation, and shared evaluation checks."""

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from wvision import Vision, bundle_run, prepare_label_budget, pretrain, resume, run_experiments
from wvision.adapters.shared.checkpoints import checkpoint_files
from wvision.datasets.detection import load_dataset
from wvision.evaluation.detection import evaluate_predictions
from wvision.models.ultralytics.yolo11 import operations as yolo_operations

pytestmark = pytest.mark.integration


@pytest.mark.ultralytics
def test_shared_coco_metric_and_split_isolation(manifest: Path, tmp_path: Path) -> None:
    pytest.importorskip("pycocotools")
    from PIL import Image

    dataset = load_dataset(manifest, ("val",))
    rows = []
    for sample in dataset.splits["val"]:
        with Image.open(sample.path) as image:
            width, height = image.size
        boxes, ids = [], []
        assert sample.label is not None
        for line in sample.label.read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            boxes.append(
                [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ]
            )
            ids.append(int(cls))
        rows.append(
            {
                "sample": str(sample.path),
                "classes": dataset.classes,
                "width": width,
                "height": height,
                "coordinate_format": "absolute_pixel_xyxy",
                "boxes": boxes,
                "scores": [1.0] * len(boxes),
                "class_ids": ids,
            }
        )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(row) for row in rows))
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"metric fixture, not a trained model")
    result = evaluate_predictions(dataset, predictions, checkpoint=checkpoint, output=tmp_path)
    assert result.metrics["wforge/coco/bbox_mAP"] == pytest.approx(1.0)
    rows[0]["sample"] = str(dataset.splits["test"][0].path)
    predictions.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="non-validation"):
        evaluate_predictions(dataset, predictions, checkpoint=checkpoint, output=tmp_path)


def executable(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to the compatible backend interpreter")
    return value


def continue_and_check(run: Path, output: Path, **budget: Any) -> Any:
    before = checkpoint_files(run)
    continued = resume(run, output=output, **budget)
    assert checkpoint_files(run) == before
    record = json.loads((continued.artifact_dir / "recovery.json").read_text())
    assert record["step"] == next(iter(budget.values()))
    assert continued.artifact_dir != run
    return continued


def relocate_and_predict(
    run: Path, sample: Path, output: Path, *, backend: str | None = None
) -> None:
    bundle = bundle_run(run, output / "bundle", include_data=True)
    relocated = output / "relocated"
    assert relocated.resolve().is_relative_to(output.resolve())
    bundle.rename(relocated)
    restored = Vision.from_run(relocated, backend_python=backend)
    result = restored.predict(sample, device="cpu", output=output)
    assert result.sample_count == 1
    assert restored._checkpoint is not None and restored._checkpoint.is_relative_to(relocated)


@pytest.mark.ultralytics
def test_yolo_resume_bundle_onnx_and_experiment_grid(manifest: Path, tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("pycocotools")
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    trained = Vision("models/detection/yolo11n/smoke.yaml").train(
        manifest,
        device=device,
        precision="fp16" if device != "cpu" else "fp32",
        output=tmp_path,
    )
    continued = continue_and_check(trained.artifact_dir, tmp_path / "continued", epochs=2)
    relocate_and_predict(
        continued.artifact_dir, manifest.parent / "images/test-0.png", tmp_path / "portable"
    )
    exported = Vision.from_run(continued.artifact_dir).export(output=tmp_path)
    assert exported.model.suffix == ".onnx"
    assert json.loads(exported.manifest.read_text())["verification"]["samples"] == 2
    plan = tmp_path / "experiments.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "data": str(manifest),
                "experiments": [
                    {
                        "name": "yolo",
                        "model": str(Path("models/detection/yolo11n/smoke.yaml").resolve()),
                        "training": {"device": device},
                        "grid": {"backend_options.lr0": [0.001, 0.01]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    compared = run_experiments(plan, output=tmp_path)
    report = json.loads(compared.report.read_text())
    assert len(report["runs"]) == 2 and report["test_evaluated"] is False
    assert all("wforge/coco/bbox_mAP" in r["validation"]["metrics"] for r in report["runs"])


@pytest.mark.ultralytics
def test_yolo_interrupted_checkpoint_recovery(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("ultralytics")
    record = yolo_operations.record_checkpoint

    def interrupt(directory: Path, checkpoint: Path, step: int, **kwargs: Any) -> None:
        record(directory, checkpoint, step, **kwargs)
        if step == 1:
            raise KeyboardInterrupt("Controlled interruption after a complete checkpoint")

    with monkeypatch.context() as context:
        context.setattr(yolo_operations, "record_checkpoint", interrupt)
        with pytest.raises(KeyboardInterrupt):
            Vision("models/detection/yolo11n/smoke.yaml").train(manifest, epochs=3, output=tmp_path)
    source = next(tmp_path.glob("train-*"))
    assert json.loads((source / "status.json").read_text())["status"] == "interrupted"
    assert not (source / "run.json").exists()
    continued = continue_and_check(source, tmp_path / "continued", epochs=3)
    assert continued.status == "completed"


@pytest.mark.mmdetection
@pytest.mark.parametrize("mixed_precision", ["fp16", "bf16"])
def test_openmmlab_resume_and_relocated_detector(
    manifest: Path, tmp_path: Path, mixed_precision: str
) -> None:
    backend = executable("WVISION_MMDET_PYTHON")
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    if device == "cpu" and mixed_precision == "bf16":
        pytest.skip("FP32 CPU coverage is provided by the first parameter case")
    precision = mixed_precision if device != "cpu" else "fp32"
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    representation = pretrain(
        data,
        model="models/representation/simsiam-r50/smoke.yaml",
        backend_python=backend,
        device=device,
        precision=precision,
        checkpoint_interval=1,
        output=tmp_path,
    )
    continue_and_check(representation.artifact_dir, tmp_path / "representation-resume", max_iters=3)
    detector = Vision(
        "models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml", backend_python=backend
    )
    if mixed_precision == "bf16":
        with pytest.raises(ValueError, match="precision"):
            detector.train(data, device=device, precision=precision, output=tmp_path)
        return
    trained = detector.train(
        data, device=device, precision=precision, checkpoint_interval=1, output=tmp_path
    )
    continued = continue_and_check(trained.artifact_dir, tmp_path / "detector-resume", max_iters=3)
    sample = next((data.parent / "test/images").iterdir())
    relocate_and_predict(continued.artifact_dir, sample, tmp_path / "portable", backend=backend)


@pytest.mark.dinov3
@pytest.mark.parametrize("freeze", [False, True])
def test_dinov3_resume_and_relocated_detector(manifest: Path, tmp_path: Path, freeze: bool) -> None:
    backend = executable("WVISION_TRANSFORMERS_PYTHON")
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    trained = Vision(
        "models/detection/dinov3-convnext-tiny-detr/smoke.yaml", backend_python=backend
    ).train(
        manifest,
        device=device,
        precision="bf16" if device != "cpu" else "fp32",
        freeze_backbone=freeze,
        checkpoint_interval=1,
        output=tmp_path,
    )
    continued = continue_and_check(trained.artifact_dir, tmp_path / "continued", max_iters=3)
    relocate_and_predict(
        continued.artifact_dir,
        manifest.parent / "images/test-0.png",
        tmp_path / "portable",
        backend=backend,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("kind", ["yolo", "simsiam", "mmdetection", "dinov3"])
def test_real_two_gpu_training_and_resume(manifest: Path, tmp_path: Path, kind: str) -> None:
    if os.environ.get("WVISION_TEST_DISTRIBUTED") != "1":
        pytest.skip("Requires explicit WVISION_TEST_DISTRIBUTED=1 and at least two GPUs")
    selected = os.environ.get("WVISION_DISTRIBUTED_BACKEND", "all")
    if selected != "all" and kind not in (
        {"simsiam", "mmdetection"} if selected == "mmdetection" else {selected}
    ):
        pytest.skip("Different distributed backend selected")
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    settings = {"device": "cuda:0", "devices": [0, 1], "checkpoint_interval": 1}
    if kind == "simsiam":
        trained = pretrain(
            data,
            model="models/representation/simsiam-r50/smoke.yaml",
            backend_python=executable("WVISION_MMDET_PYTHON"),
            output=tmp_path,
            **settings,
        )
    else:
        profiles = {
            "yolo": "models/detection/yolo11n/smoke.yaml",
            "mmdetection": "models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml",
            "dinov3": "models/detection/dinov3-convnext-tiny-detr/smoke.yaml",
        }
        backend = (
            None
            if kind == "yolo"
            else executable(
                "WVISION_MMDET_PYTHON" if kind == "mmdetection" else "WVISION_TRANSFORMERS_PYTHON"
            )
        )
        trained = Vision(profiles[kind], backend_python=backend).train(
            data, output=tmp_path, **settings
        )
    budget = {"epochs": 2} if kind == "yolo" else {"max_iters": 3}
    continued = continue_and_check(trained.artifact_dir, tmp_path / "continued", **budget)
    for rank in range(2):
        root = (
            continued.artifact_dir if rank == 0 else continued.artifact_dir / f"ranks/rank-{rank}"
        )
        device = json.loads((root / "device.json").read_text())
        assert device["world_size"] == 2 and device["rank"] == rank
        assert all(value == [f"cuda:{rank}"] for value in device["model_devices"].values())
