"""Run the added YOLO11 sizes and the upstream RTMDet detector family."""

import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from examples.make_tiny_pose_dataset import create_pose_dataset
from wvision import Vision
from wvision.models.ultralytics.yolo11.operations import load_model


def test_rtmdet_rejects_unsupported_training(manifest: Path, tmp_path: Path) -> None:
    model = Vision("mmdetection/rtmdet-tiny", backend_python=sys.executable)
    with pytest.raises(ValueError, match="supervised FP32"):
        model.train(manifest, learning="semi_supervised", output=tmp_path)
    with pytest.raises(ValueError, match="without optimizer resume"):
        model.train(manifest, resume_run=tmp_path, output=tmp_path)
    assert not list(tmp_path.glob("train-*"))


@pytest.mark.integration
@pytest.mark.mmdetection
@pytest.mark.parametrize("size", ["tiny", "s", "m", "l", "x"])
def test_rtmdet_workflow(manifest: Path, tmp_path: Path, size: str) -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_PYTHON to the pinned MMDetection environment")
    profile = f"models/detection/rtmdet-{size}/smoke.yaml"
    model = Vision(profile, backend_python=backend)
    output = tmp_path / "runs"
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    run = model.train(manifest, device=device, output=output)
    assert run.best_checkpoint.is_file() and run.last_checkpoint.is_file()
    source = json.loads((run.artifact_dir / "model-source.json").read_text(encoding="utf-8"))
    assert source["recipe"] == f"rtmdet/rtmdet_{size}_8xb32-300e_coco.py"
    assert not (run.artifact_dir / "test.coco.json").exists()
    restored = Vision.from_run(run.artifact_dir)
    for result in (restored.val(output=output), restored.test(output=output)):
        assert math.isfinite(result.metrics["mmdetection/coco/bbox_mAP"])
        assert (result.artifact_dir / f"{result.split}.coco.json").is_file()
        assert not (result.artifact_dir / "train_labeled.coco.json").exists()
    predicted = restored.predict(
        manifest.parent / "images/test-0.png", score_threshold=0.0, max_detections=5, output=output
    )
    row = json.loads(predicted.predictions.read_text(encoding="utf-8"))
    assert row["coordinate_format"] == "absolute_pixel_xyxy"
    assert 0 < len(row["boxes"]) == len(row["scores"]) == len(row["class_ids"]) <= 5
    assert all(row["classes"][str(category)] == "rectangle" for category in row["class_ids"])
    assert all(0 <= x1 <= x2 <= 128 and 0 <= y1 <= y2 <= 128 for x1, y1, x2, y2 in row["boxes"])


@pytest.mark.integration
@pytest.mark.ultralytics
@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
@pytest.mark.parametrize("task", ["detect", "pose"])
def test_yolo11_variant_workflow(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: str, task: str
) -> None:
    def offline(*args: Any, **kwargs: Any) -> None:
        raise OSError("Network disabled by the variant smoke test")

    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(socket, "create_connection", offline)
    from wvision.adapters.ultralytics.backend import backend

    if task == "pose":
        manifest = create_pose_dataset(tmp_path / "pose", "halpe26")
        profile = f"models/pose/yolo11{size}-pose/smoke.yaml"
    else:
        profile = f"models/detection/yolo11{size}/smoke.yaml"
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    output = tmp_path / "runs"
    run = Vision(profile).train(manifest, device=device, imgsz=64, output=output)
    record = json.loads((run.artifact_dir / "run.json").read_text(encoding="utf-8"))
    skeleton = record.get("skeleton")
    with backend() as factory:
        model = load_model(factory, run.best_checkpoint, task, scale=size, skeleton=skeleton)
        assert model.model.yaml["scale"] == size
        with pytest.raises(ValueError, match="YOLO11n.*architecture"):
            load_model(factory, run.best_checkpoint, task, skeleton=skeleton)
    restored = Vision.from_run(run.artifact_dir)
    metric = f"ultralytics/metrics/mAP50-95({'P' if task == 'pose' else 'B'})"
    assert math.isfinite(restored.val(output=output).metrics[metric])
    assert math.isfinite(restored.test(output=output).metrics[metric])
    prediction = restored.predict(manifest.parent / "images/test-0.png", output=output)
    row = json.loads(prediction.predictions.read_text(encoding="utf-8"))
    assert row["coordinate_format"] == "absolute_pixel_xyxy"
    assert ("persons" if task == "pose" else "boxes") in row
