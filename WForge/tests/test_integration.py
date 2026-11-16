"""Real upstream execution; select explicitly with pytest -m integration."""

import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision

pytestmark = pytest.mark.integration


def deny_network(*args: Any, **kwargs: Any) -> None:
    raise OSError("Network disabled by the integration smoke test")


def test_python_workflow(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    model = Vision("models/detection/yolo11n/smoke.yaml")
    output = tmp_path / "runs"
    run = model.train(
        manifest,
        output=output,
        epochs=1,
        imgsz=128,
        batch=2,
        backend_options={"mosaic": 0.0},
    )
    assert run.best_checkpoint.is_file() and run.last_checkpoint.is_file()
    assert (run.artifact_dir / "model-definition.json").is_file()
    restored = Vision.from_run(run.artifact_dir)
    validation = restored.val(output=output)
    tested = restored.test(output=output)
    assert validation.split == "val" and tested.split == "test"
    assert "ultralytics/metrics/mAP50-95(B)" in tested.metrics
    prediction = restored.predict(manifest.parent / "images/test-0.png", output=output)
    assert prediction.sample_count == 1
    sample = json.loads(prediction.predictions.read_text())
    assert sample["width"] == 128 and sample["height"] == 128
    assert sample["coordinate_format"] == "absolute_pixel_xyxy"
    assert len(sample["boxes"]) == len(sample["class_ids"]) == len(sample["scores"])
    fine_tuned = Vision().train(
        manifest,
        checkpoint=run.best_checkpoint,
        output=output,
        epochs=1,
        imgsz=128,
        batch=2,
        backend_options={"mosaic": 0.0},
    )
    assert fine_tuned.best_checkpoint.is_file()
    provenance = json.loads((fine_tuned.artifact_dir / "model-source.json").read_text())
    assert provenance["initialization"] == "user-provided local checkpoint"
    # Real checkpoint/data mismatch must fail rather than silently remap categories.
    from wvision.adapters.ultralytics.backend import backend
    from wvision.models.ultralytics.yolo11.operations import evaluate

    with backend() as factory:
        from ultralytics.hub.utils import events
        from ultralytics.utils import SETTINGS

        assert not events.enabled and not SETTINGS["sync"] and not SETTINGS["hub"]
        incompatible = factory("yolov8n.yaml")
        incompatible_path = tmp_path / "incompatible.pt"
        # In 8.3.4 save() expects a checkpoint mapping even for a fresh YAML model.
        incompatible.ckpt = {}
        incompatible.save(incompatible_path)
    with pytest.raises(ValueError, match="architecture"):
        Vision().val(manifest, checkpoint=incompatible_path, output=output)
    with pytest.raises(ValueError, match="class map"):
        evaluate(
            run.best_checkpoint,
            run.artifact_dir / "upstream-data.yaml",
            {0: "wrong"},
            tmp_path,
            128,
            2,
        )
    (manifest.parent / "labels/val-0.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset changed"):
        restored.test(output=output)
    run.best_checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint content has changed"):
        Vision.from_run(run.artifact_dir)


def test_cli_workflow(manifest: Path, tmp_path: Path) -> None:
    def call(*arguments: str) -> dict[str, Any]:
        # Block networking inside each real CLI process, including upstream initialization.
        wrapper = (
            "import socket, runpy; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(OSError('offline')); "
            "socket.create_connection=socket.socket.connect; "
            "runpy.run_module('wvision', run_name='__main__')"
        )
        completed = subprocess.run(
            [sys.executable, "-c", wrapper, *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(completed.stdout)

    output = str(tmp_path / "runs")
    run = call(
        "train",
        "--model",
        "models/detection/yolo11n/smoke.yaml",
        "--data",
        str(manifest),
        "--output",
        output,
    )
    for split in ("val", "test"):
        result = call(split, "--run", run["artifact_dir"], "--output", output)
        assert result["split"] == split
    prediction = call(
        "predict",
        "--run",
        run["artifact_dir"],
        "--source",
        str(manifest.parent / "images/test-0.png"),
        "--output",
        output,
    )
    assert prediction["sample_count"] == 1
