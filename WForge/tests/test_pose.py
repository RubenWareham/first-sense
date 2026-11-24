"""Pose label semantics and the offline upstream workflow."""

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from examples.make_tiny_pose_dataset import create_pose_dataset
from wvision import Vision, bundle_run
from wvision.datasets.detection import load_dataset
from wvision.datasets.pose import JOINTS, coco17, resolve_skeleton, restore_skeleton
from wvision.models.ultralytics.yolo11.operations import load_model, pose_persons

MODEL = "models/pose/yolo11n-pose/smoke.yaml"
AP = "ultralytics/metrics/mAP50-95(P)"


def test_pose_preflight_and_staging(tmp_path: Path, manifest: Path) -> None:
    source = create_pose_dataset(tmp_path / "pose")
    dataset = load_dataset(source, ("train_labeled", "val"), task="pose")
    staged = tmp_path / "staged"
    staged.mkdir()
    upstream = yaml.safe_load(dataset.write_upstream(staged).read_text(encoding="utf-8"))
    assert upstream["kpt_shape"] == [17, 3]
    assert JOINTS[upstream["flip_idx"][9]] == "right_wrist"
    assert (staged / "data/val/labels/1.txt").read_bytes() == b""
    assert not (staged / "data/test").exists()
    label = source.parent / "labels/train_labeled-0.txt"
    assert label.read_bytes() == (staged / "data/train_labeled/labels/0.txt").read_bytes()
    model = Vision(MODEL)
    with pytest.raises(ValueError, match="task=pose"):
        model.train(manifest)
    with pytest.raises(ValueError, match="bbox protocol"):
        model.val(protocol="coco")
    with pytest.raises(ValueError, match="optimizer resume"):
        model.train(source, resume_run=tmp_path)
    rows = label.read_text(encoding="utf-8").splitlines()
    fields = rows[0].split()
    invisible_x = 5 + 9 * 3
    fields[invisible_x] = "0.4"
    for bad_row in (" ".join(fields[:5]), " ".join(fields)):
        label.write_text(bad_row + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="17 x/y/visibility"):
            load_dataset(source, ("train_labeled", "val"), task="pose")


@pytest.mark.integration
@pytest.mark.ultralytics
@pytest.mark.parametrize("skeleton_id", ["coco17", "halpe26"])
def test_yolo_pose_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skeleton_id: str
) -> None:
    def offline(*args: Any, **kwargs: Any) -> None:
        raise OSError("Network disabled by the pose smoke test")

    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "upstream-settings"))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.engine.results import Results

    from wvision.adapters.ultralytics.backend import backend

    device = os.environ.get("WVISION_POSE_TEST_DEVICE", "cpu")
    source = create_pose_dataset(tmp_path / "pose", skeleton_id)
    skeleton = resolve_skeleton(skeleton_id)
    count = len(skeleton["joints"])
    output = tmp_path / "runs"
    trained = Vision(MODEL).train(source, output=output, device=device)
    run = trained.artifact_dir
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert record["skeleton"] == skeleton
    assert "pose AP50_95" in record["selection"]["metric"]
    assert not (run / "data/test").exists()
    cli = [
        sys.executable,
        "-c",
        "import socket,runpy; "
        "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(OSError('offline')); "
        "socket.create_connection=socket.socket.connect; "
        "runpy.run_module('wvision',run_name='__main__')",
    ]
    test_result = {}
    for split in ("val", "test"):
        evaluated = subprocess.run(
            [*cli, split, "--run", str(run), "--output", str(output)],
            check=True,
            text=True,
            capture_output=True,
        )
        result = json.loads(evaluated.stdout)
        assert result["split"] == split and math.isfinite(result["metrics"][AP])
        assert Path(result["checkpoint"]) == trained.best_checkpoint
        folder = Path(result["artifact_dir"])
        assert (folder / f"data/{split}/images").is_dir()
        assert not (folder / "data/train_labeled").exists()
        protocol = json.loads((folder / "pose-protocol.json").read_text(encoding="utf-8"))
        assert protocol["skeleton"] == skeleton_id and len(protocol["oks_sigmas"]) == count
        if skeleton_id == "halpe26":
            assert protocol["oks_sigmas"] == [1 / 26] * 26
        test_result = result
    with pytest.raises(ValueError, match="bundles for pose"):
        bundle_run(run, tmp_path / "bundle")

    image = source.parent / "images/test-0.png"
    restored = Vision.from_run(run)
    empty = restored.predict(image, output=output, score_threshold=1.0)
    assert json.loads(empty.predictions.read_text(encoding="utf-8"))["persons"] == []
    predicted = restored.predict(image, output=output, score_threshold=0.001, max_detections=2)
    row = json.loads(predicted.predictions.read_text(encoding="utf-8"))
    assert row["sample"] == str(image) and (row["width"], row["height"]) == (160, 96)
    assert row["skeleton"] == skeleton_id and row["keypoint_count"] == count
    with backend(device, tmp_path) as factory:
        model = load_model(factory, trained.best_checkpoint, "pose", skeleton=skeleton)
        native = model.predict(
            str(image), imgsz=128, conf=0.001, max_det=2, verbose=False, device=device
        )[0]
        assert row["persons"] == pose_persons(native, count)
        metrics = model.val(
            data=str(Path(test_result["artifact_dir"]) / "upstream-data.yaml"),
            device=device,
            imgsz=128,
            batch=2,
            workers=0,
            plots=False,
            save_json=False,
            half=False,
            project=str(tmp_path),
            name="native-validation",
        )
        assert test_result["metrics"][AP] == metrics.results_dict["metrics/mAP50-95(P)"]
        with pytest.raises(ValueError, match="architecture"):
            load_model(factory, trained.best_checkpoint)
        if skeleton_id == "halpe26":
            with pytest.raises(ValueError, match="Checkpoint keypoint shape"):
                load_model(factory, trained.best_checkpoint, "pose", skeleton=coco17())
    # Exercise matching/order with known upstream objects even when random weights yield no people.
    joints = torch.tensor([[[i + 0.25, i + 1.5, 0.75] for i in range(count)]] * 2)
    joints[0, 9, 2] = 0.25
    joints[1, :, 0] += 80
    result = Results(
        native.orig_img,
        str(image),
        {0: "person"},
        boxes=torch.tensor([[0, 0, 40, 60, 0.75, 0], [80, 0, 120, 60, 0.5, 0]]),
        keypoints=joints,
    )
    first, second = pose_persons(result, count)
    assert first["joints"][10] == {"x": 10.25, "y": 11.5, "confidence": 0.75}
    assert first["joints"][9] == {"x": None, "y": None, "confidence": 0.25}
    assert second["box"] == [80, 0, 120, 60]
    assert second["joints"][0]["x"] == 80.25
    assert second["joints"][-1]["x"] == count - 1 + 80.25


def test_custom_pose_skeleton_controls_annotation_width_and_identity(tmp_path: Path) -> None:
    source = create_pose_dataset(tmp_path / "pose")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    schema = {
        "id": "football3",
        "joints": ["left_heel", "right_heel", "hip"],
        "flip_idx": [1, 0, 2],
        "edges": [[0, 2], [1, 2]],
    }
    document["skeleton"] = schema
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="3 x/y/visibility"):
        load_dataset(source, ("val",), task="pose")
    for label in (source.parent / "labels").glob("*.txt"):
        rows = [" ".join(row.split()[:14]) + "\n" for row in label.read_text().splitlines()]
        label.write_text("".join(rows), encoding="utf-8")
    validated = load_dataset(source, ("val",), task="pose")
    assert validated.skeleton is not None
    assert restore_skeleton(validated.skeleton)["kpt_shape"] == [3, 3]
    schema["joints"][:2] = reversed(schema["joints"][:2])
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert load_dataset(source, ("val",), task="pose").fingerprint != validated.fingerprint
    schema["flip_idx"] = [1, 2, 0]
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="flip_idx"):
        load_dataset(source, ("val",), task="pose")
