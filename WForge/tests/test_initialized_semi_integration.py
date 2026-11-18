"""Actual upstream initialized Soft Teacher, including both detector stride styles."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision, prepare_label_budget, pretrain
from wvision.adapters.shared.process import stop_process_tree
from wvision.datasets.detection import sha256

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection]
MODEL = "mmdetection/faster-rcnn-r50"


def cli(arguments: list[str]) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-m", "wvision", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=1200)
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    assert process.returncode == 0, stdout + stderr
    return dict(json.loads(stdout))


@pytest.mark.parametrize(
    "source_kind,style", [("detector", "caffe"), ("detector", "pytorch"), ("backbone", "pytorch")]
)
def test_real_initialized_soft_teacher(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    style: str,
) -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_PYTHON to the verified isolated MMDetection backend")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2, seed=0)
    output = tmp_path / "runs"
    if source_kind == "detector":
        source = Vision(MODEL, backend_python=backend).train(
            data,
            max_iters=2,
            imgsz=128,
            batch=2,
            backend_options={"backbone_style": style, "lr": 0.000001},
            output=output,
        )
        source_run, checkpoint = source.artifact_dir, source.best_checkpoint
    else:
        representation = pretrain(
            data,
            model="models/representation/simsiam-r50/smoke.yaml",
            backend_python=backend,
            output=output,
        )
        source_run, checkpoint = representation.artifact_dir, representation.checkpoint
    source_hash = sha256(checkpoint)
    profile = (
        "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml"
        if style == "caffe"
        else "models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml"
    )
    if style == "caffe":
        result = cli(
            [
                "train",
                "--model",
                profile,
                "--backend-python",
                backend,
                "--data",
                str(data),
                "--detector-run",
                str(source_run),
                "--output",
                str(output),
            ]
        )
        directory = Path(result["artifact_dir"])
    else:
        initial = (
            {"detector_run": source_run}
            if source_kind == "detector"
            else {"pretraining_run": source_run}
        )
        trained = Vision(profile, backend_python=backend).train(data, output=output, **initial)
        directory = trained.artifact_dir
    transfer = json.loads((directory / "transfer.json").read_text())
    assert transfer["source_checkpoint_sha256"] == source_hash
    assert transfer["scope"] == source_kind and transfer["optimizer_state_loaded"] is False
    assert set(transfer["branches"]) == {"teacher", "student"}
    assert all(branch["all_loaded_tensors_equal"] for branch in transfer["branches"].values())
    assert transfer["all_keys_and_shapes_match"] and transfer["audit_stage"].startswith(
        "before_train"
    )
    resolved = json.loads((directory / "resolved-mmdet.json").read_text())
    detector_preprocessor = resolved["model"]["detector"]["data_preprocessor"]
    assert resolved["model"]["data_preprocessor"]["data_preprocessor"] == detector_preprocessor
    assert detector_preprocessor["bgr_to_rgb"] is (style == "pytorch")
    assert resolved["resume"] is False and resolved["load_from"] is None
    logs = json.loads((directory / "training-scalars.json").read_text())
    assert any(value > 0 for key, value in logs.items() if "unsup_loss" in key)
    assert json.loads((directory / "train_unlabeled.coco.json").read_text())["annotations"] == []
    assert not (directory / "test.coco.json").exists()
    state = json.loads((directory / "run.json").read_text())
    assert state["initialization"]["compatibility"]["scope"] == source_kind
    assert state["selection"]["metric"] == "teacher/coco/bbox_mAP"
    model = Vision.from_run(directory)
    assert model.val(output=output).metrics["mmdetection/coco/bbox_mAP"] >= 0
    assert model.test(output=output).metrics["mmdetection/coco/bbox_mAP"] >= 0
    sample = data.parent / "test/images/0.png"
    assert model.predict(sample, output=output).sample_count == 1
    if style != "caffe":
        return
    # A validly shaped detector with different class semantics must not be accepted.
    damage_classes = (
        "import sys,torch; p=sys.argv[1]; c=torch.load(p,map_location='cpu'); "
        "c['meta']['dataset_meta']['classes']=('wrong-class',); torch.save(c,p)"
    )
    subprocess.run(
        [backend, "-c", damage_classes, str(checkpoint)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    source_state = json.loads((source_run / "run.json").read_text())
    source_state["best_checkpoint_sha256"] = sha256(checkpoint)
    (source_run / "run.json").write_text(json.dumps(source_state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend.log"):
        Vision(profile, backend_python=backend).train(
            data, detector_run=source_run, output=tmp_path / "bad-class"
        )
    failed = next((tmp_path / "bad-class").glob("train-*"))
    assert "class map differs" in (failed / "backend.log").read_text()
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"
    damage_keys = (
        "import sys,torch; p=sys.argv[1]; c=torch.load(p,map_location='cpu'); "
        "c['meta']['dataset_meta']['classes']=('rectangle',); "
        "c['state_dict'].pop(next(iter(c['state_dict']))); torch.save(c,p)"
    )
    subprocess.run(
        [backend, "-c", damage_keys, str(checkpoint)], check=True, capture_output=True, timeout=180
    )
    source_state["best_checkpoint_sha256"] = sha256(checkpoint)
    (source_run / "run.json").write_text(json.dumps(source_state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="backend.log"):
        Vision(profile, backend_python=backend).train(
            data, detector_run=source_run, output=tmp_path / "bad-keys"
        )
    failed = next((tmp_path / "bad-keys").glob("train-*"))
    assert "keys/shapes" in (failed / "backend.log").read_text()
    # A trained target is independent of later source damage.
    assert Vision.from_run(directory).predict(sample, output=output).sample_count == 1
