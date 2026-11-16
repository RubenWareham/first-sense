"""Real upstream representation training, backbone transfer, and held-out evaluation."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wvision import Vision, prepare_label_budget
from wvision.adapters.shared.process import stop_process_tree
from wvision.datasets.detection import load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection]


def command(arguments: list[str]) -> dict[str, str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "wvision", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=1800)
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    assert process.returncode == 0, stdout + stderr
    return dict(json.loads(stdout))


def test_real_simsiam_and_detection_transfer(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_PYTHON to the isolated MMPreTrain/MMDetection environment")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    derived = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2, seed=0)
    output = tmp_path / "runs"
    pretrained = command(
        [
            "pretrain",
            "--data",
            str(derived),
            "--backend-python",
            backend,
            "--model",
            "models/representation/simsiam-r50/smoke.yaml",
            "--output",
            str(output),
        ]
    )
    pretraining_run = Path(pretrained["artifact_dir"])
    state = json.loads((pretraining_run / "run.json").read_text())
    assert state["model_definition"]["document"]["task"] == "representation"
    dataset = load_dataset(derived, ("test",))
    assert state["selection"] == "final_iteration"
    assert state["training_images"] == [s.pixel_hash for s in dataset.splits["train_unlabeled"]]
    assert len(list((pretraining_run / "data").rglob("*.png"))) == 2
    assert not list((pretraining_run / "data").rglob("*.txt"))
    assert list((pretraining_run / "data").iterdir())[0].name == "train_unlabeled"
    request = json.loads((pretraining_run / "backend-request.json").read_text())
    assert "classes" not in request and "data" not in request
    resolved = json.loads((pretraining_run / "resolved-mmpretrain.json").read_text())
    assert resolved["model"]["type"] == "SimSiam"
    assert resolved["train_dataloader"]["dataset"]["with_label"] is False
    assert resolved["val_dataloader"] is None and resolved["test_dataloader"] is None
    assert resolved["model"]["backbone"]["init_cfg"] is None
    loss = json.loads((pretraining_run / "metrics.json").read_text())["mmpretrain/train/loss"]
    assert math.isfinite(loss)
    comparison = command(
        [
            "compare-pretraining",
            "--data",
            str(derived),
            "--pretraining-run",
            str(pretraining_run),
            "--backend-python",
            backend,
            "--model",
            "models/detection/faster-rcnn-r50/transfer-smoke.yaml",
            "--output",
            str(output),
        ]
    )
    report = json.loads(Path(comparison["report"]).read_text())
    assert report["dataset_fingerprint"] == dataset.fingerprint
    assert report["supervision"]["fully_unsupervised_detection"] is False
    assert report["label_budget"]["labeled_count"] == 2
    assert report["seed_count"] == 1
    for name in ("random", "pretrained"):
        assert report["test"][name]["mmdetection/coco/bbox_mAP"] >= 0
        run = Path(report["runs"][name])
        assert not (run / "test.coco.json").exists()
        assert not (run / "train_unlabeled.coco.json").exists()
        cfg = json.loads((run / "resolved-mmdet.json").read_text())
        assert cfg["model"]["backbone"]["style"] == "pytorch"
        assert cfg["model"]["data_preprocessor"]["bgr_to_rgb"] is True
    transfer_run = Path(comparison["pretrained_run"])
    transfer = json.loads((transfer_run / "transfer.json").read_text())
    assert transfer["all_keys_and_shapes_match"] and transfer["all_loaded_tensors_equal"]
    assert transfer["tensor_count"] > 300
    assert transfer["source_checkpoint_sha256"] == state["checkpoint_sha256"]
    model = Vision.from_run(transfer_run)
    prediction = model.predict(dataset.splits["test"][0].path, output=output)
    assert prediction.sample_count == 1
    assert json.loads(prediction.predictions.read_text())["classes"] == {"0": "rectangle"}
    # Reloading the detector must not depend on the source pretraining checkpoint.
    # Changing it is still detected if another downstream training is requested.
    Path(pretrained["checkpoint"]).write_bytes(b"tampered test artifact")
    tested = command(["test", "--run", str(transfer_run), "--output", str(output)])
    assert Path(tested["artifact_dir"]).is_dir()
    with pytest.raises(ValueError, match="checkpoint content has changed"):
        Vision(
            "models/detection/faster-rcnn-r50/transfer-smoke.yaml", backend_python=backend
        ).train(
            derived,
            pretraining_run=pretraining_run,
            output=output,
        )
