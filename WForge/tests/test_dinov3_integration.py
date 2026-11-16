"""Real upstream DINOv3/DETR execution. Random fixtures are not pretrained SSL evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wvision import Vision
from wvision.adapters.shared.process import stop_process_tree
from wvision.datasets.detection import load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.dinov3]
MODEL = "transformers/dinov3-convnext-tiny-detr"


def run(arguments: list[str]) -> dict[str, object]:
    process = subprocess.Popen(
        [sys.executable, "-m", "wvision", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    assert process.returncode == 0, stdout + stderr
    return dict(json.loads(stdout))


def backend_python() -> str:
    backend = os.environ.get("WVISION_TRANSFORMERS_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_TRANSFORMERS_PYTHON to the isolated DINOv3 backend")
    return backend


def test_real_dinov3_train_evaluate_predict_and_reload(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backend_python()
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    output = tmp_path / "runs"
    result = run(
        [
            "train",
            "--model",
            "models/detection/dinov3-convnext-tiny-detr/smoke.yaml",
            "--data",
            str(manifest),
            "--backend-python",
            backend,
            "--output",
            str(output),
        ]
    )
    directory = Path(str(result["artifact_dir"]))
    state = json.loads((directory / "run.json").read_text())
    assert state["model_definition"]["document"]["profile"] == "smoke"
    assert state["selection"] == {"metric": "eval_loss", "direction": "minimize"}
    assert state["best_checkpoint"] == state["last_checkpoint"]
    assert Path(state["best_checkpoint"]).stat().st_size > 100_000_000
    provenance = json.loads((directory / "model-source.json").read_text())
    assert provenance["initialization"] == "random"
    assert provenance["backbone"] == "DINOv3ConvNextBackbone"
    assert provenance["parameter_count"] == 45_537_638
    assert provenance["official_pretrained_verification"] is False
    resolved = json.loads((directory / "config.json").read_text())
    assert resolved["initialization"] == "random"
    assert resolved["initialization_sources"]["backbone"] is None
    assert not provenance["weights_downloaded"] and provenance["backbone_source"] is None
    assert not (directory / "test.coco.json").exists()
    assert not (directory / "train_unlabeled.coco.json").exists()
    scalars = json.loads((directory / "training-scalars.json").read_text())
    assert [item["step"] for item in scalars if "loss" in item] == [1, 2]
    assert "There were missing keys" not in (directory / "backend.log").read_text()
    model = Vision.from_run(directory)
    assert model.val(output=output).metrics["transformers/coco/bbox_mAP"] >= 0
    assert model.test(output=output).metrics["transformers/coco/bbox_mAP"] >= 0
    sample = manifest.parent / "images/test-0.png"
    prediction = model.predict(sample, output=output)
    record = json.loads(prediction.predictions.read_text())
    assert record["width"] == record["height"] == 128
    assert record["classes"] == {"0": "rectangle"}
    assert len(record["boxes"]) == len(record["scores"]) == len(record["class_ids"]) == 100
    assert record["coordinate_format"] == "absolute_pixel_xyxy"
    assert all(label == 0 for label in record["class_ids"])
    for operation, extra in (("val", []), ("test", []), ("predict", ["--source", str(sample)])):
        report = run([operation, "--run", str(directory), "--output", str(output), *extra])
        assert Path(str(report["artifact_dir"])).is_dir()
    cfg = directory / "resolved-transformers.json"
    cfg.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="model configuration content has changed"):
        model.test(output=output)


def test_real_local_dinov3_backbone_loading_and_freezing(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = backend_python()
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    weights = tmp_path / "backbone"
    code = (
        "import sys,torch; "
        "from transformers import DINOv3ConvNextConfig,DINOv3ConvNextModel,set_seed; "
        "torch.set_num_threads(2); set_seed(17); "
        "DINOv3ConvNextModel(DINOv3ConvNextConfig()).save_pretrained(sys.argv[1])"
    )
    subprocess.run(
        [backend, "-c", code, str(weights)], check=True, capture_output=True, timeout=180
    )
    (weights / "wvision-source.json").write_text(
        json.dumps(
            {
                "source": "owned upstream random-initialization fixture",
                "supervision": "random",
                "pretraining_data": None,
                "license": "upstream code terms",
            }
        ),
        encoding="utf-8",
    )
    model = Vision(MODEL, backend_python=backend)
    trained = model.train(
        manifest,
        backbone_weights=weights,
        max_iters=2,
        imgsz=64,
        batch=2,
        freeze_backbone=True,
        output=tmp_path / "runs",
    )
    source = json.loads((trained.artifact_dir / "model-source.json").read_text())
    assert source["freeze_backbone"] is True
    assert source["backbone_source"]["provenance"]["supervision"] == "random"
    assert source["trainable_parameters"] < source["parameter_count"]
    transfer = json.loads((trained.artifact_dir / "transfer.json").read_text())
    assert transfer["tensor_count"] > 150
    assert transfer["all_loaded_tensors_equal"] and transfer["all_keys_and_shapes_match"]
    # Confirm that the actual saved detector still has the complete frozen source backbone.
    verification = (
        "import sys,torch; from transformers import DINOv3ConvNextBackbone,DetrForObjectDetection; "
        "torch.set_num_threads(2); "
        "a=DINOv3ConvNextBackbone.from_pretrained(sys.argv[1],local_files_only=True).state_dict(); "
        "b=DetrForObjectDetection.from_pretrained(sys.argv[2],local_files_only=True)"
        ".model.backbone.model.state_dict(); "
        "assert a.keys()==b.keys(); assert all(torch.equal(a[k],b[k]) for k in a)"
    )
    subprocess.run(
        [backend, "-c", verification, str(weights), str(trained.best_checkpoint.parent)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    assert (
        model.test(output=tmp_path / "runs").dataset_fingerprint
        == load_dataset(manifest, ()).fingerprint
    )
    # Exercise a checkpoint lacking most backbone tensors. It must fail, never fill them randomly.
    damage = (
        "import sys; from safetensors.torch import load_file,save_file; "
        "p=sys.argv[1]; d=load_file(p); k=next(iter(d)); save_file({k:d[k]},p)"
    )
    subprocess.run(
        [backend, "-c", damage, str(weights / "model.safetensors")],
        check=True,
        capture_output=True,
        timeout=180,
    )
    with pytest.raises(RuntimeError, match="backend.log"):
        model.train(
            manifest,
            backbone_weights=weights,
            max_iters=1,
            imgsz=64,
            batch=2,
            output=tmp_path / "failed",
        )
    failed = next((tmp_path / "failed").glob("train-*"))
    assert json.loads((failed / "status.json").read_text())["status"] == "failed"


def test_official_pretrained_dinov3_when_supplied(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate acceptance gate; never substitute an owned random fixture for official weights."""
    backend = backend_python()
    weights = os.environ.get("WVISION_DINOV3_WEIGHTS")
    if not weights:
        pytest.skip("Official licensed DINOv3 weights unavailable; set WVISION_DINOV3_WEIGHTS")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    model = Vision(MODEL, backend_python=backend)
    trained = model.train(
        manifest,
        backbone_weights=weights,
        max_iters=2,
        imgsz=64,
        batch=2,
        freeze_backbone=True,
        output=tmp_path / "runs",
    )
    assert model.val(output=tmp_path / "runs").metrics["transformers/coco/bbox_mAP"] >= 0
    assert model.test(output=tmp_path / "runs").metrics["transformers/coco/bbox_mAP"] >= 0
    restored = Vision.from_run(trained.artifact_dir)
    result = restored.predict(manifest.parent / "images/test-0.png", output=tmp_path / "runs")
    assert result.sample_count == 1
