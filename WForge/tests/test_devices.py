"""Framework-free device policy and operation propagation checks."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from wvision import Vision
from wvision.adapters.shared.devices import canonical_device, isolate_worker_device, runtime_device
from wvision.adapters.shared.process import invoke_worker
from wvision.core.registry import get_model
from wvision.models.mmdetection.faster_rcnn_r50 import adapter as mmdetection
from wvision.models.mmpretrain.simsiam_r50.config import PretrainConfig
from wvision.models.transformers.dinov3_convnext_tiny_detr.config import DinoDETRConfig
from wvision.models.ultralytics.yolo11.config import TrainConfig


@pytest.mark.parametrize(
    "value,expected",
    [("cpu", "cpu"), ("cuda", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:2", "cuda:2")],
)
def test_explicit_device_syntax(value: str, expected: str) -> None:
    assert canonical_device(value) == expected
    assert mmdetection.MMDetConfig(device=value).device == expected
    assert PretrainConfig(device=value).device == expected
    assert TrainConfig(device=value).device == expected
    assert DinoDETRConfig(device=value).device == expected


@pytest.mark.parametrize(
    "value", [None, True, 0, "auto", "mps", "0", "cuda:-1", "cuda:01", "cuda:0,1", "0,1"]
)
def test_unsupported_device_syntax(value: Any) -> None:
    with pytest.raises(ValueError, match="device must"):
        canonical_device(value)


def test_verified_models_expose_cuda_capabilities() -> None:
    get_model("mmdetection/faster-rcnn-r50").require("semi_supervised", "train", "cuda:0")
    get_model("mmpretrain/simsiam-r50").require("self_supervised", "pretrain", "cuda:1")
    for model in ("ultralytics/yolo11n", "transformers/dinov3-convnext-tiny-detr"):
        get_model(model).require("supervised", "train", "cuda:0")
        with pytest.raises(ValueError, match="not integrated"):
            get_model(model).require("semi_supervised", "train", "cuda:0")


@pytest.mark.parametrize(
    "mask,requested,selected",
    [(None, "cuda:2", "2"), ("2,3", "cuda:1", "3"), ("GPU-test", "cuda:0", "GPU-test")],
)
def test_worker_gpu_mapping_preserves_the_requested_visible_index(
    mask: str | None, requested: str, selected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setenv("ACCELERATE_TORCH_DEVICE", "cpu")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    assert isolate_worker_device(requested) == "cuda:0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == selected
    assert os.environ["ACCELERATE_TORCH_DEVICE"] == "cuda:0"
    assert os.environ["ACCELERATE_USE_CPU"] == "false"


@pytest.mark.parametrize("mask", ["", "-1"])
def test_disabled_worker_cuda_is_not_reenabled(mask: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        isolate_worker_device("cuda:0")


@pytest.mark.parametrize("name,value", [("WORLD_SIZE", "2"), ("LOCAL_RANK", "0")])
def test_distributed_worker_launches_are_rejected(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="Distributed launches"):
        isolate_worker_device("cuda:0")


def test_unavailable_cuda_has_no_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = SimpleNamespace(
        __version__="boundary-fixture",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        runtime_device("cuda:0", tmp_path)
    assert not (tmp_path / "device.json").exists()


@pytest.mark.parametrize("device,visible", [("cpu", "-1"), ("cuda:0", "2,3")])
def test_worker_device_environment_is_explicit_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, visible: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    worker = tmp_path / "environment.py"
    worker.write_text(
        "import json,os,sys; "
        "json.dump({'schema_version':1,'visible':os.environ['CUDA_VISIBLE_DEVICES']},"
        "open(sys.argv[2],'w'))",
        encoding="utf-8",
    )
    result = invoke_worker(Path(sys.executable), worker, tmp_path, {"device": device}, "fixture")
    assert result["visible"] == visible
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_saved_device_and_explicit_cpu_override(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    def invoke(python: Path, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        if request["operation"] == "train":
            checkpoint, config = directory / "fixture.pth", directory / "fixture.json"
            checkpoint.write_bytes(b"metadata boundary fixture; not a model")
            config.write_text("{}", encoding="utf-8")
            return {
                "best": str(checkpoint),
                "last": str(checkpoint),
                "model_config": str(config),
                "metrics": {},
            }
        if request["operation"] == "predict":
            (directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
            return {"sample_count": 1}
        return {"metrics": {"fixture/AP": 0.0}}

    monkeypatch.setattr(mmdetection, "invoke", invoke)
    result = Vision("mmdetection/faster-rcnn-r50", backend_python=sys.executable).train(
        manifest, device="cuda", output=tmp_path / "runs"
    )
    restored = Vision.from_run(result.artifact_dir)
    validation = restored.val(device="cpu", output=tmp_path / "runs")
    restored.test(output=tmp_path / "runs")
    restored.predict(manifest.parent / "images/test-0.png", device="cpu", output=tmp_path / "runs")
    assert requests[0]["config"]["device"] == "cuda:0"
    assert [r["device"] for r in requests[1:]] == ["cpu", "cuda:0", "cpu"]
    assert json.loads((validation.artifact_dir / "config.json").read_text())["device"] == "cpu"
    assert (
        json.loads((result.artifact_dir / "run.json").read_text())["config"]["device"] == "cuda:0"
    )
