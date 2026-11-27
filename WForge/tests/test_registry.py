"""Adapter seam checks; mocked boundaries do not establish model support."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision, list_models
from wvision.adapters.base import Capability, ModelSpec
from wvision.core import registry
from wvision.models.transformers.dinov3_convnext_tiny_detr.adapter import DinoDETRAdapter
from wvision.models.transformers.dinov3_convnext_tiny_detr.config import (
    DinoDETRConfig,
    local_backbone,
)
from wvision.models.ultralytics.yolo11 import operations as ultralytics


def test_catalog_and_cli_are_framework_free() -> None:
    code = (
        "import sys; from wvision import Vision, list_models; "
        "Vision('ultralytics/yolo11n'); Vision('ultralytics/yolo11n-pose'); "
        "Vision('mmpose/rtmpose-t', backend_python=sys.executable); "
        "Vision('mmdetection/faster-rcnn-r50', backend_python=sys.executable); "
        "Vision('transformers/dinov3-convnext-tiny-detr', backend_python=sys.executable); "
        "assert not {'torch','transformers','mmdet','mmpose','ultralytics'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run(
        [sys.executable, "-m", "wvision", "models"], capture_output=True, text=True, check=True
    )
    assert {item["model_id"] for item in json.loads(result.stdout)} == {
        spec.model_id for spec in list_models()
    }


def test_joint_capability_gate() -> None:
    spec = ModelSpec(
        "example/model", "example", "detect", (Capability("supervised", "predict"),), "test only"
    )
    spec.require("supervised", "predict")
    for learning, operation, device in (
        ("supervised", "train", "cpu"),
        ("semi_supervised", "predict", "cpu"),
        ("supervised", "predict", "cuda:0"),
    ):
        with pytest.raises(ValueError, match="not integrated"):
            spec.require(learning, operation, device)
    with pytest.raises(ValueError, match="representation model"):
        Vision("mmpretrain/simsiam-r50")


def test_catalog_extension_uses_existing_facade(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A new reviewed catalog entry can reuse an adapter without any Vision/CLI branch.
    spec = replace(registry.get_model("ultralytics/yolo11n"), model_id="test/yolo11n")
    monkeypatch.setattr(
        registry,
        "_MODELS",
        (*registry._MODELS, (spec, "wvision.models.ultralytics.yolo11.adapter:create_adapter")),
    )
    monkeypatch.setattr(ultralytics, "check_dependency", lambda: None)
    called = []

    def train(
        checkpoint: Any, data: Path, settings: Any, directory: Path, *, task: str, scale: str
    ) -> dict[str, Any]:
        assert task == "detect" and scale == "n"
        called.append("train")
        weights = directory / "best.pt"
        weights.write_bytes(b"fixture checkpoint, not a model")
        return {"best": weights, "last": weights, "metrics": {"fixture/loss": 1.0}}

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, float]:
        called.append("evaluate")
        return {"fixture/AP": 0.0}

    def predict(
        checkpoint: Path, samples: list[str], directory: Path, imgsz: int, **kwargs: Any
    ) -> int:
        called.append("predict")
        (directory / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
        return len(samples)

    monkeypatch.setattr(ultralytics, "train", train)
    monkeypatch.setattr(ultralytics, "evaluate", evaluate)
    monkeypatch.setattr(ultralytics, "predict", predict)
    run = Vision(spec.model_id).train(manifest, output=tmp_path / "runs", imgsz=128, batch=2)
    restored = Vision.from_run(run.artifact_dir)
    assert restored.val(output=tmp_path / "runs").split == "val"
    assert restored.test(output=tmp_path / "runs").split == "test"
    assert (
        restored.predict(
            manifest.parent / "images/test-0.png", output=tmp_path / "runs"
        ).sample_count
        == 1
    )
    assert called == ["train", "evaluate", "evaluate", "predict"]


@pytest.mark.parametrize(
    "options",
    [
        {"learning": "self_supervised"},
        {"device": "mps"},
        {"initialization": "auto"},
        {"freeze_backbone": "yes"},
        {"max_iters": True},
        {"imgsz": 33},
        {"lr": float("nan")},
        {"epochs": 2},
    ],
)
def test_invalid_dinov3_settings(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DinoDETRConfig.resolve(options)


def test_dinov3_initialization_never_falls_back(manifest: Path) -> None:
    model = Vision("transformers/dinov3-convnext-tiny-detr", backend_python=sys.executable)
    with pytest.raises(ValueError, match="Provide backbone_weights"):
        model.train(manifest)
    with pytest.raises(ValueError, match="cannot also receive"):
        model.train(manifest, initialization="random", backbone_weights="missing")
    with pytest.raises(ValueError, match="not checkpoint/pretraining_run"):
        model.train(manifest, initialization="random", pretraining_run="missing")


def test_initialization_provenance_does_not_overwrite_adapter_settings(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inspect orchestration artifacts without claiming model execution support.
    def invoke(self: DinoDETRAdapter, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        assert request["config"]["initialization"] == "random" and request["backbone"] is None
        checkpoint = directory / "fixture.safetensors"
        checkpoint.write_bytes(b"boundary fixture")
        config = directory / "model.json"
        config.write_text("{}", encoding="utf-8")
        return {
            "best": str(checkpoint),
            "last": str(checkpoint),
            "metrics": {},
            "model_config": str(config),
        }

    monkeypatch.setattr(DinoDETRAdapter, "invoke", invoke)
    result = Vision(
        "models/detection/dinov3-convnext-tiny-detr/smoke.yaml", backend_python=sys.executable
    ).train(manifest, output=tmp_path / "runs")
    resolved = json.loads((result.artifact_dir / "config.json").read_text())
    state = json.loads((result.artifact_dir / "run.json").read_text())
    assert resolved["initialization"] == state["config"]["initialization"] == "random"
    assert resolved["initialization_sources"] == state["initialization"] == {"backbone": None}


def test_local_backbone_rejects_wrong_architecture_and_executable_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires config.json"):
        local_backbone(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    values = {
        "model_type": "dinov3_convnext",
        "hidden_sizes": [96, 192, 384, 768],
        "depths": [3, 3, 9, 3],
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**values, "auto_map": {"AutoModel": "custom.py"}}))
    with pytest.raises(ValueError, match="Only the upstream"):
        local_backbone(tmp_path)
    config.write_text(json.dumps(values))
    snapshot = local_backbone(tmp_path)
    assert snapshot["provenance"]["supervision"] == "unknown"
    assert snapshot["provenance_verified"] is False
    assert set(snapshot["files"]) == {"config.json", "model.safetensors"}


@pytest.mark.parametrize(
    "worker",
    [
        "mmdetection/faster_rcnn_r50/operations.py",
        "mmpretrain/simsiam_r50/operations.py",
        "transformers/dinov3_convnext_tiny_detr/operations.py",
    ],
)
def test_worker_helper_imports_do_not_shadow_upstream_packages(worker: str) -> None:
    package = Path(__file__).resolve().parents[1] / "src/wvision"
    code = """
import importlib.util
import runpy
import sys
from pathlib import Path

worker = Path(sys.argv[1])
package = worker.parents[3]
adapters = package / "adapters"
sys.path.insert(0, str(worker.parent))
runpy.run_path(str(worker), run_name="_worker_import_probe")
assert Path(sys.modules["devices"].__file__) == adapters / "shared/devices.py"
assert str(adapters) not in sys.path
assert str(package / "models") not in sys.path
assert not {"wvision", "torch", "mmdet", "mmpretrain"} & sys.modules.keys()
for name in ("transformers", "ultralytics"):
    spec = importlib.util.find_spec(name)
    assert spec is None or not Path(spec.origin).is_relative_to(package)
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(package / "models" / worker)], check=True)
