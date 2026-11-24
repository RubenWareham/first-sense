"""Framework-free initialization rules and source-provenance rejection checks."""

import json
import sys
from pathlib import Path

import pytest

from wvision import Vision
from wvision.adapters.base import Initialization
from wvision.core.registry import get_model
from wvision.datasets.detection import load_dataset, sha256
from wvision.experiments.artifacts import write_json
from wvision.models.mmdetection.faster_rcnn_r50.weights import detector_source

MODEL = "mmdetection/faster-rcnn-r50"


def test_initialization_rules_are_specific_to_source_scope_and_method() -> None:
    model = get_model(MODEL)
    rule = model.require_initialization(
        "mmpretrain/simsiam-r50", "pretraining_run", "semi_supervised", "pytorch"
    )
    assert rule.scope == "backbone"
    assert (
        model.require_initialization(MODEL, "detector_run", "semi_supervised").scope == "detector"
    )
    with pytest.raises(ValueError, match="backbone_style=pytorch"):
        model.require_initialization(
            "mmpretrain/simsiam-r50", "pretraining_run", "semi_supervised", "caffe"
        )
    for source, kind, learning in (("ultralytics/yolo11n", "detector_run", "semi_supervised"),):
        with pytest.raises(ValueError, match="not integrated"):
            model.require_initialization(source, kind, learning)


def test_only_one_initializer_is_accepted() -> None:
    with pytest.raises(ValueError, match="only one"):
        Initialization(pretraining_run=Path("ssl"), detector_run=Path("detector"))


@pytest.mark.parametrize("model", ["ultralytics/yolo11n", "transformers/dinov3-convnext-tiny-detr"])
def test_unsupported_detector_initialization_fails_before_reading_source(
    manifest: Path, model: str
) -> None:
    backend = None if model.startswith("ultralytics/") else sys.executable
    with pytest.raises(ValueError, match="detector_run"):
        Vision(model, backend_python=backend).train(manifest, detector_run="missing")


@pytest.fixture
def detector_artifact(manifest: Path, tmp_path: Path) -> tuple[Path, Path]:
    dataset = load_dataset(manifest, ())
    root = tmp_path / "source-run"
    root.mkdir()
    checkpoint, config = root / "best.pth", root / "model.json"
    checkpoint.write_bytes(b"metadata boundary fixture, not a model")
    write_json(config, {"model": {"type": "FasterRCNN"}})
    write_json(root / "status.json", {"status": "completed"})
    write_json(root / "dataset.json", dataset.snapshot())
    write_json(root / "backend-environment.json", {"packages": {"mmdet": "3.3.0"}})
    write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "model": MODEL,
            "dataset_fingerprint": dataset.fingerprint,
            "config": {"learning": "supervised", "backend_options": {}},
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": sha256(checkpoint),
            "model_config": str(config),
            "model_config_sha256": sha256(config),
        },
    )
    return root, manifest


@pytest.mark.parametrize(
    "damage",
    [
        "status",
        "model",
        "learning",
        "dataset",
        "classes",
        "style",
        "version",
        "checkpoint",
        "config",
    ],
)
def test_detector_artifact_incompatibilities_are_rejected(
    detector_artifact: tuple[Path, Path],
    damage: str,
) -> None:
    root, manifest = detector_artifact
    dataset = load_dataset(manifest, ())
    assert detector_source(root, dataset, "caffe")["scope"] == "complete_detector"
    state = json.loads((root / "run.json").read_text())
    if damage == "status":
        write_json(root / "status.json", {"status": "failed"})
    elif damage == "model":
        state["model"] = "ultralytics/yolo11n"
    elif damage == "learning":
        state["config"]["learning"] = "semi_supervised"
    elif damage == "dataset":
        state["dataset_fingerprint"] = "different"
    elif damage == "classes":
        write_json(root / "dataset.json", {**dataset.snapshot(), "classes": {0: "different"}})
    elif damage == "style":
        state["config"]["backend_options"] = {"backbone_style": "pytorch"}
    elif damage == "version":
        write_json(root / "backend-environment.json", {"packages": {"mmdet": "other"}})
    elif damage == "checkpoint":
        (root / "best.pth").write_bytes(b"changed")
    else:
        (root / "model.json").write_text("{}", encoding="utf-8")
    write_json(root / "run.json", state)
    with pytest.raises(ValueError):
        detector_source(root, dataset, "caffe")


def test_detector_cross_dataset_and_new_class_backbone_transfer(
    detector_artifact: tuple[Path, Path], tmp_path: Path
) -> None:
    import yaml
    from PIL import Image

    from examples.make_tiny_dataset import create_dataset

    root, _ = detector_artifact
    manifest = create_dataset(tmp_path / "different")
    for image in (manifest.parent / "images").iterdir():
        with Image.open(image) as source_image:
            source_image.putpixel((0, 0), (254, 0, 254))
            source_image.save(image)
    target = load_dataset(manifest, ())
    assert detector_source(root, target, "caffe")["transfer_data"]["cross_dataset"]
    config = yaml.safe_load(manifest.read_text())
    config["classes"] = {0: "new-class", 1: "second-class"}
    manifest.write_text(yaml.safe_dump(config))
    target = load_dataset(manifest, ())
    with pytest.raises(ValueError, match="class map"):
        detector_source(root, target, "caffe")
    source = detector_source(root, target, "caffe", "backbone")
    assert source["scope"] == "backbone" and source["transfer_data"]["cross_dataset"]
