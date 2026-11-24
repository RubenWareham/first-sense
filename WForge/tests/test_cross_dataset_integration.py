"""Actual upstream detector/backbone transfer across datasets and target class sets."""

import json
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from examples.make_tiny_dataset import create_dataset
from wvision import Vision, prepare_label_budget, pretrain
from wvision.adapters.shared.checkpoints import checkpoint_files

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection]


def test_real_cross_dataset_transfer(manifest: Path, tmp_path: Path) -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_PYTHON to the verified backend")
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    settings = {
        "max_iters": 2,
        "imgsz": 128,
        "batch": 2,
        "device": device,
        "backend_options": {"backbone_style": "pytorch", "lr": 0.000001},
    }
    model_id = "mmdetection/faster-rcnn-r50"
    source = Vision(model_id, backend_python=backend).train(manifest, output=tmp_path, **settings)
    original = checkpoint_files(source.artifact_dir)
    target = create_dataset(tmp_path / "target")
    for image in (target.parent / "images").iterdir():
        with Image.open(image) as decoded:
            decoded.putpixel((0, 0), (254, 0, 254))
            decoded.save(image)
    full = Vision(model_id, backend_python=backend).train(
        target, detector_run=source.artifact_dir, output=tmp_path, **settings
    )
    audit = json.loads((full.artifact_dir / "transfer.json").read_text())
    assert audit["scope"] == "detector" and audit["all_loaded_tensors_equal"]
    assert audit["data_compatibility"]["cross_dataset"]
    config = yaml.safe_load(target.read_text())
    config["classes"] = {0: "target-object", 1: "other-object"}
    target.write_text(yaml.safe_dump(config))
    budget = prepare_label_budget(target, tmp_path / "target-budget", labeled_count=2)
    for learning, data in [("supervised", target), ("semi_supervised", budget)]:
        trained = Vision(model_id, backend_python=backend).train(
            data,
            learning=learning,
            detector_run=source.artifact_dir,
            transfer_scope="backbone",
            output=tmp_path,
            **settings,
        )
        audit = json.loads((trained.artifact_dir / "transfer.json").read_text())
        assert audit["scope"] == "backbone" and audit["all_loaded_tensors_equal"]
        assert audit["detection_heads"] == "random initialization"
        assert audit["data_compatibility"]["cross_dataset"]
        assert set(audit["branches"]) == (
            {"detector"} if learning == "supervised" else {"teacher", "student"}
        )
        restored = Vision.from_run(trained.artifact_dir)
        sample = next(
            (data.parent / ("images" if learning == "supervised" else "test/images")).glob("*.png")
        )
        prediction = restored.predict(sample, device="cpu", output=tmp_path)
        assert prediction.sample_count == 1
        row = json.loads(prediction.predictions.read_text().splitlines()[0])
        assert row["classes"] == {"0": "target-object", "1": "other-object"}
    source_budget = prepare_label_budget(manifest, tmp_path / "source-budget", labeled_count=2)
    representation = pretrain(
        source_budget,
        model="models/representation/simsiam-r50/smoke.yaml",
        backend_python=backend,
        max_iters=2,
        imgsz=64,
        batch=2,
        device=device,
        output=tmp_path,
    )
    represented = Vision(model_id, backend_python=backend).train(
        target, pretraining_run=representation.artifact_dir, output=tmp_path, **settings
    )
    audit = json.loads((represented.artifact_dir / "transfer.json").read_text())
    assert audit["data_compatibility"]["cross_dataset"] and audit["all_loaded_tensors_equal"]
    assert checkpoint_files(source.artifact_dir) == original
