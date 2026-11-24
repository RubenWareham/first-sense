"""Actual Faster R-CNN / Soft Teacher ONNX inference, without custom runtime ops."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from wvision import Vision, prepare_label_budget
from wvision.adapters.openmmlab import runtime
from wvision.adapters.shared import checkpoints
from wvision.adapters.shared.checkpoints import checkpoint_files
from wvision.models.mmdetection.faster_rcnn_r50 import adapter

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection]


@pytest.mark.parametrize("style", ["caffe", "pytorch"])
@pytest.mark.parametrize("learning", ["supervised", "semi_supervised"])
def test_mmdet_onnx_export(manifest: Path, tmp_path: Path, style: str, learning: str) -> None:
    backend = os.environ.get("WVISION_MMDEPLOY_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDEPLOY_PYTHON to the compatible MMDeploy environment")
    document = yaml.safe_load(manifest.read_text())
    document["classes"][1] = "second-object"
    manifest.write_text(yaml.safe_dump(document))
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    profile = (
        "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml"
        if learning == "semi_supervised"
        else "models/detection/faster-rcnn-r50/supervised-smoke.yaml"
    )
    trained = Vision(profile, backend_python=backend).train(
        data,
        device=os.environ.get("WVISION_TEST_DEVICE", "cpu"),
        backend_options={"backbone_style": style, "lr": 0.000001},
        output=tmp_path,
    )
    before = checkpoint_files(trained.artifact_dir)
    images = sorted((data.parent / "val/images").glob("*.png"))
    restored = Vision.from_run(trained.artifact_dir, backend_python=backend)
    exported = restored.export(verification_images=images, output=tmp_path)
    record = json.loads(exported.manifest.read_text())
    assert record["source_branch"] == ("teacher" if learning == "semi_supervised" else "detector")
    assert record["source_state_verified"] and record["custom_op_domains"] == []
    assert record["classes"] == {"0": "rectangle", "1": "second-object"}
    assert record["verification"]["samples"] == 2
    assert sum(case["detections"] for case in record["verification"]["cases"]) > 0
    assert checkpoint_files(trained.artifact_dir) == before


def test_mmdet_export_parity_rejects_corruption() -> None:
    backend = os.environ.get("WVISION_MMDEPLOY_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDEPLOY_PYTHON to the compatible MMDeploy environment")
    # Exercise the real backend checker, including padding and reordered detections.
    code = """
from types import SimpleNamespace
import numpy as np
import torch
from deployment import compare_detections
expected = SimpleNamespace(
    bboxes=torch.tensor([[1., 2., 5., 6.], [3., 4., 8., 9.]]),
    scores=torch.tensor([.8, .7]), labels=torch.tensor([0, 1]))
valid = np.array([[3., 4., 8., 9., .7], [1., 2., 5., 6., .8], [0., 0., 0., 0., 0.]])
labels = np.array([1, 0, 0])
assert compare_detections(expected, valid, labels)['detections'] == 2
cases = []
boxes = valid.copy(); boxes[0, 0] += 1; cases.append((boxes, labels))
scores = valid.copy(); scores[0, 4] = .5; cases.append((scores, labels))
classes = labels.copy(); classes[0] = 0; cases.append((valid, classes))
cases.append((valid[1:], labels[1:]))
nonfinite = valid.copy(); nonfinite[2, 4] = np.nan; cases.append((nonfinite, labels))
for dets, classes in cases:
    try:
        compare_detections(expected, dets, classes)
    except ValueError as error:
        assert 'ONNX parity failed' in str(error)
    else:
        raise AssertionError('Corrupted output was accepted')
"""
    paths = [Path(adapter.__file__).parent]
    paths.extend([Path(checkpoints.__file__).parent, Path(runtime.__file__).parent])
    subprocess.run(
        [backend, "-c", code],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
        check=True,
        capture_output=True,
        text=True,
    )
