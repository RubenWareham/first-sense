"""Check the RTMPose boundary and the real isolated MMPose workflow."""

import json
import os
import sys
from pathlib import Path

import pytest

from examples.make_tiny_pose_dataset import create_pose_dataset
from wvision import Vision
from wvision.datasets.detection import load_dataset
from wvision.datasets.pose import read_person_boxes

PROFILE = "models/pose/rtmpose-t/smoke.yaml"


def test_topdown_inputs_and_pose_conversion(tmp_path: Path) -> None:
    manifest = create_pose_dataset(tmp_path / "pose", "halpe26")
    dataset = load_dataset(manifest, ("val",), task="pose")
    path = dataset.write_coco(tmp_path, ("val",), keypoints=True)["val"]
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    first = coco["annotations"][0]
    assert len(first["keypoints"]) == 26 * 3
    assert first["keypoints"][:3] == [28, 20, 2]
    assert first["keypoints"][27:30] == [0, 0, 0]
    assert first["num_keypoints"] == 25
    assert {row["image_id"] for row in coco["annotations"]} == {coco["images"][0]["id"]}

    image = manifest.parent / "images" / "val-0.png"
    inputs = tmp_path / "persons.json"
    inputs.write_text(json.dumps({"pose/images/val-0.png": [[1, 2, 161, 40]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid xyxy"):
        read_person_boxes(inputs, {image: (160, 96)})
    inputs.write_text(json.dumps({"pose/images/val-0.png": []}), encoding="utf-8")
    assert read_person_boxes(inputs, {image: (160, 96)}) == {image: []}
    with pytest.raises(ValueError, match="requires person_boxes"):
        Vision(PROFILE, backend_python=sys.executable).predict(image)


@pytest.mark.integration
@pytest.mark.parametrize("skeleton,count", [("coco17", 17), ("halpe26", 26)])
def test_rtmpose_workflow(tmp_path: Path, skeleton: str, count: int) -> None:
    backend = os.environ.get("WVISION_MMPOSE_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMPOSE_PYTHON to the pinned MMPose environment")
    device = os.environ.get("WVISION_TEST_DEVICE", "cpu")
    data = create_pose_dataset(tmp_path / "pose", skeleton)
    output = tmp_path / "runs"
    trained = Vision(PROFILE, backend_python=backend).train(data, device=device, output=output)
    assert trained.best_checkpoint.is_file() and trained.last_checkpoint.is_file()
    assert 0 <= trained.metrics["mmpose/coco/AP"] <= 1
    assert not (trained.artifact_dir / "data/test").exists()
    model = Vision.from_run(trained.artifact_dir)
    for split, execute in (("val", model.val), ("test", model.test)):
        result = execute(output=output)
        assert result.split == split and 0 <= result.metrics["mmpose/coco/AP"] <= 1
        protocol = json.loads(
            (result.artifact_dir / "pose-protocol.json").read_text(encoding="utf-8")
        )
        assert protocol["boxes"] == "ground_truth" and len(protocol["oks_sigmas"]) == count
        assert (result.artifact_dir / f"{split}.coco.json").is_file()
        assert not (result.artifact_dir / "train_labeled.coco.json").exists()

    images = data.parent / "images"
    boxes = [[8, 4, 48, 82], [85, 4, 124, 82]]
    person_boxes = tmp_path / "boxes.json"
    person_boxes.write_text(
        json.dumps({"pose/images/test-0.png": boxes, "pose/images/test-1.png": []}),
        encoding="utf-8",
    )
    result = model.predict(
        [images / "test-0.png", images / "test-1.png"], person_boxes=person_boxes, output=output
    )
    assert result.sample_count == 2
    frames = [json.loads(line) for line in result.predictions.read_text().splitlines()]
    assert frames[0]["skeleton"] == skeleton and frames[0]["keypoint_count"] == count
    assert (frames[0]["width"], frames[0]["height"]) == (160, 96)
    assert [person["box"] for person in frames[0]["persons"]] == boxes
    assert all(len(person["joints"]) == count for person in frames[0]["persons"])
    assert frames[1]["persons"] == []
