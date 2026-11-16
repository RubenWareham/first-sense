import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wvision import Vision, prepare_label_budget
from wvision.datasets.detection import load_dataset
from wvision.models.mmdetection.faster_rcnn_r50.adapter import invoke
from wvision.models.mmdetection.faster_rcnn_r50.config import MODEL_ID, MMDetConfig


def test_budget_is_reproducible_and_withholds_labels(manifest: Path, tmp_path: Path) -> None:
    first = prepare_label_budget(manifest, tmp_path / "first", labeled_count=2, seed=17)
    second = prepare_label_budget(manifest, tmp_path / "second", labeled_count=2, seed=17)
    one, two = load_dataset(first, ("test",)), load_dataset(second, ("test",))
    assert one.fingerprint == two.fingerprint
    assert len(one.splits["train_labeled"]) == 2
    assert len(one.splits["train_unlabeled"]) == 2
    assert all(sample.label is None for sample in one.splits["train_unlabeled"])
    assert not list((first.parent / "train_unlabeled").rglob("*.txt"))
    budget = json.loads((first.parent / "budget.json").read_text())
    assert budget["labeled_fraction"] == 0.5
    assert budget["selection"]["val"] == ["images/val-0.png", "images/val-1.png"]
    assert budget["selection"]["test"] == ["images/test-0.png", "images/test-1.png"]
    assert not any(
        "val-" in item or "test-" in item for item in budget["selection"]["train_unlabeled"]
    )
    with pytest.raises(FileExistsError):
        prepare_label_budget(manifest, first.parent, labeled_count=2)


def test_budget_changes_with_seed(manifest: Path, tmp_path: Path) -> None:
    selections = []
    for seed in (0, 1, 2):
        target = prepare_label_budget(manifest, tmp_path / str(seed), labeled_count=2, seed=seed)
        budget = json.loads((target.parent / "budget.json").read_text())
        selections.append(tuple(budget["selection"]["train_labeled"]))
    assert len(set(selections)) > 1


@pytest.mark.parametrize("count", [0, 4, -1, True])
def test_invalid_budget(manifest: Path, tmp_path: Path, count: int) -> None:
    with pytest.raises(ValueError, match="labeled_count"):
        prepare_label_budget(manifest, tmp_path / "derived", labeled_count=count)
    assert not (tmp_path / "derived").exists()


def test_coco_conversion_and_unlabeled_isolation(manifest: Path, tmp_path: Path) -> None:
    derived = prepare_label_budget(manifest, tmp_path / "derived", labeled_count=2)
    dataset = load_dataset(derived, ("train_unlabeled",))
    paths = dataset.write_coco(tmp_path, ("train_labeled", "train_unlabeled", "val"))
    labeled = json.loads(Path(paths["train_labeled"]).read_text())
    unlabeled = json.loads(Path(paths["train_unlabeled"]).read_text())
    assert len(labeled["annotations"]) == 2
    assert unlabeled["annotations"] == []
    assert not list((tmp_path / "data/train_unlabeled").rglob("*.txt"))
    sample = dataset.splits["train_labeled"][0]
    assert sample.label is not None
    cls, x, y, width, height = map(float, sample.label.read_text().split())
    box = labeled["annotations"][0]
    assert box["category_id"] == cls + 1
    assert box["bbox"] == pytest.approx(
        [(x - width / 2) * 128, (y - height / 2) * 128, width * 128, height * 128]
    )
    assert "test" not in paths


def test_mmdet_configuration_and_missing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WVISION_MMDET_PYTHON", raising=False)
    with pytest.raises(ValueError, match="isolated environment"):
        Vision(MODEL_ID)
    assert MMDetConfig(learning="semi_supervised", max_iters=2).max_iters == 2
    with pytest.raises(ValueError, match="Unknown"):
        MMDetConfig.resolve({"epochs": 2})
    with pytest.raises(ValueError, match="semi_supervised"):
        MMDetConfig(backend_options={"cls_pseudo_thr": 0.5})


def test_semi_requires_unlabeled_pool_before_worker(manifest: Path) -> None:
    model = Vision(MODEL_ID, backend_python=sys.executable)
    with pytest.raises(ValueError, match="train_unlabeled"):
        model.train(manifest, learning="semi_supervised", max_iters=1)


def test_mmdet_rejects_foreign_checkpoint_before_loading(manifest: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "foreign.pth"
    checkpoint.write_bytes(b"foreign checkpoint")
    model = Vision(MODEL_ID, backend_python=sys.executable)
    with pytest.raises(ValueError, match="selected checkpoint of a completed run"):
        model.val(manifest, checkpoint=checkpoint)


def test_worker_failure_has_a_durable_log(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.log"):
        # The core interpreter has the wrong Python version for this worker.
        invoke(Path(sys.executable), tmp_path, {"operation": "train"})
    assert "requires Python 3.10" in (tmp_path / "backend.log").read_text()


def test_budget_cli(manifest: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wvision",
            "prepare-budget",
            "--data",
            str(manifest),
            "--destination",
            str(tmp_path / "derived"),
            "--labeled-count",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(json.loads(result.stdout)["manifest"]).is_file()


def test_class_map_coco_order_is_stable(manifest: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(manifest.read_text())
    config["classes"] = {1: "other", 0: "rectangle"}
    manifest.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    dataset = load_dataset(manifest, ("val",))
    result = json.loads(Path(dataset.write_coco(tmp_path, ("val",))["val"]).read_text())
    assert result["categories"] == [{"id": 1, "name": "rectangle"}, {"id": 2, "name": "other"}]
