"""Group separation, derived data provenance, and cross-dataset transfer boundaries."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from examples.make_tiny_dataset import create_dataset
from wvision import Vision, prepare_group_split, prepare_label_budget
from wvision.adapters.shared.checkpoints import checkpoint_files
from wvision.datasets.detection import Dataset, load_dataset
from wvision.experiments.comparison import _jobs, run_experiments
from wvision.training.transfer import transfer_data


def other_dataset(directory: Path) -> Dataset:
    path = create_dataset(directory)
    for image in (directory / "images").iterdir():
        with Image.open(image) as source:
            source.putpixel((0, 0), (254, 0, 254))
            source.save(image)
    return load_dataset(path, ())


def explicit_groups(dataset: Dataset, path: Path) -> dict[str, str]:
    mapping = {
        s.path.relative_to(dataset.root).as_posix(): name
        for name, pool in dataset.splits.items()
        for s in pool
    }
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return mapping


def test_group_split_preserves_source_and_budget_groups(manifest: Path, tmp_path: Path) -> None:
    dataset = load_dataset(manifest, ())
    before = checkpoint_files(dataset.root)
    mapping = tmp_path / "groups.yaml"
    explicit_groups(dataset, mapping)
    prepared = prepare_group_split(
        manifest, tmp_path / "grouped", groups=mapping, val_groups=["val"], test_groups=["test"]
    )
    grouped = load_dataset(prepared, ())
    assert grouped.groups and grouped.fingerprint != dataset.fingerprint
    assert checkpoint_files(dataset.root) == before
    assert {s.pixel_hash for pool in grouped.splits.values() for s in pool} == {
        s.pixel_hash for pool in dataset.splits.values() for s in pool
    }
    budget = prepare_label_budget(prepared, tmp_path / "budget", labeled_count=2)
    derived = load_dataset(budget, ())
    assert set(derived.groups.values()) == set(grouped.groups.values())
    assert derived.groups[next(iter(derived.groups))] in set(grouped.groups.values())


def test_group_metadata_rejects_leakage_and_incomplete_mapping(
    manifest: Path, tmp_path: Path
) -> None:
    dataset = load_dataset(manifest, ())
    file = dataset.root / "groups.yaml"
    mapping = explicit_groups(dataset, file)
    config = yaml.safe_load(manifest.read_text())
    config["groups"] = file.name
    manifest.write_text(yaml.safe_dump(config))
    assert load_dataset(manifest, ()).groups
    mapping["images/val-0.png"] = "train_labeled"
    file.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError, match="crosses"):
        load_dataset(manifest, ())
    mapping.pop("images/val-0.png")
    file.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError, match="every dataset-relative"):
        load_dataset(manifest, ())


def test_group_selection_rejects_unknown_or_shared_groups(manifest: Path, tmp_path: Path) -> None:
    mapping = tmp_path / "groups.yaml"
    explicit_groups(load_dataset(manifest, ()), mapping)
    for validation, testing in [(["val"], ["val"]), (["missing"], ["test"])]:
        with pytest.raises(ValueError, match="distinct known"):
            prepare_group_split(
                manifest,
                tmp_path / "invalid",
                groups=mapping,
                val_groups=validation,
                test_groups=testing,
            )
    assert not (tmp_path / "invalid").exists()


def test_group_selection_cannot_turn_unlabeled_images_into_heldout_labels(
    manifest: Path, tmp_path: Path
) -> None:
    budget = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    mapping = tmp_path / "groups.yaml"
    explicit_groups(load_dataset(budget, ()), mapping)
    with pytest.raises(ValueError, match="unlabeled images"):
        prepare_group_split(
            budget,
            tmp_path / "invalid",
            groups=mapping,
            val_groups=["train_unlabeled"],
            test_groups=["test"],
        )
    assert not (tmp_path / "invalid").exists()


def test_cross_dataset_transfer_checks_source_training_and_selection(
    manifest: Path, tmp_path: Path
) -> None:
    source = load_dataset(manifest, ())
    target = other_dataset(tmp_path / "different")
    state = {"dataset_fingerprint": source.fingerprint}
    audit = transfer_data(source.snapshot(), state, target)
    assert audit["cross_dataset"] and audit["source_samples_checked"] == 6
    snapshot = source.snapshot()
    snapshot["splits"]["train_labeled"][0]["pixel_sha256"] = target.splits["test"][0].pixel_hash
    with pytest.raises(ValueError, match="held-out"):
        transfer_data(snapshot, state, target)
    inherited = {
        **state,
        "initialization": {
            "pretraining": {"training_images": [target.splits["test"][0].pixel_hash]}
        },
    }
    with pytest.raises(ValueError, match="held-out"):
        transfer_data(source.snapshot(), inherited, target)
    snapshot = source.snapshot()
    snapshot["splits"]["val"][0]["pixel_sha256"] = target.splits["test"][0].pixel_hash
    with pytest.raises(ValueError, match="held-out"):
        transfer_data(snapshot, state, target)


def test_cross_dataset_transfer_rejects_shared_video_groups(manifest: Path, tmp_path: Path) -> None:
    source = load_dataset(manifest, ())
    target = other_dataset(tmp_path / "different")
    target.groups.update(
        {
            s.path.relative_to(target.root).as_posix(): name
            for name, pool in target.splits.items()
            for s in pool
        }
    )
    snapshot = source.snapshot()
    snapshot["groups"] = {
        s.path.relative_to(source.root).as_posix(): f"source-{name}"
        for name, pool in source.splits.items()
        for s in pool
    }
    snapshot["groups"]["images/train_labeled-0.png"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        transfer_data(snapshot, {"dataset_fingerprint": source.fingerprint}, target)


def test_experiment_group_requirement_and_transfer_scope(manifest: Path, tmp_path: Path) -> None:
    from unittest.mock import patch

    from wvision.experiments import comparison

    plan: dict[str, Any] = {
        "schema_version": 1,
        "data": str(manifest),
        "require_groups": True,
        "experiments": [{"name": "model", "model": "ultralytics/yolo11n"}],
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    with patch.object(comparison, "require_evaluator"), pytest.raises(ValueError, match="isolated"):
        run_experiments(path, output=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
    plan["experiments"][0]["initialization"] = {
        "transfer_scope": "backbone",
        "detector_run": str(tmp_path),
    }
    job = _jobs(plan, tmp_path)[0]
    assert job["initialization"]["transfer_scope"] == "backbone"
    with pytest.raises(ValueError, match="transfer_scope"):
        Vision().train(manifest, transfer_scope="backbone")
