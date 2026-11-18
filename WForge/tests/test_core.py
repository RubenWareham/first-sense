import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wvision import Vision
from wvision.adapters.ultralytics import backend as adapter
from wvision.datasets.detection import load_dataset, sha256
from wvision.experiments.artifacts import environment, operation
from wvision.models.ultralytics.yolo11 import operations
from wvision.models.ultralytics.yolo11.config import TrainConfig


def test_import_and_help_without_loading_framework() -> None:
    code = (
        "import sys; from wvision import Vision; from wvision.cli import parser; "
        "parser().format_help(); assert 'torch' not in sys.modules; "
        "assert 'ultralytics' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run(
        [sys.executable, "-m", "wvision", "--help"], capture_output=True, text=True, check=True
    )
    assert "train" in result.stdout


def test_unborn_repository_does_not_record_head_as_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "wvision.experiments.artifacts.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 128, stdout="HEAD\n"),
    )
    assert environment()["git_commit"] is None


def test_unknown_model_and_unsupported_mode_fail_before_backend(manifest: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        Vision("unknown")
    with pytest.raises(ValueError, match="not integrated"):
        Vision().train(manifest, learning="semi_supervised")


def test_training_defaults_and_unknown_fields() -> None:
    resolved = TrainConfig.resolve({"epochs": 1, "imgsz": 128})
    assert resolved.epochs == 1 and resolved.imgsz == 128 and resolved.device == "cpu"
    with pytest.raises(ValueError, match="Unknown"):
        TrainConfig.resolve({"epohs": 2})
    with pytest.raises(ValueError, match="backend_options"):
        TrainConfig(backend_options={"data": "other.yaml"})


@pytest.mark.parametrize("options", [{"lr0": "secret"}, {"mixup": 2}, {"patience": -1}])
def test_invalid_backend_options(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="backend_options"):
        TrainConfig(backend_options=options)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_epochs(value: object) -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig.resolve({"epochs": value})


def test_missing_extra_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter.metadata, "version", missing)
    with pytest.raises(RuntimeError, match=r"pip install.*\[ultralytics\]"):
        adapter.check_dependency()


def test_wrong_backend_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "0.0.0")
    with pytest.raises(RuntimeError, match="Expected ultralytics"):
        adapter.check_dependency()


def test_data_fingerprint_and_training_excludes_test(manifest: Path, tmp_path: Path) -> None:
    dataset = load_dataset(manifest, ("train_labeled", "val"))
    assert len(dataset.splits["train_labeled"]) == 4
    assert load_dataset(manifest, ("test",)).fingerprint == dataset.fingerprint
    target = dataset.write_upstream(tmp_path)
    config = yaml.safe_load(target.read_text())
    assert "test" not in config
    assert all(
        "train_labeled" in Path(line).parts
        for line in (tmp_path / "train_labeled.txt").read_text().splitlines()
    )
    (manifest.parent / "labels/train_labeled-0.txt").write_text("", encoding="utf-8")
    assert load_dataset(manifest, ("val",)).fingerprint != dataset.fingerprint


def test_test_evaluation_uses_test_list(manifest: Path, tmp_path: Path) -> None:
    dataset = load_dataset(manifest, ("test",))
    generated = dataset.write_upstream(tmp_path, "test")
    config = yaml.safe_load(generated.read_text())
    assert Path(config["val"]).name == "test.txt"
    staged = [Path(line) for line in Path(config["val"]).read_text().splitlines()]
    assert [sha256(path) for path in staged] == [s.image_hash for s in dataset.splits["test"]]


@pytest.mark.parametrize("format", ["yolo", "coco"])
def test_staging_long_source_names(manifest: Path, tmp_path: Path, format: str) -> None:
    original = manifest.parent / "images/val-0.png"
    # The source fits Windows' legacy path limit; adding run folders must not
    # make a valid source filename unusable in an upstream snapshot.
    long_name = "v" * min(180, 225 - len(str(original.parent))) + ".png"
    renamed = original.with_name(long_name)
    original.rename(renamed)
    (manifest.parent / "labels/val-0.txt").rename(
        manifest.parent / "labels" / renamed.with_suffix(".txt").name
    )
    listing = manifest.parent / "splits/val.txt"
    listing.write_text(listing.read_text().replace(original.name, long_name), encoding="utf-8")
    dataset = load_dataset(manifest, ("val",))
    target = tmp_path / ("nested-" * 6)
    target.mkdir()
    if format == "yolo":
        dataset.write_upstream(target, "val")
    else:
        dataset.write_coco(target, ("val",))
    images = sorted((target / "data/val/images").iterdir())
    assert [sha256(path) for path in images] == [s.image_hash for s in dataset.splits["val"]]
    assert renamed.is_file()
    for image, sample in zip(images, dataset.splits["val"], strict=True):
        label = image.parent.parent / "labels" / image.with_suffix(".txt").name
        assert sha256(label) == sample.label_hash


def test_staging_preserves_originals_and_checks_changes(manifest: Path, tmp_path: Path) -> None:
    dataset = load_dataset(manifest, ("val",))
    target = tmp_path / "stage"
    target.mkdir()
    dataset.write_upstream(target, "val")
    staged_image = next((target / "data/val/images").iterdir())
    original = dataset.splits["val"][0].path
    staged_image.write_bytes(b"changed")
    assert original.read_bytes() != staged_image.read_bytes()
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Image changed after preflight"):
        dataset.write_upstream(target, "val")


def test_split_overlap_is_rejected(manifest: Path) -> None:
    root = manifest.parent
    (root / "splits/val.txt").write_text("images/train_labeled-0.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_dataset(manifest, ("val",))


def test_duplicate_pixels_are_rejected(manifest: Path) -> None:
    from PIL import Image

    root = manifest.parent
    with Image.open(root / "images/train_labeled-0.png") as image:
        image.save(root / "images/val-0.png")
    with pytest.raises(ValueError, match="Duplicate image content"):
        load_dataset(manifest, ("val",))


def test_missing_annotation_is_not_a_negative(manifest: Path) -> None:
    (manifest.parent / "labels/val-0.txt").unlink()
    with pytest.raises(ValueError, match="Missing annotation"):
        load_dataset(manifest, ("val",))


@pytest.mark.parametrize("annotation", ["0 0.5 0.5 nan 0.2", "1 0.5 0.5 0.2 0.2", "0 0 0 1 1"])
def test_invalid_boxes(manifest: Path, annotation: str) -> None:
    (manifest.parent / "labels/val-0.txt").write_text(annotation, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        load_dataset(manifest, ("val",))


def test_test_split_is_never_substituted(manifest: Path) -> None:
    config = yaml.safe_load(manifest.read_text())
    del config["splits"]["test"]
    manifest.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Required split is missing: test"):
        load_dataset(manifest, ("test",))


def test_run_failure_and_non_overwriting(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="example"):
        with operation(tmp_path, "train", {}) as first:
            raise RuntimeError("example")
    assert json.loads((first / "status.json").read_text())["status"] == "failed"
    with operation(tmp_path, "train", {}) as second:
        assert first != second
    assert json.loads((second / "status.json").read_text())["status"] == "completed"


def test_interrupted_run(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        with operation(tmp_path, "train", {}) as directory:
            raise KeyboardInterrupt
    assert json.loads((directory / "status.json").read_text())["status"] == "interrupted"


def test_backend_failure_records_status(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(operations, "check_dependency", lambda: None)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("training failed")

    monkeypatch.setattr(operations, "train", fail)
    output = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="training failed"):
        Vision().train(manifest, output=output, epochs=1, batch=2, imgsz=128)
    status = next(output.glob("*/status.json"))
    assert json.loads(status.read_text())["status"] == "failed"


def test_cli_invalid_mode_is_an_error(manifest: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wvision",
            "train",
            "--data",
            str(manifest),
            "--learning",
            "unsupervised",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not integrated" in result.stderr
