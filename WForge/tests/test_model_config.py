"""Model documents, configuration layering, and external dataset catalog behavior."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from examples.make_tiny_dataset import create_dataset
from wvision import Vision, compare_learning, compare_pretraining, prepare_label_budget, pretrain
from wvision.cli import parser
from wvision.core.config import load_yaml
from wvision.core.model_config import ModelConfig, comparison_model
from wvision.core.registry import get_model, vision_adapter
from wvision.datasets.budgets import load_budget_metadata
from wvision.datasets.detection import load_dataset
from wvision.models.mmpretrain.simsiam_r50.config import PretrainConfig
from wvision.models.ultralytics.yolo11 import operations as ultralytics

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "models/detection/yolo11n/smoke.yaml"


def model_document(tmp_path: Path, **changes: Any) -> Path:
    value = load_yaml(SMOKE)
    value.update(changes)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "models").rglob("*.yaml")),
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_all_catalog_configs_select_valid_registered_settings(path: Path) -> None:
    definition = ModelConfig.load(path)
    spec = get_model(definition.model_id)
    if spec.task == "representation":
        PretrainConfig.resolve(definition.training())
    else:
        python = None if spec.backend == "ultralytics" else sys.executable
        adapter = vision_adapter(spec.model_id, python)
        settings = adapter.resolve(definition.training())
        spec.require(settings.learning, "train", settings.device)


def test_constructor_settings_overrides_and_self_contained_run(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = model_document(
        tmp_path,
        training={
            "epochs": 5,
            "imgsz": 128,
            "batch": 2,
            "backend_options": {"mosaic": 0.0, "fliplr": 0.2},
        },
    )
    monkeypatch.setattr(ultralytics, "check_dependency", lambda: None)

    def train(
        checkpoint: Any, data: Path, settings: Any, directory: Path, *, task: str
    ) -> dict[str, Any]:
        assert task == "detect"
        assert settings.epochs == 1 and settings.imgsz == 128 and settings.batch == 2
        assert settings.backend_options == {"mixup": 0.3}
        saved = directory / "best.pt"
        saved.write_bytes(b"boundary fixture; not model support evidence")
        return {"best": saved, "last": saved, "metrics": {"fixture/loss": 1.0}}

    def evaluate(
        checkpoint: Path,
        data: Path,
        classes: Any,
        directory: Path,
        imgsz: int,
        batch: int,
        device: str = "cpu",
        *,
        task: str,
    ) -> dict[str, float]:
        assert imgsz == 128 and batch == 2 and task == "detect"
        return {"fixture/AP": 0.0}

    monkeypatch.setattr(ultralytics, "train", train)
    monkeypatch.setattr(ultralytics, "evaluate", evaluate)
    model = Vision(source)
    trained = model.train(
        manifest, epochs=1, backend_options={"mixup": 0.3}, output=tmp_path / "runs"
    )
    snapshot = json.loads((trained.artifact_dir / "model-definition.json").read_text())
    assert snapshot["document"]["training"]["epochs"] == 5
    assert snapshot["path"] == str(source.resolve()) and len(snapshot["sha256"]) == 64
    resolved = json.loads((trained.artifact_dir / "config.json").read_text())
    assert resolved["epochs"] == 1 and resolved["imgsz"] == 128
    state = json.loads((trained.artifact_dir / "run.json").read_text())
    assert state["model"] == "ultralytics/yolo11n" and state["model_definition"] == snapshot
    source.unlink()
    assert Vision.from_run(trained.artifact_dir).val(output=tmp_path / "runs").split == "val"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"task": "pose"},
        {"profile": "state_of_the_art"},
        {"description": ""},
        {"upstream": {"version": 123}},
        {"upstream": {"execute": "code"}},
        {"training": []},
        {"training": {"epohs": 1}},
        {"model": "not-registered"},
        {"data": "dataset.yaml"},
    ],
)
def test_invalid_model_documents_fail_before_backend(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    path = model_document(tmp_path, **changes)
    with pytest.raises(ValueError):
        Vision(path)


def test_changed_source_requires_new_instance(manifest: Path, tmp_path: Path) -> None:
    path = model_document(tmp_path)
    model = Vision(path)
    path.write_text(path.read_text() + "\n# changed after construction\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since construction"):
        model.train(manifest, output=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_model_yaml_is_safe_and_missing_path_does_not_fall_back(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Vision(tmp_path / "missing.yaml")
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.print ['should not execute']", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        Vision(path)


def test_model_document_rejects_flat_settings_and_wrong_task(tmp_path: Path) -> None:
    flat = tmp_path / "flat.yaml"
    flat.write_text("epochs: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model configuration fields"):
        Vision(flat)
    with pytest.raises(ValueError, match="representation model"):
        Vision(ROOT / "models/representation/simsiam-r50/smoke.yaml")
    with pytest.raises(ValueError, match="comparison requires model"):
        comparison_model(SMOKE, "mmdetection/faster-rcnn-r50")


@pytest.mark.parametrize("operation", ["train", "pretrain", "compare", "compare-pretraining"])
def test_removed_config_argument_is_rejected_before_work(manifest: Path, operation: str) -> None:
    arguments = [operation, "--data", str(manifest), "--config", "removed.yaml"]
    if operation == "compare-pretraining":
        arguments += ["--pretraining-run", "missing-run"]
    with pytest.raises(SystemExit) as error:
        parser().parse_args(arguments)
    assert error.value.code == 2
    call = {
        "train": Vision().train,
        "pretrain": pretrain,
        "compare": compare_learning,
        "compare-pretraining": compare_pretraining,
    }[operation]
    kwargs = {"pretraining_run": "missing-run"} if operation == "compare-pretraining" else {}
    with pytest.raises(ValueError, match="Unknown.*config"):
        call(manifest, config="removed.yaml", **kwargs)


def test_separate_dataset_catalog_keeps_data_identity_and_budget(tmp_path: Path) -> None:
    catalog = tmp_path / "datasets/examples"
    catalog.mkdir(parents=True)
    supervised = catalog / "tiny-detection.yaml"
    semi = catalog / "tiny-semi-supervised.yaml"
    shutil.copyfile(ROOT / "datasets/examples/tiny-detection.yaml", supervised)
    shutil.copyfile(ROOT / "datasets/examples/tiny-semi-supervised.yaml", semi)
    generated = create_dataset(tmp_path / ".wvision/tiny")
    assert (
        load_dataset(supervised, ("test",)).fingerprint == load_dataset(generated, ()).fingerprint
    )
    derived = prepare_label_budget(supervised, tmp_path / ".wvision/tiny-budget", labeled_count=2)
    dataset = load_dataset(semi, ("train_unlabeled",))
    assert dataset.fingerprint == load_dataset(derived, ()).fingerprint
    metadata = load_budget_metadata(dataset)
    assert metadata is not None and metadata["policy"] == "seeded_sample_v1"
    assert metadata["labeled_count"] == 2 and not metadata["withheld_labels_copied"]
    budget_file = dataset.root / "budget.json"
    budget_file.write_text('{"derived_fingerprint": "changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_budget_metadata(dataset)
    budget_file.unlink()
    (catalog / "budget.json").write_text('{"derived_fingerprint": "unrelated"}', encoding="utf-8")
    assert load_budget_metadata(dataset) is None


def test_representation_model_config_uses_pretraining_entry_point(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)

    def invoke(
        python: Path, directory: Path, request: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        assert request["config"]["max_iters"] == 1
        assert request["config"]["imgsz"] == 64 and kwargs == {}
        checkpoint = directory / "fixture.pth"
        checkpoint.write_bytes(b"boundary fixture")
        return {"checkpoint": str(checkpoint), "metrics": {"fixture/loss": -0.1}}

    monkeypatch.setattr("wvision.training.pretraining.invoke", invoke)
    result = pretrain(
        data,
        model=ROOT / "models/representation/simsiam-r50/smoke.yaml",
        backend_python=sys.executable,
        max_iters=1,
        output=tmp_path / "runs",
    )
    state = json.loads((result.artifact_dir / "run.json").read_text())
    assert state["model"] == "mmpretrain/simsiam-r50"
    assert state["model_definition"]["document"]["profile"] == "smoke"
