import json
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision, compare_pretraining, prepare_label_budget, pretrain
from wvision.datasets.detection import load_dataset
from wvision.experiments.artifacts import write_json
from wvision.models.mmdetection.faster_rcnn_r50.config import MODEL_ID, MMDetConfig
from wvision.models.mmpretrain.simsiam_r50.adapter import load_pretraining
from wvision.models.mmpretrain.simsiam_r50.config import PretrainConfig


@pytest.mark.parametrize(
    "values",
    [
        {"batch": 1},
        {"max_iters": True},
        {"imgsz": 33},
        {"seed": -1},
        {"device": "mps"},
        {"lr": float("nan")},
        {"lr": True},
        {"learning": "unsupervised"},
    ],
)
def test_invalid_pretraining_settings(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PretrainConfig.resolve(values)


def test_pretraining_defaults() -> None:
    config = PretrainConfig.resolve({"max_iters": 2, "imgsz": 128})
    assert config.max_iters == 2 and config.imgsz == 128


def test_pretraining_requires_explicit_unlabeled_pool(manifest: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_unlabeled"):
        pretrain(manifest, backend_python=sys.executable, output=tmp_path / "runs")
    with pytest.raises(ValueError, match="Unsupported pretraining model"):
        pretrain(manifest, model="unknown")
    derived = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=3)
    with pytest.raises(ValueError, match="at least batch"):
        pretrain(derived, backend_python=sys.executable)
    assert not (tmp_path / "runs").exists()


def test_staged_pretraining_has_no_annotations_or_heldout_images(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A boundary test, not evidence of backend support: inspect the actual staged files.
    derived = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    dataset = load_dataset(derived, ("train_unlabeled",))

    def inspect(
        python: Path, directory: Path, request: dict[str, Any], **kw: Any
    ) -> dict[str, Any]:
        assert kw == {} and request["operation"] == "pretrain"
        listing = Path(request["images"])
        images = list((directory / "data").rglob("*.png"))
        assert len(images) == 2 and len(listing.read_text().splitlines()) == 2
        assert all("train_unlabeled" in str(path) for path in images)
        assert not list((directory / "data").rglob("*.txt"))
        assert "classes" not in request and "val" not in request and "test" not in request
        checkpoint = directory / "fixture.pth"
        checkpoint.write_bytes(b"boundary fixture, not a real model")
        return {"checkpoint": str(checkpoint), "metrics": {"mmpretrain/train/loss": -0.1}}

    monkeypatch.setattr("wvision.training.pretraining.invoke", inspect)
    result = pretrain(derived, backend_python=sys.executable, output=tmp_path / "runs")
    state = load_pretraining(result.artifact_dir, dataset.fingerprint)
    assert state["selection"] == "final_iteration"
    assert state["training_images"] == [s.pixel_hash for s in dataset.splits["train_unlabeled"]]
    with pytest.raises(ValueError, match="representation run"):
        Vision.from_run(result.artifact_dir)
    with pytest.raises(ValueError, match="identity differs"):
        load_pretraining(result.artifact_dir, "other")
    result.checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint content has changed"):
        load_pretraining(result.artifact_dir, dataset.fingerprint)
    write_json(result.artifact_dir / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed"):
        load_pretraining(result.artifact_dir, dataset.fingerprint)


@pytest.mark.parametrize(
    "model,settings",
    [
        ("ultralytics/yolo11n", {}),
        (MODEL_ID, {}),
        (
            MODEL_ID,
            {"learning": "semi_supervised"},
        ),
    ],
)
def test_transfer_rejects_unsupported_combinations_before_loading(
    manifest: Path,
    model: str,
    settings: dict[str, Any],
) -> None:
    backend = sys.executable if model == MODEL_ID else None
    with pytest.raises(ValueError, match="pretraining_run requires"):
        Vision(model, backend_python=backend).train(
            manifest,
            pretraining_run="does-not-exist",
            **settings,
        )


def test_transfer_comparison_requires_matching_protocol(manifest: Path) -> None:
    with pytest.raises(ValueError, match="backbone_style=pytorch"):
        compare_pretraining(manifest, pretraining_run="does-not-exist")
    with pytest.raises(ValueError, match="backbone_style"):
        MMDetConfig(backend_options={"backbone_style": []})
    assert MMDetConfig(learning="semi_supervised", backend_options={"backbone_style": "pytorch"})


def test_pretraining_import_is_framework_free() -> None:
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from wvision import pretrain, compare_pretraining; "
                "from wvision.cli import parser; parser().format_help(); "
                "assert not {'torch','mmpretrain','mmengine','mmdet'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )


def test_pretraining_cli_rejects_invalid_mode(manifest: Path) -> None:
    from wvision.cli import main

    assert main(["pretrain", "--data", str(manifest), "--model", "unknown"]) == 2


def test_invalid_pretraining_run_kind(tmp_path: Path) -> None:
    write_json(tmp_path / "run.json", {"schema_version": 1, "run_kind": "detection"})
    write_json(tmp_path / "status.json", {"status": "completed"})
    with pytest.raises(ValueError, match="completed WVision SimSiam"):
        load_pretraining(tmp_path, "fingerprint")
    assert json.loads((tmp_path / "run.json").read_text())["run_kind"] == "detection"
