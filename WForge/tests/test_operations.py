"""Recovery, portability, and experiment protocol regressions without ML imports."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision, bundle_run, resume
from wvision.adapters.base import DetectionContext
from wvision.adapters.shared.checkpoints import checkpoint_files, file_lock, record_checkpoint
from wvision.adapters.shared.devices import distributed_visibility, validate_devices, worker_context
from wvision.experiments.artifacts import run_lock, write_json
from wvision.experiments.comparison import _jobs, _report
from wvision.experiments.portable import verify_bundle
from wvision.models.mmdetection.faster_rcnn_r50.config import MMDetConfig
from wvision.models.transformers.dinov3_convnext_tiny_detr.config import DinoDETRConfig
from wvision.models.ultralytics.yolo11 import operations
from wvision.models.ultralytics.yolo11.config import TrainConfig
from wvision.training.recovery import resolve_resume


@pytest.fixture
def recoverable(manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(operations, "check_dependency", lambda: None)

    def train(
        checkpoint: Any, data: Path, settings: Any, directory: Path, **kwargs: Any
    ) -> dict[str, Any]:
        weights = directory / "weights.pt"
        weights.write_bytes(b"checkpoint boundary fixture, not model weights")
        write_json(directory / "model-source.json", {"license": "fixture declaration"})
        record_checkpoint(directory, weights, settings.epochs)
        return {"best": weights, "last": weights, "metrics": {"fixture/loss": 1.0}}

    monkeypatch.setattr(operations, "train", train)
    return Vision().train(manifest, epochs=1, output=tmp_path / "runs").artifact_dir


def test_resume_preserves_source_and_validates_protocol(recoverable: Path, tmp_path: Path) -> None:
    state = json.loads((recoverable / "training.json").read_text())
    before = (recoverable / "weights.pt").read_bytes()
    continued = resume(recoverable, epochs=2, output=tmp_path / "continued")
    assert continued.artifact_dir != recoverable
    assert (recoverable / "weights.pt").read_bytes() == before
    assert continued.best_checkpoint.is_relative_to(continued.artifact_dir)
    resolved = json.loads((continued.artifact_dir / "config.json").read_text())
    assert (
        resolved["initialization_sources"]["resume"]["provenance"]["license"]
        == "fixture declaration"
    )
    with pytest.raises(ValueError, match="budget must exceed"):
        resume(recoverable)
    with pytest.raises(ValueError, match="identity differs"):
        resolve_resume(
            recoverable, "different/model", state["dataset_fingerprint"], state["config"]
        )
    with pytest.raises(ValueError, match="only the total"):
        resolve_resume(
            recoverable,
            state["model"],
            state["dataset_fingerprint"],
            {**state["config"], "epochs": 2, "imgsz": 128},
        )
    recovery = json.loads((recoverable / "recovery.json").read_text())
    (recoverable / recovery["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="content has changed"):
        resume(recoverable, epochs=2)


def test_active_run_rejected_and_stale_run_recoverable(recoverable: Path, tmp_path: Path) -> None:
    with run_lock(recoverable), pytest.raises(ValueError, match="active"):
        resume(recoverable, epochs=2)
    write_json(recoverable / "status.json", {"status": "running"})
    result = resume(recoverable, epochs=2, output=tmp_path / "recovered")
    assert result.artifact_dir.is_dir()


def test_missing_checkpoint_fails_clearly(recoverable: Path) -> None:
    (recoverable / "recovery.json").unlink()
    with pytest.raises(ValueError, match="no recoverable checkpoint"):
        resume(recoverable, epochs=2)


def test_invalid_source_does_not_create_coordination_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="predates recoverable"):
        resume(tmp_path, epochs=2)
    with pytest.raises(ValueError, match="completed training"):
        bundle_run(tmp_path, tmp_path / "destination")
    assert not (tmp_path / ".lock").exists()


def test_orphan_backend_writer_blocks_recovery(recoverable: Path) -> None:
    write_json(recoverable / "status.json", {"status": "running"})
    with file_lock(recoverable / ".worker.lock"), pytest.raises(ValueError, match="active"):
        resume(recoverable, epochs=2)


def test_partial_new_snapshot_preserves_committed_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wvision.adapters.shared import checkpoints

    source = tmp_path / "weights.pt"
    source.write_bytes(b"complete first checkpoint")
    record_checkpoint(tmp_path, source, 1)
    committed = json.loads((tmp_path / "recovery.json").read_text())
    source.write_bytes(b"new upstream checkpoint")
    assert (tmp_path / committed["checkpoint"]).read_bytes() == b"complete first checkpoint"

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated interrupted snapshot write")

    with monkeypatch.context() as context:
        context.setattr(checkpoints.shutil, "copyfile", fail)
        with pytest.raises(OSError, match="interrupted snapshot"):
            record_checkpoint(tmp_path, source, 2)
    assert json.loads((tmp_path / "recovery.json").read_text()) == committed
    assert checkpoint_files(tmp_path / committed["checkpoint"]) == committed["files"]
    record_checkpoint(tmp_path, source, 2)
    latest = json.loads((tmp_path / "recovery.json").read_text())
    assert latest["step"] == 2
    assert checkpoint_files(tmp_path / latest["checkpoint"]) == latest["files"]
    assert checkpoint_files(tmp_path / committed["checkpoint"]) == committed["files"]


def test_bundle_survives_relocation_and_retains_dataset_identity(
    recoverable: Path, tmp_path: Path
) -> None:
    bundle = bundle_run(recoverable, tmp_path / "bundle", include_data=True)
    state = json.loads((bundle / "run.json").read_text())
    assert not Path(state["best_checkpoint"]).is_absolute()
    assert state["backend_python"] is None
    moved = tmp_path / "moved"
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    bundle.rename(moved)
    model = Vision.from_run(moved)
    from wvision.datasets.detection import load_dataset

    assert model._data is not None
    assert load_dataset(model._data, ("test",)).fingerprint == model._fingerprint
    assert model._checkpoint is not None and model._checkpoint.is_relative_to(moved)
    verify_bundle(moved)
    model._checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Bundle content has changed"):
        Vision.from_run(moved)


def test_inference_bundle_works_without_source_dataset(recoverable: Path, tmp_path: Path) -> None:
    bundle = bundle_run(recoverable, tmp_path / "inference")
    model = Vision.from_run(bundle)
    assert model._data is None
    with pytest.raises(ValueError, match="dataset manifest"):
        model.val()
    with pytest.raises(ValueError, match="must be new"):
        bundle_run(recoverable, bundle)
    with pytest.raises(ValueError, match="outside the source"):
        bundle_run(recoverable, recoverable / "nested")


def test_grouped_bundle_preserves_group_identity(recoverable: Path, tmp_path: Path) -> None:
    import yaml

    from wvision.datasets.detection import load_dataset

    original = json.loads((recoverable / "run.json").read_text())["data"]
    dataset = load_dataset(original, ())
    mapping = {
        sample.path.relative_to(dataset.root).as_posix(): name
        for name, pool in dataset.splits.items()
        for sample in pool
    }
    (dataset.root / "groups.yaml").write_text(yaml.safe_dump(mapping))
    manifest = yaml.safe_load(dataset.manifest.read_text())
    manifest["groups"] = "groups.yaml"
    dataset.manifest.write_text(yaml.safe_dump(manifest))
    grouped = Vision().train(dataset.manifest, output=tmp_path / "grouped")
    source = load_dataset(dataset.manifest, ())
    bundle = bundle_run(grouped.artifact_dir, tmp_path / "grouped-bundle", include_data=True)
    moved = tmp_path / "grouped-moved"
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    bundle.rename(moved)
    restored = Vision.from_run(moved)
    assert restored._data is not None
    copied = load_dataset(restored._data, ())
    assert copied.fingerprint == source.fingerprint and copied.groups == source.groups


def test_portable_optimizer_state_resumes_after_relocation(
    recoverable: Path, tmp_path: Path
) -> None:
    bundle = bundle_run(
        recoverable, tmp_path / "portable-training", include_data=True, include_training=True
    )
    moved = tmp_path / "training-moved"
    assert moved.resolve().is_relative_to(tmp_path.resolve())
    bundle.rename(moved)
    state = json.loads((moved / "training.json").read_text())
    assert not Path(state["data"]).is_absolute() and state["backend_python"] is None
    before = checkpoint_files(moved)
    continued = resume(moved, epochs=2, output=tmp_path / "resumed-bundle")
    assert checkpoint_files(moved) == before
    assert continued.best_checkpoint.is_relative_to(continued.artifact_dir)


@pytest.mark.parametrize(
    "changes",
    [
        {"precision": "fp16"},
        {"precision": "bf16"},
        {"checkpoint_interval": 0},
        {"checkpoint_interval": True},
    ],
)
def test_precision_and_checkpoint_settings_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TrainConfig.resolve(changes)


def test_backend_precision_boundaries_fail_before_execution() -> None:
    with pytest.raises(ValueError, match="precision"):
        MMDetConfig(device="cuda:0", precision="bf16")
    with pytest.raises(ValueError, match="precision"):
        DinoDETRConfig(device="cuda:0", precision="fp16")


def test_parameter_grid_preserves_backend_options_and_interpreters(tmp_path: Path) -> None:
    plan = {
        "schema_version": 1,
        "data": "unused.yaml",
        "experiments": [
            {
                "name": "yolo",
                "model": "ultralytics/yolo11n",
                "python": sys.executable,
                "training": {"epochs": 1, "backend_options": {"mosaic": 0.0}},
                "grid": {"seed": [0, 1], "backend_options.lr0": [0.001, 0.01]},
            }
        ],
    }
    jobs = _jobs(plan, tmp_path)
    assert len(jobs) == 4 and len({job["name"] for job in jobs}) == 4
    assert {job["training"]["seed"] for job in jobs} == {0, 1}
    assert all(job["training"]["backend_options"]["mosaic"] == 0 for job in jobs)
    assert all(job["python"] == str(Path(sys.executable).resolve()) for job in jobs)
    plan["experiments"][0]["grid"] = {"epochs": []}  # type: ignore[index]
    with pytest.raises(ValueError, match="nonempty lists"):
        _jobs(plan, tmp_path)


def test_comparison_retains_metric_namespaces_and_never_ranks_test(tmp_path: Path) -> None:
    results = [
        {
            "name": "first",
            "model": "example",
            "evaluator": "protocol-a",
            "training": {"seed": 0},
            "validation": {"metrics": {"backend-a/AP": 0.4}},
        },
        {
            "name": "second",
            "model": "example",
            "evaluator": "protocol-b",
            "training": {"seed": 0},
            "validation": {"metrics": {"backend-b/AP": 0.6}},
        },
    ]
    _report(tmp_path, results, "dataset-fingerprint")
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["test_evaluated"] is False and report["selection_split"] == "val"
    assert "backend-a/AP" in (tmp_path / "comparison.csv").read_text()
    assert "backend-b/AP" in (tmp_path / "comparison.csv").read_text()
    assert "cross-protocol ranking" in json.dumps(report)


def test_export_rejects_unsupported_format_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="not integrated"):
        Vision().export(format="tensorrt")


def test_mmdet_export_requires_representative_images(tmp_path: Path) -> None:
    from wvision.core.registry import vision_adapter

    adapter = vision_adapter("mmdetection/faster-rcnn-r50", sys.executable)
    context = DetectionContext(tmp_path / "weights.pth", None, "supervised", 128, 2)
    with pytest.raises(ValueError, match="two representative"):
        adapter.export(context, tmp_path, "onnx")


def test_seed_aggregation_preserves_undefined_metrics(tmp_path: Path) -> None:
    results = [
        {
            "name": f"seed-{seed}",
            "model": "example",
            "evaluator": "common",
            "training": {"seed": seed, "batch": 2},
            "validation": {
                "metrics": {"wforge/coco/bbox_mAP": value, "wforge/coco/bbox_mAP_l": -1.0}
            },
        }
        for seed, value in [(0, 0.2), (1, 0.4)]
    ]
    _report(tmp_path, results, "fingerprint")
    aggregate = json.loads((tmp_path / "comparison.json").read_text())["aggregates"][0]
    assert aggregate["distinct_seeds"] == 2
    assert aggregate["metrics"]["wforge/coco/bbox_mAP"]["mean"] == pytest.approx(0.3)
    assert aggregate["metrics"]["wforge/coco/bbox_mAP"]["sample_stddev"] == pytest.approx(
        0.141421356
    )
    assert aggregate["metrics"]["wforge/coco/bbox_mAP_l"] == {
        "count": 0,
        "mean": None,
        "sample_stddev": None,
    }


@pytest.mark.parametrize(
    "devices,device",
    [
        ([0], "cuda:0"),
        ([0, 0], "cuda:0"),
        ([0, True], "cuda:0"),
        ([0, -1], "cuda:0"),
        ([0, 1], "cpu"),
        ([1, 2], "cuda:0"),
    ],
)
def test_invalid_distributed_requests(devices: list[int], device: str) -> None:
    with pytest.raises(ValueError):
        validate_devices(devices, device)


def test_distributed_mask_and_rank_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c")
    assert distributed_visibility([2, 0]) == "GPU-c,GPU-a"
    monkeypatch.setenv("WVISION_MANAGED_DISTRIBUTED", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    request: dict[str, Any] = {
        "directory": str(tmp_path),
        "config": {"device": "cuda:0", "devices": [0, 1]},
    }
    directory, device = worker_context(request)
    assert device == "cuda:1" and directory == tmp_path / "ranks/rank-1"
    assert request["upstream_directory"] == str(tmp_path / "upstream")
    assert request["data_directory"] == str(tmp_path / "data")
    monkeypatch.delenv("WVISION_MANAGED_DISTRIBUTED")
    with pytest.raises(ValueError, match="must be launched"):
        worker_context({"directory": str(tmp_path), "config": request["config"]})
