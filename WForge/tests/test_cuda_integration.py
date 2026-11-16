"""Real CUDA training of upstream Soft Teacher and SimSiam; no synthetic weight stubs."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision, compare_pretraining, prepare_label_budget, pretrain
from wvision.adapters.shared.process import stop_process_tree

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection, pytest.mark.cuda]


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def cli(arguments: list[str]) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-m", "wvision", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=1200)
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    assert process.returncode == 0, stdout + stderr
    return dict(json.loads(stdout))


def check_cuda(directory: Path, branches: set[str]) -> None:
    device = read(directory / "device.json")
    assert device["requested"] == device["allocation_device"] == "cuda:0"
    assert device["gpu_name"] and device["peak_memory_allocated_bytes"] > 0
    assert set(device["model_devices"]) == branches
    assert all(value == ["cuda:0"] for value in device["model_devices"].values())


@pytest.fixture
def cuda_backend(monkeypatch: pytest.MonkeyPatch) -> str:
    backend = os.environ.get("WVISION_MMDET_CUDA_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_CUDA_PYTHON to the verified CUDA backend")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    return backend


def test_cuda_simsiam_to_soft_teacher(manifest: Path, tmp_path: Path, cuda_backend: str) -> None:
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    output = tmp_path / "runs"
    representation = pretrain(
        data,
        model="models/representation/simsiam-r50/smoke.yaml",
        device="cuda:0",
        backend_python=cuda_backend,
        output=output,
    )
    check_cuda(representation.artifact_dir, {"model"})
    model = Vision(
        "models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml",
        backend_python=cuda_backend,
    )
    trained = model.train(
        data, device="cuda:0", pretraining_run=representation.artifact_dir, output=output
    )
    check_cuda(trained.artifact_dir, {"teacher", "student"})
    transfer = read(trained.artifact_dir / "transfer.json")
    assert transfer["tensor_count"] == 318 and transfer["all_loaded_tensors_equal"]
    scalars = read(trained.artifact_dir / "training-scalars.json")
    assert any(value > 0 for key, value in scalars.items() if "unsup_loss" in key)
    assert read(trained.artifact_dir / "train_unlabeled.coco.json")["annotations"] == []
    assert not (trained.artifact_dir / "test.coco.json").exists()
    restored = Vision.from_run(trained.artifact_dir)
    check_cuda(restored.val(output=output).artifact_dir, {"teacher", "student"})
    check_cuda(restored.test(output=output).artifact_dir, {"teacher", "student"})
    sample = data.parent / "test/images/0-test-0.png"
    predicted = restored.predict(sample, output=output)
    assert predicted.sample_count == 1
    check_cuda(predicted.artifact_dir, {"teacher", "student"})
    cpu = cli(
        [
            "predict",
            "--run",
            str(trained.artifact_dir),
            "--source",
            str(sample),
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    assert cpu["sample_count"] == 1
    assert read(Path(cpu["artifact_dir"]) / "device.json")["model_devices"] == {
        "teacher": ["cpu"],
        "student": ["cpu"],
    }
    cpu_backend = os.environ.get("WVISION_MMDET_PYTHON")
    if cpu_backend:
        portable = Vision.from_run(trained.artifact_dir, backend_python=cpu_backend).predict(
            sample, device="cpu", output=output
        )
        assert portable.sample_count == 1
        assert read(portable.artifact_dir / "device.json")["requested"] == "cpu"


def test_cuda_comparison_and_supervised_initialization(
    manifest: Path, tmp_path: Path, cuda_backend: str
) -> None:
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    comparison = cli(
        [
            "compare",
            "--data",
            str(data),
            "--model",
            "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
            "--device",
            "cuda:0",
            "--backend-python",
            cuda_backend,
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    report = read(Path(comparison["report"]))
    assert report["compute"]["device"] == "cuda:0" and report["label_budget"]["labeled_count"] == 2
    check_cuda(Path(comparison["supervised_run"]), {"model"})
    check_cuda(Path(comparison["semi_supervised_run"]), {"teacher", "student"})
    initialized = Vision(
        "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml", backend_python=cuda_backend
    ).train(
        data, detector_run=comparison["supervised_run"], device="cuda:0", output=tmp_path / "runs"
    )
    check_cuda(initialized.artifact_dir, {"teacher", "student"})
    transfer = read(initialized.artifact_dir / "transfer.json")
    assert transfer["tensor_count"] == 348 and transfer["all_loaded_tensors_equal"]


def test_cuda_failures_do_not_fall_back(manifest: Path, tmp_path: Path, cuda_backend: str) -> None:
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    cases = [(cuda_backend, "cuda:99", "CUDA devices are visible")]
    cpu_backend = os.environ.get("WVISION_MMDET_PYTHON")
    if cpu_backend:
        cases.append((cpu_backend, "cuda:0", "CUDA is unavailable"))
    for index, (backend, device, message) in enumerate(cases):
        output = tmp_path / f"failure-{index}"
        with pytest.raises(RuntimeError, match="backend.log"):
            pretrain(
                data,
                model="models/representation/simsiam-r50/smoke.yaml",
                device=device,
                backend_python=backend,
                output=output,
            )
        directory = next(output.glob("pretrain-*"))
        assert message in (directory / "backend.log").read_text(encoding="utf-8")
        assert read(directory / "status.json")["status"] == "failed"
        assert not (directory / "resolved-mmpretrain.json").exists()


def test_cuda_pretraining_comparison(manifest: Path, tmp_path: Path, cuda_backend: str) -> None:
    data = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2)
    output = tmp_path / "runs"
    representation = pretrain(
        data,
        model="models/representation/simsiam-r50/smoke.yaml",
        device="cuda:0",
        backend_python=cuda_backend,
        output=output,
    )
    result = compare_pretraining(
        data,
        pretraining_run=representation.artifact_dir,
        model="models/detection/faster-rcnn-r50/transfer-smoke.yaml",
        device="cuda:0",
        backend_python=cuda_backend,
        output=output,
    )
    report = read(result.report)
    assert report["compute"]["downstream_config"]["device"] == "cuda:0"
    assert report["label_budget"]["labeled_count"] == 2
    for run in (result.random_run, result.pretrained_run):
        check_cuda(run, {"model"})
    transfer = read(result.pretrained_run / "transfer.json")
    assert transfer["tensor_count"] == 318 and transfer["all_loaded_tensors_equal"]
