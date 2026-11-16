"""Real CUDA YOLO and DINOv3/DETR workflows; optional licensed weights are separate."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wvision import Vision
from wvision.adapters.shared.process import stop_process_tree

pytestmark = [pytest.mark.integration, pytest.mark.cuda]


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def cli(arguments: list[str]) -> dict[str, Any]:
    wrapper = (
        "import socket,runpy; "
        "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(OSError('offline')); "
        "socket.create_connection=socket.socket.connect; "
        "runpy.run_module('wvision',run_name='__main__')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", wrapper, *arguments],
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


def check_device(directory: Path, expected: str = "cuda:0") -> None:
    device = read(directory / "device.json")
    assert device["requested"] == expected
    assert device["model_devices"] == {"model": [expected]}
    if expected != "cpu":
        assert device["allocation_device"] == expected and device["gpu_name"]
        assert device["peak_memory_allocated_bytes"] > 0


def test_cuda_yolo_train_reload_and_cpu_gpu_switching(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("WVISION_YOLO_CUDA_TEST") != "1":
        pytest.skip("Set WVISION_YOLO_CUDA_TEST=1 inside the isolated YOLO CUDA environment")
    import socket

    import torch

    assert torch.cuda.is_available()
    original_device = torch.cuda.current_device()
    monkeypatch.setattr(
        socket.socket, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline"))
    )
    output = tmp_path / "runs"
    result = cli(
        [
            "train",
            "--model",
            "models/detection/yolo11n/smoke.yaml",
            "--data",
            str(manifest),
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
    )
    directory = Path(result["artifact_dir"])
    check_device(directory)
    restored = Vision.from_run(directory)
    check_device(restored.val(output=output).artifact_dir)
    check_device(restored.test(output=output).artifact_dir)
    sample = manifest.parent / "images/test-0.png"
    predicted = restored.predict(sample, output=output)
    check_device(predicted.artifact_dir)
    assert predicted.sample_count == 1
    assert read(directory / "run.json")["config"]["device"] == "cuda:0"
    # One process must support CPU -> CUDA -> CPU without masking or remapping GPUs.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    check_device(restored.predict(sample, device="cpu", output=output).artifact_dir, "cpu")
    check_device(restored.predict(sample, output=output).artifact_dir)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    assert torch.cuda.current_device() == original_device
    fine_tuned = Vision("models/detection/yolo11n/smoke.yaml").train(
        manifest, checkpoint=Path(result["best_checkpoint"]), device="cuda:0", output=output
    )
    check_device(fine_tuned.artifact_dir)
    invalid = tmp_path / "invalid"
    with pytest.raises(ValueError, match="CUDA devices are visible"):
        Vision().train(manifest, device="cuda:99", output=invalid)
    assert read(next(invalid.glob("train-*")) / "status.json")["status"] == "failed"


def test_yolo_cuda_request_with_cpu_torch_fails(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("WVISION_YOLO_CPU_TEST") != "1":
        pytest.skip("Set WVISION_YOLO_CPU_TEST=1 inside the verified CPU YOLO environment")
    import socket

    import torch

    assert torch.version.cuda is None
    monkeypatch.setattr(
        socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        Vision("models/detection/yolo11n/smoke.yaml").train(
            manifest, device="cuda:0", output=tmp_path / "failed"
        )
    directory = next((tmp_path / "failed").glob("train-*"))
    assert read(directory / "status.json")["status"] == "failed"
    assert not (directory / "upstream").exists()


@pytest.fixture
def dino_cuda(monkeypatch: pytest.MonkeyPatch) -> str:
    backend = os.environ.get("WVISION_TRANSFORMERS_CUDA_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_TRANSFORMERS_CUDA_PYTHON to the isolated CUDA DINOv3 backend")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    return backend


def test_cuda_dinov3_random_detection(manifest: Path, tmp_path: Path, dino_cuda: str) -> None:
    output = tmp_path / "runs"
    result = cli(
        [
            "train",
            "--model",
            "models/detection/dinov3-convnext-tiny-detr/smoke.yaml",
            "--data",
            str(manifest),
            "--backend-python",
            dino_cuda,
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
    )
    directory = Path(result["artifact_dir"])
    check_device(directory)
    assert read(directory / "model-source.json")["parameter_count"] == 45_537_638
    model = Vision.from_run(directory)
    check_device(model.val(output=output).artifact_dir)
    check_device(model.test(output=output).artifact_dir)
    sample = manifest.parent / "images/test-0.png"
    prediction = model.predict(sample, output=output)
    check_device(prediction.artifact_dir)
    record = read(prediction.predictions)
    assert record["width"] == record["height"] == 128 and len(record["boxes"]) == 100
    assert record["classes"] == {"0": "rectangle"}
    cpu = cli(
        [
            "predict",
            "--run",
            str(directory),
            "--source",
            str(sample),
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    check_device(Path(cpu["artifact_dir"]), "cpu")
    cpu_backend = os.environ.get("WVISION_TRANSFORMERS_PYTHON")
    if cpu_backend:
        portable = Vision.from_run(directory, backend_python=cpu_backend).predict(
            sample, device="cpu", output=output
        )
        check_device(portable.artifact_dir, "cpu")
    assert not (directory / "test.coco.json").exists()


def test_cuda_dinov3_local_frozen_backbone(manifest: Path, tmp_path: Path, dino_cuda: str) -> None:
    weights = tmp_path / "owned-backbone"
    subprocess.run(
        [
            dino_cuda,
            "-c",
            "import sys; "
            "from transformers import DINOv3ConvNextConfig,DINOv3ConvNextModel,set_seed; "
            "set_seed(17); "
            "DINOv3ConvNextModel(DINOv3ConvNextConfig()).save_pretrained(sys.argv[1])",
            str(weights),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    (weights / "wvision-source.json").write_text(
        json.dumps(
            {
                "source": "owned random fixture",
                "supervision": "random",
                "license": "upstream code terms",
            }
        ),
        encoding="utf-8",
    )
    trained = Vision(
        "models/detection/dinov3-convnext-tiny-detr/frozen-backbone.yaml", backend_python=dino_cuda
    ).train(
        manifest,
        backbone_weights=weights,
        device="cuda:0",
        max_iters=2,
        imgsz=64,
        batch=2,
        output=tmp_path / "runs",
    )
    check_device(trained.artifact_dir)
    transfer = read(trained.artifact_dir / "transfer.json")
    assert transfer["tensor_count"] == 178 and transfer["all_loaded_tensors_equal"]
    verification = (
        "import sys,torch; from transformers import DINOv3ConvNextBackbone,DetrForObjectDetection; "
        "a=DINOv3ConvNextBackbone.from_pretrained(sys.argv[1],local_files_only=True).state_dict(); "
        "b=DetrForObjectDetection.from_pretrained(sys.argv[2],local_files_only=True)"
        ".model.backbone.model.state_dict(); "
        "assert a.keys()==b.keys(); assert all(torch.equal(a[k],b[k]) for k in a)"
    )
    subprocess.run(
        [dino_cuda, "-c", verification, str(weights), str(trained.best_checkpoint.parent)],
        check=True,
        capture_output=True,
        timeout=180,
    )


def test_cuda_dinov3_missing_cuda_does_not_fall_back(
    manifest: Path, tmp_path: Path, dino_cuda: str
) -> None:
    cases = [(dino_cuda, "cuda:99")]
    cpu_backend = os.environ.get("WVISION_TRANSFORMERS_PYTHON")
    if cpu_backend:
        cases.append((cpu_backend, "cuda:0"))
    for index, (backend, device) in enumerate(cases):
        output = tmp_path / str(index)
        with pytest.raises(RuntimeError, match="backend.log"):
            Vision(
                "models/detection/dinov3-convnext-tiny-detr/smoke.yaml", backend_python=backend
            ).train(manifest, device=device, output=output)
        directory = next(output.glob("train-*"))
        assert "CUDA" in (directory / "backend.log").read_text(encoding="utf-8")
        assert not (directory / "resolved-transformers.json").exists()
