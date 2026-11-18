"""Explicit device selection; also importable by standalone backend workers."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


def validate_precision(value: str, device: str, supported: tuple[str, ...]) -> None:
    if value not in supported:
        raise ValueError(f"precision must be one of {supported}")
    if value != "fp32" and device == "cpu":
        raise ValueError("Mixed precision requires CUDA; use precision='fp32' for CPU")


def validate_devices(value: list[int] | None, device: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, list)
        or len(value) < 2
        or any(type(index) is not int or index < 0 for index in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("devices must be a list of at least two distinct nonnegative GPU indices")
    if device != f"cuda:{value[0]}":
        raise ValueError("For distributed training, device must select the first entry in devices")


def distributed_visibility(devices: list[int]) -> str:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return ",".join(map(str, devices))
    visible = [item.strip() for item in mask.split(",") if item.strip()]
    if visible == ["-1"] or max(devices) >= len(visible):
        raise ValueError("Requested distributed GPUs are unavailable under CUDA_VISIBLE_DEVICES")
    return ",".join(visible[index] for index in devices)


def worker_context(request: dict[str, Any], *, isolate_single: bool = False) -> tuple[Path, str]:
    """Prepare rank-local audits while upstream owns the shared training directory."""
    root = Path(request["directory"])
    selected = request.get("device", request.get("config", {}).get("device", "cpu"))
    devices = request.get("config", {}).get("devices")
    if devices is None:
        require_single_process()
        execution = isolate_worker_device(selected) if isolate_single else selected
        request["execution_device"] = execution
        return root, execution
    validate_devices(devices, selected)
    rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        os.environ.get("WVISION_MANAGED_DISTRIBUTED") != "1"
        or int(os.environ.get("WORLD_SIZE", "1")) != len(devices)
        or rank not in range(len(devices))
    ):
        raise ValueError(
            "Distributed training must be launched through the WVision worker transport"
        )
    directory = root if rank == 0 else root / "ranks" / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=True)
    request.update(
        directory=str(directory),
        upstream_directory=str(root / "upstream"),
        data_directory=str(root / "data"),
        execution_device=f"cuda:{rank}",
        requested_device=f"cuda:{devices[rank]}",
        distributed=True,
        rank=rank,
    )
    if rank and request.get("resume"):
        source = request["resume"]
        path = Path(source["checkpoint"])
        copy = directory / "resume-source" / path.name
        copy.parent.mkdir()
        if path.is_dir():
            shutil.copytree(path, copy)
        else:
            shutil.copyfile(path, copy)
        request["resume"] = {**source, "checkpoint": str(copy)}
    os.environ["ACCELERATE_TORCH_DEVICE"] = f"cuda:{rank}"
    os.environ["ACCELERATE_USE_CPU"] = "false"
    return directory, request["execution_device"]


def runtime_precision(value: str, directory: Path) -> None:
    import torch

    if value == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested BF16, but the selected CUDA device does not support it")
    path = directory / "device.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info.update(
        precision=value,
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def canonical_device(value: str) -> str:
    if value == "cuda":
        return "cuda:0"
    if isinstance(value, str) and (value == "cpu" or re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value)):
        return value
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'; use devices=[...] for DDP")


def device_family(value: str) -> str:
    return canonical_device(value).split(":")[0]


def require_single_process() -> None:
    if os.environ.get("WORLD_SIZE", "1") != "1" or os.environ.get("LOCAL_RANK", "-1") != "-1":
        raise ValueError(
            "Distributed launches must be managed by WVision; use devices=[...] without torchrun"
        )


def isolate_worker_device(value: str) -> str:
    """Pin one visible GPU before torch imports; prevent Trainer's automatic DataParallel."""
    require_single_process()
    selected = canonical_device(value)
    if selected == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        index = int(selected.split(":")[1])
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is None:
            visible = str(index)
        else:
            devices = [item.strip() for item in mask.split(",") if item.strip()]
            if not devices or devices == ["-1"]:
                raise ValueError(
                    f"Requested {selected}, but CUDA is unavailable under the visibility mask"
                )
            if index >= len(devices):
                raise ValueError(
                    f"Requested {selected}; only {len(devices)} CUDA devices are visible"
                )
            visible = devices[index]
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        selected = "cuda:0"
    os.environ["ACCELERATE_TORCH_DEVICE"] = selected
    os.environ["ACCELERATE_USE_CPU"] = "true" if selected == "cpu" else "false"
    return selected


def runtime_device(value: str, directory: Path, *, requested: str | None = None) -> None:
    """Verify actual CUDA allocation before invoking upstream training. Never fall back."""
    import torch

    selected = canonical_device(value)
    original = canonical_device(requested) if requested is not None else selected
    info: dict[str, Any] = {
        "requested": original,
        "execution_device": selected,
        "torch_version": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "memory_scope": "CUDA allocator peak at audit; upstream logging may reset peak counters",
    }
    if selected != "cpu":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested {original}, but CUDA is unavailable. Install the documented CUDA "
                "backend and check the NVIDIA driver; use device='cpu' explicitly for CPU."
            )
        index = int(selected.split(":")[1])
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {selected}; only {torch.cuda.device_count()} CUDA devices are visible"
            )
        torch.cuda.set_device(index)
        probe = torch.empty(1, device=selected)
        torch.cuda.synchronize(index)
        info.update(
            allocation_device=str(probe.device),
            gpu_name=torch.cuda.get_device_name(index),
            total_memory_bytes=torch.cuda.get_device_properties(index).total_memory,
            compute_capability=list(torch.cuda.get_device_capability(index)),
        )
        del probe
        torch.cuda.reset_peak_memory_stats(index)
    (directory / "device.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def record_model_device(model: Any, value: str, directory: Path) -> None:
    """Audit placement only; all model/training behavior remains upstream-owned."""
    import torch

    selected = canonical_device(value)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    branches = (
        {"teacher": model.teacher, "student": model.student}
        if hasattr(model, "teacher")
        else {"model": model}
    )
    placements = {}
    for name, branch in branches.items():
        devices = sorted({str(parameter.device) for parameter in branch.parameters()})
        if devices != [selected]:
            raise RuntimeError(
                f"{name} is on {devices}, expected {selected}; refusing device fallback"
            )
        placements[name] = devices
    path = directory / "device.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["model_devices"] = placements
    if selected != "cpu":
        torch.cuda.synchronize(selected)
        info["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(selected)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
