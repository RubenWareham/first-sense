"""Shared versioned subprocess transport; adapters choose fixed packaged workers."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from wvision.adapters.shared.devices import (
    distributed_visibility,
    require_single_process,
    validate_devices,
)
from wvision.experiments.artifacts import write_json


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
    """Stop our process and its Windows venv redirector children, then reap it."""
    if process.poll() is None:
        if os.name == "nt":
            command = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
            subprocess.run(
                [str(command), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    process.wait()


def invoke_worker(
    python: Path, worker: Path, directory: Path, request: dict[str, Any], label: str
) -> dict[str, Any]:
    request_file, result_file = (
        directory / "backend-request.json",
        directory / "backend-result.json",
    )
    write_json(request_file, {"schema_version": 1, "directory": str(directory), **request})
    font_cache = directory.parent / ".mmdet-matplotlib-cache"
    font_cache.mkdir(exist_ok=True)
    with (directory / "backend.log").open("w", encoding="utf-8") as log:
        device = request.get("device", request.get("config", {}).get("device", "cpu"))
        device_env = (
            {"CUDA_VISIBLE_DEVICES": "-1"}
            if device == "cpu"
            else {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
        )
        environment = {
            **os.environ,
            **device_env,
            "MPLCONFIGDIR": str(font_cache),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Standalone workers must not import application catalog folders
            # (e.g. datasets/) as optional third-party framework packages.
            "PYTHONPATH": "",
        }
        command = [str(python), str(worker), str(request_file), str(result_file)]
        devices = request.get("config", {}).get("devices")
        if devices is not None:
            require_single_process()
            validate_devices(devices, device)
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import sys,torch; "
                    "n=torch.cuda.device_count(); "
                    "print(f'CUDA devices visible: {n}',flush=True); "
                    "sys.exit(0 if torch.cuda.is_available() and n>int(sys.argv[1]) else 2)",
                    str(max(devices)),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=60,
                check=False,
            )
            if probe.returncode:
                raise RuntimeError(
                    "Requested distributed CUDA devices are unavailable; "
                    f"see {directory / 'backend.log'}"
                )
            environment.update(
                CUDA_VISIBLE_DEVICES=distributed_visibility(devices),
                WVISION_MANAGED_DISTRIBUTED="1",
                OMP_NUM_THREADS="2",
            )
            if os.name == "nt":
                environment["USE_LIBUV"] = "0"
            command = [
                str(python),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={len(devices)}",
                str(worker),
                str(request_file),
                str(result_file),
            ]
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            process.wait()
        except BaseException:
            stop_process_tree(process)
            raise
    if process.returncode != 0 or not result_file.is_file():
        raise RuntimeError(
            f"{label} worker failed (exit {process.returncode}); see {directory / 'backend.log'}"
        )
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported {label} worker result schema")
    return dict(result)
