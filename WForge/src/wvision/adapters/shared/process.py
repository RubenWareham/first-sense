"""Shared versioned subprocess transport; adapters choose fixed packaged workers."""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

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
        process = subprocess.Popen(
            [str(python), str(worker), str(request_file), str(result_file)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                **device_env,
                "MPLCONFIGDIR": str(font_cache),
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
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
