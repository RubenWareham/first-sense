"""Shared pinned Ultralytics runtime, dependency checks, and metric normalization."""

import os
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

from wvision.adapters.shared.devices import canonical_device, require_single_process, runtime_device

BACKEND_VERSION = "8.3.4"
_LOCK = threading.RLock()


def check_dependency() -> None:
    try:
        version = metadata.version("ultralytics")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install the matching PyTorch build first, then from WForge/ (the package root) run: "
            "python -m pip install -e '.[ultralytics]'. Use the pinned CPU setup in "
            "docs/DEVELOPMENT.md or the CUDA setup in docs/GPU.md."
        ) from error
    if version != BACKEND_VERSION:
        raise RuntimeError(
            f"Expected ultralytics=={BACKEND_VERSION}, found {version}. "
            "Use the documented backend environment."
        )


@contextmanager
def backend(
    device: str | None = None,
    directory: Path | None = None,
    *,
    managed_distributed: bool = False,
    requested_device: str | None = None,
) -> Iterator[Any]:
    """Disable upstream networking/loggers without changing installed upstream source.

    Scoped overrides affect process globals, so WVision serializes its own backend calls.
    Applications using Ultralytics concurrently should isolate WVision in another process.
    """
    if not managed_distributed:
        require_single_process()
    check_dependency()
    with _LOCK, ExitStack() as stack:
        settings_dir = Path.cwd() / ".wvision" / "ultralytics"
        settings_dir.mkdir(parents=True, exist_ok=True)
        stack.enter_context(
            patch.dict(
                os.environ, {"YOLO_CONFIG_DIR": str(settings_dir), "YOLO_AUTOINSTALL": "false"}
            )
        )
        from ultralytics import YOLO
        from ultralytics.hub.utils import events
        from ultralytics.utils import SETTINGS

        integrations = (
            "sync",
            "hub",
            "clearml",
            "comet",
            "dvc",
            "mlflow",
            "neptune",
            "raytune",
            "tensorboard",
            "wandb",
        )
        stack.enter_context(patch.dict(SETTINGS, {key: False for key in integrations}))
        stack.enter_context(patch.object(events, "enabled", False))
        stack.enter_context(patch("ultralytics.utils.checks.AUTOINSTALL", False))
        # No plot fonts are needed because this adapter disables upstream plotting.
        stack.enter_context(patch("ultralytics.data.utils.check_font", return_value=None))
        stack.enter_context(
            patch("ultralytics.utils.checks.check_pip_update_available", return_value=False)
        )
        if device is not None:
            import torch
            from ultralytics.utils.torch_utils import select_device

            if directory is None:
                raise ValueError("Device execution requires an operation directory")
            selected = canonical_device(device)
            if selected != "cpu":
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            previous = (
                torch.cuda.current_device()
                if selected != "cpu" and torch.cuda.is_available()
                else None
            )
            if previous is not None:
                stack.callback(torch.cuda.set_device, previous)
            runtime_device(selected, directory, requested=requested_device)

            # Upstream accepts torch.device directly, without rewriting CUDA visibility.
            # Force that documented path at every selector, including final validation.
            def explicit_device(*args: Any, **kwargs: Any) -> Any:
                return select_device(torch.device(selected))

            for module in ("trainer", "validator", "predictor"):
                stack.enter_context(
                    patch(f"ultralytics.engine.{module}.select_device", explicit_device)
                )
        yield YOLO


def metrics(result: Any) -> dict[str, float]:
    return {f"ultralytics/{name}": float(value) for name, value in result.results_dict.items()}
