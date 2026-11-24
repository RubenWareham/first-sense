"""Shared explicit OpenMMLab interpreter selection; no model-specific configuration."""

import os
from pathlib import Path


def interpreter(
    value: str | Path | None,
    *,
    env: str = "WVISION_MMDET_PYTHON",
    backend: str = "MMDetection",
    guide: str = "SEMI_SUPERVISED.md",
) -> Path:
    reference = value or os.environ.get(env)
    if not reference:
        raise ValueError(
            f"{backend} requires its isolated environment. Pass backend_python=... or "
            f"set {env}; see docs/{guide}."
        )
    path = Path(reference).resolve(strict=True)
    if not path.is_file():
        raise ValueError("backend_python must point to the isolated Python executable")
    return path
