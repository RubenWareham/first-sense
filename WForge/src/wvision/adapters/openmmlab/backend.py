"""Shared explicit OpenMMLab interpreter selection; no model-specific configuration."""

import os
from pathlib import Path


def interpreter(value: str | Path | None) -> Path:
    reference = value or os.environ.get("WVISION_MMDET_PYTHON")
    if not reference:
        raise ValueError(
            "MMDetection requires its isolated environment. Pass backend_python=... or "
            "set WVISION_MMDET_PYTHON; see docs/SEMI_SUPERVISED.md."
        )
    path = Path(reference).resolve(strict=True)
    if not path.is_file():
        raise ValueError("backend_python must point to the isolated Python executable")
    return path
