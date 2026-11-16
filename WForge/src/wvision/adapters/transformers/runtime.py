"""Standalone pinned Transformers environment preflight and file-integrity helpers."""

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dependencies(directory: Path) -> None:
    expected = {
        "transformers": "5.22.2",
        "accelerate": "1.10.1",
        "pycocotools": "2.0.11",
        "safetensors": "0.8.0",
    }
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The DINOv3 worker requires Python 3.12")
    builds = {"2.8.0+cpu": "0.23.0+cpu", "2.8.0+cu128": "0.23.0+cu128"}
    torch_version = metadata.version("torch")
    if torch_version not in builds:
        raise RuntimeError("DINOv3 requires torch==2.8.0+cpu or torch==2.8.0+cu128")
    expected["torchvision"] = builds[torch_version]
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"Install {name}=={version}; see docs/DINOV3.md") from error
        if actual != version:
            raise RuntimeError(f"DINOv3 worker requires {name}=={version}; found {actual}")
    dump(
        directory / "backend-environment.json",
        {
            "python": sys.version,
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        },
    )
