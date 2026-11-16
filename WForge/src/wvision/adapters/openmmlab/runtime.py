"""Shared standalone OpenMMLab environment preflight and JSON writing."""

import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dependencies(directory: Path) -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The initial MMDetection worker requires Python 3.10")
    expected = {"mmdet": "3.3.0", "mmcv": "2.1.0", "mmengine": "0.10.7"}
    torch_version = metadata.version("torch")
    builds = {"2.1.0+cpu": "0.16.0+cpu", "2.1.0+cu121": "0.16.0+cu121"}
    if torch_version not in builds:
        raise RuntimeError("OpenMMLab requires torch==2.1.0+cpu or torch==2.1.0+cu121")
    expected["torchvision"] = builds[torch_version]
    for name, version in expected.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"MMDetection worker requires {name}=={version}")
    import mmcv.ops  # noqa: F401 -- importing compiled operators is part of the preflight

    dump(
        directory / "backend-environment.json",
        {
            "python": sys.version,
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        },
    )
