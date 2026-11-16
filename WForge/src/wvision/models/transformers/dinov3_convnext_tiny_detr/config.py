"""DINOv3/DETR settings and strict local-backbone source validation."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wvision.adapters.shared.devices import canonical_device
from wvision.core.config import reject_unknown
from wvision.datasets.detection import sha256


@dataclass(frozen=True)
class DinoDETRConfig:
    schema_version: int = 1
    learning: str = "supervised"
    max_iters: int = 1000
    imgsz: int = 640
    batch: int = 2
    seed: int = 0
    device: str = "cpu"
    lr: float = 0.0001
    initialization: str = "local_backbone"
    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("DINOv3 schema_version must be 1")
        if self.learning != "supervised":
            raise ValueError("Only downstream supervised detection is integrated for DINOv3/DETR")
        for key in ("max_iters", "imgsz", "batch"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.imgsz < 64 or self.imgsz % 32:
            raise ValueError("imgsz must be at least 64 and divisible by 32")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "device", canonical_device(self.device))
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if self.initialization not in ("local_backbone", "random"):
            raise ValueError("initialization must be local_backbone or explicit random")
        if type(self.freeze_backbone) is not bool:
            raise ValueError("freeze_backbone must be a boolean")

    @classmethod
    def resolve(cls, values: dict[str, Any]) -> "DinoDETRConfig":
        reject_unknown(values, set(cls.__dataclass_fields__), "DINOv3 configuration")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def local_backbone(path: Path) -> dict[str, Any]:
    """Inspect only local, single-file safetensors checkpoints. Never resolve remote IDs."""
    directory = path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("backbone_weights must be a local Hugging Face model directory")
    config, weights = directory / "config.json", directory / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("backbone_weights requires config.json and model.safetensors")
    values = json.loads(config.read_text(encoding="utf-8"))
    if (
        values.get("model_type") != "dinov3_convnext"
        or values.get("hidden_sizes") != [96, 192, 384, 768]
        or values.get("depths") != [3, 3, 9, 3]
        or values.get("auto_map")
    ):
        raise ValueError("Only the upstream DINOv3 ConvNeXt Tiny configuration is accepted")
    provenance_file = directory / "wvision-source.json"
    provenance = (
        json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance_file.is_file()
        else {
            "supervision": "unknown",
            "source": "user-provided local backbone",
            "pretraining_data": "unknown",
            "license": "supplied by owner",
        }
    )
    if not isinstance(provenance, dict):
        raise ValueError("wvision-source.json must contain a provenance mapping")
    files = {"config.json": sha256(config), "model.safetensors": sha256(weights)}
    if provenance_file.is_file():
        files[provenance_file.name] = sha256(provenance_file)
    return {
        "directory": str(directory),
        "files": files,
        "provenance": provenance,
        "provenance_verified": False,
    }
