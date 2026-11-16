"""Validated configuration for the isolated upstream SimSiam workflow."""

import math
from dataclasses import asdict, dataclass
from typing import Any

from wvision.adapters.shared.devices import canonical_device
from wvision.core.config import reject_unknown

MODEL_ID = "mmpretrain/simsiam-r50"


@dataclass(frozen=True)
class PretrainConfig:
    schema_version: int = 1
    max_iters: int = 1000
    imgsz: int = 224
    batch: int = 2
    seed: int = 0
    device: str = "cpu"
    lr: float = 0.05

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Pretraining schema_version must be 1")
        for name in ("max_iters", "imgsz", "batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.imgsz < 64 or self.imgsz % 32:
            raise ValueError("imgsz must be at least 64 and a multiple of 32")
        if self.batch < 2:
            raise ValueError("SimSiam batch must be at least 2 for batch normalization")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "device", canonical_device(self.device))
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")

    @classmethod
    def resolve(cls, values: dict[str, Any]) -> "PretrainConfig":
        reject_unknown(values, set(cls.__dataclass_fields__), "pretraining configuration")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
