"""Validated settings for the registered YOLO11n detection workflow."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from wvision.adapters.shared.devices import canonical_device, validate_devices, validate_precision
from wvision.core.config import reject_unknown


@dataclass(frozen=True)
class TrainConfig:
    schema_version: int = 1
    learning: str = "supervised"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    devices: list[int] | None = None
    checkpoint_interval: int = 1
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Training schema_version must be 1")
        if self.learning != "supervised":
            raise ValueError(
                f"Learning mode {self.learning!r} is not integrated. This adapter supports "
                "supervised YOLO detection only; inspect 'wvision models' for other workflows."
            )
        for name in ("epochs", "imgsz", "batch"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch < 2:
            raise ValueError("batch must be at least 2 for the initial training integration")
        if self.imgsz < 64 or self.imgsz % 32:
            raise ValueError("imgsz must be at least 64 and a multiple of 32")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "device", canonical_device(self.device))
        validate_precision(self.precision, self.device, ("fp32", "fp16"))
        validate_devices(self.devices, self.device)
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        if not isinstance(self.backend_options, dict):
            raise ValueError("backend_options must be a mapping")
        # Restrict escape hatches so backend options cannot override split and run safety.
        allowed = {"lr0", "patience", "mosaic", "mixup", "degrees", "fliplr", "optimizer"}
        reject_unknown(self.backend_options, allowed, "backend_options")
        for key, option in self.backend_options.items():
            if key == "optimizer":
                if option not in ("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"):
                    raise ValueError("backend_options.optimizer is not a supported optimizer")
                continue
            if type(option) not in (int, float) or not math.isfinite(option):
                raise ValueError(f"backend_options.{key} must be a finite number")
            if key == "patience" and (type(option) is not int or option < 0):
                raise ValueError("backend_options.patience must be a nonnegative integer")
            if key in {"mosaic", "mixup", "fliplr"} and not 0 <= option <= 1:
                raise ValueError(f"backend_options.{key} must be between 0 and 1")
            if key == "lr0" and option <= 0:
                raise ValueError("backend_options.lr0 must be positive")
            if key == "degrees" and not 0 <= option <= 180:
                raise ValueError("backend_options.degrees must be between 0 and 180")

    @classmethod
    def resolve(cls, values: dict[str, Any]) -> "TrainConfig":
        reject_unknown(values, set(cls.__dataclass_fields__), "training configuration")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
