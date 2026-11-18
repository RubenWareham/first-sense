"""Validated Faster R-CNN / Soft Teacher settings."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from wvision.adapters.shared.devices import canonical_device, validate_devices, validate_precision
from wvision.core.config import reject_unknown

MODEL_ID = "mmdetection/faster-rcnn-r50"


@dataclass(frozen=True)
class MMDetConfig:
    schema_version: int = 1
    learning: str = "supervised"
    max_iters: int = 1000
    imgsz: int = 640
    batch: int = 2
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    devices: list[int] | None = None
    checkpoint_interval: int = 100
    backend_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Training schema_version must be 1")
        if self.learning not in {"supervised", "semi_supervised"}:
            raise ValueError(f"Learning mode {self.learning!r} is not integrated for MMDetection")
        for key in ("max_iters", "imgsz", "batch"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.imgsz < 64 or self.imgsz % 32:
            raise ValueError("imgsz must be at least 64 and a multiple of 32")
        if self.batch < 2 or self.batch % 2:
            raise ValueError("MMDetection batch must be an even integer of at least 2")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "device", canonical_device(self.device))
        validate_precision(self.precision, self.device, ("fp32", "fp16"))
        validate_devices(self.devices, self.device)
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        if not isinstance(self.backend_options, dict):
            raise ValueError("backend_options must be a mapping")
        allowed = {
            "lr",
            "backbone_style",
            "pseudo_label_initial_score_thr",
            "rpn_pseudo_thr",
            "cls_pseudo_thr",
            "reg_pseudo_thr",
        }
        reject_unknown(self.backend_options, allowed, "MMDetection backend_options")
        for key, value in self.backend_options.items():
            if key == "backbone_style":
                if not isinstance(value, str) or value not in {"caffe", "pytorch"}:
                    raise ValueError("backend_options.backbone_style must be caffe or pytorch")
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"backend_options.{key} must be finite")
            if key == "lr":
                if value <= 0:
                    raise ValueError("backend_options.lr must be positive")
            elif self.learning != "semi_supervised" or not 0 <= value <= 1:
                raise ValueError(
                    f"{key} requires semi_supervised learning and a value between 0 and 1"
                )

    @classmethod
    def resolve(cls, values: dict[str, Any]) -> "MMDetConfig":
        reject_unknown(values, set(cls.__dataclass_fields__), "MMDetection configuration")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
