"""Adapt MMDetection's supervised RTMDet family to the Vision workflow."""

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from wvision.adapters.base import (
    DetectionAdapter,
    DetectionContext,
    DetectionSettings,
    Initialization,
    ModelSpec,
)
from wvision.adapters.openmmlab.detector import MMDetectionAdapter
from wvision.adapters.shared.devices import canonical_device
from wvision.adapters.shared.process import invoke_worker
from wvision.core.config import reject_unknown
from wvision.datasets.detection import Dataset


@dataclass(frozen=True)
class RTMDetSettings:
    schema_version: int = 1
    learning: str = "supervised"
    max_iters: int = 1000
    imgsz: int = 640
    batch: int = 2
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    checkpoint_interval: int = 100
    backend_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RTMDet(MMDetectionAdapter):
    def resolve(self, values: dict[str, Any]) -> RTMDetSettings:
        reject_unknown(values, set(RTMDetSettings.__dataclass_fields__), "RTMDet settings")
        settings = RTMDetSettings(**values)
        if type(settings.schema_version) is not int or settings.schema_version != 1:
            raise ValueError("RTMDet requires training schema_version=1")
        if settings.learning != "supervised" or settings.precision != "fp32":
            raise ValueError("RTMDet supports supervised FP32 training")
        if type(settings.imgsz) is not int or settings.imgsz < 64 or settings.imgsz % 32:
            raise ValueError("RTMDet imgsz must be at least 64 and a multiple of 32")
        for name, minimum in (
            ("max_iters", 1),
            ("batch", 1),
            ("seed", 0),
            ("checkpoint_interval", 1),
        ):
            value = getattr(settings, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"RTMDet {name} must be an integer >= {minimum}")
        if not isinstance(settings.backend_options, dict):
            raise ValueError("RTMDet backend_options must be a mapping")
        reject_unknown(settings.backend_options, {"lr"}, "RTMDet backend_options")
        if "lr" in settings.backend_options:
            lr = settings.backend_options["lr"]
            if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
                raise ValueError("RTMDet backend_options.lr must be positive and finite")
        return replace(settings, device=canonical_device(settings.device))

    def selection(self, learning: str) -> dict[str, str]:
        return {"metric": "coco/bbox_mAP", "direction": "maximize"}

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if any(
            source is not None
            for source in (
                inputs.checkpoint,
                inputs.pretraining_run,
                inputs.backbone_weights,
                inputs.detector_run,
                inputs.resume_run,
            )
        ):
            raise ValueError("RTMDet supports random initialization without optimizer resume")

    def initialization(
        self, inputs: Initialization, dataset: Dataset, settings: DetectionSettings
    ) -> dict[str, Any]:
        return {"kind": "random"}

    def _invoke(self, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        return invoke_worker(
            self.python,
            Path(__file__).with_name("operations.py"),
            directory,
            {**request, "model_id": self.spec.model_id},
            "RTMDet",
        )

    def export(self, context: DetectionContext, directory: Path, format: str) -> Path:
        self.spec.require_export(format)
        raise ValueError("RTMDet export is not supported")


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return RTMDet(spec, python)
