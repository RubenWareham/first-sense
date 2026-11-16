"""Ultralytics implementation of the shared detection adapter contract."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from wvision.adapters.base import (
    DetectionAdapter,
    DetectionContext,
    DetectionSettings,
    Initialization,
    ModelSpec,
    TrainingOutput,
)
from wvision.datasets.detection import Dataset, sha256
from wvision.models.ultralytics.yolo11 import operations as yolo
from wvision.models.ultralytics.yolo11.config import TrainConfig


class UltralyticsDetection:
    checkpoint_suffix = ".pt"
    external_checkpoint = True
    needs_model_config = False
    evaluator = "ultralytics 8.3.4 detection AP"
    max_detections = 300
    python: Path | None = None

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        if python:
            raise ValueError("backend_python is not used by the in-process Ultralytics adapter")
        self.spec = spec

    def resolve(self, values: dict[str, Any]) -> TrainConfig:
        return TrainConfig.resolve(values)

    def selection(self, learning: str) -> dict[str, str]:
        return dict(yolo.SELECTION)

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if inputs.detector_run is not None:
            raise ValueError(
                "YOLO does not support detector_run initialization; use its checkpoint input"
            )
        if inputs.pretraining_run is not None:
            raise ValueError("pretraining_run requires MMDetection with backbone_style=pytorch")
        if inputs.backbone_weights is not None:
            raise ValueError("YOLO initialization uses checkpoint, not backbone_weights")
        if inputs.checkpoint is not None:
            self.spec.require_initialization(self.spec.model_id, "checkpoint", settings.learning)

    def initialization(
        self, inputs: Initialization, dataset: Dataset, settings: DetectionSettings
    ) -> dict[str, Any]:
        return {
            "checkpoint": str(inputs.checkpoint) if inputs.checkpoint else None,
            "checkpoint_sha256": sha256(inputs.checkpoint) if inputs.checkpoint else None,
            "compatibility": asdict(
                self.spec.require_initialization(
                    self.spec.model_id,
                    "checkpoint",
                    settings.learning,
                )
            )
            if inputs.checkpoint
            else None,
        }

    def check_dependency(self) -> None:
        yolo.check_dependency()

    def train(
        self,
        dataset: Dataset,
        settings: DetectionSettings,
        initialization: dict[str, Any],
        directory: Path,
    ) -> TrainingOutput:
        assert isinstance(settings, TrainConfig)
        checkpoint = Path(initialization["checkpoint"]) if initialization["checkpoint"] else None
        result = yolo.train(checkpoint, dataset.write_upstream(directory), settings, directory)
        return TrainingOutput(Path(result["best"]), Path(result["last"]), result["metrics"])

    def evaluate(
        self, dataset: Dataset, split: str, context: DetectionContext, directory: Path
    ) -> dict[str, float]:
        return yolo.evaluate(
            context.checkpoint,
            dataset.write_upstream(directory, split),
            dataset.classes,
            directory,
            context.imgsz,
            context.batch,
            device=context.device,
        )

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int:
        return yolo.predict(
            context.checkpoint,
            [str(path) for path in samples],
            directory,
            context.imgsz,
            device=context.device,
        )


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return UltralyticsDetection(spec, python)
