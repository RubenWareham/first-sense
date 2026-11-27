"""Ultralytics YOLO11 implementation of the shared vision workflow."""

import sys
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
from wvision.adapters.shared.process import invoke_worker
from wvision.adapters.ultralytics.backend import BACKEND_VERSION
from wvision.datasets.detection import Dataset, sha256
from wvision.datasets.pose import JOINTS
from wvision.models.ultralytics.yolo11 import operations as yolo
from wvision.models.ultralytics.yolo11.config import TrainConfig


class UltralyticsYOLO11:
    checkpoint_suffix = ".pt"
    needs_model_config = False
    max_detections = 300
    python: Path | None = None

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        if python:
            raise ValueError("backend_python is not used by the in-process Ultralytics adapter")
        self.spec = spec
        self.scale = spec.model_id.removesuffix("-pose")[-1]
        self.external_checkpoint = spec.task == "detect"
        kind = "pose" if spec.task == "pose" else "detection"
        self.evaluator = f"ultralytics {BACKEND_VERSION} {kind} AP"

    def resolve(self, values: dict[str, Any]) -> TrainConfig:
        config = TrainConfig.resolve(values)
        if self.spec.task == "pose" and (
            config.precision != "fp32" or config.devices or config.checkpoint_interval != 1
        ):
            raise ValueError("Pose supports single-device FP32 training without recovery options")
        return config

    def selection(self, learning: str) -> dict[str, str]:
        return dict(yolo.POSE_SELECTION if self.spec.task == "pose" else yolo.SELECTION)

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if self.spec.task == "pose" and any(
            source is not None
            for source in (
                inputs.checkpoint,
                inputs.pretraining_run,
                inputs.backbone_weights,
                inputs.detector_run,
                inputs.resume_run,
            )
        ):
            raise ValueError("Pose supports random initialization only, without optimizer resume")
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
        if dataset.skeleton is not None:
            joints = tuple(dataset.skeleton["joints"])
            if len(joints) == len(JOINTS) and joints != JOINTS:
                raise ValueError(
                    "YOLO11-pose uses COCO sigmas for 17 joints and requires COCO joint order"
                )
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
        if settings.devices:
            result = invoke_worker(
                Path(sys.executable),
                Path(__file__).with_name("worker.py"),
                directory,
                {
                    "operation": "train",
                    "scale": self.scale,
                    "config": settings.to_dict(),
                    "data": str(dataset.write_upstream(directory)),
                    **initialization,
                },
                "YOLO DDP",
            )
            return TrainingOutput(Path(result["best"]), Path(result["last"]), result["metrics"])
        checkpoint = (
            Path(initialization["checkpoint"]) if initialization.get("checkpoint") else None
        )
        result = yolo.train(
            checkpoint,
            dataset.write_upstream(directory),
            settings,
            directory,
            task=self.spec.task,
            scale=self.scale,
            **({"resume": initialization["resume"]} if initialization.get("resume") else {}),
        )
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
            task=self.spec.task,
            scale=self.scale,
            **({"skeleton": dataset.skeleton} if dataset.skeleton is not None else {}),
        )

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int:
        options = {}
        if context.score_threshold is not None:
            options["conf"] = context.score_threshold
        if context.max_detections is not None:
            options["max_det"] = context.max_detections
        return yolo.predict(
            context.checkpoint,
            [str(path) for path in samples],
            directory,
            context.imgsz,
            device=context.device,
            task=self.spec.task,
            scale=self.scale,
            skeleton=context.skeleton,
            **options,
        )

    def export(self, context: DetectionContext, directory: Path, format: str) -> Path:
        self.spec.require_export(format)
        if context.verification_images:
            raise ValueError(
                "YOLO export verifies tensor inputs; verification_images is for MMDetection"
            )
        return yolo.export(context.checkpoint, context.imgsz, directory, scale=self.scale)


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return UltralyticsYOLO11(spec, python)
