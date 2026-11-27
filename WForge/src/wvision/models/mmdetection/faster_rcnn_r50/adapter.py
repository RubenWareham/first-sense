"""MMDetection implementation of the shared detection adapter contract."""

from dataclasses import asdict
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
from wvision.adapters.shared.process import invoke_worker
from wvision.datasets.detection import Dataset
from wvision.models.mmdetection.faster_rcnn_r50.config import MMDetConfig
from wvision.models.mmdetection.faster_rcnn_r50.weights import detector_source
from wvision.models.mmpretrain.simsiam_r50.adapter import load_pretraining


class MMDetection(MMDetectionAdapter):
    def _invoke(self, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        return invoke(self.python, directory, request)

    def resolve(self, values: dict[str, Any]) -> MMDetConfig:
        return MMDetConfig.resolve(values)

    def selection(self, learning: str) -> dict[str, str]:
        return {
            "metric": "teacher/coco/bbox_mAP" if learning == "semi_supervised" else "coco/bbox_mAP",
            "direction": "maximize",
        }

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if inputs.checkpoint is not None:
            raise ValueError(
                "MMDetection initialization uses detector_run; recovery uses resume_run"
            )
        if inputs.backbone_weights is not None:
            raise ValueError("MMDetection backbone transfer uses pretraining_run")
        if inputs.pretraining_run is not None:
            self.spec.require_initialization(
                "mmpretrain/simsiam-r50",
                "pretraining_run",
                settings.learning,
                settings.to_dict()["backend_options"].get("backbone_style", "caffe"),
            )
        if inputs.detector_run is not None:
            self.spec.require_initialization(
                self.spec.model_id, "detector_run", settings.learning, scope=inputs.transfer_scope
            )

    def initialization(
        self, inputs: Initialization, dataset: Dataset, settings: DetectionSettings
    ) -> dict[str, Any]:
        style = settings.to_dict()["backend_options"].get("backbone_style", "caffe")
        rule = None
        if inputs.pretraining_run is not None:
            rule = self.spec.require_initialization(
                "mmpretrain/simsiam-r50", "pretraining_run", settings.learning, style
            )
        if inputs.detector_run is not None:
            rule = self.spec.require_initialization(
                self.spec.model_id, "detector_run", settings.learning, scope=inputs.transfer_scope
            )
        return {
            "pretraining": load_pretraining(inputs.pretraining_run, dataset)
            if inputs.pretraining_run
            else None,
            "detector": detector_source(inputs.detector_run, dataset, style, inputs.transfer_scope)
            if inputs.detector_run
            else None,
            "compatibility": asdict(rule) if rule else None,
        }

    def export(self, context: DetectionContext, directory: Path, format: str) -> Path:
        self.spec.require_export(format)
        if len(set(context.verification_images)) < 2:
            raise ValueError(
                "MMDetection export requires at least two representative verification_images"
            )
        result = self._invoke(
            directory,
            {
                "operation": "export",
                "device": "cpu",
                "learning": context.learning,
                "checkpoint": str(context.checkpoint),
                "model_config": str(context.model_config),
                "imgsz": context.imgsz,
                "verification_images": [str(path) for path in context.verification_images],
            },
        )
        return Path(result["model"])


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return MMDetection(spec, python)


def invoke(python: Path, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
    return invoke_worker(
        python, Path(__file__).with_name("operations.py"), directory, request, "OpenMMLab"
    )
