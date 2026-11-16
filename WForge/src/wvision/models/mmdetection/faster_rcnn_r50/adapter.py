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
    TrainingOutput,
)
from wvision.adapters.openmmlab.backend import interpreter
from wvision.adapters.shared.process import invoke_worker
from wvision.datasets.detection import Dataset
from wvision.models.mmdetection.faster_rcnn_r50.config import MMDetConfig
from wvision.models.mmdetection.faster_rcnn_r50.weights import detector_source
from wvision.models.mmpretrain.simsiam_r50.adapter import load_pretraining


class MMDetection:
    checkpoint_suffix = ".pth"
    external_checkpoint = False
    needs_model_config = True
    evaluator = "mmdetection/pycocotools"
    max_detections = 100

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        self.spec = spec
        self.python = interpreter(python)

    def resolve(self, values: dict[str, Any]) -> MMDetConfig:
        return MMDetConfig.resolve(values)

    def selection(self, learning: str) -> dict[str, str]:
        return {
            "metric": "teacher/coco/bbox_mAP" if learning == "semi_supervised" else "coco/bbox_mAP",
            "direction": "maximize",
        }

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if inputs.checkpoint is not None:
            raise ValueError("MMDetection checkpoint initialization/resume is not yet exposed")
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
            self.spec.require_initialization(self.spec.model_id, "detector_run", settings.learning)

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
                self.spec.model_id, "detector_run", settings.learning
            )
        return {
            "pretraining": load_pretraining(inputs.pretraining_run, dataset.fingerprint)
            if inputs.pretraining_run
            else None,
            "detector": detector_source(inputs.detector_run, dataset, style)
            if inputs.detector_run
            else None,
            "compatibility": asdict(rule) if rule else None,
        }

    def check_dependency(self) -> None:
        # The isolated worker checks exact dependencies before importing model code.
        pass

    def train(
        self,
        dataset: Dataset,
        settings: DetectionSettings,
        initialization: dict[str, Any],
        directory: Path,
    ) -> TrainingOutput:
        names = ("train_labeled", "val") + (
            ("train_unlabeled",) if settings.learning == "semi_supervised" else ()
        )
        result = invoke(
            self.python,
            directory,
            {
                "operation": "train",
                "config": settings.to_dict(),
                "data": dataset.write_coco(directory, names),
                "classes": [dataset.classes[key] for key in sorted(dataset.classes)],
                **initialization,
            },
        )
        return TrainingOutput(
            Path(result["best"]),
            Path(result["last"]),
            result["metrics"],
            Path(result["model_config"]),
        )

    def evaluate(
        self, dataset: Dataset, split: str, context: DetectionContext, directory: Path
    ) -> dict[str, float]:
        result = invoke(
            self.python,
            directory,
            {
                "operation": split,
                "device": context.device,
                "checkpoint": str(context.checkpoint),
                "model_config": str(context.model_config),
                "classes": [dataset.classes[key] for key in sorted(dataset.classes)],
                "data": dataset.write_coco(directory, (split,))[split],
            },
        )
        return dict(result["metrics"])

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int:
        result = invoke(
            self.python,
            directory,
            {
                "operation": "predict",
                "device": context.device,
                "checkpoint": str(context.checkpoint),
                "model_config": str(context.model_config),
                "source": [str(path) for path in samples],
            },
        )
        return int(result["sample_count"])


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return MMDetection(spec, python)


def invoke(python: Path, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
    return invoke_worker(
        python, Path(__file__).with_name("operations.py"), directory, request, "OpenMMLab"
    )
