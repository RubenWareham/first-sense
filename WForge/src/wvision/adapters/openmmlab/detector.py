"""COCO staging and worker transport shared by MMDetection detector adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from wvision.adapters.base import DetectionContext, DetectionSettings, ModelSpec, TrainingOutput
from wvision.adapters.openmmlab.backend import interpreter
from wvision.datasets.detection import Dataset


class MMDetectionAdapter(ABC):
    checkpoint_suffix = ".pth"
    external_checkpoint = False
    needs_model_config = True
    evaluator = "mmdetection/pycocotools"
    max_detections = 100

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        self.spec = spec
        self.python = interpreter(python)

    @abstractmethod
    def _invoke(self, directory: Path, request: dict[str, Any]) -> dict[str, Any]: ...

    def check_dependency(self) -> None:
        # The isolated worker checks exact dependencies before importing model code.
        return None

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
        result = self._invoke(
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
        result = self._invoke(
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
        result = self._invoke(
            directory,
            {
                "operation": "predict",
                "device": context.device,
                "checkpoint": str(context.checkpoint),
                "model_config": str(context.model_config),
                "source": [str(path) for path in samples],
                "score_threshold": context.score_threshold,
                "max_detections": context.max_detections,
            },
        )
        return int(result["sample_count"])
