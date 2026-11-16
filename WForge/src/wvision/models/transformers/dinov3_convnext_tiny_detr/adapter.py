"""DINOv3 ConvNeXt Tiny / DETR public detection adapter."""

import os
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
from wvision.datasets.detection import Dataset
from wvision.models.transformers.dinov3_convnext_tiny_detr.config import (
    DinoDETRConfig,
    local_backbone,
)


class DinoDETRAdapter:
    checkpoint_suffix = ".safetensors"
    external_checkpoint = False
    needs_model_config = True
    evaluator = "transformers DETR postprocessing + pycocotools COCO AP"
    max_detections = 100

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        reference = python or os.environ.get("WVISION_TRANSFORMERS_PYTHON")
        if not reference:
            raise ValueError(
                "DINOv3 requires its isolated environment; pass backend_python or set "
                "WVISION_TRANSFORMERS_PYTHON; see docs/DINOV3.md"
            )
        self.python = Path(reference).resolve(strict=True)
        if not self.python.is_file():
            raise ValueError("backend_python must reference a Python executable")
        self.spec = spec

    def resolve(self, values: dict[str, Any]) -> DinoDETRConfig:
        return DinoDETRConfig.resolve(values)

    def selection(self, learning: str) -> dict[str, str]:
        return {"metric": "eval_loss", "direction": "minimize"}

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        assert isinstance(settings, DinoDETRConfig)
        if inputs.detector_run is not None:
            raise ValueError("DINOv3/DETR does not support detector_run initialization")
        if inputs.checkpoint is not None or inputs.pretraining_run is not None:
            raise ValueError(
                "DINOv3 initialization uses backbone_weights, not checkpoint/pretraining_run"
            )
        if settings.initialization == "local_backbone" and inputs.backbone_weights is None:
            raise ValueError(
                "Provide backbone_weights for local_backbone initialization; "
                "random initialization must be explicitly configured"
            )
        if settings.initialization == "random" and inputs.backbone_weights is not None:
            raise ValueError("random initialization cannot also receive backbone_weights")
        if inputs.backbone_weights is not None:
            self.spec.require_initialization(
                "dinov3/convnext-tiny", "backbone_weights", settings.learning
            )

    def initialization(
        self, inputs: Initialization, dataset: Dataset, settings: DetectionSettings
    ) -> dict[str, Any]:
        return {
            "backbone": local_backbone(inputs.backbone_weights) if inputs.backbone_weights else None
        }

    def check_dependency(self) -> None:
        # Exact package and Python checks execute in the isolated worker.
        pass

    def invoke(self, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        return invoke_worker(
            self.python, Path(__file__).with_name("operations.py"), directory, request, "DINOv3"
        )

    def train(
        self,
        dataset: Dataset,
        settings: DetectionSettings,
        initialization: dict[str, Any],
        directory: Path,
    ) -> TrainingOutput:
        result = self.invoke(
            directory,
            {
                "operation": "train",
                "config": settings.to_dict(),
                **initialization,
                "data": dataset.write_coco(directory, ("train_labeled", "val")),
                "classes": [dataset.classes[key] for key in sorted(dataset.classes)],
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
        return dict(
            self.invoke(
                directory,
                {
                    "operation": split,
                    "device": context.device,
                    "checkpoint": str(context.checkpoint),
                    "model_config": str(context.model_config),
                    "imgsz": context.imgsz,
                    "classes": [dataset.classes[key] for key in sorted(dataset.classes)],
                    "data": dataset.write_coco(directory, (split,))[split],
                },
            )["metrics"]
        )

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int:
        return int(
            self.invoke(
                directory,
                {
                    "operation": "predict",
                    "device": context.device,
                    "checkpoint": str(context.checkpoint),
                    "model_config": str(context.model_config),
                    "imgsz": context.imgsz,
                    "source": [str(path) for path in samples],
                },
            )["sample_count"]
        )


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return DinoDETRAdapter(spec, python)
