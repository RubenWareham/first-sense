"""Core-side validation and artifact transport for RTMPose."""

from dataclasses import asdict, dataclass, replace
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
from wvision.adapters.shared.devices import canonical_device
from wvision.adapters.shared.process import invoke_worker
from wvision.core.config import reject_unknown
from wvision.datasets.detection import Dataset


@dataclass(frozen=True)
class RTMPoseSettings:
    schema_version: int = 1
    learning: str = "supervised"
    epochs: int = 1
    imgsz: int = 256
    batch: int = 2
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RTMPose:
    checkpoint_suffix = ".pth"
    external_checkpoint = False
    needs_model_config = True
    evaluator = "mmpose 1.3.2 COCO keypoint AP (ground-truth boxes)"
    max_detections = 20

    def __init__(self, spec: ModelSpec, python: str | Path | None) -> None:
        self.spec = spec
        self.imgsz = 384 if spec.model_id == "mmpose/rtmpose-x" else 256
        self.python = interpreter(
            python, env="WVISION_MMPOSE_PYTHON", backend="MMPose", guide="RTMPOSE.md"
        )

    def resolve(self, values: dict[str, Any]) -> RTMPoseSettings:
        reject_unknown(values, set(RTMPoseSettings.__dataclass_fields__), "RTMPose settings")
        settings = RTMPoseSettings(**{"imgsz": self.imgsz, **values})
        if type(settings.schema_version) is not int or settings.schema_version != 1:
            raise ValueError("RTMPose requires training schema_version=1")
        if settings.learning != "supervised" or settings.precision != "fp32":
            raise ValueError("RTMPose supports supervised FP32 training")
        if type(settings.imgsz) is not int or settings.imgsz != self.imgsz:
            raise ValueError(f"{self.spec.model_id} requires imgsz={self.imgsz}")
        for name, minimum in (("epochs", 1), ("batch", 1), ("seed", 0)):
            value = getattr(settings, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"RTMPose {name} must be an integer >= {minimum}")
        return replace(settings, device=canonical_device(settings.device))

    def selection(self, learning: str) -> dict[str, str]:
        return {"metric": "mmpose/coco/AP", "direction": "maximize"}

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None:
        if any(
            value is not None
            for value in (
                inputs.checkpoint,
                inputs.detector_run,
                inputs.backbone_weights,
                inputs.pretraining_run,
                inputs.resume_run,
            )
        ):
            raise ValueError("RTMPose trains from random initialization in this workflow")

    def initialization(
        self, inputs: Initialization, dataset: Dataset, settings: DetectionSettings
    ) -> dict[str, Any]:
        if dataset.skeleton is None or dataset.skeleton["id"] not in {"coco17", "halpe26"}:
            raise ValueError("RTMPose requires skeleton: coco17 or skeleton: halpe26")
        return {"kind": "random"}

    def check_dependency(self) -> None:
        # The isolated worker checks the pinned stack before loading MMPose.
        pass

    def _invoke(self, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
        return invoke_worker(
            self.python, Path(__file__).with_name("operations.py"), directory, request, "MMPose"
        )

    def train(
        self,
        dataset: Dataset,
        settings: DetectionSettings,
        initialization: dict[str, Any],
        directory: Path,
    ) -> TrainingOutput:
        result = self._invoke(
            directory,
            {
                "operation": "train",
                "model_id": self.spec.model_id,
                "config": settings.to_dict(),
                "skeleton": dataset.skeleton,
                "data": dataset.write_coco(directory, ("train_labeled", "val"), keypoints=True),
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
                "skeleton": context.skeleton,
                "data": dataset.write_coco(directory, (split,), keypoints=True)[split],
            },
        )
        return dict(result["metrics"])

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int:
        if context.person_boxes is None:
            raise ValueError("RTMPose prediction requires explicit person boxes")
        result = self._invoke(
            directory,
            {
                "operation": "predict",
                "device": context.device,
                "checkpoint": str(context.checkpoint),
                "model_config": str(context.model_config),
                "skeleton": context.skeleton,
                "images": [
                    {"path": str(path), "boxes": context.person_boxes[path]} for path in samples
                ],
            },
        )
        return int(result["sample_count"])

    def export(self, context: DetectionContext, directory: Path, format: str) -> Path:
        self.spec.require_export(format)
        raise ValueError("RTMPose export is not supported")


def create_adapter(spec: ModelSpec, python: str | Path | None) -> DetectionAdapter:
    return RTMPose(spec, python)
