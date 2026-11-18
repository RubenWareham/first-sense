"""Framework-free contracts for detection adapters and explicit model capabilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wvision.adapters.shared.devices import device_family
from wvision.datasets.detection import Dataset


@dataclass(frozen=True)
class Capability:
    learning: str
    operation: str
    device: str = "cpu"


@dataclass(frozen=True)
class InitializationRule:
    source_model: str
    source_kind: str
    scope: str
    learning: str
    backbone_style: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: str
    task: str
    capabilities: tuple[Capability, ...]
    notes: str
    initializations: tuple[InitializationRule, ...] = ()
    exports: tuple[str, ...] = ()

    def require_export(self, format: str) -> None:
        if format not in self.exports:
            raise ValueError(
                f"{self.model_id}: export format {format!r} is not integrated; "
                f"supported: {self.exports}"
            )

    def require_initialization(
        self,
        source_model: str,
        source_kind: str,
        learning: str,
        backbone_style: str | None = None,
    ) -> InitializationRule:
        rules = [
            r
            for r in self.initializations
            if (r.source_model, r.source_kind, r.learning) == (source_model, source_kind, learning)
        ]
        if not rules:
            raise ValueError(
                f"{self.model_id}: {source_kind} from {source_model} is not integrated "
                f"for {learning}; inspect 'wvision models' initialization rules"
            )
        for rule in rules:
            if rule.backbone_style is None or rule.backbone_style == backbone_style:
                return rule
        styles = ", ".join(str(r.backbone_style) for r in rules)
        raise ValueError(f"{source_kind} requires backbone_style={styles} for {self.model_id}")

    def require(self, learning: str, operation: str, device: str = "cpu") -> None:
        if Capability(learning, operation, device_family(device)) not in self.capabilities:
            raise ValueError(
                f"{self.model_id}: learning={learning}, operation={operation}, device={device} "
                "is not integrated; use list_models() or 'wvision models' for capabilities"
            )


class DetectionSettings(Protocol):
    @property
    def learning(self) -> str: ...

    @property
    def imgsz(self) -> int: ...

    @property
    def batch(self) -> int: ...

    @property
    def device(self) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Initialization:
    checkpoint: Path | None = None
    pretraining_run: Path | None = None
    backbone_weights: Path | None = None
    detector_run: Path | None = None
    resume_run: Path | None = None

    def __post_init__(self) -> None:
        if (
            sum(
                value is not None
                for value in (
                    self.checkpoint,
                    self.pretraining_run,
                    self.backbone_weights,
                    self.detector_run,
                    self.resume_run,
                )
            )
            > 1
        ):
            raise ValueError("Select only one initialization source")


@dataclass(frozen=True)
class TrainingOutput:
    best: Path
    last: Path
    metrics: dict[str, float]
    model_config: Path | None = None


@dataclass(frozen=True)
class DetectionContext:
    checkpoint: Path
    model_config: Path | None
    learning: str
    imgsz: int
    batch: int
    device: str = "cpu"
    score_threshold: float | None = None
    max_detections: int | None = None


class DetectionAdapter(Protocol):
    spec: ModelSpec

    @property
    def python(self) -> Path | None: ...

    checkpoint_suffix: str
    external_checkpoint: bool
    needs_model_config: bool
    evaluator: str
    max_detections: int

    def resolve(self, values: dict[str, Any]) -> DetectionSettings: ...

    def selection(self, learning: str) -> dict[str, str]: ...

    def validate_training(self, settings: DetectionSettings, inputs: Initialization) -> None: ...

    def initialization(
        self,
        inputs: Initialization,
        dataset: Dataset,
        settings: DetectionSettings,
    ) -> dict[str, Any]: ...

    def check_dependency(self) -> None: ...

    def train(
        self,
        dataset: Dataset,
        settings: DetectionSettings,
        initialization: dict[str, Any],
        directory: Path,
    ) -> TrainingOutput: ...

    def evaluate(
        self,
        dataset: Dataset,
        split: str,
        context: DetectionContext,
        directory: Path,
    ) -> dict[str, float]: ...

    def predict(self, samples: list[Path], context: DetectionContext, directory: Path) -> int: ...

    def export(self, context: DetectionContext, directory: Path, format: str) -> Path: ...
