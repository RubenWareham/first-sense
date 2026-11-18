"""Small, serializable public result contracts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    run_id: str
    artifact_dir: Path
    status: str
    best_checkpoint: Path
    last_checkpoint: Path
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationResult:
    artifact_dir: Path
    split: str
    checkpoint: Path
    dataset_fingerprint: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class PredictionResult:
    artifact_dir: Path
    predictions: Path
    sample_count: int


@dataclass(frozen=True)
class ComparisonResult:
    artifact_dir: Path
    supervised_run: Path
    semi_supervised_run: Path
    report: Path


@dataclass(frozen=True)
class PretrainResult:
    artifact_dir: Path
    checkpoint: Path
    dataset_fingerprint: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class TransferComparisonResult:
    artifact_dir: Path
    random_run: Path
    pretrained_run: Path
    report: Path


@dataclass(frozen=True)
class ExperimentResult:
    artifact_dir: Path
    report: Path
    table: Path
    html: Path


@dataclass(frozen=True)
class ExportResult:
    artifact_dir: Path
    model: Path
    manifest: Path
