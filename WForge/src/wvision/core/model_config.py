"""Safe, framework-free model-config documents and training-setting precedence."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wvision.core.config import load_yaml, reject_unknown
from wvision.core.registry import get_model
from wvision.datasets.detection import sha256


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    sha256: str
    model_id: str
    document: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        source = Path(path).resolve(strict=True)
        before = sha256(source)
        values = load_yaml(source)
        if before != sha256(source):
            raise ValueError("Model configuration changed while loading")
        reject_unknown(
            values,
            {"schema_version", "model", "task", "profile", "description", "upstream", "training"},
            "model configuration",
        )
        if type(values.get("schema_version")) is not int or values["schema_version"] != 1:
            raise ValueError("Model configuration schema_version must be 1")
        model = values.get("model")
        if not isinstance(model, str):
            raise ValueError("Model configuration requires a registered model identifier")
        spec = get_model(model)
        if values.get("task") != spec.task:
            raise ValueError(f"Model configuration task must be {spec.task!r} for {model}")
        if values.get("profile") not in ("smoke", "baseline", "local_weights"):
            raise ValueError(
                "Model configuration profile must be smoke, baseline, or local_weights"
            )
        if not isinstance(values.get("description"), str) or not values["description"].strip():
            raise ValueError("Model configuration requires a nonempty description")
        upstream = values.get("upstream")
        if (
            not isinstance(upstream, dict)
            or not upstream
            or any(
                not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                for k, v in upstream.items()
            )
        ):
            raise ValueError("upstream must contain nonempty string metadata")
        reject_unknown(upstream, {"repository", "version", "reference", "license"}, "upstream")
        training = values.get("training")
        if not isinstance(training, dict):
            raise ValueError("Model configuration training must be a mapping")
        if any(not isinstance(key, str) for key in training):
            raise ValueError("Model configuration training field names must be strings")
        return cls(source, before, model, deepcopy(values))

    def assert_unchanged(self) -> None:
        if not self.path.is_file() or sha256(self.path) != self.sha256:
            raise ValueError(
                "Model configuration changed since construction; create a new instance"
            )

    def training(self) -> dict[str, Any]:
        return deepcopy(dict(self.document["training"]))

    def snapshot(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "document": deepcopy(self.document)}


def model_reference(value: str | Path) -> tuple[str, ModelConfig | None]:
    """YAML paths are model configs; other strings are exact registered model IDs."""
    if isinstance(value, Path) or Path(value).suffix.lower() in {".yaml", ".yml"}:
        definition = ModelConfig.load(value)
        return definition.model_id, definition
    return value, None


def training_values(definition: ModelConfig | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Replace individual fields in order; backend_options is one field, not deep-merged."""
    values = definition.training() if definition else {}
    values.update(overrides)
    return values


def comparison_model(model: str | Path, model_id: str) -> tuple[dict[str, Any], ModelConfig | None]:
    """Resolve an exact model ID or model document for a fixed comparison protocol."""
    selected, definition = model_reference(model)
    if selected != model_id:
        raise ValueError(f"This comparison requires model {model_id}")
    return training_values(definition, {}), definition
