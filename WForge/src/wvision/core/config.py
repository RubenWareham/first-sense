"""Safe YAML parsing and common field-name validation."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML document: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def reject_unknown(value: dict[str, Any], allowed: set[str], context: str) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} field names must be strings")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {context} fields: {sorted(unknown)}")
