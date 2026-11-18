"""Run lifecycle and provenance, independent of ML frameworks."""

import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from wvision.adapters.shared.checkpoints import file_lock


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_reference(directory: Path, value: str, *, strict: bool = True) -> Path:
    reference = Path(value)
    resolved = (directory / reference).resolve(strict=strict)
    if not reference.is_absolute() and not resolved.is_relative_to(directory):
        raise ValueError("Artifact reference escapes its run directory")
    return resolved


@contextmanager
def run_lock(directory: Path) -> Iterator[None]:
    """Reject live writers while permitting recovery after an ungraceful exit."""
    with file_lock(directory / ".lock"):
        yield


def environment() -> dict[str, Any]:
    names = {dist.metadata["Name"] for dist in metadata.distributions() if dist.metadata["Name"]}
    packages = {name: metadata.version(name) for name in sorted(names)}
    source_root = Path(__file__).resolve().parents[3]
    revision = None
    try:
        if (source_root / "pyproject.toml").is_file():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            revision = (result.stdout.strip() or None) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": revision,
    }


@contextmanager
def operation(output: str | Path, kind: str, config: dict[str, Any]) -> Iterator[Path]:
    directory = Path(output).resolve() / f"{kind}-{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    write_json(directory / "config.json", config)
    write_json(directory / "environment.json", environment())
    write_json(directory / "status.json", {"status": "running", "started": started})
    try:
        with run_lock(directory):
            yield directory
    except BaseException as error:
        # Preserve a failure/interruption status, then propagate the original exception.
        write_json(
            directory / "status.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "started": started,
                "finished": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
            },
        )
        raise
    else:
        write_json(
            directory / "status.json",
            {"status": "completed", "started": started, "finished": datetime.now(UTC).isoformat()},
        )
