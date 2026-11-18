"""Checkpoint integrity records shared with standalone upstream workers."""

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """An OS-released lock also protects recovery when an application process dies."""
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Run is active in another process: {path.parent}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def worker_lock(request_file: Path) -> Iterator[None]:
    request = json.loads(request_file.read_text(encoding="utf-8"))
    root = Path(request["directory"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    directory = (
        root / "ranks" / f"rank-{rank}"
        if rank > 0 and request.get("config", {}).get("devices")
        else root
    )
    directory.mkdir(parents=True, exist_ok=True)
    with file_lock(directory / ".worker.lock"):
        yield


def checkpoint_files(path: Path) -> dict[str, str]:
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = {}
    for file in files:
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[file.relative_to(path).as_posix() if path.is_dir() else file.name] = (
                digest.hexdigest()
            )
    if not result:
        raise ValueError(f"Checkpoint contains no files: {path}")
    return result


def _snapshot(directory: Path, source: Path, role: str) -> dict[str, Any]:
    if not source.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Checkpoint escapes its owned run directory")
    files = checkpoint_files(source)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:20]
    root = directory / "recovery"
    root.mkdir(exist_ok=True)
    if not root.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Recovery storage escapes its run directory")
    destination = root / f"{role}-{identity}"
    target = destination if source.is_dir() else destination / source.name
    if destination.exists():
        if checkpoint_files(target) != files:
            raise ValueError("Recovery snapshot identity collision or changed content")
    else:
        temporary = root / f".{uuid4().hex[:12]}.tmp"
        temporary.mkdir()
        staged = temporary if source.is_dir() else temporary / source.name
        if source.is_dir():
            shutil.copytree(source, staged, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, staged)
        if checkpoint_files(staged) != files:
            raise ValueError("Checkpoint changed while creating its recovery snapshot")
        # Both paths are fixed children of the verified, owned recovery directory.
        if not temporary.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid recovery snapshot path")
        temporary.replace(destination)
    return {
        "checkpoint": target.resolve().relative_to(directory.resolve()).as_posix(),
        "files": files,
    }


def record_checkpoint(
    directory: Path, checkpoint: Path, step: int, *, selected: Path | None = None
) -> None:
    manifest = directory / "recovery.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    value: dict[str, Any] = {
        "schema_version": 1,
        **_snapshot(directory, checkpoint, "checkpoint"),
        "step": step,
    }
    if selected is not None and selected.is_file():
        value["selected"] = _snapshot(directory, selected, "selected")
    temporary = directory / "recovery.json.tmp"
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    # Keep current and previous committed snapshots, including their selected
    # weights. Upstream retention/rewrites cannot invalidate these owned copies.
    retained = set()
    for record in (value, previous):
        for entry in (record, record.get("selected", {})):
            parts = Path(entry.get("checkpoint", "")).parts
            if len(parts) > 1 and parts[0] == "recovery":
                retained.add(parts[1])
    root = directory / "recovery"
    for entry in root.iterdir():
        if (
            entry.is_dir()
            and entry.name.startswith(("checkpoint-", "selected-"))
            and not entry.name.endswith(".tmp")
            and entry.name not in retained
        ):
            if not entry.resolve().is_relative_to(root.resolve()) or entry.is_symlink():
                raise ValueError("Invalid obsolete recovery snapshot path")
            shutil.rmtree(entry)
