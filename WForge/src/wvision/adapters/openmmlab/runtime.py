"""Shared standalone OpenMMLab environment preflight and JSON writing."""

import json
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def checkpoint_audit(runner: Any, directory: Path) -> None:
    """Observe upstream checkpoint saves without replacing the training loop."""
    from checkpoints import record_checkpoint  # type: ignore[import-not-found]
    from mmengine.hooks import Hook

    class Audit(Hook):  # type: ignore[misc]
        last: Path | None = None

        def save(self, runner: Any, step: int, *, force: bool = False) -> None:
            if runner.rank != 0:
                return
            pointer = Path(runner.work_dir) / "last_checkpoint"
            if pointer.is_file():
                checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
                if force or checkpoint != self.last:
                    selected = runner.message_hub.runtime_info.get("best_ckpt")
                    record_checkpoint(
                        directory, checkpoint, step, selected=Path(selected) if selected else None
                    )
                    self.last = checkpoint

        def after_train_iter(self, runner: Any, **kwargs: Any) -> None:
            self.save(runner, runner.iter + 1)

        def after_val_epoch(self, runner: Any, **kwargs: Any) -> None:
            # Upstream may rewrite the last checkpoint after updating best selection.
            self.save(runner, runner.iter, force=True)

    runner.register_hook(Audit(), priority="LOWEST")


def resume_checkpoint(request: dict[str, Any], directory: Path) -> str | None:
    source = request.get("resume")
    if not source:
        return None
    import torch

    path = Path(source["checkpoint"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer" not in state:
        raise ValueError("Checkpoint does not contain optimizer state")
    info = state["message_hub"]["runtime_info"]
    if info.get("best_ckpt"):
        if not source.get("selected"):
            raise ValueError("Recovery record has no selected checkpoint")
        selected = directory / "upstream/selected.pth"
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["selected"]["checkpoint"], selected)
        info["best_ckpt"] = str(selected)
    info["last_ckpt"] = str(path)
    info["keep_ckpt_ids"] = []
    torch.save(state, path)
    return str(path)


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dependencies(directory: Path) -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The initial MMDetection worker requires Python 3.10")
    expected = {"mmdet": "3.3.0", "mmcv": "2.1.0", "mmengine": "0.10.7"}
    torch_version = metadata.version("torch")
    builds = {"2.1.0+cpu": "0.16.0+cpu", "2.1.0+cu121": "0.16.0+cu121"}
    if torch_version not in builds:
        raise RuntimeError("OpenMMLab requires torch==2.1.0+cpu or torch==2.1.0+cu121")
    expected["torchvision"] = builds[torch_version]
    for name, version in expected.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"MMDetection worker requires {name}=={version}")
    import mmcv.ops  # noqa: F401 -- importing compiled operators is part of the preflight

    dump(
        directory / "backend-environment.json",
        {
            "python": sys.version,
            "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
        },
    )
