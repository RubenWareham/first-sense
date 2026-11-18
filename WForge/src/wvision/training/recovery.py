"""Validated continuation of trusted local upstream checkpoints."""

import json
import shutil
from pathlib import Path
from typing import Any

from wvision.adapters.shared.checkpoints import checkpoint_files, file_lock
from wvision.core.contracts import PretrainResult, RunResult
from wvision.experiments.artifacts import run_lock


def training_state(run: str | Path) -> dict[str, Any]:
    from wvision.experiments.artifacts import artifact_reference
    from wvision.experiments.portable import verify_bundle

    directory = Path(run).resolve(strict=True)
    verify_bundle(directory)
    path = directory / "training.json"
    if not path.is_file():
        if (directory / "bundle.json").is_file():
            raise ValueError("Bundle has no recovery state; create it with include_training=True")
        raise ValueError("Run predates recoverable checkpoints; start a new training run")
    with run_lock(directory):
        for worker in directory.rglob(".worker.lock"):
            with file_lock(worker):
                pass
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported training state schema")
        if state.get("data"):
            state["data"] = str(artifact_reference(directory, state["data"], strict=False))
    return dict(state)


def resolve_resume(
    run: str | Path, model: str, fingerprint: str, settings: dict[str, Any]
) -> dict[str, Any]:
    directory = Path(run).resolve(strict=True)
    state = training_state(directory)
    with run_lock(directory):
        if state["model"] != model or state["dataset_fingerprint"] != fingerprint:
            raise ValueError("Resume model or dataset identity differs from the source run")
        previous = state["config"]
        budget = "epochs" if "epochs" in previous else "max_iters"
        if {k: v for k, v in settings.items() if k != budget} != {
            k: v for k, v in previous.items() if k != budget
        }:
            raise ValueError("Resume permits changing only the total epochs/max_iters budget")
        manifest = directory / "recovery.json"
        if not manifest.is_file():
            raise ValueError("Run has no recoverable checkpoint; wait for a checkpoint interval")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported recovery checkpoint schema")
        checkpoint = (directory / record["checkpoint"]).resolve(strict=True)
        if not checkpoint.is_relative_to(directory):
            raise ValueError("Recovery checkpoint escapes its run directory")
        if checkpoint_files(checkpoint) != record["files"]:
            raise ValueError("Recovery checkpoint content has changed")
        if record.get("selected"):
            selected = (directory / record["selected"]["checkpoint"]).resolve(strict=True)
            if (
                not selected.is_relative_to(directory)
                or checkpoint_files(selected) != record["selected"]["files"]
            ):
                raise ValueError("Selected checkpoint content has changed")
            record["selected"] = {**record["selected"], "checkpoint": str(selected)}
        if settings[budget] <= record["step"]:
            raise ValueError("Resume budget must exceed the checkpoint's completed steps")
        provenance = directory / "model-source.json"
        return {
            **record,
            "checkpoint": str(checkpoint),
            "source_run": str(directory),
            "provenance": json.loads(provenance.read_text(encoding="utf-8"))
            if provenance.is_file()
            else None,
        }


def stage_resume(source: dict[str, Any], directory: Path) -> dict[str, Any]:
    checkpoint = Path(source["checkpoint"])
    target = directory / "resume-source" / checkpoint.name
    target.parent.mkdir()
    if checkpoint.is_dir():
        shutil.copytree(checkpoint, target)
    else:
        shutil.copyfile(checkpoint, target)
    if checkpoint_files(target) != source["files"]:
        raise ValueError("Recovery checkpoint changed while copying")
    result = {**source, "checkpoint": str(target)}
    if source.get("selected"):
        selected = Path(source["selected"]["checkpoint"])
        copy = target.parent / ("selected" + selected.suffix)
        shutil.copyfile(selected, copy)
        if checkpoint_files(copy) != {copy.name: next(iter(source["selected"]["files"].values()))}:
            raise ValueError("Selected checkpoint changed while copying")
        result["selected"] = {"checkpoint": str(copy)}
    return result


def resume(
    run: str | Path,
    *,
    epochs: int | None = None,
    max_iters: int | None = None,
    backend_python: str | Path | None = None,
    data: str | Path | None = None,
    output: str | Path = "runs",
) -> RunResult | PretrainResult:
    """Continue a completed/interrupted run in a new directory; budgets are totals."""
    from wvision.api import Vision
    from wvision.core.registry import get_model
    from wvision.training.pretraining import pretrain

    state = training_state(run)
    if data is not None:
        state["data"] = str(Path(data).resolve(strict=True))
    if not state.get("data"):
        raise ValueError("Portable recovery requires data=... or a bundle containing its dataset")
    settings = dict(state["config"])
    for name, value in (("epochs", epochs), ("max_iters", max_iters)):
        if value is not None:
            if name not in settings:
                raise ValueError(f"This model does not use {name}")
            settings[name] = value
    executable = backend_python or state.get("backend_python")
    if get_model(state["model"]).task == "representation":
        return pretrain(
            state["data"],
            model=state["model"],
            backend_python=executable,
            resume_run=run,
            output=output,
            **settings,
        )
    return Vision(state["model"], backend_python=executable).train(
        state["data"], resume_run=run, output=output, **settings
    )
