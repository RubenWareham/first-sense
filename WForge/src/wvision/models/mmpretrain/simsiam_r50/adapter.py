"""SimSiam worker launch and representation-source integrity checks."""

import json
from pathlib import Path
from typing import Any

from wvision.adapters.shared.process import invoke_worker
from wvision.datasets.detection import Dataset, sha256
from wvision.experiments.artifacts import artifact_reference
from wvision.experiments.portable import verify_bundle
from wvision.models.mmpretrain.simsiam_r50.config import MODEL_ID
from wvision.training.transfer import transfer_data


def invoke(python: Path, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
    return invoke_worker(
        python, Path(__file__).with_name("operations.py"), directory, request, "OpenMMLab"
    )


def load_pretraining(run: str | Path, dataset: Dataset) -> dict[str, Any]:
    """Transfer a completed, untampered representation with held-out overlap checks."""
    directory = Path(run).resolve(strict=True)
    verify_bundle(directory)
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != 1
        or state.get("run_kind") != "self_supervised_pretrain"
        or state.get("model") != MODEL_ID
        or status.get("status") != "completed"
    ):
        raise ValueError("Expected a completed WVision SimSiam pretraining run")
    snapshot = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    audit = transfer_data(snapshot, state, dataset)
    checkpoint = artifact_reference(directory, state["checkpoint"])
    if sha256(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Pretraining checkpoint content has changed")
    return {"run": str(directory), **state, "checkpoint": str(checkpoint), "transfer_data": audit}
