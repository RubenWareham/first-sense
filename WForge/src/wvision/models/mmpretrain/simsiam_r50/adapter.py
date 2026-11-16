"""SimSiam worker launch and representation-source integrity checks."""

import json
from pathlib import Path
from typing import Any

from wvision.adapters.shared.process import invoke_worker
from wvision.datasets.detection import sha256
from wvision.models.mmpretrain.simsiam_r50.config import MODEL_ID


def invoke(python: Path, directory: Path, request: dict[str, Any]) -> dict[str, Any]:
    return invoke_worker(
        python, Path(__file__).with_name("operations.py"), directory, request, "OpenMMLab"
    )


def load_pretraining(run: str | Path, fingerprint: str) -> dict[str, Any]:
    """Only transfer a completed, untampered WVision SimSiam run on this dataset."""
    directory = Path(run).resolve(strict=True)
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != 1
        or state.get("run_kind") != "self_supervised_pretrain"
        or state.get("model") != MODEL_ID
        or status.get("status") != "completed"
    ):
        raise ValueError("Expected a completed WVision SimSiam pretraining run")
    if state["dataset_fingerprint"] != fingerprint:
        raise ValueError("Pretraining dataset identity differs from downstream dataset")
    checkpoint = Path(state["checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Pretraining checkpoint content has changed")
    return {"run": str(directory), **state}
