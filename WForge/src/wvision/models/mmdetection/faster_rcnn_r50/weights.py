"""Framework-free validation of completed supervised MMDetection source artifacts."""

import json
from pathlib import Path
from typing import Any

from wvision.datasets.detection import Dataset, sha256
from wvision.experiments.artifacts import artifact_reference
from wvision.experiments.portable import verify_bundle


def detector_source(run: Path, dataset: Dataset, backbone_style: str) -> dict[str, Any]:
    directory = run.resolve(strict=True)
    verify_bundle(directory)
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if (
        state.get("schema_version") != 1
        or status.get("status") != "completed"
        or state.get("model") != "mmdetection/faster-rcnn-r50"
        or state.get("config", {}).get("learning") != "supervised"
    ):
        raise ValueError("detector_run requires a completed supervised WVision Faster R-CNN run")
    if state.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError(
            "Detector source dataset identity differs; use the same labeled/unlabeled splits"
        )
    source_style = state["config"].get("backend_options", {}).get("backbone_style", "caffe")
    if source_style != backbone_style:
        raise ValueError("Detector source backbone style differs from target backbone_style")
    snapshot = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    if snapshot.get("fingerprint") != dataset.fingerprint or snapshot.get("classes") != {
        str(k): v for k, v in dataset.classes.items()
    }:
        raise ValueError("Detector source class map/dataset snapshot differs from target")
    environment = json.loads((directory / "backend-environment.json").read_text(encoding="utf-8"))
    if environment.get("packages", {}).get("mmdet") != "3.3.0":
        raise ValueError("Detector source requires the verified mmdet==3.3.0 backend")
    checkpoint = artifact_reference(directory, state["best_checkpoint"])
    config = artifact_reference(directory, state["model_config"])
    if sha256(checkpoint) != state["best_checkpoint_sha256"]:
        raise ValueError("Detector source checkpoint content has changed")
    if sha256(config) != state["model_config_sha256"]:
        raise ValueError("Detector source model configuration content has changed")
    return {
        "run": str(directory),
        "model": state["model"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": state["best_checkpoint_sha256"],
        "model_config": str(config),
        "model_config_sha256": state["model_config_sha256"],
        "dataset_fingerprint": dataset.fingerprint,
        "scope": "complete_detector",
        "backbone_style": source_style,
        "source_learning": "supervised",
        "initialization": state.get("initialization"),
        "policy": "selected weights only; no optimizer/scheduler/iteration restoration",
    }
