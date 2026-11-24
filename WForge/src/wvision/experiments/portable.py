"""Portable inference/representation bundles with explicit optional dataset copies."""

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from wvision.adapters.shared.checkpoints import checkpoint_files
from wvision.core.registry import get_model
from wvision.datasets.detection import load_dataset, sha256
from wvision.experiments.artifacts import artifact_reference, run_lock, write_json


def verify_bundle(directory: Path) -> None:
    manifest = directory / "bundle.json"
    if not manifest.is_file():
        return
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    if value.get("status", "completed") != "completed":
        raise ValueError("Bundle is incomplete; use a completed bundle")
    for name, expected in value["files"].items():
        if Path(name).is_absolute():
            raise ValueError("Bundle files must use relative paths")
        path = artifact_reference(directory, name)
        if sha256(path) != expected:
            raise ValueError(f"Bundle content has changed: {name}")


def bundle_run(
    run: str | Path,
    destination: str | Path,
    *,
    include_data: bool = False,
    include_training: bool = False,
) -> Path:
    """Copy a completed run for prediction/transfer, optionally including evaluation data.

    Backend software is installed separately. Recovery/optimizer state and dataset
    copies are explicit because they can be much larger than inference weights.
    """
    source = Path(run).resolve(strict=True)
    verify_bundle(source)
    if not (source / "run.json").is_file() or not (source / "status.json").is_file():
        raise ValueError("Bundle requires a completed training/pretraining run")
    if include_training and (
        not (source / "training.json").is_file() or not (source / "recovery.json").is_file()
    ):
        raise ValueError("Source run has no recoverable training state")
    target = Path(destination).resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError("Bundle destination must be outside the source run")
    if target.exists():
        raise ValueError("Bundle destination must be new")
    with run_lock(source):
        state: dict[str, Any] = json.loads((source / "run.json").read_text(encoding="utf-8"))
        status = json.loads((source / "status.json").read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or status.get("status") != "completed":
            raise ValueError("Bundle requires a completed training/pretraining run")
        if get_model(state["model"]).task == "pose":
            raise ValueError("Portable bundles for pose runs are not supported yet")
        references = (
            ["checkpoint"]
            if state.get("run_kind") == "self_supervised_pretrain"
            else ["best_checkpoint", "last_checkpoint", "model_config"]
        )
        # Validate every required source before creating the destination.
        files = {
            key: artifact_reference(source, state[key]) for key in references if state.get(key)
        }
        for key, path in files.items():
            if sha256(path) != state[key + "_sha256"]:
                raise ValueError(f"Run {key} content has changed")
        dataset = None
        if include_data:
            data = state.get("data") or json.loads(
                (source / "dataset.json").read_text(encoding="utf-8")
            ).get("manifest")
            if not data:
                raise ValueError("Source bundle has no dataset; bundle the original training run")
            dataset = load_dataset(artifact_reference(source, data), ())
            if dataset.fingerprint != state["dataset_fingerprint"]:
                raise ValueError("Dataset changed since training")
        target.mkdir(parents=True, exist_ok=False)
        with run_lock(target):
            pass
        write_json(target / "bundle.json", {"schema_version": 1, "status": "preparing"})
        for name in (
            "status.json",
            "dataset.json",
            "environment.json",
            "backend-environment.json",
            "metrics.json",
            "config.json",
            "model-source.json",
            "model-definition.json",
        ):
            path = source / name
            if path.is_file():
                shutil.copyfile(path, target / name)
        copied_paths: dict[Path, str] = {}
        for key, path in files.items():
            if path in copied_paths:
                state[key] = copied_paths[path]
                continue
            copied = target / "artifacts" / key / path.name
            copied.parent.mkdir(parents=True)
            shutil.copyfile(path, copied)
            if sha256(copied) != state[key + "_sha256"]:
                raise ValueError(f"Run {key} changed while copying")
            state[key] = copied.relative_to(target).as_posix()
            copied_paths[path] = state[key]
        state["backend_python"] = None
        state["data"] = None
        if dataset is not None:
            root = target / "dataset"
            root.mkdir(exist_ok=True)
            for name in (
                "budget.json",
                "source.json",
                "README.md",
                "README.dataset.txt",
                "README.roboflow.txt",
                "LICENSE",
                "LICENSE.txt",
            ):
                sidecar = dataset.root / name
                if sidecar.is_file():
                    shutil.copyfile(sidecar, root / name)
            splits = {}
            for split, samples in dataset.splits.items():
                listing = []
                for sample in samples:
                    relative = sample.path.relative_to(dataset.root)
                    image = root / relative
                    image.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(sample.path, image)
                    if sha256(image) != sample.image_hash:
                        raise ValueError("Dataset changed while bundling")
                    if sample.label is not None:
                        label = root / sample.label.relative_to(dataset.root)
                        label.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(sample.label, label)
                        if sha256(label) != sample.label_hash:
                            raise ValueError("Annotations changed while bundling")
                    listing.append(relative.as_posix())
                file = root / f"{split}.txt"
                file.write_text("\n".join(listing) + "\n", encoding="utf-8")
                splits[split] = file.name
            manifest = root / "dataset.yaml"
            if dataset.groups:
                (root / "groups.yaml").write_text(yaml.safe_dump(dataset.groups), encoding="utf-8")
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "task": "detect",
                        "format": "yolo",
                        "root": ".",
                        "classes": dataset.classes,
                        "splits": splits,
                        **({"groups": "groups.yaml"} if dataset.groups else {}),
                    }
                ),
                encoding="utf-8",
            )
            state["data"] = manifest.relative_to(target).as_posix()
        write_json(target / "run.json", state)
        if include_training:
            training = source / "training.json"
            recovery = source / "recovery.json"
            record = json.loads(recovery.read_text(encoding="utf-8"))
            for item in (record, record.get("selected")):
                if item is None:
                    continue
                checkpoint = artifact_reference(source, item["checkpoint"])
                if (
                    not checkpoint.is_relative_to(source)
                    or checkpoint_files(checkpoint) != item["files"]
                ):
                    raise ValueError("Recovery checkpoint content has changed")
                copied = (
                    target
                    / "recovery"
                    / ("last" if item is record else "selected")
                    / checkpoint.name
                )
                copied.parent.mkdir(parents=True)
                if checkpoint.is_dir():
                    shutil.copytree(checkpoint, copied)
                else:
                    shutil.copyfile(checkpoint, copied)
                if checkpoint_files(copied) != item["files"]:
                    raise ValueError("Recovery checkpoint changed while bundling")
                item["checkpoint"] = copied.relative_to(target).as_posix()
            write_json(target / "recovery.json", record)
            metadata = json.loads(training.read_text(encoding="utf-8"))
            metadata.update(data=state["data"], backend_python=None)
            write_json(target / "training.json", metadata)
        write_json(
            target / "bundle.json",
            {
                "schema_version": 1,
                "status": "completed",
                "source_run": str(source),
                "include_data": include_data,
                "include_training": include_training,
                "purpose": "portable run; install compatible backend separately",
                "files": {
                    p.relative_to(target).as_posix(): sha256(p)
                    for p in sorted(target.rglob("*"))
                    if p.is_file() and p.name not in {"bundle.json", ".lock", ".worker.lock"}
                },
            },
        )
    return target
