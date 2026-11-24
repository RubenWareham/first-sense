"""Reproducible training-only label budgets; withheld labels never enter derived data."""

import json
import random
from pathlib import Path
from typing import Any

import yaml

from wvision.datasets.detection import Dataset, load_dataset, stage_sample
from wvision.experiments.artifacts import write_json


def load_budget_metadata(dataset: Dataset) -> dict[str, Any] | None:
    """Catalog manifests may live separately from the generated budget's data root."""
    source = dataset.root / "budget.json"
    if not source.is_file():
        return None
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("derived_fingerprint") != dataset.fingerprint:
        raise ValueError("Label-budget metadata does not match this dataset")
    return dict(value)


def prepare_label_budget(
    data: str | Path,
    destination: str | Path,
    *,
    labeled_count: int,
    seed: int = 0,
) -> Path:
    """Select from labeled training data only and copy a leakage-checked derived dataset.

    Selection is sample-level. Correlated sequences/subjects must already be grouped in
    the source training/held-out splits; this helper does not infer those groups.
    """
    dataset = load_dataset(data, ("train_labeled", "val"))
    candidates = sorted(dataset.splits["train_labeled"], key=lambda sample: sample.path.as_posix())
    if type(labeled_count) is not int or not 0 < labeled_count < len(candidates):
        raise ValueError(
            "labeled_count must leave at least one labeled and one withheld training image"
        )
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    chosen = set(random.Random(seed).sample(range(len(candidates)), labeled_count))
    labeled = tuple(sample for index, sample in enumerate(candidates) if index in chosen)
    withheld = tuple(sample for index, sample in enumerate(candidates) if index not in chosen)
    pools = {
        "train_labeled": labeled,
        "train_unlabeled": withheld + dataset.splits.get("train_unlabeled", ()),
        **{name: samples for name, samples in dataset.splits.items() if name in {"val", "test"}},
    }
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "status.json", {"status": "preparing"})
    source_root = dataset.root
    selections: dict[str, Any] = {}
    references = {}
    groups = {}
    try:
        for name, samples in pools.items():
            listing = []
            for index, sample in enumerate(samples):
                image = target / name / "images" / f"{index}{sample.path.suffix}"
                stage_sample(sample, image, include_label=name != "train_unlabeled")
                listing.append(image.relative_to(target).as_posix())
                if dataset.groups:
                    groups[listing[-1]] = dataset.groups[
                        sample.path.relative_to(source_root).as_posix()
                    ]
            split_file = target / f"{name}.txt"
            split_file.write_text("".join(f"{item}\n" for item in listing), encoding="utf-8")
            references[name] = split_file.name
            selections[name] = [
                sample.path.relative_to(source_root).as_posix() for sample in samples
            ]
        manifest = target / "dataset.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "task": "detect",
                    "format": "yolo",
                    "root": ".",
                    "classes": dataset.classes,
                    "splits": references,
                    **({"groups": "groups.yaml"} if groups else {}),
                }
            ),
            encoding="utf-8",
        )
        if groups:
            (target / "groups.yaml").write_text(yaml.safe_dump(groups), encoding="utf-8")
        derived = load_dataset(manifest, ("train_labeled", "train_unlabeled", "val"))
        object_counts = {name: 0 for name in dataset.classes.values()}
        image_counts = {name: 0 for name in dataset.classes.values()}
        for sample in derived.splits["train_labeled"]:
            assert sample.label is not None
            present: set[int] = set()
            for row in sample.label.read_text(encoding="utf-8").splitlines():
                if row.strip():
                    category = int(float(row.split()[0]))
                    object_counts[dataset.classes[category]] += 1
                    present.add(category)
            for category in present:
                image_counts[dataset.classes[category]] += 1
        denominator = len(labeled) + len(pools["train_unlabeled"])
        write_json(
            target / "budget.json",
            {
                "schema_version": 1,
                "policy": "seeded_sample_v1",
                "seed": seed,
                "source_manifest": str(dataset.manifest),
                "source_fingerprint": dataset.fingerprint,
                "derived_fingerprint": derived.fingerprint,
                "labelable_training_pool": len(candidates),
                "training_pool_size": denominator,
                "labeled_count": labeled_count,
                "labeled_fraction": labeled_count / denominator,
                "selection": selections,
                "labeled_class_object_counts": object_counts,
                "labeled_class_image_counts": image_counts,
                "withheld_labels_copied": False,
                "group_policy": "source-defined held-out splits",
            },
        )
    except BaseException as error:
        write_json(target / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    write_json(target / "status.json", {"status": "completed"})
    return manifest
