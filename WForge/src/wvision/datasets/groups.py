"""Explicit video/match/subject-separated copies; original datasets stay intact."""

from pathlib import Path

import yaml

from wvision.datasets.detection import group_mapping, load_dataset, stage_sample
from wvision.experiments.artifacts import operation, write_json


def prepare_group_split(
    data: str | Path,
    destination: str | Path,
    *,
    groups: str | Path,
    val_groups: list[str],
    test_groups: list[str],
) -> Path:
    """Assign complete explicit groups before training and retain a provenance map."""
    dataset = load_dataset(data, ("train_labeled",))
    mapping = group_mapping(Path(groups).resolve(strict=True), dataset.root, dataset.splits)
    if any(
        not isinstance(pool, list) or any(not isinstance(item, str) for item in pool)
        for pool in (val_groups, test_groups)
    ):
        raise ValueError("Validation/test groups must be lists of group IDs")
    validation, testing = set(val_groups), set(test_groups)
    known = set(mapping.values())
    if (
        not validation
        or not testing
        or validation & testing
        or not (validation | testing) < known
        or len(validation) != len(val_groups)
        or len(testing) != len(test_groups)
    ):
        raise ValueError("Choose distinct known validation/test groups and leave training groups")
    if not any(
        mapping[s.path.relative_to(dataset.root).as_posix()] not in validation | testing
        for s in dataset.splits["train_labeled"]
    ):
        raise ValueError("Leave at least one labeled training group")
    target = Path(destination).resolve()
    if target.exists():
        raise ValueError("Group-split destination must be new")
    # Validate annotations for every requested held-out sample before copying.
    for samples in dataset.splits.values():
        for sample in samples:
            group = mapping[sample.path.relative_to(dataset.root).as_posix()]
            if group in validation | testing and sample.label is None:
                raise ValueError(
                    "Held-out groups contain unlabeled images; provide verified labels first"
                )
    target.mkdir(parents=True)
    with operation(target, "prepare-groups", {"source": str(dataset.manifest)}) as audit:
        listings: dict[str, list[str]] = {}
        derived_groups, provenance = {}, {}
        for samples in dataset.splits.values():
            for sample in samples:
                original = sample.path.relative_to(dataset.root).as_posix()
                group = mapping[original]
                name = (
                    "val"
                    if group in validation
                    else "test"
                    if group in testing
                    else "train_labeled"
                    if sample.label is not None
                    else "train_unlabeled"
                )
                pool = listings.setdefault(name, [])
                image = target / name / "images" / f"{len(pool)}{sample.path.suffix}"
                stage_sample(sample, image, include_label=name != "train_unlabeled")
                relative = image.relative_to(target).as_posix()
                pool.append(relative)
                derived_groups[relative] = group
                provenance[relative] = original
        for name, listing in listings.items():
            (target / f"{name}.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
        (target / "groups.yaml").write_text(yaml.safe_dump(derived_groups), encoding="utf-8")
        manifest = target / "dataset.yaml"
        manifest.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "task": "detect",
                    "format": "yolo",
                    "root": ".",
                    "classes": dataset.classes,
                    "groups": "groups.yaml",
                    "splits": {name: f"{name}.txt" for name in listings},
                }
            ),
            encoding="utf-8",
        )
        derived = load_dataset(manifest, ("train_labeled", "val", "test"))
        write_json(
            audit / "selection.json",
            {
                "policy": "explicit_groups_v1",
                "source_fingerprint": dataset.fingerprint,
                "derived_fingerprint": derived.fingerprint,
                "groups": mapping,
                "val_groups": sorted(validation),
                "test_groups": sorted(testing),
                "source_samples": provenance,
            },
        )
    return manifest
