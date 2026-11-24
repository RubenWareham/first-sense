"""Framework-free cross-dataset transfer provenance and held-out leakage checks."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from wvision.datasets.detection import Dataset


def transfer_data(
    snapshot: dict[str, Any], state: dict[str, Any], target: Dataset
) -> dict[str, Any]:
    if snapshot.get("fingerprint") != state.get("dataset_fingerprint"):
        raise ValueError("Source dataset snapshot identity differs from its run")
    cross_dataset = state["dataset_fingerprint"] != target.fingerprint
    source_splits = snapshot.get("splits", {})
    target_heldout = [sample for name in ("val", "test") for sample in target.splits.get(name, ())]
    protected_pixels = {sample.pixel_hash for sample in target_heldout}
    protected_groups = {
        target.groups[sample.path.relative_to(target.root).as_posix()]
        for sample in target_heldout
        if target.groups
    }
    source_root = snapshot.get("root")
    if not source_root:
        raise ValueError("Source dataset snapshot has no root")
    source_paths = PureWindowsPath if PureWindowsPath(source_root).is_absolute() else PurePosixPath
    group_map = snapshot.get("groups", {})
    examined = 0
    exposed_pixels: set[str] = set(state.get("training_images", []))
    exposed_groups: set[str] = set()
    pending = [state.get("initialization", {})]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            exposed_pixels.update(value.get("training_images", []))
            exposure = value.get("exposure", {})
            exposed_pixels.update(exposure.get("pixel_sha256", []))
            exposed_groups.update(exposure.get("groups", []))
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    pools = (
        ("train_unlabeled",)
        if state.get("run_kind") == "self_supervised_pretrain"
        else (
            ("train_labeled", "train_unlabeled", "val")
            if cross_dataset
            else ("train_labeled", "train_unlabeled")
        )
    )
    for name in pools:
        for sample in source_splits.get(name, []):
            pixels = sample.get("pixel_sha256")
            if not pixels:
                raise ValueError("Source snapshot lacks training image content provenance")
            relative = (
                sample.get("relative_path")
                or source_paths(sample["path"]).relative_to(source_root).as_posix()
            )
            if group_map and relative not in group_map:
                raise ValueError("Source group metadata is incomplete for its training exposure")
            group = group_map.get(relative)
            exposed_pixels.add(pixels)
            if group is not None:
                exposed_groups.add(group)
            examined += 1
    if exposed_pixels & protected_pixels or exposed_groups & protected_groups:
        raise ValueError(
            "Transfer source or recorded initialization overlaps target held-out images or groups"
        )
    if not examined:
        raise ValueError("Source snapshot has no declared training samples")
    return {
        "source_fingerprint": state["dataset_fingerprint"],
        "target_fingerprint": target.fingerprint,
        "cross_dataset": cross_dataset,
        "source_samples_checked": examined,
        "target_heldout_overlap": False,
        "exposure": {"pixel_sha256": sorted(exposed_pixels), "groups": sorted(exposed_groups)},
        "group_check": "verified"
        if group_map and target.groups
        else "unverified: groups not supplied for both datasets",
        "limitations": "Exact decoded-content/group checks do not detect unknown "
        "near-duplicates or external pretraining overlap",
    }
