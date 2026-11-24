"""Detection manifest preflight, split isolation, and upstream dataset snapshots."""

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from wvision.core.config import load_yaml, reject_unknown
from wvision.datasets.pose import PoseSkeleton, resolve_skeleton, valid_joints


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_sample(sample: "Sample", image: Path, *, include_label: bool = True) -> Path | None:
    """Copy a preflighted sample, checking for changes without touching source caches."""
    image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample.path, image)
    if sha256(image) != sample.image_hash:
        raise ValueError(f"Image changed after preflight: {sample.path}")
    if not include_label or sample.label is None:
        return None
    label = image.parent.parent / "labels" / image.with_suffix(".txt").name
    label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample.label, label)
    if sha256(label) != sample.label_hash:
        raise ValueError(f"Annotation changed after preflight: {sample.label}")
    return label


@dataclass(frozen=True)
class Sample:
    path: Path
    label: Path | None
    image_hash: str
    pixel_hash: str
    label_hash: str | None


@dataclass(frozen=True)
class Dataset:
    manifest: Path
    root: Path
    classes: dict[int, str]
    splits: dict[str, tuple[Sample, ...]]
    fingerprint: str
    groups: dict[str, str] = field(default_factory=dict)
    skeleton: PoseSkeleton | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest": str(self.manifest),
            "root": str(self.root),
            "classes": self.classes,
            "fingerprint": self.fingerprint,
            "groups": self.groups,
            **({"task": "pose", "skeleton": self.skeleton} if self.skeleton else {}),
            "splits": {
                name: [
                    {
                        "path": str(s.path),
                        "relative_path": s.path.relative_to(self.root).as_posix(),
                        "label": str(s.label) if s.label else None,
                        "image_sha256": s.image_hash,
                        "pixel_sha256": s.pixel_hash,
                        "label_sha256": s.label_hash,
                    }
                    for s in samples
                ]
                for name, samples in self.splits.items()
            },
        }

    def write_upstream(self, directory: Path, split: str | None = None) -> Path:
        """Generate lists, excluding test/unlabeled samples entirely during training."""
        names = ("train_labeled", "val") if split is None else (split,)
        paths: dict[str, str] = {}
        for name in names:
            staged: list[Path] = []
            for index, sample in enumerate(self.splits[name]):
                image = directory / "data" / name / "images" / f"{index}{sample.path.suffix}"
                stage_sample(sample, image)
                staged.append(image)
            file = directory / f"{name}.txt"
            file.write_text("".join(f"{path.as_posix()}\n" for path in staged), encoding="utf-8")
            paths[name] = str(file)
        # Ultralytics requires both train and val keys, even for standalone evaluation.
        selected = paths[split] if split else paths["val"]
        upstream: dict[str, Any] = {
            "path": str(directory),
            "train": paths.get("train_labeled", selected),
            "val": selected,
            "names": self.classes,
        }
        if self.skeleton is not None:
            upstream.update({key: self.skeleton[key] for key in ("kpt_shape", "flip_idx")})
        target = directory / "upstream-data.yaml"
        target.write_text(yaml.safe_dump(upstream), encoding="utf-8")
        return target

    def write_coco(
        self, directory: Path, names: tuple[str, ...], *, keypoints: bool = False
    ) -> dict[str, str]:
        """Stage selected splits as COCO; unlabeled samples always have zero annotations."""
        if keypoints != (self.skeleton is not None):
            raise ValueError("The COCO annotation type must match the dataset task")
        paths: dict[str, str] = {}
        for name in names:
            images: list[dict[str, Any]] = []
            annotations: list[dict[str, Any]] = []
            for index, sample in enumerate(self.splits[name], 1):
                image = directory / "data" / name / "images" / f"{index}{sample.path.suffix}"
                label = stage_sample(sample, image, include_label=name != "train_unlabeled")
                with Image.open(image) as decoded:
                    width, height = decoded.size
                images.append(
                    {
                        "id": index,
                        "file_name": image.as_posix(),
                        "width": width,
                        "height": height,
                    }
                )
                if label is not None:
                    for row in label.read_text(encoding="utf-8").splitlines():
                        if not row.strip():
                            continue
                        cls, x, y, w, h, *joints = map(float, row.split())
                        box = [(x - w / 2) * width, (y - h / 2) * height, w * width, h * height]
                        annotations.append(
                            {
                                "id": len(annotations) + 1,
                                "image_id": index,
                                "category_id": int(cls) + 1,
                                "bbox": box,
                                "area": box[2] * box[3],
                                "iscrowd": 0,
                            }
                        )
                        if keypoints:
                            labels = list(zip(joints[::3], joints[1::3], joints[2::3], strict=True))
                            annotations[-1].update(
                                keypoints=[
                                    value
                                    for x, y, v in labels
                                    for value in (x * width, y * height, int(v))
                                ],
                                num_keypoints=sum(v > 0 for _, _, v in labels),
                            )
            target = directory / f"{name}.coco.json"
            target.write_text(
                json.dumps(
                    {
                        "info": {"description": "WVision validated dataset snapshot"},
                        "images": images,
                        "annotations": annotations,
                        "categories": [
                            {"id": key + 1, "name": value}
                            for key, value in sorted(self.classes.items())
                        ],
                    },
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            paths[name] = str(target)
        return paths


def _annotation(path: Path, classes: dict[int, str], *, keypoints: int = 0) -> Path:
    # Mirror upstream: replace the last /images/ component with /labels/.
    parts = list(path.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise ValueError(f"Image must be under an images/ directory: {path}")
    parts[indices[-1]] = "labels"
    label = Path(*parts).with_suffix(".txt")
    if not label.is_file():
        raise ValueError(f"Missing annotation: {label}; use an empty file for a verified negative")
    rows: set[str] = set()
    for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            cls, x, y, width, height = map(float, fields[:5] if keypoints else fields)
            valid = (
                all(math.isfinite(v) for v in (cls, x, y, width, height))
                and cls in classes
                and cls == int(cls)
                and 0 < width <= 1
                and 0 < height <= 1
                and x - width / 2 >= -1e-6
                and x + width / 2 <= 1 + 1e-6
                and y - height / 2 >= -1e-6
                and y + height / 2 <= 1 + 1e-6
            )
            if keypoints:
                valid = valid and valid_joints(fields[5:], keypoints)
        except (ValueError, OverflowError):
            valid = False
        normalized = " ".join(fields)
        if not valid or normalized in rows:
            kind = (
                f"pose row (expected box and {keypoints} x/y/visibility triples)"
                if keypoints
                else "box"
            )
            raise ValueError(f"Invalid or duplicate YOLO {kind}: {label}:{line_number}")
        rows.add(normalized)
    return label


def group_mapping(path: Path, root: Path, splits: dict[str, tuple[Sample, ...]]) -> dict[str, str]:
    """Require an explicit group ID for every sample; never infer IDs from filenames."""
    values = load_yaml(path)
    expected = {s.path.relative_to(root).as_posix() for pool in splits.values() for s in pool}
    if set(values) != expected or any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise ValueError("Groups must map every dataset-relative image path to a nonempty group ID")
    return dict(values)


def load_dataset(
    manifest: str | Path, required: tuple[str, ...], *, task: str = "detect"
) -> Dataset:
    source = Path(manifest).resolve(strict=True)
    config = load_yaml(source)
    reject_unknown(
        config,
        {"schema_version", "task", "format", "root", "classes", "splits", "groups"}
        | ({"skeleton"} if task == "pose" else set()),
        "manifest",
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Manifest schema_version must be 1")
    if (
        task not in {"detect", "pose"}
        or config.get("task") != task
        or config.get("format") != "yolo"
    ):
        raise ValueError(f"Expected manifest task={task} and format=yolo")
    classes = config.get("classes")
    if (
        not isinstance(classes, dict)
        or not classes
        or any(type(key) is not int for key in classes)
        or set(classes) != set(range(len(classes)))
        or any(not isinstance(v, str) or not v.strip() for v in classes.values())
        or len(set(classes.values())) != len(classes)
    ):
        raise ValueError("classes must contain unique names with consecutive integer keys from 0")
    skeleton = None
    if task == "pose":
        if classes != {0: "person"}:
            raise ValueError("Pose requires classes: {0: person}")
        skeleton = resolve_skeleton(config.get("skeleton"))
    keypoints = len(skeleton["joints"]) if skeleton else 0
    root_ref = config.get("root")
    if not isinstance(root_ref, str) or not root_ref:
        raise ValueError("Manifest root must be a nonempty path string")
    root = (source.parent / root_ref).resolve(strict=True)
    references = config.get("splits")
    if not isinstance(references, dict):
        raise ValueError("splits must be a mapping")
    reject_unknown(references, {"train_labeled", "train_unlabeled", "val", "test"}, "split")
    for name in required:
        if name not in references:
            raise ValueError(f"Required split is missing: {name}")
    seen_paths: set[Path] = set()
    seen_pixels: dict[str, str] = {}
    splits: dict[str, tuple[Sample, ...]] = {}
    identity: dict[str, Any] = {"classes": classes, "splits": {}}
    if skeleton is not None:
        identity.update(task=task, skeleton=skeleton)
    for name, reference in references.items():
        if not isinstance(reference, str):
            raise ValueError(f"Split {name} must reference a text file")
        listing = (root / reference).resolve(strict=True)
        samples: list[Sample] = []
        for item in listing.read_text(encoding="utf-8").splitlines():
            if not item.strip():
                continue
            path = (root / item.strip()).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"Sample escapes dataset root: {path}")
            if path in seen_paths:
                raise ValueError(f"Repeated sample or split overlap: {path}")
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"Unsupported image format: {path}")
            with Image.open(path) as image:
                image.load()
                if min(image.size) < 10 or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Expected single-frame image at least 10x10: {path}")
                pixels = hashlib.sha256(
                    str(image.size).encode() + image.convert("RGB").tobytes()
                ).hexdigest()
            if pixels in seen_pixels:
                raise ValueError(
                    f"Duplicate image content in {name} and {seen_pixels[pixels]}: {path}"
                )
            seen_pixels[pixels] = name
            seen_paths.add(path)
            label = (
                _annotation(path, classes, keypoints=keypoints)
                if name != "train_unlabeled"
                else None
            )
            samples.append(
                Sample(path, label, sha256(path), pixels, sha256(label) if label else None)
            )
        if not samples:
            raise ValueError(f"Split {name} is empty")
        splits[name] = tuple(samples)
        identity["splits"][name] = [
            [s.path.relative_to(root).as_posix(), s.image_hash, s.label_hash] for s in samples
        ]
    groups = {}
    if "groups" in config:
        if not isinstance(config["groups"], str) or not config["groups"]:
            raise ValueError("groups must reference a YAML mapping relative to the dataset root")
        groups = group_mapping((root / config["groups"]).resolve(strict=True), root, splits)
        owners: dict[str, str] = {}
        for name, pool in splits.items():
            partition = "train" if name.startswith("train_") else name
            for sample in pool:
                group = groups[sample.path.relative_to(root).as_posix()]
                if group in owners and owners[group] != partition:
                    raise ValueError(
                        f"Group {group!r} crosses {owners[group]} and {partition} splits"
                    )
                owners[group] = partition
        identity["groups"] = groups
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Dataset(source, root, classes, splits, fingerprint, groups, skeleton)
