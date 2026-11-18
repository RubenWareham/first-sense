"""Detection manifest preflight, split isolation, and upstream dataset snapshots."""

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from wvision.core.config import load_yaml, reject_unknown


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

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest": str(self.manifest),
            "root": str(self.root),
            "classes": self.classes,
            "fingerprint": self.fingerprint,
            "splits": {
                name: [
                    {
                        "path": str(s.path),
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
        upstream = {
            "path": str(directory),
            "train": paths.get("train_labeled", selected),
            "val": selected,
            "names": self.classes,
        }
        target = directory / "upstream-data.yaml"
        target.write_text(yaml.safe_dump(upstream), encoding="utf-8")
        return target

    def write_coco(self, directory: Path, names: tuple[str, ...]) -> dict[str, str]:
        """Stage selected splits as COCO; unlabeled samples always have zero annotations."""
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
                        cls, x, y, w, h = map(float, row.split())
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


def _annotation(path: Path, classes: dict[int, str]) -> Path:
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
            cls, x, y, width, height = map(float, fields)
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
        except (ValueError, OverflowError):
            valid = False
        normalized = " ".join(fields)
        if not valid or normalized in rows:
            raise ValueError(f"Invalid or duplicate YOLO box: {label}:{line_number}")
        rows.add(normalized)
    return label


def load_dataset(manifest: str | Path, required: tuple[str, ...]) -> Dataset:
    source = Path(manifest).resolve(strict=True)
    config = load_yaml(source)
    reject_unknown(
        config, {"schema_version", "task", "format", "root", "classes", "splits"}, "manifest"
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Manifest schema_version must be 1")
    if config.get("task") != "detect" or config.get("format") != "yolo":
        raise ValueError("Only task=detect and format=yolo are integrated")
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
            label = _annotation(path, classes) if name != "train_unlabeled" else None
            samples.append(
                Sample(path, label, sha256(path), pixels, sha256(label) if label else None)
            )
        if not samples:
            raise ValueError(f"Split {name} is empty")
        splits[name] = tuple(samples)
        identity["splits"][name] = [
            [s.path.relative_to(root).as_posix(), s.image_hash, s.label_hash] for s in samples
        ]
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Dataset(source, root, classes, splits, fingerprint)
