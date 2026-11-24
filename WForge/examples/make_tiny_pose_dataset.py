"""Generate an original CC0-1.0 pose fixture for offline execution checks."""

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from wvision.datasets.pose import resolve_skeleton

# The order follows the WVision COCO-17 skeleton.
JOINT_PIXELS = (
    (16, 15),
    (14, 12),
    (18, 12),
    (11, 14),
    (21, 14),
    (10, 24),
    (22, 24),
    (7, 34),
    (26, 29),
    (4, 44),
    (29, 36),
    (12, 42),
    (21, 42),
    (10, 56),
    (23, 53),
    (7, 68),
    (27, 65),
)


def create_pose_dataset(destination: Path, skeleton: str = "coco17") -> Path:
    if skeleton not in {"coco17", "halpe26"}:
        raise ValueError("Fixture supports coco17 or halpe26")
    definition = resolve_skeleton(skeleton)
    joint_pixels = JOINT_PIXELS
    if skeleton == "halpe26":
        joint_pixels += (
            (16, 9),
            (16, 23),
            (16, 42),
            (5, 72),
            (29, 69),
            (2, 71),
            (31, 68),
            (9, 70),
            (25, 67),
        )
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("images", "labels", "splits"):
        (destination / name).mkdir()
    width, height = 160, 96
    splits = {}
    edges = definition["edges"]
    for split_index, (split, count) in enumerate((("train_labeled", 4), ("val", 2), ("test", 2))):
        images = []
        for index in range(count):
            stem = f"{split}-{index}"
            image = Image.new("RGB", (width, height), (20 + split_index * 10 + index, 30, 35))
            draw = ImageDraw.Draw(image)
            rows = []
            negative = split == "val" and index == 1
            for left in () if negative else (12 + index, 88 - index):
                points = [(left + x, y + 4 + split_index) for x, y in joint_pixels]
                for a, b in edges:
                    if b != 9 and a != 9:
                        draw.line((points[a], points[b]), fill=(200, 150, 95), width=3)
                fields = [0, (left + 16) / width, 44 / height, 34 / width, 72 / height]
                for joint, (x, y) in enumerate(points):
                    fields.extend(
                        (0, 0, 0)
                        if joint == 9
                        else (x / width, y / height, 1 if joint == 10 else 2)
                    )
                rows.append(" ".join(map(str, fields)) + "\n")
            relative = f"images/{stem}.png"
            image.save(destination / relative)
            (destination / "labels" / f"{stem}.txt").write_text("".join(rows), encoding="utf-8")
            images.append(relative + "\n")
        splits[split] = f"splits/{split}.txt"
        (destination / splits[split]).write_text("".join(images), encoding="utf-8")
    manifest = destination / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "task": "pose",
                "format": "yolo",
                "root": ".",
                "classes": {0: "person"},
                "skeleton": skeleton,
                "splits": splits,
            }
        ),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--skeleton", choices=("coco17", "halpe26"), default="coco17")
    args = parser.parse_args()
    print(create_pose_dataset(args.destination, args.skeleton))
