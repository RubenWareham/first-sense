"""Create original synthetic detection fixtures; no downloaded assets or weights.

The generated rectangle task tests execution only, not real-world model quality.
The generated images and labels are dedicated to the public domain under CC0-1.0.
"""

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def create_dataset(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "labels", "splits"):
        (destination / folder).mkdir()
    rng = random.Random(17)
    for split, count in (("train_labeled", 4), ("val", 2), ("test", 2)):
        listing = []
        for index in range(count):
            stem = f"{split}-{index}"
            image = Image.new("RGB", (128, 128))
            image.putdata(
                [(rng.randrange(30), rng.randrange(30), rng.randrange(30)) for _ in range(128**2)]
            )
            x, y = rng.randrange(10, 60), rng.randrange(10, 60)
            width, height = rng.randrange(24, 50), rng.randrange(24, 50)
            ImageDraw.Draw(image).rectangle(
                (x, y, x + width - 1, y + height - 1), fill=(220, 80, 40)
            )
            image.save(destination / "images" / f"{stem}.png")
            (destination / "labels" / f"{stem}.txt").write_text(
                f"0 {(x + width / 2) / 128} {(y + height / 2) / 128} "
                f"{width / 128} {height / 128}\n",
                encoding="utf-8",
            )
            listing.append(f"images/{stem}.png\n")
        (destination / "splits" / f"{split}.txt").write_text("".join(listing), encoding="utf-8")
    manifest = destination / "dataset.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "task": "detect",
                "format": "yolo",
                "root": ".",
                "classes": {0: "rectangle"},
                "splits": {name: f"splits/{name}.txt" for name in ("train_labeled", "val", "test")},
            }
        ),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(create_dataset(parser.parse_args().destination))
