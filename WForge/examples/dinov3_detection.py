"""Offline random-initialization smoke example; local-weight usage is in docs/DINOV3.md."""

import argparse
from pathlib import Path

from wvision import Vision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend-python", type=Path)
    args = parser.parse_args()
    model = Vision(
        "models/detection/dinov3-convnext-tiny-detr/smoke.yaml",
        backend_python=args.backend_python,
    )
    run = model.train(".wvision/dinov3-data/dataset.yaml", device=args.device)
    print(run)
    print(model.val())
    print(model.test())
    print(model.predict(".wvision/dinov3-data/images/test-0.png"))


if __name__ == "__main__":
    main()
