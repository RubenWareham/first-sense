"""Run after generating the tiny dataset; requires the optional backend."""

import argparse
from pathlib import Path

from wvision import Vision
from wvision.datasets.detection import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = Vision("models/detection/yolo11n/smoke.yaml")
    run = model.train(args.data, device=args.device)
    print(run)
    print(model.val())
    print(model.test())
    sample = load_dataset(args.data, ("test",)).splits["test"][0].path
    print(model.predict(sample))


if __name__ == "__main__":
    main()
