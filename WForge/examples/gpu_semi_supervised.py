"""Pretrain SimSiam and initialize upstream Soft Teacher on one NVIDIA GPU."""

import argparse
from pathlib import Path

from wvision import Vision, pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Prepared labeled/unlabeled manifest")
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source = pretrain(
        args.data,
        model="models/representation/simsiam-r50/smoke.yaml",
        backend_python=args.backend_python,
        device=args.device,
    )
    model = Vision(
        "models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml",
        backend_python=args.backend_python,
    )
    result = model.train(args.data, device=args.device, pretraining_run=source.artifact_dir)
    print(result.artifact_dir)
    print(model.val().metrics)
    print(model.test().metrics)


if __name__ == "__main__":
    main()
