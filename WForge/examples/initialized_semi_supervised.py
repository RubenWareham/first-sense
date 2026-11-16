"""Initialize upstream Soft Teacher from a supervised detector on the same label budget."""

import argparse
from pathlib import Path

from wvision import Vision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Prepared labeled/unlabeled dataset manifest")
    parser.add_argument("--backend-python", required=True, type=Path)
    args = parser.parse_args()
    source = Vision(
        "models/detection/faster-rcnn-r50/supervised-smoke.yaml",
        backend_python=args.backend_python,
    ).train(args.data)
    model = Vision(
        "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
        backend_python=args.backend_python,
    )
    result = model.train(args.data, detector_run=source.artifact_dir)
    print(result.artifact_dir)
    print(model.val().metrics)
    print(model.test().metrics)


if __name__ == "__main__":
    main()
