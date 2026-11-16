"""Compare existing supervised Faster R-CNN and upstream Soft Teacher on a label budget."""

import argparse
from pathlib import Path

from wvision import compare_learning, prepare_label_budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Source WVision detection manifest")
    parser.add_argument("destination", type=Path, help="New derived dataset directory")
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--labeled-count", default=2, type=int)
    args = parser.parse_args()
    manifest = prepare_label_budget(args.data, args.destination, labeled_count=args.labeled_count)
    result = compare_learning(
        manifest,
        backend_python=args.backend_python,
        model="models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
    )
    print(result.report)


if __name__ == "__main__":
    main()
