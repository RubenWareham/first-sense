"""One COCO metric protocol for comparisons across registered detection models."""

import json
import math
from pathlib import Path
from typing import Any

from wvision.core.contracts import EvaluationResult
from wvision.datasets.detection import Dataset, sha256
from wvision.experiments.artifacts import operation, write_json

COCO_PROTOCOL = "pycocotools bbox AP; IoU=.50:.95; maxDets=1,10,100; score>=.001"


def require_evaluator() -> None:
    try:
        import pycocotools.cocoeval  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Experiment comparison requires: pip install '.[evaluation]'") from error


def evaluate_predictions(
    dataset: Dataset, predictions: Path, *, checkpoint: Path, output: Path
) -> EvaluationResult:
    """Evaluate exactly the validation samples; no test-based model selection."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    samples = dataset.splits["val"]
    with operation(
        output,
        "common-val",
        {
            "protocol": COCO_PROTOCOL,
            "dataset_fingerprint": dataset.fingerprint,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "predictions": str(predictions),
            "predictions_sha256": sha256(predictions),
        },
    ) as directory:
        annotations = dataset.write_coco(directory, ("val",))["val"]
        coco = COCO(annotations)
        identities = {str(sample.path): index for index, sample in enumerate(samples, 1)}
        seen = set()
        detections = []
        for line in predictions.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sample = str(Path(row["sample"]).resolve(strict=True))
            if sample not in identities or sample in seen:
                raise ValueError(
                    "Comparison predictions contain duplicate or non-validation samples"
                )
            seen.add(sample)
            image_id = identities[sample]
            image = coco.imgs[image_id]
            if (
                row["classes"] != {str(k): v for k, v in dataset.classes.items()}
                or row.get("coordinate_format") != "absolute_pixel_xyxy"
                or (row["width"], row["height"]) != (image["width"], image["height"])
            ):
                raise ValueError(
                    "Prediction classes/coordinates/dimensions differ from the dataset"
                )
            if len(row["boxes"]) > 100:
                raise ValueError("Comparison predictions exceed the 100-detection protocol")
            for box, score, category in zip(
                row["boxes"], row["scores"], row["class_ids"], strict=True
            ):
                if (
                    len(box) != 4
                    or not all(math.isfinite(v) for v in [*box, score])
                    or not 0.001 <= score <= 1
                    or type(category) is not int
                    or category not in dataset.classes
                ):
                    raise ValueError("Invalid comparison prediction")
                x1, y1, x2, y2 = box
                if not (0 <= x1 <= x2 <= image["width"] and 0 <= y1 <= y2 <= image["height"]):
                    raise ValueError("Comparison box is outside the source image")
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": category + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
        if seen != set(identities):
            raise ValueError("Comparison predictions are missing validation samples")
        write_json(directory / "detections.json", detections)
        if detections:
            result = coco.loadRes(detections)
        else:
            result = COCO()
            result.dataset = {
                "images": list(coco.imgs.values()),
                "categories": list(coco.cats.values()),
                "annotations": [],
            }
            result.createIndex()
        evaluator = COCOeval(coco, result, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        names = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l")
        metrics: dict[str, Any] = {
            f"wforge/coco/{key}": float(value)
            for key, value in zip(names, evaluator.stats[:6], strict=True)
        }
        write_json(directory / "metrics.json", metrics)
    return EvaluationResult(directory, "val", checkpoint, dataset.fingerprint, metrics)
