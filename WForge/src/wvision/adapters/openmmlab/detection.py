"""Standalone MMDetection helpers used with the isolated backend interpreter."""

import json
from pathlib import Path
from typing import Any

from devices import record_model_device  # type: ignore[import-not-found]
from runtime import dump  # type: ignore[import-not-found]


def resize_pipeline(pipeline: Any, size: int) -> None:
    if isinstance(pipeline, list):
        for item in pipeline:
            resize_pipeline(item, size)
    elif isinstance(pipeline, dict):
        kind = pipeline.get("type")
        if kind == "RandomResize":
            shape = (size, size)
            pipeline["scale"] = shape if "ratio_range" in pipeline else [shape, shape]
        elif kind == "Resize":
            pipeline["scale"] = (size, size)
        elif kind == "Pad":
            if "size" in pipeline:
                pipeline["size"] = (size, size)
            pad = pipeline.get("pad_val")
            if isinstance(pad, dict) and isinstance(pad.get("img"), list):
                pad["img"] = tuple(pad["img"])
        elif kind == "RandomCrop":
            pipeline["crop_size"] = (size, size)
        for value in pipeline.values():
            if isinstance(value, (dict, list)):
                resize_pipeline(value, size)


def dataset_config(annotation: str, classes: list[str], pipeline: list[Any]) -> dict[str, Any]:
    return {
        "type": "CocoDataset",
        "ann_file": annotation,
        "data_root": "",
        "data_prefix": {"img": ""},
        "metainfo": {"classes": tuple(classes)},
        "filter_cfg": {"filter_empty_gt": False, "min_size": 1},
        "pipeline": pipeline,
    }


def train_detector(runner: Any, directory: Path, device: str, max_iters: int) -> dict[str, Any]:
    runner.train()
    record_model_device(runner.model, device, directory)
    if runner.rank != 0:
        return {}
    selected = runner.message_hub.get_info("best_ckpt")
    if selected is None:
        raise RuntimeError(
            "Upstream produced no validation-selected checkpoint; inspect validation metrics "
            "and training stability in backend.log"
        )
    best = Path(selected)
    last = Path(runner.work_dir) / f"iter_{max_iters}.pth"
    logs = {
        key: float(history.current()) for key, history in runner.message_hub.log_scalars.items()
    }
    dump(directory / "training-scalars.json", logs)
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Upstream did not produce selected and final checkpoints")
    return {
        "best": str(best),
        "last": str(last),
        "model_config": str(directory / "resolved-mmdet.json"),
        "metrics": {
            f"mmdetection/{key}": value for key, value in logs.items() if key.startswith("val/")
        },
    }


def evaluate(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from mmdet.apis import inference_detector
    from mmdet.evaluation.metrics import CocoMetric

    model = inference_model(request)
    if list(model.dataset_meta["classes"]) != request["classes"]:
        raise ValueError("Checkpoint class map does not match evaluation data")
    metric = CocoMetric(
        ann_file=request["data"],
        metric="bbox",
        outfile_prefix=str(directory / "coco-results"),
    )
    metric.dataset_meta = {"classes": tuple(request["classes"])}
    annotations = json.loads(Path(request["data"]).read_text(encoding="utf-8"))
    for image in annotations["images"]:
        result = inference_detector(model, image["file_name"])
        result.set_metainfo({"img_id": image["id"]})
        metric.process({}, [result.to_dict()])
    values = metric.evaluate(len(annotations["images"]))
    if "coco/bbox_mAP" not in values:
        raise RuntimeError("Upstream COCO evaluation returned no AP metrics; inspect backend.log")
    record_model_device(model, request.get("device", "cpu"), directory)
    return {"metrics": {f"mmdetection/{key}": float(value) for key, value in values.items()}}


def predict(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from mmdet.apis import inference_detector

    model = inference_model(request)
    classes = dict(enumerate(model.dataset_meta["classes"]))
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for source in request["source"]:
            result = inference_detector(model, source)
            instances = result.pred_instances.cpu()
            handle.write(
                json.dumps(
                    {
                        "sample": source,
                        "height": result.ori_shape[0],
                        "width": result.ori_shape[1],
                        "coordinate_format": "absolute_pixel_xyxy",
                        "classes": classes,
                        "boxes": instances.bboxes.tolist(),
                        "scores": instances.scores.tolist(),
                        "class_ids": instances.labels.tolist(),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    record_model_device(model, request.get("device", "cpu"), directory)
    return {"sample_count": len(request["source"])}


def inference_model(request: dict[str, Any]) -> Any:
    from mmdet.apis import init_detector

    cfg = load_config(request["model_config"])
    detector = (
        cfg.model.detector
        if request.get("learning") == "semi_supervised" or cfg.model.type == "SoftTeacher"
        else cfg.model
    )
    test_cfg = detector.test_cfg.get("rcnn", detector.test_cfg)
    if request.get("score_threshold") is not None:
        test_cfg.score_thr = request["score_threshold"]
    if request.get("max_detections") is not None:
        test_cfg.max_per_img = request["max_detections"]
    cfg.test_dataloader.dataset.pipeline = [
        item for item in cfg.test_dataloader.dataset.pipeline if item["type"] != "LoadAnnotations"
    ]
    # The default upstream path assumes a top-level backbone, absent on SoftTeacher.
    model = init_detector(
        cfg, request["checkpoint"], device=request.get("device", "cpu"), cfg_options={}
    )
    record_model_device(model, request.get("device", "cpu"), Path(request["directory"]))
    return model


def load_config(path: str) -> Any:
    from mmengine.config import Config

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    size = values.get("wvision_imgsz") or next(
        item["scale"][0] for item in values["test_pipeline"] if item["type"] == "Resize"
    )
    # JSON turns tuples into lists. Upstream Resize requires tuple-valued image scales.
    resize_pipeline(values, size)
    return Config(values)
