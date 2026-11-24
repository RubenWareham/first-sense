"""Reviewed built-in catalog. No optional ML imports or automatic plugin discovery."""

from importlib import import_module
from pathlib import Path
from typing import cast

from wvision.adapters.base import Capability, DetectionAdapter, InitializationRule, ModelSpec


def _vision_capabilities(
    learning: tuple[str, ...], devices: tuple[str, ...] = ("cpu",)
) -> tuple[Capability, ...]:
    return tuple(
        Capability(mode, operation, device)
        for mode in learning
        for operation in ("train", "val", "test", "predict")
        for device in devices
    )


# Factories are fixed, reviewed source references. They cannot be supplied through YAML.
_MODELS: tuple[tuple[ModelSpec, str | None], ...] = (
    (
        ModelSpec(
            "ultralytics/yolo11n",
            "ultralytics",
            "detect",
            _vision_capabilities(("supervised",), ("cpu", "cuda")),
            "Verified Windows CPU/CUDA; random or trusted local YOLO11n initialization",
            (InitializationRule("ultralytics/yolo11n", "checkpoint", "detector", "supervised"),),
            exports=("onnx",),
        ),
        "wvision.models.ultralytics.yolo11.adapter:create_adapter",
    ),
    (
        ModelSpec(
            "ultralytics/yolo11n-pose",
            "ultralytics",
            "pose",
            _vision_capabilities(("supervised",), ("cpu", "cuda")),
            "Human pose with a dataset-defined skeleton; supervised FP32 on CPU and single GPU",
        ),
        "wvision.models.ultralytics.yolo11.adapter:create_adapter",
    ),
    (
        ModelSpec(
            "mmpose/rtmpose-t",
            "mmpose",
            "pose",
            _vision_capabilities(("supervised",), ("cpu", "cuda")),
            "RTMPose-t with COCO-17 or Halpe-26; prediction takes explicit person boxes",
            requires_person_boxes=True,
        ),
        "wvision.models.mmpose.rtmpose_t.adapter:create_adapter",
    ),
    (
        ModelSpec(
            "mmdetection/faster-rcnn-r50",
            "mmdetection",
            "detect",
            _vision_capabilities(("supervised", "semi_supervised"), ("cpu", "cuda")),
            "Verified Windows CPU/single CUDA GPU; upstream Soft Teacher in semi-supervised mode",
            (
                InitializationRule(
                    "mmpretrain/simsiam-r50", "pretraining_run", "backbone", "supervised", "pytorch"
                ),
                InitializationRule(
                    "mmpretrain/simsiam-r50",
                    "pretraining_run",
                    "backbone",
                    "semi_supervised",
                    "pytorch",
                ),
                InitializationRule(
                    "mmdetection/faster-rcnn-r50", "detector_run", "detector", "semi_supervised"
                ),
                InitializationRule(
                    "mmdetection/faster-rcnn-r50", "detector_run", "detector", "supervised"
                ),
                InitializationRule(
                    "mmdetection/faster-rcnn-r50", "detector_run", "backbone", "supervised"
                ),
                InitializationRule(
                    "mmdetection/faster-rcnn-r50", "detector_run", "backbone", "semi_supervised"
                ),
            ),
            exports=("onnx",),
        ),
        "wvision.models.mmdetection.faster_rcnn_r50.adapter:create_adapter",
    ),
    (
        ModelSpec(
            "mmpretrain/simsiam-r50",
            "mmpretrain",
            "representation",
            (
                Capability("self_supervised", "pretrain"),
                Capability("self_supervised", "pretrain", "cuda"),
            ),
            "Verified Windows CPU/CUDA; backbone transfer to Faster R-CNN or Soft Teacher",
        ),
        None,
    ),
    (
        ModelSpec(
            "transformers/dinov3-convnext-tiny-detr",
            "transformers",
            "detect",
            _vision_capabilities(("supervised",), ("cpu", "cuda")),
            "Verified CPU/CUDA random/local-fixture paths; official pretrained weights unverified",
            (
                InitializationRule(
                    "dinov3/convnext-tiny", "backbone_weights", "backbone", "supervised"
                ),
            ),
            exports=("onnx",),
        ),
        "wvision.models.transformers.dinov3_convnext_tiny_detr.adapter:create_adapter",
    ),
)


def list_models() -> tuple[ModelSpec, ...]:
    """Return exact registered variants and joint capabilities without loading frameworks."""
    return tuple(spec for spec, _ in _MODELS)


def get_model(model_id: str) -> ModelSpec:
    for spec in list_models():
        if spec.model_id == model_id:
            return spec
    raise ValueError(
        f"Unknown model {model_id!r}; registered models: "
        + ", ".join(spec.model_id for spec in list_models())
    )


def vision_adapter(model_id: str, python: str | Path | None) -> DetectionAdapter:
    spec = get_model(model_id)
    factory = next(reference for candidate, reference in _MODELS if candidate == spec)
    if factory is None:
        raise ValueError(f"{model_id} is a representation model; use pretrain()")
    module, name = factory.split(":")
    create = getattr(import_module(module), name)
    return cast(DetectionAdapter, create(spec, python))
