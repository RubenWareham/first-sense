"""Standalone Python 3.10 worker. Execute only through the WVision process adapter.

This module deliberately imports no WVision modules: the core uses Python 3.12.
Upstream owns all model, pseudo-label, teacher-update, optimizer, and loop behavior.
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# Load the packaged framework-free helper without importing the application.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/openmmlab"))
from checkpoints import worker_lock  # type: ignore[import-not-found]
from detection import (  # type: ignore[import-not-found]
    dataset_config,
    evaluate,
    inference_model,
    load_config,
    predict,
    resize_pipeline,
    train_detector,
)
from devices import record_model_device, runtime_device, runtime_precision, worker_context  # type: ignore[import-not-found]
from runtime import (  # type: ignore[import-not-found]
    checkpoint_audit,
    dependencies,
    disable_network_for_tests,
    dump,
    resume_checkpoint,
)


def configure(request: dict[str, Any], directory: Path) -> Any:
    import mmdet
    from mmengine.config import Config

    settings = request["config"]
    semi = settings["learning"] == "semi_supervised"
    recipe = (
        Path(mmdet.__file__).parent
        / ".mim"
        / "configs"
        / "soft_teacher"
        / "soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py"
    )
    cfg = Config.fromfile(str(recipe))
    # Fixed upstream recipe only; callers cannot supply executable Python config files.
    cfg.model.detector.backbone.init_cfg = None
    style = settings["backend_options"].get("backbone_style", "caffe")
    if style == "pytorch":
        cfg.model.detector.backbone.style = "pytorch"
        cfg.model.detector.data_preprocessor.update(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
        )
    pretraining = request.get("pretraining")
    if pretraining:
        if style != "pytorch":
            raise ValueError("SimSiam transfer requires PyTorch-style ResNet")
        source = Path(pretraining["checkpoint"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != pretraining["checkpoint_sha256"]:
            raise ValueError("Pretraining checkpoint content has changed")
        cfg.model.detector.backbone.init_cfg = dict(
            type="Pretrained", checkpoint=str(source), prefix="backbone."
        )
    cfg.model.detector.roi_head.bbox_head.num_classes = len(request["classes"])
    if semi:
        # Multi-branch inputs are normalized by the wrapper before reaching either detector.
        cfg.model.data_preprocessor.data_preprocessor = copy.deepcopy(
            cfg.model.detector.data_preprocessor
        )
    detector = request.get("detector")
    if detector:
        validate_detector_structure(detector, cfg.model.detector)
        if detector["scope"] == "backbone":
            cfg.model.detector.backbone.init_cfg = dict(
                type="Pretrained", checkpoint=detector["checkpoint"], prefix="backbone."
            )
    if semi:
        cfg.model.semi_train_cfg.update(
            {
                k: v
                for k, v in settings["backend_options"].items()
                if k not in {"lr", "backbone_style"}
            }
        )
    else:
        cfg.model = copy.deepcopy(cfg.model.detector)
        cfg.custom_hooks = []
        cfg.val_cfg = dict(type="ValLoop")
    resize_pipeline(cfg.sup_pipeline, settings["imgsz"])
    resize_pipeline(cfg.unsup_pipeline, settings["imgsz"])
    resize_pipeline(cfg.test_pipeline, settings["imgsz"])
    # For the supervised baseline use exactly the upstream supervised augmentation branch.
    supervised_pipeline = copy.deepcopy(cfg.sup_pipeline)
    if not semi:
        supervised_pipeline[-1] = dict(type="PackDetInputs")
    labeled = dataset_config(
        request["data"]["train_labeled"], request["classes"], supervised_pipeline
    )
    if semi:
        unlabeled = dataset_config(
            request["data"]["train_unlabeled"], request["classes"], cfg.unsup_pipeline
        )
        training = dict(type="ConcatDataset", datasets=[labeled, unlabeled])
        sampler = dict(type="MultiSourceSampler", batch_size=settings["batch"], source_ratio=[1, 1])
    else:
        training = labeled
        sampler = dict(type="InfiniteSampler", shuffle=True)
    cfg.train_dataloader = dict(
        batch_size=settings["batch"],
        num_workers=0,
        persistent_workers=False,
        sampler=sampler,
        dataset=training,
    )
    validation = dataset_config(request["data"]["val"], request["classes"], cfg.test_pipeline)
    validation["test_mode"] = True
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=validation,
    )
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.val_evaluator = dict(type="CocoMetric", ann_file=request["data"]["val"], metric="bbox")
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    interval = min(settings["checkpoint_interval"], settings["max_iters"])
    cfg.train_cfg = dict(
        type="IterBasedTrainLoop",
        max_iters=settings["max_iters"],
        val_interval=interval,
    )
    cfg.param_scheduler = []
    cfg.optim_wrapper.optimizer.lr = settings["backend_options"].get("lr", 0.01)
    if settings["precision"] != "fp32":
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.dtype = "float16"
        cfg.optim_wrapper.loss_scale = "dynamic"
    selection = "teacher/coco/bbox_mAP" if semi else "coco/bbox_mAP"
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=interval,
        save_best=selection,
        rule="greater",
        max_keep_ckpts=1,
        save_optimizer=True,
        save_param_scheduler=True,
    )
    cfg.default_hooks.logger.interval = 1
    cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]
    cfg.work_dir = request.get("upstream_directory", str(directory / "upstream"))
    cfg.launcher = "pytorch" if request.get("distributed") else "none"
    cfg.env_cfg.dist_cfg = dict(
        backend="nccl" if request.get("distributed") and os.name != "nt" else "gloo"
    )
    if request.get("distributed"):
        cfg.model_wrapper_cfg = dict(type="MMDistributedDataParallel", find_unused_parameters=True)
    cfg.env_cfg.mp_cfg = dict(mp_start_method="spawn", opencv_num_threads=0)
    cfg.randomness = dict(seed=settings["seed"], deterministic=True)
    cfg.load_from = resume_checkpoint(request, directory)
    cfg.resume = bool(request.get("resume"))
    cfg.device = request.get("execution_device", settings["device"])
    cfg.wvision_imgsz = settings["imgsz"]
    cfg.dump(str(directory / "resolved-mmdet.py"))
    dump(directory / "resolved-mmdet.json", cfg.to_dict())
    dump(
        directory / "model-source.json",
        {
            "backend": "mmdet",
            "release": "v3.3.0",
            "recipe": recipe.name,
            "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
            "resume_source": request.get("resume"),
            "initialization": "optimizer-state resume"
            if request.get("resume")
            else (
                "supervised detector backbone; random neck/proposal/detection heads"
                if detector["scope"] == "backbone"
                else "complete supervised detector weights"
            )
            if detector
            else ("SimSiam backbone; random detection heads" if pretraining else "random"),
            "pretraining_data": pretraining,
            "detector_source": detector,
            "initialization_rule": request.get("compatibility"),
            "backbone_style": style,
            "learning": settings["learning"],
            "method": "SoftTeacher" if semi else "FasterRCNN",
            "teacher": {"update": "upstream MeanTeacherHook", "momentum": 0.001} if semi else None,
            "pseudo_label_policy": dict(cfg.model.semi_train_cfg) if semi else None,
            "pseudo_labels_serialized": False,
            "pseudo_label_visibility": (
                "upstream generates labels on the fly; no per-image dump" if semi else None
            ),
            "recipe_changes": [
                "supervised detector initialization"
                if detector
                else (
                    "SimSiam backbone initialization" if pretraining else "random initialization"
                ),
                f"{style} backbone and normalization",
                "fixed image scale",
                "1:1 source ratio" if semi else "labeled-only batches",
                "iteration budget",
                "no warmup/decay",
            ],
        },
    )
    return cfg


def train(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from mmengine.runner import Runner

    cfg = configure(request, directory)
    runner = Runner.from_cfg(cfg)
    checkpoint_audit(runner, directory)
    record_model_device(
        runner.model, request.get("execution_device", request["config"]["device"]), directory
    )
    if request.get("pretraining"):
        verify_transfer(runner, request["pretraining"], directory)
    elif request.get("detector"):
        if request["detector"]["scope"] == "backbone":
            verify_transfer(runner, request["detector"], directory)
        else:
            initialize_detector(runner, request, directory)
    result: dict[str, Any] = train_detector(
        runner,
        directory,
        request.get("execution_device", request["config"]["device"]),
        request["config"]["max_iters"],
    )
    return result


def verify_transfer(runner: Any, source: dict[str, Any], directory: Path) -> None:
    """Check the complete backbone contract, then let upstream initialize its weights."""
    from mmengine.runner.checkpoint import _load_checkpoint_with_prefix

    weights = _load_checkpoint_with_prefix("backbone", source["checkpoint"], map_location="cpu")
    for detector in initialization_targets(runner.model).values():
        check_state(detector.backbone, weights)
        detector.backbone.init_weights()
    audit_initialization(runner, weights, "backbone", source, directory)


def initialization_targets(model: Any) -> dict[str, Any]:
    if hasattr(model, "teacher"):
        return {"teacher": model.teacher, "student": model.student}
    return {"detector": model}


def check_state(module: Any, weights: dict[str, Any]) -> None:
    expected = module.state_dict()
    if set(expected) != set(weights) or any(
        expected[k].shape != weights[k].shape or expected[k].dtype != weights[k].dtype
        for k in expected
    ):
        raise ValueError(
            "Source checkpoint does not contain complete compatible state keys/shapes/dtypes"
        )


def model_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: model_structure(item) for key, item in value.items() if key != "init_cfg"}
    if isinstance(value, (list, tuple)):
        return [model_structure(item) for item in value]
    return value


def validate_detector_structure(source: dict[str, Any], target: Any) -> None:
    config = Path(source["model_config"])
    if hashlib.sha256(config.read_bytes()).hexdigest() != source["model_config_sha256"]:
        raise ValueError("Detector source model configuration content has changed")
    values = json.loads(config.read_text(encoding="utf-8"))
    keys = ("backbone", "data_preprocessor") if source["scope"] == "backbone" else tuple(target)
    if model_structure({k: values["model"][k] for k in keys}) != model_structure(
        {k: target[k] for k in keys}
    ):
        raise ValueError("Detector source structure/preprocessing differs from target detector")


def initialize_detector(runner: Any, request: dict[str, Any], directory: Path) -> None:
    from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint

    source = request["detector"]
    path = Path(source["checkpoint"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["checkpoint_sha256"]:
        raise ValueError("Detector source checkpoint content has changed")
    checkpoint = _load_checkpoint(str(path), map_location="cpu")
    if (
        list(checkpoint.get("meta", {}).get("dataset_meta", {}).get("classes", []))
        != request["classes"]
    ):
        raise ValueError("Detector checkpoint class map differs from target dataset")
    weights = checkpoint["state_dict"]
    for detector in initialization_targets(runner.model).values():
        check_state(detector, weights)
    runner.model.init_weights()
    # Upstream SemiBaseDetector expands supervised keys into teacher/student branches.
    load_checkpoint(runner.model, str(path), map_location="cpu", strict=True)
    audit_initialization(runner, weights, "detector", source, directory)


def audit_initialization(
    runner: Any, weights: dict[str, Any], scope: str, source: dict[str, Any], directory: Path
) -> None:
    """Read-only audit after upstream initialization/hooks, before the first training step."""
    import torch
    from mmengine.hooks import Hook

    # Optional framework types are unavailable to the core's type checker.
    class InitializationAudit(Hook):  # type: ignore[misc]
        def before_train(self, runner: Any) -> None:
            branches = {}
            for name, detector in initialization_targets(runner.model).items():
                module = detector.backbone if scope == "backbone" else detector
                check_state(module, weights)
                if not all(
                    torch.equal(v.detach().cpu(), weights[k])
                    for k, v in module.state_dict().items()
                ):
                    raise RuntimeError(f"{name} {scope} initialization differs from source weights")
                branches[name] = {"tensor_count": len(weights), "all_loaded_tensors_equal": True}
            dump(
                directory / "transfer.json",
                {
                    "source_checkpoint_sha256": source["checkpoint_sha256"],
                    "prefix": "backbone" if scope == "backbone" else None,
                    "scope": scope,
                    "tensor_count": len(weights),
                    "all_keys_and_shapes_match": True,
                    "all_loaded_tensors_equal": True,
                    "audit_stage": "before_train; after upstream initialization hooks",
                    "branches": branches,
                    "detection_heads": "random initialization"
                    if scope == "backbone"
                    else "transferred",
                    "optimizer_state_loaded": False,
                    "data_compatibility": source.get("transfer_data"),
                },
            )

    runner.register_hook(InitializationAudit(), priority="LOW")


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported worker request schema")
    directory, device = worker_context(request)
    disable_network_for_tests()
    print("WVision: checking backend environment and compiled operators", flush=True)
    dependencies(directory)
    runtime_device(device, directory, requested=request.get("requested_device"))
    runtime_precision(request.get("config", {}).get("precision", "fp32"), directory)
    import torch
    from mmdet.utils import register_all_modules

    torch.set_num_threads(2)
    register_all_modules(init_default_scope=True)
    print(f"WVision: executing {request['operation']}", flush=True)
    if request["operation"] == "train":
        result = train(request, directory)
    elif request["operation"] in {"val", "test"}:
        result = evaluate(request, directory)
    elif request["operation"] == "export":
        try:
            from deployment import export_detector  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "MMDetection export requires its MMDeploy environment; "
                "see docs/TRAINING_OPERATIONS.md"
            ) from error
        result = export_detector(
            inference_model(request), load_config(request["model_config"]), request, directory
        )
    elif request["operation"] == "predict":
        result = predict(request, directory)
    else:
        raise ValueError("Unsupported MMDetection operation")
    if request.get("rank", 0) == 0:
        dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    with worker_lock(Path(sys.argv[1])):
        main()
