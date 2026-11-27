"""Isolated RTMDet worker using MMDetection 3.3.0's packaged recipes."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/openmmlab"))
from checkpoints import worker_lock  # type: ignore[import-not-found]
from detection import (  # type: ignore[import-not-found]
    dataset_config,
    evaluate,
    predict,
    resize_pipeline,
    train_detector,
)
from devices import record_model_device, runtime_device, runtime_precision, worker_context  # type: ignore[import-not-found]
from runtime import dependencies, disable_network_for_tests, dump  # type: ignore[import-not-found]

RECIPES = {
    f"mmdetection/rtmdet-{size}": f"rtmdet_{size}_8xb32-300e_coco.py"
    for size in ("tiny", "s", "m", "l", "x")
}


def train(request: dict[str, Any], directory: Path, device: str) -> dict[str, Any]:
    import mmdet
    from mmengine.config import Config
    from mmengine.runner import Runner

    model_id = request["model_id"]
    recipe = Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / RECIPES[model_id]
    cfg = Config.fromfile(str(recipe))
    settings, data, classes = request["config"], request["data"], request["classes"]
    size, max_iters = settings["imgsz"], settings["max_iters"]
    interval = min(settings["checkpoint_interval"], max_iters)
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.num_classes = len(classes)
    cfg.model.test_cfg.max_per_img = 100
    for part in (cfg.model.backbone, cfg.model.neck, cfg.model.bbox_head):
        part.norm_cfg = dict(type="BN")
    # Use the upstream stage-two transforms at the requested size, with the
    # resize ratio fixed so all crops and pads follow the same image budget.
    pipeline = cfg.train_pipeline_stage2
    for transform in pipeline:
        if transform["type"] == "RandomResize":
            transform["ratio_range"] = (1.0, 1.0)
        elif transform["type"] == "RandomCrop":
            transform["allow_negative_crop"] = True
    resize_pipeline(pipeline, size)
    resize_pipeline(cfg.test_pipeline, size)
    cfg.train_dataloader = dict(
        batch_size=settings["batch"],
        num_workers=0,
        sampler=dict(type="InfiniteSampler", shuffle=True),
        dataset=dataset_config(data["train_labeled"], classes, pipeline),
    )
    validation = dataset_config(data["val"], classes, cfg.test_pipeline)
    validation["test_mode"] = True
    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=0,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=validation,
    )
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.val_evaluator = dict(type="CocoMetric", ann_file=data["val"], metric="bbox")
    cfg.test_evaluator = copy.deepcopy(cfg.val_evaluator)
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=interval)
    cfg.param_scheduler = []
    cfg.custom_hooks = []
    cfg.auto_scale_lr = dict(enable=False)
    cfg.optim_wrapper.optimizer.lr = settings["backend_options"].get("lr", 0.004)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=interval,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=1,
        save_optimizer=False,
    )
    cfg.default_hooks.logger.interval = 1
    cfg.env_cfg.mp_cfg = dict(mp_start_method="spawn", opencv_num_threads=0)
    cfg.randomness = dict(seed=settings["seed"], deterministic=True)
    cfg.launcher = "none"
    cfg.work_dir = str(directory / "upstream")
    cfg.load_from = None
    cfg.resume = False
    cfg.wvision_imgsz = size
    dump(directory / "resolved-mmdet.json", cfg.to_dict())
    dump(
        directory / "model-source.json",
        {
            "backend": "mmdet",
            "release": "v3.3.0",
            "model": model_id,
            "recipe": f"rtmdet/{recipe.name}",
            "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
            "initialization": "random",
            "learning": "supervised",
            "recipe_changes": [
                "single-device batch normalization",
                "fixed-scale stage-two transforms",
                "iteration budget at fixed learning rate",
                "no EMA or pipeline switching",
            ],
        },
    )
    runner = Runner.from_cfg(cfg)
    record_model_device(runner.model, device, directory)
    result: dict[str, Any] = train_detector(runner, directory, device, max_iters)
    return result


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    operation = request.get("operation")
    if request.get("schema_version") != 1 or operation not in {"train", "val", "test", "predict"}:
        raise ValueError("Unsupported RTMDet worker request")
    directory, device = worker_context(request)
    disable_network_for_tests()
    dependencies(directory)
    runtime_device(device, directory)
    runtime_precision("fp32", directory)
    import torch
    from mmdet.utils import register_all_modules

    torch.set_num_threads(2)
    register_all_modules(init_default_scope=True)
    if operation == "train":
        result = train(request, directory, device)
    elif operation == "predict":
        result = predict(request, directory)
    else:
        result = evaluate(request, directory)
    dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    with worker_lock(Path(sys.argv[1])):
        main()
