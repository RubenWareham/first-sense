"""Standalone Python 3.10 worker: upstream owns SimSiam and its training loop."""

import hashlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

# Load the packaged framework-free helper without importing the application.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/openmmlab"))
from devices import record_model_device, runtime_device  # type: ignore[import-not-found]
from runtime import dependencies, dump  # type: ignore[import-not-found]


def train(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    import mmpretrain
    from mmengine.config import Config
    from mmengine.runner import Runner

    settings = request["config"]
    recipe = (
        Path(mmpretrain.__file__).parent
        / ".mim/configs/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py"
    )
    print("WVision: loading installed SimSiam recipe", flush=True)
    cfg = Config.fromfile(str(recipe))
    cfg.model.backbone.norm_cfg = dict(type="BN")
    cfg.model.backbone.init_cfg = None
    cfg.view_pipeline[0]["scale"] = settings["imgsz"]
    cfg.train_pipeline[1]["transforms"] = [cfg.view_pipeline]
    cfg.train_dataloader = dict(
        batch_size=settings["batch"],
        num_workers=0,
        persistent_workers=False,
        drop_last=True,
        sampler=dict(type="InfiniteSampler", shuffle=True),
        collate_fn=dict(type="default_collate"),
        dataset=dict(
            type="CustomDataset",
            data_root="",
            data_prefix=str(directory / "data/train_unlabeled/images"),
            ann_file=request["images"],
            with_label=False,
            pipeline=cfg.train_pipeline,
        ),
    )
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=settings["max_iters"])
    cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
    cfg.test_cfg = cfg.test_dataloader = cfg.test_evaluator = None
    cfg.param_scheduler = []
    cfg.optim_wrapper.optimizer.lr = settings["lr"]
    cfg.custom_hooks[0].update(lr=settings["lr"], adjust_by_epoch=False)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=settings["max_iters"],
        max_keep_ckpts=1,
        save_optimizer=False,
        save_param_scheduler=False,
    )
    cfg.default_hooks.logger.interval = 1
    cfg.env_cfg.mp_cfg = dict(mp_start_method="spawn", opencv_num_threads=0)
    cfg.env_cfg.dist_cfg = dict(backend="gloo")
    cfg.work_dir = str(directory / "upstream")
    cfg.launcher = "none"
    cfg.device = settings["device"]
    cfg.randomness = dict(seed=settings["seed"], deterministic=True)
    cfg.load_from = None
    cfg.resume = False
    cfg.dump(str(directory / "resolved-mmpretrain.py"))
    dump(directory / "resolved-mmpretrain.json", cfg.to_dict())
    # Include inherited files so the recipe identity covers more than its leaf file.
    sources = [
        recipe,
        *[
            (recipe.parent / name).resolve()
            for name in [
                "../_base_/datasets/imagenet_bs32_mocov2.py",
                "../_base_/schedules/imagenet_sgd_coslr_200e.py",
                "../_base_/default_runtime.py",
            ]
        ],
    ]
    dump(
        directory / "model-source.json",
        {
            "backend": "mmpretrain",
            "version": "1.2.0",
            "method": "SimSiam",
            "recipe": recipe.name,
            "recipe_sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "initialization": "random",
            "external_pretraining": None,
            "supervision": "training images only; no human labels",
            "selection": "final iteration; no validation or test selection",
            "recipe_changes": [
                "single-device BatchNorm instead of SyncBN",
                "fixed crop size",
                "iteration budget",
                "constant learning rate",
                "unlabeled local list",
            ],
        },
    )
    print("WVision: building upstream SimSiam runner", flush=True)
    runner = Runner.from_cfg(cfg)
    record_model_device(runner.model, settings["device"], directory)
    runner.train()
    record_model_device(runner.model, settings["device"], directory)
    logs = {k: float(v.current()) for k, v in runner.message_hub.log_scalars.items()}
    dump(directory / "training-scalars.json", logs)
    checkpoint = Path(runner.work_dir) / f"iter_{settings['max_iters']}.pth"
    if not checkpoint.is_file() or "train/loss" not in logs:
        raise RuntimeError("Upstream produced no checkpoint or training loss")
    return {
        "checkpoint": str(checkpoint),
        "metrics": {f"mmpretrain/{k}": v for k, v in logs.items() if k.startswith("train/")},
    }


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schema_version") != 1 or request.get("operation") != "pretrain":
        raise ValueError("Unsupported pretraining request")
    if metadata.version("mmpretrain") != "1.2.0":
        raise RuntimeError("Pretraining requires mmpretrain==1.2.0")
    directory = Path(request["directory"])
    if os.environ.get("WVISION_TEST_OFFLINE") == "1":
        import socket

        def deny_network(*args: Any, **kwargs: Any) -> None:
            raise OSError("Network disabled by WVision integration test")

        socket.socket.connect = deny_network  # type: ignore[method-assign]
        socket.create_connection = deny_network  # type: ignore[assignment]
    print("WVision: checking pretraining dependencies", flush=True)
    dependencies(directory)
    runtime_device(request["config"]["device"], directory)
    import torch
    from mmpretrain.utils import register_all_modules

    torch.set_num_threads(2)
    print("WVision: registering upstream modules", flush=True)
    register_all_modules(init_default_scope=True)
    result = train(request, directory)
    dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    main()
