"""Isolated RTMPose worker using the installed MMPose recipe and runner."""

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/openmmlab"))
from checkpoints import worker_lock  # type: ignore[import-not-found]
from devices import record_model_device, runtime_device, runtime_precision, worker_context  # type: ignore[import-not-found]
from runtime import dependencies, disable_network_for_tests, dump  # type: ignore[import-not-found]

RECIPES = {
    f"mmpose/rtmpose-{size}": f"coco/rtmpose-{size}_8xb256-420e_coco-256x192.py"
    for size in ("t", "s", "m", "l")
}
RECIPES["mmpose/rtmpose-x"] = "body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py"
METAINFO = {"coco17": "coco", "halpe26": "halpe26"}


def dataloader(cfg: Any, path: str, batch: int, metainfo: Path, *, training: bool = False) -> Any:
    # Use the upstream transforms and model. This slice omits the optional
    # Albumentations pipeline, pretrained initialization, and long schedule.
    pipeline = cfg.train_pipeline if training else cfg.val_pipeline
    return dict(
        batch_size=batch,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type="DefaultSampler", shuffle=training),
        dataset=dict(
            type="CocoDataset",
            data_mode="topdown",
            ann_file=path,
            metainfo=dict(from_file=str(metainfo)),
            data_prefix=dict(img=""),
            test_mode=not training,
            pipeline=[step for step in pipeline if step["type"] != "Albumentation"],
        ),
    )


def evaluator(path: str) -> dict[str, Any]:
    return dict(type="CocoMetric", ann_file=path, score_mode="bbox", nms_mode="none")


def train(request: dict[str, Any], directory: Path, configs: Path, device: str) -> dict[str, Any]:
    from mmengine.config import Config
    from mmengine.runner import Runner

    model_id = request["model_id"]
    recipe = configs / "body_2d_keypoint/rtmpose" / RECIPES[model_id]
    cfg = Config.fromfile(str(recipe))
    settings = request["config"]
    skeleton = request["skeleton"]
    data = request["data"]
    metainfo = configs / "_base_/datasets" / (METAINFO[skeleton["id"]] + ".py")
    if cfg.codec.input_size[1] != settings["imgsz"]:
        raise ValueError("RTMPose input size differs from the selected upstream recipe")
    cfg.model.backbone.init_cfg = None
    cfg.model.backbone.norm_cfg = dict(type="BN")
    cfg.model.head.out_channels = len(skeleton["joints"])
    cfg.train_dataloader = dataloader(
        cfg, data["train_labeled"], settings["batch"], metainfo, training=True
    )
    cfg.val_dataloader = dataloader(cfg, data["val"], settings["batch"], metainfo)
    cfg.val_evaluator = evaluator(data["val"])
    cfg.test_dataloader = cfg.test_evaluator = cfg.test_cfg = None
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=settings["epochs"], val_interval=1)
    cfg.param_scheduler = []
    cfg.custom_hooks = []
    cfg.auto_scale_lr = dict(enable=False)
    cfg.default_hooks.checkpoint = dict(
        type="CheckpointHook", interval=1, save_best="coco/AP", rule="greater", max_keep_ckpts=1
    )
    cfg.env_cfg.mp_cfg = dict(mp_start_method="spawn", opencv_num_threads=0)
    cfg.randomness = dict(seed=settings["seed"])
    cfg.launcher = "none"
    cfg.work_dir = str(directory / "upstream")
    resolved = directory / "resolved-mmpose.json"
    dump(resolved, cfg.to_dict())
    dump(
        directory / "model-source.json",
        {
            "backend": "mmpose",
            "version": "1.3.2",
            "model": model_id,
            "recipe": recipe.relative_to(configs).as_posix(),
            "recipe_sources": {
                path.relative_to(configs).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (recipe, configs / "_base_/default_runtime.py", metainfo)
            },
            "initialization": "random",
            "crop_size_wh": list(cfg.codec.input_size),
            "skeleton": skeleton["id"],
        },
    )
    runner = Runner.from_cfg(cfg)
    runner.train()
    record_model_device(runner.model, device, directory)
    best = runner.message_hub.get_info("best_ckpt")
    last = Path(runner.work_dir) / f"epoch_{settings['epochs']}.pth"
    if best is None or not Path(best).is_file() or not last.is_file():
        raise RuntimeError("RTMPose produced no validation-selected checkpoint")
    return {
        "best": str(best),
        "last": str(last),
        "model_config": str(resolved),
        "metrics": {"mmpose/coco/AP": float(runner.message_hub.get_info("best_score"))},
    }


def load_config(request: dict[str, Any]) -> Any:
    from mmengine.config import Config

    cfg = Config(json.loads(Path(request["model_config"]).read_text(encoding="utf-8")))
    if cfg.model.head.out_channels != len(request["skeleton"]["joints"]):
        raise ValueError("RTMPose head width must match the saved skeleton")
    return cfg


def evaluate(
    request: dict[str, Any], directory: Path, configs: Path, device: str
) -> dict[str, Any]:
    from mmengine.runner import Runner
    from mmpose.datasets.datasets.utils import parse_pose_metainfo

    cfg = load_config(request)
    skeleton = request["skeleton"]["id"]
    metainfo = configs / "_base_/datasets" / (METAINFO[skeleton] + ".py")
    batch = cfg.val_dataloader.batch_size
    cfg.train_dataloader = cfg.train_cfg = cfg.optim_wrapper = cfg.param_scheduler = None
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.test_cfg = {}
    cfg.test_dataloader = dataloader(cfg, request["data"], batch, metainfo)
    cfg.test_evaluator = evaluator(request["data"])
    cfg.work_dir = str(directory / "upstream")
    cfg.load_from = request["checkpoint"]
    runner = Runner.from_cfg(cfg)
    scores = runner.test()
    record_model_device(runner.model, device, directory)
    dump(
        directory / "pose-protocol.json",
        {
            "backend": "mmpose",
            "version": "1.3.2",
            "skeleton": skeleton,
            "boxes": "ground_truth",
            "oks_sigmas": parse_pose_metainfo({"from_file": str(metainfo)})["sigmas"].tolist(),
            "score_mode": "bbox",
            "nms_mode": "none",
        },
    )
    return {"metrics": {f"mmpose/{name}": float(score) for name, score in scores.items()}}


def predict(request: dict[str, Any], directory: Path, device: str) -> dict[str, int]:
    import mmcv
    from mmpose.apis import inference_topdown, init_model

    cfg = load_config(request)
    skeleton = request["skeleton"]
    count = len(skeleton["joints"])
    # The inference API reads the transform recipe from this field.
    cfg.test_dataloader = dict(dataset=dict(pipeline=cfg.val_pipeline))
    model = init_model(cfg, request["checkpoint"], device=device)
    names = [model.dataset_meta["keypoint_id2name"][i] for i in range(count)]
    if names != skeleton["joints"]:
        raise ValueError("RTMPose checkpoint joint order must match the saved skeleton")
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in request["images"]:
            image = mmcv.imread(item["path"], flag="color_ignore_orientation")
            if image is None:
                raise ValueError(f"Cannot decode {item['path']}")
            boxes = item["boxes"]
            results = inference_topdown(model, image, boxes, bbox_format="xyxy") if boxes else []
            persons = []
            for box, result in zip(boxes, results, strict=True):
                points = result.pred_instances
                joints = [
                    {"x": x, "y": y, "confidence": score}
                    for (x, y), score in zip(
                        points.keypoints[0].tolist(),
                        points.keypoint_scores[0].tolist(),
                        strict=True,
                    )
                ]
                if len(joints) != count:
                    raise ValueError("RTMPose prediction does not match the skeleton")
                persons.append({"class_id": 0, "box": box, "confidence": None, "joints": joints})
            stream.write(
                json.dumps(
                    {
                        "sample": item["path"],
                        "height": image.shape[0],
                        "width": image.shape[1],
                        "classes": {0: "person"},
                        "coordinate_format": "absolute_pixel_xyxy",
                        "joint_coordinate_format": "absolute_pixel_xy",
                        "skeleton": skeleton["id"],
                        "keypoint_count": count,
                        "persons": persons,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    record_model_device(model, device, directory)
    return {"sample_count": len(request["images"])}


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    operation = request.get("operation")
    if request.get("schema_version") != 1 or operation not in {"train", "val", "test", "predict"}:
        raise ValueError("Unsupported RTMPose worker request")
    directory, device = worker_context(request)
    disable_network_for_tests()
    dependencies(directory)
    for name, version in {"mmpose": "1.3.2", "xtcocotools": "1.14.3"}.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"RTMPose requires {name}=={version}")
    import mmpose
    from mmpose.utils import register_all_modules

    runtime_device(device, directory)
    runtime_precision("fp32", directory)
    register_all_modules()
    configs = Path(mmpose.__file__).parent / ".mim/configs"
    if operation == "train":
        result = train(request, directory, configs, device)
    elif operation == "predict":
        result = predict(request, directory, device)
    else:
        result = evaluate(request, directory, configs, device)
    dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    with worker_lock(Path(sys.argv[1])):
        main()
