"""Standalone Transformers worker; all learning/model/loss code comes from upstream."""

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

# Load the packaged framework-free helper without importing the application.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/transformers"))
from devices import isolate_worker_device, record_model_device, runtime_device  # type: ignore[import-not-found]
from runtime import dependencies, digest, dump  # type: ignore[import-not-found]


def processor(size: int) -> Any:
    from transformers import DetrImageProcessor

    # Construct locally: no remote pretrained processor or hidden network lookup.
    return DetrImageProcessor(
        size={"height": size, "width": size},
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )


def load_coco(path: str) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def coco_dataset(path: str) -> Any:
    from torch.utils.data import Dataset

    coco = load_coco(path)
    annotations: dict[int, list[Any]] = {image["id"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        # Upstream DETR requires zero-based labels; COCO staging uses IDs 1..N.
        annotations[annotation["image_id"]].append(
            {**annotation, "category_id": annotation["category_id"] - 1}
        )

    # Optional framework types are unavailable in the core-only environment.
    class CocoImages(Dataset[dict[str, Any]]):  # type: ignore[misc]
        def __len__(self) -> int:
            return len(coco["images"])

        def __getitem__(self, index: int) -> dict[str, Any]:
            image = coco["images"][index]
            return {
                "path": image["file_name"],
                "image_id": image["id"],
                "annotations": annotations[image["id"]],
            }

    return CocoImages()


def collator(image_processor: Any) -> Any:
    from PIL import Image

    def collate(examples: list[dict[str, Any]]) -> Any:
        images = []
        for sample in examples:
            with Image.open(sample["path"]) as image:
                images.append(image.convert("RGB"))
        labels = [
            {"image_id": item["image_id"], "annotations": item["annotations"]} for item in examples
        ]
        return image_processor(images=images, annotations=labels, return_tensors="pt")

    return collate


def build_model(request: dict[str, Any], directory: Path) -> Any:
    import torch
    from transformers import (
        DetrConfig,
        DetrForObjectDetection,
        DINOv3ConvNextBackbone,
        DINOv3ConvNextConfig,
    )

    source = request.get("backbone")
    if source:
        root = Path(source["directory"])
        for name, expected in source["files"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Backbone source changed after preflight: {name}")
        # The pinned AutoBackbone.from_pretrained probes Hub repo_exists even for local paths.
        # The concrete upstream class avoids that lookup without patching upstream code.
        backbone, loading = DINOv3ConvNextBackbone.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            out_features=["stage4"],
            output_loading_info=True,
        )
        # A representation model has a final pooling norm which a dense backbone does not use.
        allowed_unused = {"layer_norm.weight", "layer_norm.bias"}
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or set(loading.get("unexpected_keys", [])) - allowed_unused
        ):
            raise ValueError(f"Incomplete/incompatible DINOv3 backbone checkpoint: {loading}")
        config = backbone.config
    else:
        config = DINOv3ConvNextConfig(out_features=["stage4"])
        backbone = None
    classes = dict(enumerate(request["classes"]))
    cfg = DetrConfig(
        backbone_config=config,
        num_labels=len(classes),
        id2label=classes,
        label2id={name: key for key, name in classes.items()},
    )
    model = DetrForObjectDetection(cfg)
    if backbone is not None:
        assert source is not None
        # The upstream DETR feature wrapper preserves masks and positional encoding.
        target = model.model.backbone.model
        expected = target.state_dict()
        weights = backbone.state_dict()
        if set(expected) != set(weights) or any(
            expected[k].shape != weights[k].shape for k in expected
        ):
            raise ValueError("Backbone state does not match the configured DETR feature extractor")
        target.load_state_dict(weights, strict=True)
        if not all(torch.equal(v, weights[k]) for k, v in target.state_dict().items()):
            raise RuntimeError("Backbone tensor verification failed")
        dump(
            directory / "transfer.json",
            {
                "tensor_count": len(weights),
                "all_keys_and_shapes_match": True,
                "all_loaded_tensors_equal": True,
                "source_files": source["files"],
                "upstream_loading_info": {
                    key: sorted(loading.get(key, []))
                    for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
                },
            },
        )
    if request["config"]["freeze_backbone"]:
        model.model.freeze_backbone()
    else:
        model.model.unfreeze_backbone()
    # Persist the complete upstream configuration; inference never executes Python config.
    dump(directory / "resolved-transformers.json", model.config.to_dict())
    dump(
        directory / "model-source.json",
        {
            "backend": "transformers",
            "version": "5.22.2",
            "repository": "https://github.com/huggingface/transformers",
            "release": "v5.22.2",
            "upstream_source_sha256": {
                "detr": digest(Path(inspect.getfile(DetrForObjectDetection))),
                "dinov3_convnext": digest(Path(inspect.getfile(DINOv3ConvNextBackbone))),
            },
            "detector": "DetrForObjectDetection",
            "backbone": "DINOv3ConvNextBackbone",
            "variant": "tiny",
            "backbone_source": source,
            "initialization": "user-provided local backbone; random detection head"
            if source
            else "random",
            "supervision": "supervised downstream detection; see source supervision separately",
            "official_pretrained_verification": False,
            "weights_downloaded": False,
            "freeze_backbone": request["config"]["freeze_backbone"],
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "augmentation": "square resize; ImageNet RGB normalization; no random augmentation",
            "selection": "validation loss at final step via upstream Trainer",
        },
    )
    return model


def train(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments, set_seed

    settings = request["config"]
    set_seed(settings["seed"])
    model = build_model(request, directory)
    arguments = TrainingArguments(
        output_dir=str(directory / "upstream"),
        use_cpu=request["execution_device"] == "cpu",
        fp16=False,
        bf16=False,
        max_steps=settings["max_iters"],
        learning_rate=settings["lr"],
        per_device_train_batch_size=settings["batch"],
        per_device_eval_batch_size=settings["batch"],
        seed=settings["seed"],
        data_seed=settings["seed"],
        eval_strategy="steps",
        eval_steps=settings["max_iters"],
        save_strategy="steps",
        save_steps=settings["max_iters"],
        save_total_limit=1,
        # Only one validation/save occurs, at the final step. Avoid an unnecessary
        # reload through Trainer's legacy key path; inference uses from_pretrained.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_only_model=True,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        report_to="none",
        push_to_hub=False,
        disable_tqdm=True,
        remove_unused_columns=False,
        prediction_loss_only=True,
        label_names=["labels"],
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_steps=1,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=coco_dataset(request["data"]["train_labeled"]),
        eval_dataset=coco_dataset(request["data"]["val"]),
        data_collator=collator(processor(settings["imgsz"])),
    )
    record_model_device(trainer.model, request["execution_device"], directory)
    trainer.train()
    record_model_device(trainer.model, request["execution_device"], directory)
    dump(directory / "training-scalars.json", trainer.state.log_history)
    checkpoint = Path(trainer.state.best_model_checkpoint) / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("Trainer did not produce the selected detector checkpoint")
    # Evaluation occurs once at the final step, so the selected and final checkpoint coincide.
    return {
        "best": str(checkpoint),
        "last": str(checkpoint),
        "model_config": str(directory / "resolved-transformers.json"),
        "metrics": {"transformers/eval_loss": float(trainer.state.best_metric)},
    }


def restore(request: dict[str, Any]) -> Any:
    from transformers import DetrConfig, DetrForObjectDetection

    values = json.loads(Path(request["model_config"]).read_text(encoding="utf-8"))
    model, loading = DetrForObjectDetection.from_pretrained(
        str(Path(request["checkpoint"]).parent),
        config=DetrConfig.from_dict(values),
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"Incomplete/incompatible detector checkpoint: {loading}")
    model.to(request["execution_device"])
    model.eval()
    record_model_device(model, request["execution_device"], Path(request["directory"]))
    return model


def predictions(model: Any, paths: list[str], size: int) -> Any:
    import torch
    from PIL import Image

    image_processor = processor(size)
    device = next(model.parameters()).device
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            result = model(**image_processor(images=image, return_tensors="pt").to(device))
            boxes = image_processor.post_process_object_detection(
                result,
                threshold=0.0,
                target_sizes=torch.tensor([[image.height, image.width]], device=device),
            )[0]
        yield {
            "sample": path,
            "height": image.height,
            "width": image.width,
            "coordinate_format": "absolute_pixel_xyxy",
            "classes": model.config.id2label,
            "boxes": boxes["boxes"].tolist(),
            "scores": boxes["scores"].tolist(),
            "class_ids": boxes["labels"].tolist(),
        }


def evaluate(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model = restore(request)
    if [model.config.id2label[k] for k in sorted(model.config.id2label)] != request["classes"]:
        raise ValueError("Checkpoint class map differs from evaluation data")
    coco = COCO(request["data"])
    images = list(coco.imgs.values())
    detections = []
    for image, result in zip(
        images, predictions(model, [i["file_name"] for i in images], request["imgsz"]), strict=True
    ):
        for box, score, cls in zip(
            result["boxes"], result["scores"], result["class_ids"], strict=True
        ):
            x1, y1, x2, y2 = box
            detections.append(
                {
                    "image_id": image["id"],
                    "category_id": cls + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
    dump(directory / "coco-predictions.json", detections)
    if detections:
        predicted = coco.loadRes(detections)
    else:
        predicted = COCO()
        predicted.dataset = {
            "images": images,
            "categories": coco.dataset["categories"],
            "annotations": [],
        }
        predicted.createIndex()
    evaluator = COCOeval(coco, predicted, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    record_model_device(model, request["execution_device"], directory)
    names = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAP_s", "bbox_mAP_m", "bbox_mAP_l")
    return {
        "metrics": {
            f"transformers/coco/{name}": float(value)
            for name, value in zip(names, evaluator.stats[:6], strict=True)
        }
    }


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schema_version") != 1:
        raise ValueError("Unsupported worker request schema")
    if os.environ.get("WVISION_TEST_OFFLINE") == "1":
        import socket

        def deny_network(*args: Any, **kwargs: Any) -> None:
            raise OSError("Network disabled by WVision integration test")

        socket.socket.connect = deny_network  # type: ignore[method-assign]
        socket.create_connection = deny_network  # type: ignore[assignment]
    directory = Path(request["directory"])
    requested = request.get("device", request.get("config", {}).get("device", "cpu"))
    request["execution_device"] = isolate_worker_device(requested)
    dependencies(directory)
    runtime_device(request["execution_device"], directory, requested=requested)
    import torch

    torch.set_num_threads(2)
    if request["operation"] == "train":
        result = train(request, directory)
    elif request["operation"] in ("val", "test"):
        result = evaluate(request, directory)
    elif request["operation"] == "predict":
        model = restore(request)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for item in predictions(model, request["source"], request["imgsz"]):
                handle.write(json.dumps(item, allow_nan=False) + "\n")
        result = {"sample_count": len(request["source"])}
        record_model_device(model, request["execution_device"], directory)
    else:
        raise ValueError("Unsupported DINOv3 operation")
    dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    main()
