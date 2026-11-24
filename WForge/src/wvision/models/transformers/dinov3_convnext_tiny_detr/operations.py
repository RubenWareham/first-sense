"""Standalone Transformers worker; all learning/model/loss code comes from upstream."""

import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Load the packaged framework-free helper without importing the application.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "adapters/transformers"))
from checkpoints import record_checkpoint, worker_lock  # type: ignore[import-not-found]
from devices import record_model_device, runtime_device, runtime_precision, worker_context  # type: ignore[import-not-found]
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
            "selection": "minimum checkpoint-interval validation loss via upstream Trainer",
        },
    )
    return model


def train(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    settings = request["config"]
    set_seed(settings["seed"])
    if request.get("resume"):
        from transformers import DetrForObjectDetection

        model = DetrForObjectDetection.from_pretrained(
            request["resume"]["checkpoint"],
            local_files_only=True,
            use_safetensors=True,
        )
        if settings["freeze_backbone"]:
            model.model.freeze_backbone()
        else:
            model.model.unfreeze_backbone()
        source = request["resume"]
        if source.get("selected"):
            selected = directory / "upstream/selected"
            selected.mkdir(parents=True)
            shutil.copyfile(source["selected"]["checkpoint"], selected / "model.safetensors")
            shutil.copyfile(Path(source["checkpoint"]) / "config.json", selected / "config.json")
            state_file = Path(source["checkpoint"]) / "trainer_state.json"
            state = json.loads(state_file.read_text(encoding="utf-8"))
            state["best_model_checkpoint"] = str(selected)
            dump(state_file, state)
        dump(directory / "resolved-transformers.json", model.config.to_dict())
        dump(
            directory / "model-source.json",
            {"initialization": "optimizer-state resume", "source": request["resume"]},
        )
    else:
        model = build_model(request, directory)
    interval = min(settings["checkpoint_interval"], settings["max_iters"])

    class Audit(TrainerCallback):  # type: ignore[misc]
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not state.is_world_process_zero:
                return
            record_checkpoint(
                directory,
                Path(args.output_dir) / f"checkpoint-{state.global_step}",
                state.global_step,
                selected=Path(state.best_model_checkpoint) / "model.safetensors"
                if state.best_model_checkpoint
                else None,
            )

    arguments = TrainingArguments(
        output_dir=request.get("upstream_directory", str(directory / "upstream")),
        ddp_backend=("gloo" if os.name == "nt" else "nccl") if request.get("distributed") else None,
        use_cpu=request["execution_device"] == "cpu",
        fp16=False,
        bf16=settings["precision"] == "bf16",
        max_steps=settings["max_iters"],
        learning_rate=settings["lr"],
        per_device_train_batch_size=settings["batch"],
        per_device_eval_batch_size=settings["batch"],
        seed=settings["seed"],
        data_seed=settings["seed"],
        eval_strategy="steps",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        save_total_limit=2,
        # Inference loads the selected checkpoint through upstream from_pretrained,
        # which applies the required checkpoint key conversions.
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_only_model=False,
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
        callbacks=[Audit()],
    )
    record_model_device(trainer.model, request["execution_device"], directory)
    trainer.train(
        resume_from_checkpoint=request["resume"]["checkpoint"] if request.get("resume") else None
    )
    record_model_device(trainer.model, request["execution_device"], directory)
    if not trainer.is_world_process_zero():
        return {}
    dump(directory / "training-scalars.json", trainer.state.log_history)
    checkpoint = Path(trainer.state.best_model_checkpoint) / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("Trainer did not produce the selected detector checkpoint")
    return {
        "best": str(checkpoint),
        "last": str(
            Path(trainer.args.output_dir)
            / f"checkpoint-{trainer.state.global_step}"
            / "model.safetensors"
        ),
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


def predictions(
    model: Any,
    paths: list[str],
    size: int,
    *,
    score_threshold: float = 0.0,
    max_detections: int = 100,
) -> Any:
    import torch
    from PIL import Image
    from torchvision.ops import clip_boxes_to_image

    image_processor = processor(size)
    device = next(model.parameters()).device
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            result = model(**image_processor(images=image, return_tensors="pt").to(device))
            boxes = image_processor.post_process_object_detection(
                result,
                threshold=score_threshold,
                target_sizes=torch.tensor([[image.height, image.width]], device=device),
            )[0]
            keep = boxes["scores"].argsort(descending=True)[:max_detections]
            boxes = {key: value[keep] for key, value in boxes.items()}
            boxes["boxes"] = clip_boxes_to_image(boxes["boxes"], (image.height, image.width))
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


def export(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Export upstream forward outputs and verify them using ONNX Runtime on CPU."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Install onnx==1.17.0 and onnxruntime==1.20.1 in the DINOv3 backend"
        ) from error

    model = restore(request).float().eval()
    size = request["imgsz"]

    class Outputs(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.detector = model

        def forward(self, pixel_values: Any, pixel_mask: Any) -> Any:
            values = self.detector(
                pixel_values=pixel_values, pixel_mask=pixel_mask, return_dict=True
            )
            return values.logits, values.pred_boxes

    wrapper = Outputs().eval()
    pixels = torch.zeros((1, 3, size, size), dtype=torch.float32)
    mask = torch.ones((1, size, size), dtype=torch.int64)
    target = directory / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (pixels, mask),
            str(target),
            opset_version=17,
            dynamo=False,
            input_names=["pixel_values", "pixel_mask"],
            output_names=["logits", "pred_boxes"],
        )
    onnx.checker.check_model(onnx.load(str(target)))
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    errors = {"logits": 0.0, "pred_boxes": 0.0}
    for sample in (pixels, torch.linspace(-1, 1, 3 * size * size).reshape(1, 3, size, size)):
        with torch.inference_mode():
            native = wrapper(sample, mask)
        feeds = {"pixel_values": sample.numpy(), "pixel_mask": mask.numpy()}
        actual = session.run(None, {item.name: feeds[item.name] for item in session.get_inputs()})
        for name, expected, observed in zip(errors, native, actual, strict=True):
            reference = expected.cpu().numpy()
            if not np.isfinite(observed).all():
                raise ValueError("DINOv3 export produced non-finite values")
            np.testing.assert_allclose(observed, reference, rtol=1e-3, atol=1e-4)
            errors[name] = max(errors[name], float(np.max(np.abs(observed - reference))))
    dump(
        directory / "export.json",
        {
            "schema_version": 1,
            "format": "onnx",
            "model": target.name,
            "sha256": digest(target),
            "source_checkpoint_sha256": digest(Path(request["checkpoint"])),
            "classes": model.config.id2label,
            "precision": "fp32",
            "opset": 17,
            "inputs": [
                {"name": item.name, "shape": item.shape, "type": item.type}
                for item in session.get_inputs()
            ],
            "preprocessing": {
                "resize": [size, size],
                "preserve_aspect_ratio": False,
                "resampling": "PIL bilinear",
                "layout": "NCHW RGB",
                "rescale": 1 / 255,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "pixel_mask": "int64 ones; fixed square input",
            },
            "outputs": {
                "logits": "batch, queries, classes+1; final class is no-object",
                "pred_boxes": "batch, queries, normalized cxcywh",
                "postprocessing": "softmax logits over classes+1; discard final no-object column; "
                "select maximum foreground score/class per query; convert cxcywh to xyxy "
                "and scale to original image dimensions; clip to bounds; "
                "threshold and sort by score",
                "nms_included": False,
            },
            "verification": {
                "runtime": "onnxruntime CPUExecutionProvider",
                "samples": 2,
                "rtol": 1e-3,
                "atol": 1e-4,
                "max_absolute_error": errors,
            },
        },
    )
    record_model_device(model, "cpu", directory)
    return {"model": str(target)}


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
    directory, execution_device = worker_context(request, isolate_single=True)
    requested = request.get(
        "requested_device", request.get("device", request.get("config", {}).get("device", "cpu"))
    )
    dependencies(directory)
    runtime_device(execution_device, directory, requested=requested)
    runtime_precision(request.get("config", {}).get("precision", "fp32"), directory)
    import torch

    torch.set_num_threads(2)
    if request["operation"] == "train":
        result = train(request, directory)
    elif request["operation"] in ("val", "test"):
        result = evaluate(request, directory)
    elif request["operation"] == "export":
        result = export(request, directory)
    elif request["operation"] == "predict":
        model = restore(request)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for item in predictions(
                model,
                request["source"],
                request["imgsz"],
                score_threshold=request.get("score_threshold") or 0.0,
                max_detections=request.get("max_detections") or 100,
            ):
                handle.write(json.dumps(item, allow_nan=False) + "\n")
        result = {"sample_count": len(request["source"])}
        record_model_device(model, request["execution_device"], directory)
    else:
        raise ValueError("Unsupported DINOv3 operation")
    if request.get("rank", 0) == 0:
        dump(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    with worker_lock(Path(sys.argv[1])):
        main()
