"""YOLO11 detection and pose operations. Training stays upstream."""

import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

from wvision.adapters.shared.checkpoints import record_checkpoint
from wvision.adapters.shared.devices import record_model_device, runtime_precision
from wvision.adapters.ultralytics.backend import BACKEND_VERSION, backend, metrics
from wvision.adapters.ultralytics.backend import check_dependency as check_dependency
from wvision.datasets.detection import sha256
from wvision.datasets.pose import JOINTS, PoseSkeleton
from wvision.experiments.artifacts import write_json
from wvision.models.ultralytics.yolo11.config import TrainConfig

SELECTION = {"metric": "0.1 * AP50 + 0.9 * AP50_95", "direction": "maximize"}
POSE_SELECTION = {
    "metric": "0.1 * (box AP50 + pose AP50) + 0.9 * (box AP50_95 + pose AP50_95)",
    "direction": "maximize",
}


def load_model(
    factory: Any,
    checkpoint: Path | None,
    task: str = "detect",
    *,
    scale: str = "n",
    skeleton: PoseSkeleton | None = None,
) -> Any:
    if task == "pose" and checkpoint is not None and skeleton is None:
        raise ValueError("A pose checkpoint requires its resolved skeleton")
    suffix = "-pose" if task == "pose" else ""
    model = factory(str(checkpoint) if checkpoint else f"yolo11{scale}{suffix}.yaml", task=task)
    definition = model.model.yaml
    if (
        model.task != task
        or "yolo11" not in Path(definition.get("yaml_file", "")).stem
        or definition.get("scale") != scale
        or (task == "pose" and checkpoint is not None and model.names != {0: "person"})
    ):
        raise ValueError(
            f"Checkpoint must belong to the registered YOLO11{scale} {task} architecture"
        )
    if (
        skeleton is not None
        and getattr(model.model.model[-1], "kpt_shape", None) != skeleton["kpt_shape"]
    ):
        raise ValueError(f"Checkpoint keypoint shape differs from skeleton {skeleton['id']!r}")
    return model


def train(
    checkpoint: Path | None,
    data: Path,
    config: TrainConfig,
    directory: Path,
    *,
    task: str = "detect",
    scale: str = "n",
    resume: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_device = distributed["execution_device"] if distributed else config.device
    upstream = Path(distributed["upstream_directory"]) if distributed else directory / "upstream"
    with backend(
        selected_device,
        directory,
        managed_distributed=bool(distributed),
        requested_device=distributed.get("requested_device") if distributed else None,
    ) as factory:
        import torch

        runtime_precision(config.precision, directory)

        if resume:
            checkpoint = Path(resume["checkpoint"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state.get("optimizer") is None:
                raise ValueError("YOLO checkpoint has no optimizer state")
            # The pinned upstream resume path restores these values from its checkpoint.
            # Rebase only the owned copy, so it cannot write into the source run.
            state["train_args"].update(
                epochs=config.epochs,
                data=str(data),
                project=str(upstream.parent),
                name="upstream",
                exist_ok=False,
                save_dir=str(upstream),
            )
            torch.save(state, checkpoint)
        model = load_model(factory, checkpoint, task, scale=scale)

        def save_recovery(trainer: Any) -> None:
            if (
                trainer.epoch + 1
            ) % config.checkpoint_interval and trainer.epoch + 1 != config.epochs:
                return
            target = directory / "recovery.pt"
            saved = torch.load(trainer.last, map_location="cpu", weights_only=False)
            saved["wvision_scaler"] = trainer.scaler.state_dict()
            temporary = target.with_suffix(".tmp")
            torch.save(saved, temporary)
            temporary.replace(target)
            record_checkpoint(directory, target, trainer.epoch + 1, selected=trainer.best)

        if task == "detect":
            model.add_callback("on_model_save", save_recovery)
        if resume:
            if resume.get("selected") and (not distributed or distributed["rank"] == 0):
                weights = upstream / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resume["selected"]["checkpoint"], weights / "best.pt")
            model.add_callback(
                "on_train_start",
                lambda trainer: trainer.scaler.load_state_dict(state["wvision_scaler"]),
            )
        model.add_callback(
            "on_train_start",
            lambda trainer: record_model_device(trainer.model, selected_device, directory),
        )
        from ultralytics.utils import ROOT

        name = "yolo11-pose.yaml" if task == "pose" else "yolo11.yaml"
        definition = ROOT / "cfg" / "models" / "11" / name
        shutil.copyfile(definition, directory / "upstream-model.yaml")
        write_json(
            directory / "model-source.json",
            {
                "backend": "ultralytics",
                "version": BACKEND_VERSION,
                "repository": "https://github.com/ultralytics/ultralytics",
                "release": "v8.3.4",
                "model": f"ultralytics/yolo11{scale}" + ("-pose" if task == "pose" else ""),
                "scale": scale,
                "bundled_definition": f"cfg/models/11/{name}",
                "definition_sha256": sha256(definition),
                "initialization": "optimizer-state resume"
                if resume
                else ("user-provided local checkpoint" if checkpoint else "random"),
                "resume_source": resume,
                "pretraining_data": "retained source-run provenance"
                if resume
                else ("unknown; supplied by caller" if checkpoint else None),
                "weights_downloaded": False,
            },
        )
        # Upstream's AMP probe downloads reference weights. CUDA and AMP are
        # explicitly requested here; verify numerical behavior in backend tests.
        with patch("ultralytics.engine.trainer.check_amp", return_value=True):
            result = model.train(
                data=str(data),
                epochs=config.epochs,
                imgsz=config.imgsz,
                batch=config.batch * len(distributed["config"]["devices"])
                if distributed
                else config.batch,
                seed=config.seed,
                device=list(range(len(distributed["config"]["devices"])))
                if distributed
                else config.device,
                workers=0,
                pretrained=False,
                amp=config.precision == "fp16",
                resume=bool(resume),
                plots=False,
                save=True,
                val=True,
                deterministic=True,
                project=str(upstream.parent),
                name="upstream",
                exist_ok=False,
                **config.backend_options,
            )
        record_model_device(model.trainer.model, selected_device, directory)
        if distributed and distributed["rank"] != 0:
            return {}
        weights = Path(model.trainer.save_dir) / "weights"
        best, last = weights / "best.pt", weights / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Upstream training did not produce best.pt and last.pt")
        if task == "detect":
            record_checkpoint(directory, directory / "recovery.pt", config.epochs, selected=best)
        return {"best": best, "last": last, "metrics": metrics(result)}


def evaluate(
    checkpoint: Path,
    data: Path,
    classes: dict[int, str],
    directory: Path,
    imgsz: int,
    batch: int,
    device: str = "cpu",
    *,
    task: str = "detect",
    scale: str = "n",
    skeleton: PoseSkeleton | None = None,
) -> dict[str, float]:
    with backend(device, directory) as factory:
        model = load_model(factory, checkpoint, task, scale=scale, skeleton=skeleton)
        if model.names != classes:
            raise ValueError("Checkpoint class map does not match the evaluation dataset")
        result = model.val(
            data=str(data),
            split="val",
            device=device,
            imgsz=imgsz,
            batch=batch,
            workers=0,
            plots=False,
            save_json=False,
            half=False,
            project=str(directory),
            name="upstream",
            exist_ok=False,
            max_det=300,
        )
        record_model_device(model.model, device, directory)
        if skeleton is not None:
            from ultralytics.utils.metrics import OKS_SIGMA

            count = len(skeleton["joints"])
            # Ultralytics 8.3.4 selects COCO sigmas for [17, 3] and 1/K otherwise.
            coco = count == len(JOINTS)
            write_json(
                directory / "pose-protocol.json",
                {
                    "version": BACKEND_VERSION,
                    "skeleton": skeleton["id"],
                    "keypoint_count": count,
                    "oks_sigmas": OKS_SIGMA.tolist() if coco else [1.0 / count] * count,
                    "oks_sigma_source": "coco17" if coco else "ultralytics_uniform",
                    "oks_thresholds": [value / 100 for value in range(50, 100, 5)],
                    "oks_area": "0.53 * ground-truth bounding box area in pixels",
                },
            )
        return metrics(result)


def pose_persons(result: Any, count: int) -> list[dict[str, Any]]:
    """Read upstream keypoints in original pixels, preserving their person association."""
    points = result.keypoints
    boxes = result.boxes
    if not len(boxes):
        return []
    if points is None or tuple(points.data.shape[1:]) != (count, 3):
        raise ValueError(f"Expected {count} pose joints with confidence for each person")
    return [
        {
            "class_id": int(category),
            "box": box,
            "confidence": confidence,
            "joints": [
                {
                    "x": x if score >= 0.5 else None,
                    "y": y if score >= 0.5 else None,
                    "confidence": score,
                }
                for x, y, score in joints
            ],
        }
        for box, category, confidence, joints in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            points.data.cpu().tolist(),
            strict=True,
        )
    ]


def predict(
    checkpoint: Path,
    source: list[str],
    directory: Path,
    imgsz: int,
    device: str = "cpu",
    *,
    task: str = "detect",
    scale: str = "n",
    skeleton: PoseSkeleton | None = None,
    **options: Any,
) -> int:
    import json

    count = 0
    with backend(device, directory) as factory:
        model = load_model(factory, checkpoint, task, scale=scale, skeleton=skeleton)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in model.predict(
                source=source,
                stream=True,
                device=device,
                imgsz=imgsz,
                save=False,
                verbose=False,
                **options,
            ):
                value = {
                    "sample": str(Path(result.path).resolve()),
                    "height": int(result.orig_shape[0]),
                    "width": int(result.orig_shape[1]),
                    "coordinate_format": "absolute_pixel_xyxy",
                    "classes": result.names,
                }
                if skeleton is not None:
                    keypoints = len(skeleton["joints"])
                    value.update(
                        skeleton=skeleton["id"],
                        keypoint_count=keypoints,
                        joint_coordinate_format="absolute_pixel_xy",
                        joint_confidence_threshold=0.5,
                        persons=pose_persons(result, keypoints),
                    )
                else:
                    boxes = result.boxes
                    value.update(
                        boxes=boxes.xyxy.cpu().tolist(),
                        scores=boxes.conf.cpu().tolist(),
                        class_ids=[int(v) for v in boxes.cls.cpu().tolist()],
                    )
                handle.write(json.dumps(value, allow_nan=False) + "\n")
                count += 1
        record_model_device(model.predictor.model, device, directory)
    return count


def export(checkpoint: Path, size: int, directory: Path, *, scale: str = "n") -> Path:
    """Use upstream ONNX export and check raw outputs against the FP32 model."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as error:
        raise RuntimeError(
            "ONNX verification requires the optional extra: pip install '.[onnx]'"
        ) from error

    local = directory / "model.pt"
    shutil.copyfile(checkpoint, local)
    with backend("cpu", directory) as factory:
        model = load_model(factory, local, scale=scale)
        target = Path(
            model.export(
                format="onnx",
                imgsz=size,
                batch=1,
                device="cpu",
                half=False,
                dynamic=False,
                simplify=False,
                opset=17,
            )
        ).resolve(strict=True)
        onnx.checker.check_model(onnx.load(str(target)))
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        error_max = 0.0
        model.model.eval()
        for array in (
            np.zeros((1, 3, size, size), dtype=np.float32),
            np.linspace(0, 1, 3 * size * size, dtype=np.float32).reshape(1, 3, size, size),
        ):
            with torch.inference_mode():
                reference = model.model(torch.from_numpy(array))[0].cpu().numpy()
            actual = session.run(None, {session.get_inputs()[0].name: array})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Export produced non-finite values")
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
            error_max = max(error_max, float(np.max(np.abs(actual - reference))))
        write_json(
            directory / "export.json",
            {
                "schema_version": 1,
                "format": "onnx",
                "model": target.name,
                "sha256": sha256(target),
                "source_checkpoint_sha256": sha256(checkpoint),
                "classes": model.names,
                "precision": "fp32",
                "opset": 17,
                "input": {
                    "name": session.get_inputs()[0].name,
                    "shape": [1, 3, size, size],
                    "dtype": "float32",
                    "layout": "NCHW RGB",
                    "range": [0, 1],
                    "preprocessing": "resize/letterbox to input size, then normalize",
                },
                "output": {
                    "shape": list(actual.shape),
                    "layout": "batch, xywh coordinates plus class scores, candidates",
                    "coordinates": "input-image pixels",
                    "nms_included": False,
                },
                "verification": {
                    "runtime": "onnxruntime CPUExecutionProvider",
                    "samples": 2,
                    "rtol": 1e-3,
                    "atol": 1e-4,
                    "max_absolute_error": error_max,
                },
            },
        )
    return target
