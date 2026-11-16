"""YOLO11n train/evaluate/predict calls; all learning remains upstream."""

import shutil
from pathlib import Path
from typing import Any

from wvision.adapters.shared.devices import record_model_device
from wvision.adapters.ultralytics.backend import BACKEND_VERSION, backend, metrics
from wvision.adapters.ultralytics.backend import check_dependency as check_dependency
from wvision.datasets.detection import sha256
from wvision.experiments.artifacts import write_json
from wvision.models.ultralytics.yolo11.config import TrainConfig

MODEL_ID = "ultralytics/yolo11n"
SELECTION = {"metric": "0.1 * AP50 + 0.9 * AP50_95", "direction": "maximize"}


def load_model(factory: Any, checkpoint: Path | None) -> Any:
    model = factory(str(checkpoint) if checkpoint else "yolo11n.yaml", task="detect")
    definition = model.model.yaml
    if (
        model.task != "detect"
        or "yolo11" not in Path(definition.get("yaml_file", "")).stem
        or definition.get("scale") != "n"
    ):
        raise ValueError("Checkpoint must belong to the registered YOLO11n detection architecture")
    return model


def train(
    checkpoint: Path | None, data: Path, config: TrainConfig, directory: Path
) -> dict[str, Any]:
    with backend(config.device, directory) as factory:
        model = load_model(factory, checkpoint)
        model.add_callback(
            "on_train_start",
            lambda trainer: record_model_device(trainer.model, config.device, directory),
        )
        from ultralytics.utils import ROOT

        definition = ROOT / "cfg" / "models" / "11" / "yolo11.yaml"
        shutil.copyfile(definition, directory / "upstream-model.yaml")
        write_json(
            directory / "model-source.json",
            {
                "backend": "ultralytics",
                "version": BACKEND_VERSION,
                "repository": "https://github.com/ultralytics/ultralytics",
                "release": "v8.3.4",
                "model": MODEL_ID,
                "scale": "n",
                "bundled_definition": "cfg/models/11/yolo11.yaml",
                "definition_sha256": sha256(definition),
                "initialization": "user-provided local checkpoint" if checkpoint else "random",
                "pretraining_data": "unknown; supplied by caller" if checkpoint else None,
                "weights_downloaded": False,
            },
        )
        result = model.train(
            data=str(data),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            seed=config.seed,
            device=config.device,
            workers=0,
            pretrained=False,
            amp=False,
            plots=False,
            save=True,
            val=True,
            deterministic=True,
            project=str(directory),
            name="upstream",
            exist_ok=False,
            **config.backend_options,
        )
        record_model_device(model.trainer.model, config.device, directory)
        weights = Path(model.trainer.save_dir) / "weights"
        best, last = weights / "best.pt", weights / "last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("Upstream training did not produce best.pt and last.pt")
        return {"best": best, "last": last, "metrics": metrics(result)}


def evaluate(
    checkpoint: Path,
    data: Path,
    classes: dict[int, str],
    directory: Path,
    imgsz: int,
    batch: int,
    device: str = "cpu",
) -> dict[str, float]:
    with backend(device, directory) as factory:
        model = load_model(factory, checkpoint)
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
        return metrics(result)


def predict(
    checkpoint: Path,
    source: list[str],
    directory: Path,
    imgsz: int,
    device: str = "cpu",
) -> int:
    import json

    count = 0
    with backend(device, directory) as factory:
        model = load_model(factory, checkpoint)
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for result in model.predict(
                source=source,
                stream=True,
                device=device,
                imgsz=imgsz,
                save=False,
                verbose=False,
            ):
                boxes = result.boxes
                value = {
                    "sample": str(Path(result.path).resolve()),
                    "height": int(result.orig_shape[0]),
                    "width": int(result.orig_shape[1]),
                    "coordinate_format": "absolute_pixel_xyxy",
                    "classes": result.names,
                    "boxes": boxes.xyxy.cpu().tolist(),
                    "scores": boxes.conf.cpu().tolist(),
                    "class_ids": [int(v) for v in boxes.cls.cpu().tolist()],
                }
                handle.write(json.dumps(value, allow_nan=False) + "\n")
                count += 1
        record_model_device(model.predictor.model, device, directory)
    return count
