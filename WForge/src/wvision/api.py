"""Public application layer shared by Python and the CLI."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from wvision.adapters.base import DetectionContext, Initialization
from wvision.adapters.shared.devices import canonical_device
from wvision.core.contracts import EvaluationResult, ExportResult, PredictionResult, RunResult
from wvision.core.model_config import model_reference, training_values
from wvision.core.registry import vision_adapter
from wvision.datasets.detection import load_dataset, sha256
from wvision.datasets.pose import PoseSkeleton, read_person_boxes, restore_skeleton
from wvision.experiments.artifacts import artifact_reference, operation, write_json
from wvision.experiments.portable import verify_bundle
from wvision.training.recovery import resolve_resume, stage_resume


class Vision:
    """Run a registered vision model. Heavy dependencies load only during execution."""

    def __init__(
        self,
        model: str | Path = "ultralytics/yolo11n",
        *,
        backend_python: str | Path | None = None,
    ) -> None:
        model_id, self._definition = model_reference(model)
        self._adapter = vision_adapter(model_id, backend_python)
        self.model = model_id
        self._backend_python = self._adapter.python
        self._model_config: Path | None = None
        self._model_config_hash: str | None = None
        self._learning = "supervised"
        self._checkpoint: Path | None = None
        self._checkpoint_hash: str | None = None
        self._data: Path | None = None
        self._fingerprint: str | None = None
        self._skeleton: PoseSkeleton | None = None
        self._imgsz = 640
        self._batch = 16
        self._device = "cpu"
        if self._definition is not None:
            defaults = self._adapter.resolve(self._definition.training())
            self._adapter.spec.require(defaults.learning, "train", defaults.device)
            self._learning, self._device = defaults.learning, defaults.device
            self._imgsz, self._batch = defaults.imgsz, defaults.batch

    @classmethod
    def from_run(
        cls,
        run: str | Path,
        *,
        backend_python: str | Path | None = None,
        data: str | Path | None = None,
    ) -> "Vision":
        directory = Path(run).resolve(strict=True)
        verify_bundle(directory)
        state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if state.get("run_kind") == "self_supervised_pretrain":
            raise ValueError(
                "Pretraining is a representation run; pass it as pretraining_run to train"
            )
        if state.get("schema_version") != 1 or status["status"] != "completed":
            raise ValueError("Expected a completed WVision training run with schema_version=1")
        instance = cls(state["model"], backend_python=backend_python or state.get("backend_python"))
        instance._learning = state["config"]["learning"]
        if state.get("model_config"):
            instance._model_config = artifact_reference(directory, state["model_config"])
            if sha256(instance._model_config) != state["model_config_sha256"]:
                raise ValueError("Run model configuration content has changed")
            instance._model_config_hash = state["model_config_sha256"]
        instance._checkpoint = artifact_reference(directory, state["best_checkpoint"])
        if sha256(instance._checkpoint) != state["best_checkpoint_sha256"]:
            raise ValueError("Run checkpoint content has changed")
        instance._checkpoint_hash = state["best_checkpoint_sha256"]
        instance._data = (
            Path(data).resolve(strict=True)
            if data
            else (
                artifact_reference(directory, state["data"], strict=False)
                if state.get("data")
                else None
            )
        )
        instance._fingerprint = state["dataset_fingerprint"]
        if instance._adapter.spec.task == "pose":
            instance._skeleton = restore_skeleton(state.get("skeleton"))
        instance._imgsz = state["config"]["imgsz"]
        instance._batch = state["config"]["batch"]
        instance._device = state["config"].get("device", "cpu")
        return instance

    def _resolve_checkpoint(self, checkpoint: str | Path | None) -> Path:
        path = Path(checkpoint).resolve(strict=True) if checkpoint else self._checkpoint
        if not self._adapter.external_checkpoint and path != self._checkpoint:
            raise ValueError(
                "This adapter only supports the selected checkpoint of a completed run"
            )
        suffix = self._adapter.checkpoint_suffix
        if path is None or not path.is_file() or path.suffix != suffix:
            raise ValueError(f"Provide a trusted local {suffix} checkpoint or load a completed run")
        if path == self._checkpoint and sha256(path) != self._checkpoint_hash:
            raise ValueError("Run checkpoint content has changed")
        if self._model_config is not None and sha256(self._model_config) != self._model_config_hash:
            raise ValueError("Run model configuration content has changed")
        return path

    def _context(self, checkpoint: Path, device: str) -> DetectionContext:
        if self._adapter.needs_model_config and self._model_config is None:
            raise ValueError(
                "This adapter requires a completed training run for evaluation/prediction"
            )
        return DetectionContext(
            checkpoint,
            self._model_config,
            self._learning,
            self._imgsz,
            self._batch,
            device,
            skeleton=self._skeleton,
        )

    def train(
        self,
        data: str | Path,
        *,
        checkpoint: str | Path | None = None,
        pretraining_run: str | Path | None = None,
        backbone_weights: str | Path | None = None,
        detector_run: str | Path | None = None,
        resume_run: str | Path | None = None,
        transfer_scope: str = "detector",
        output: str | Path = "runs",
        **overrides: Any,
    ) -> RunResult:
        if self._definition is not None:
            self._definition.assert_unchanged()
        settings = self._adapter.resolve(training_values(self._definition, overrides))
        self._adapter.spec.require(settings.learning, "train", settings.device)
        inputs = Initialization(
            Path(checkpoint) if checkpoint is not None else None,
            Path(pretraining_run) if pretraining_run is not None else None,
            Path(backbone_weights) if backbone_weights is not None else None,
            Path(detector_run) if detector_run is not None else None,
            Path(resume_run) if resume_run is not None else None,
            transfer_scope,
        )
        self._adapter.validate_training(settings, inputs)
        required: tuple[str, ...] = ("train_labeled", "val")
        if settings.learning == "semi_supervised":
            required += ("train_unlabeled",)
        dataset = load_dataset(data, required, task=self._adapter.spec.task)
        initial = self._resolve_checkpoint(checkpoint) if checkpoint else None
        initialization = (
            {
                "resume": resolve_resume(
                    inputs.resume_run, self.model, dataset.fingerprint, settings.to_dict()
                )
            }
            if inputs.resume_run
            else self._adapter.initialization(
                Initialization(
                    initial,
                    inputs.pretraining_run,
                    inputs.backbone_weights,
                    inputs.detector_run,
                    transfer_scope=inputs.transfer_scope,
                ),
                dataset,
                settings,
            )
        )
        self._adapter.check_dependency()
        resolved = {"model": self.model, "data": str(dataset.manifest), **settings.to_dict()}
        resolved["initialization_sources"] = initialization
        selection = self._adapter.selection(settings.learning)
        resolved["selection"] = selection
        resolved["backend_python"] = str(self._backend_python) if self._backend_python else None
        with operation(output, "train", resolved) as directory:
            write_json(
                directory / "training.json",
                {
                    "schema_version": 1,
                    "model": self.model,
                    "data": str(dataset.manifest),
                    "dataset_fingerprint": dataset.fingerprint,
                    "config": settings.to_dict(),
                    "backend_python": resolved["backend_python"],
                },
            )
            if inputs.resume_run:
                initialization = {"resume": stage_resume(initialization["resume"], directory)}
            write_json(directory / "dataset.json", dataset.snapshot())
            if self._definition is not None:
                write_json(directory / "model-definition.json", self._definition.snapshot())
            result = self._adapter.train(dataset, settings, initialization, directory)
            if not all(p.resolve().is_relative_to(directory) for p in (result.best, result.last)):
                raise ValueError("Backend returned checkpoints outside its owned run directory")
            self._model_config = result.model_config
            write_json(directory / "metrics.json", result.metrics)
            state: dict[str, Any] = {
                "schema_version": 1,
                "model": self.model,
                **({"task": "pose", "skeleton": dataset.skeleton} if dataset.skeleton else {}),
                "data": str(dataset.manifest),
                "dataset_fingerprint": dataset.fingerprint,
                "config": settings.to_dict(),
                "best_checkpoint": str(result.best),
                "last_checkpoint": str(result.last),
                "best_checkpoint_sha256": sha256(result.best),
                "last_checkpoint_sha256": sha256(result.last),
                "selection": selection,
                "initialization": initialization,
                "model_definition": self._definition.snapshot() if self._definition else None,
                "backend_python": str(self._backend_python) if self._backend_python else None,
                "model_config": str(self._model_config) if self._model_config else None,
                "model_config_sha256": sha256(self._model_config) if self._model_config else None,
            }
            write_json(directory / "run.json", state)
        self._checkpoint = result.best
        self._checkpoint_hash = state["best_checkpoint_sha256"]
        self._model_config_hash = state["model_config_sha256"]
        self._data = dataset.manifest
        self._fingerprint = dataset.fingerprint
        self._skeleton = dataset.skeleton
        self._imgsz, self._batch = settings.imgsz, settings.batch
        self._learning = settings.learning
        self._device = settings.device
        return RunResult(
            directory.name,
            directory,
            "completed",
            result.best,
            result.last,
            result.metrics,
        )

    def val(
        self,
        data: str | Path | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        protocol: str = "native",
        output: str | Path = "runs",
    ) -> EvaluationResult:
        return self._evaluate("val", data, checkpoint, output, device, protocol)

    def test(
        self,
        data: str | Path | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        protocol: str = "native",
        output: str | Path = "runs",
    ) -> EvaluationResult:
        return self._evaluate("test", data, checkpoint, output, device, protocol)

    def _evaluate(
        self,
        split: str,
        data: str | Path | None,
        checkpoint: str | Path | None,
        output: str | Path,
        device: str | None,
        protocol: str,
    ) -> EvaluationResult:
        if not isinstance(protocol, str) or protocol not in {"native", "coco"}:
            raise ValueError("protocol must be native or coco")
        if protocol == "coco" and self._adapter.spec.task != "detect":
            raise ValueError("The shared COCO bbox protocol only supports detection")
        selected_device = canonical_device(self._device if device is None else device)
        self._adapter.spec.require(self._learning, split, selected_device)
        selected = self._resolve_checkpoint(checkpoint)
        source = Path(data) if data is not None else self._data
        if source is None:
            raise ValueError("Provide a dataset manifest or load a completed run")
        dataset = load_dataset(source, (split,), task=self._adapter.spec.task)
        if self._fingerprint is not None and (
            dataset.fingerprint != self._fingerprint or dataset.skeleton != self._skeleton
        ):
            raise ValueError(
                "Dataset changed since training; use a new Vision instance for a new dataset"
            )
        if protocol == "coco":
            from wvision.evaluation.detection import evaluate_predictions, require_evaluator

            require_evaluator()
            predicted = self.predict(
                [sample.path for sample in dataset.splits[split]],
                checkpoint=selected,
                device=selected_device,
                score_threshold=0.001,
                max_detections=100,
                output=output,
            )
            return evaluate_predictions(
                dataset,
                predicted.predictions,
                checkpoint=selected,
                output=Path(output),
                split=split,
            )
        self._adapter.check_dependency()
        context = self._context(selected, selected_device)
        with operation(
            output,
            split,
            {
                "model": self.model,
                "checkpoint": str(selected),
                "checkpoint_sha256": sha256(selected),
                "data": str(dataset.manifest),
                "split": split,
                "device": selected_device,
                "imgsz": self._imgsz,
                "batch": self._batch,
                "max_det": self._adapter.max_detections,
                "evaluator": self._adapter.evaluator,
            },
        ) as directory:
            write_json(directory / "dataset.json", dataset.snapshot())
            values = self._adapter.evaluate(dataset, split, context, directory)
            write_json(directory / "metrics.json", values)
        return EvaluationResult(directory, split, selected, dataset.fingerprint, values)

    def predict(
        self,
        source: str | Path | list[str | Path],
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        score_threshold: float | None = None,
        max_detections: int | None = None,
        person_boxes: str | Path | None = None,
        output: str | Path = "runs",
    ) -> PredictionResult:
        selected_device = canonical_device(self._device if device is None else device)
        self._adapter.spec.require(self._learning, "predict", selected_device)
        if self._adapter.spec.requires_person_boxes:
            if person_boxes is None:
                raise ValueError("This top-down pose model requires person_boxes (pixel xyxy)")
            if score_threshold is not None or max_detections is not None:
                raise ValueError(
                    "Filter person_boxes before passing them to top-down pose prediction"
                )
        elif person_boxes is not None:
            raise ValueError("person_boxes only applies to top-down pose models")
        selected = self._resolve_checkpoint(checkpoint)
        if score_threshold is not None and (
            type(score_threshold) not in (int, float)
            or not math.isfinite(score_threshold)
            or not 0 <= score_threshold <= 1
        ):
            raise ValueError("score_threshold must be a finite number between 0 and 1")
        if max_detections is not None and (
            type(max_detections) is not int or not 1 <= max_detections <= 100
        ):
            raise ValueError("max_detections must be an integer between 1 and 100")
        if isinstance(source, list):
            samples = [Path(item).resolve(strict=True) for item in source]
            description: str | list[str] = [str(path) for path in samples]
        else:
            location = Path(source).resolve(strict=True)
            samples = sorted(location.iterdir()) if location.is_dir() else [location]
            description = str(location)
        samples = [p for p in samples if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not samples:
            raise ValueError("Prediction source contains no supported local images")
        sizes = {}
        for path in samples:
            with Image.open(path) as image:
                sizes[path] = image.size
                image.verify()
        boxes = read_person_boxes(person_boxes, sizes) if person_boxes is not None else None
        self._adapter.check_dependency()
        context = self._context(selected, selected_device)
        context = replace(
            context,
            score_threshold=score_threshold,
            max_detections=max_detections,
            person_boxes=boxes,
        )
        with operation(
            output,
            "predict",
            {
                "model": self.model,
                "checkpoint": str(selected),
                "checkpoint_sha256": sha256(selected),
                "source": description,
                **(
                    {"person_boxes": {str(path): rows for path, rows in boxes.items()}}
                    if boxes is not None
                    else {}
                ),
                **({"skeleton": self._skeleton} if self._skeleton else {}),
                "score_threshold": score_threshold,
                "max_detections": max_detections,
                "device": selected_device,
                "imgsz": self._imgsz,
            },
        ) as directory:
            write_json(
                directory / "inputs.json",
                [{"path": str(path), "sha256": sha256(path)} for path in samples],
            )
            count = self._adapter.predict(samples, context, directory)
        return PredictionResult(directory, directory / "predictions.jsonl", count)

    def export(
        self,
        *,
        format: str = "onnx",
        output: str | Path = "runs",
        verification_images: list[str | Path] | None = None,
    ) -> ExportResult:
        """Export the selected checkpoint through a verified model-owned exporter."""
        self._adapter.spec.require_export(format)
        checkpoint = self._resolve_checkpoint(None)
        if verification_images is not None and not isinstance(verification_images, list):
            raise ValueError("verification_images must be a list of local image paths")
        images = tuple(Path(path).resolve(strict=True) for path in (verification_images or []))
        for path in images:
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError("Export verification images must be local PNG/JPEG files")
            with Image.open(path) as image:
                image.verify()
        with operation(
            output,
            "export",
            {
                "model": self.model,
                "format": format,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "imgsz": self._imgsz,
                "verification_images": [str(path) for path in images],
            },
        ) as directory:
            context = replace(self._context(checkpoint, "cpu"), verification_images=images)
            model = self._adapter.export(context, directory, format)
        return ExportResult(directory, model, directory / "export.json")
