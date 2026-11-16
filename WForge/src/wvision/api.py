"""Public application layer shared by Python and the CLI."""

import json
from pathlib import Path
from typing import Any

from PIL import Image

from wvision.adapters.base import DetectionContext, Initialization
from wvision.adapters.shared.devices import canonical_device
from wvision.core.contracts import EvaluationResult, PredictionResult, RunResult
from wvision.core.model_config import model_reference, training_values
from wvision.core.registry import detection_adapter
from wvision.datasets.detection import load_dataset, sha256
from wvision.experiments.artifacts import operation, write_json


class Vision:
    """Run the registered detection model. Heavy dependencies load only during execution."""

    def __init__(
        self,
        model: str | Path = "ultralytics/yolo11n",
        *,
        backend_python: str | Path | None = None,
    ) -> None:
        model_id, self._definition = model_reference(model)
        self._adapter = detection_adapter(model_id, backend_python)
        self.model = model_id
        self._backend_python = self._adapter.python
        self._model_config: Path | None = None
        self._model_config_hash: str | None = None
        self._learning = "supervised"
        self._checkpoint: Path | None = None
        self._checkpoint_hash: str | None = None
        self._data: Path | None = None
        self._fingerprint: str | None = None
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
    ) -> "Vision":
        directory = Path(run).resolve(strict=True)
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
            instance._model_config = Path(state["model_config"]).resolve(strict=True)
            if sha256(instance._model_config) != state["model_config_sha256"]:
                raise ValueError("Run model configuration content has changed")
            instance._model_config_hash = state["model_config_sha256"]
        instance._checkpoint = Path(state["best_checkpoint"]).resolve(strict=True)
        if sha256(instance._checkpoint) != state["best_checkpoint_sha256"]:
            raise ValueError("Run checkpoint content has changed")
        instance._checkpoint_hash = state["best_checkpoint_sha256"]
        instance._data = Path(state["data"])
        instance._fingerprint = state["dataset_fingerprint"]
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
            checkpoint, self._model_config, self._learning, self._imgsz, self._batch, device
        )

    def train(
        self,
        data: str | Path,
        *,
        checkpoint: str | Path | None = None,
        pretraining_run: str | Path | None = None,
        backbone_weights: str | Path | None = None,
        detector_run: str | Path | None = None,
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
        )
        self._adapter.validate_training(settings, inputs)
        required: tuple[str, ...] = ("train_labeled", "val")
        if settings.learning == "semi_supervised":
            required += ("train_unlabeled",)
        dataset = load_dataset(data, required)
        initial = self._resolve_checkpoint(checkpoint) if checkpoint else None
        initialization = self._adapter.initialization(
            Initialization(
                initial, inputs.pretraining_run, inputs.backbone_weights, inputs.detector_run
            ),
            dataset,
            settings,
        )
        self._adapter.check_dependency()
        resolved = {"model": self.model, "data": str(dataset.manifest), **settings.to_dict()}
        resolved["initialization_sources"] = initialization
        selection = self._adapter.selection(settings.learning)
        resolved["selection"] = selection
        resolved["backend_python"] = str(self._backend_python) if self._backend_python else None
        with operation(output, "train", resolved) as directory:
            write_json(directory / "dataset.json", dataset.snapshot())
            if self._definition is not None:
                write_json(directory / "model-definition.json", self._definition.snapshot())
            result = self._adapter.train(dataset, settings, initialization, directory)
            self._model_config = result.model_config
            write_json(directory / "metrics.json", result.metrics)
            state: dict[str, Any] = {
                "schema_version": 1,
                "model": self.model,
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
        output: str | Path = "runs",
    ) -> EvaluationResult:
        return self._evaluate("val", data, checkpoint, output, device)

    def test(
        self,
        data: str | Path | None = None,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        output: str | Path = "runs",
    ) -> EvaluationResult:
        return self._evaluate("test", data, checkpoint, output, device)

    def _evaluate(
        self,
        split: str,
        data: str | Path | None,
        checkpoint: str | Path | None,
        output: str | Path,
        device: str | None,
    ) -> EvaluationResult:
        selected_device = canonical_device(self._device if device is None else device)
        self._adapter.spec.require(self._learning, split, selected_device)
        selected = self._resolve_checkpoint(checkpoint)
        source = Path(data) if data is not None else self._data
        if source is None:
            raise ValueError("Provide a dataset manifest or load a completed run")
        dataset = load_dataset(source, (split,))
        if self._fingerprint is not None and dataset.fingerprint != self._fingerprint:
            raise ValueError(
                "Dataset changed since training; use a new Vision instance for a new dataset"
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
        source: str | Path,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        output: str | Path = "runs",
    ) -> PredictionResult:
        selected_device = canonical_device(self._device if device is None else device)
        self._adapter.spec.require(self._learning, "predict", selected_device)
        selected = self._resolve_checkpoint(checkpoint)
        location = Path(source).resolve(strict=True)
        samples = sorted(location.iterdir()) if location.is_dir() else [location]
        samples = [p for p in samples if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not samples:
            raise ValueError("Prediction source contains no supported local images")
        for path in samples:
            with Image.open(path) as image:
                image.verify()
        self._adapter.check_dependency()
        context = self._context(selected, selected_device)
        with operation(
            output,
            "predict",
            {
                "model": self.model,
                "checkpoint": str(selected),
                "checkpoint_sha256": sha256(selected),
                "source": str(location),
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
