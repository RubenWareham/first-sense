"""An experiment's explicitly selected application interpreter; no shell execution."""

import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from wvision.api import Vision
from wvision.datasets.detection import load_dataset
from wvision.experiments.artifacts import artifact_reference, write_json


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with redirect_stdout(sys.stderr):
        started = time.monotonic()
        if job.get("run"):
            model = Vision.from_run(
                job["run"], backend_python=job.get("backend_python"), data=job["data"]
            )
            run = Path(job["run"])
            seconds = None
        else:
            model = Vision(job["model"], backend_python=job.get("backend_python"))
            trained = model.train(
                job["data"], output=job["output"], **job["initialization"], **job["training"]
            )
            run = trained.artifact_dir
            seconds = time.monotonic() - started
        validation = model.val(output=job["output"])
        dataset = load_dataset(job["data"], ("val",))
        prediction = model.predict(
            [sample.path for sample in dataset.splits["val"]],
            score_threshold=0.001,
            max_detections=100,
            output=job["output"],
        )
        state = json.loads((run / "run.json").read_text(encoding="utf-8"))
        result = {
            "name": job["name"],
            "model": model.model,
            "run": str(run),
            "dataset_fingerprint": validation.dataset_fingerprint,
            "training": state["config"],
            "global_batch": state["config"]["batch"] * len(state["config"].get("devices") or [0]),
            "train_seconds": seconds,
            "evaluator": model._adapter.evaluator,
            "max_detections": model._adapter.max_detections,
            "validation": {
                **asdict(validation),
                "artifact_dir": str(validation.artifact_dir),
                "checkpoint": str(validation.checkpoint),
            },
            "checkpoint": str(artifact_reference(run, state["best_checkpoint"])),
            "checkpoint_sha256": state["best_checkpoint_sha256"],
            "comparison_predictions": str(prediction.predictions),
            "selection": state["selection"],
            "initialization": state.get("initialization"),
            "device": json.loads((run / "device.json").read_text(encoding="utf-8"))
            if (run / "device.json").exists()
            else None,
        }
    write_json(Path(job["result"]), result)


if __name__ == "__main__":
    main()
