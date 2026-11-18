"""Managed upstream YOLO DDP entry point using the application interpreter."""

import json
import sys
from pathlib import Path

# The transport clears caller PYTHONPATH; add only this package's src directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from wvision.adapters.shared.checkpoints import worker_lock
from wvision.adapters.shared.devices import worker_context
from wvision.experiments.artifacts import write_json
from wvision.models.ultralytics.yolo11.config import TrainConfig
from wvision.models.ultralytics.yolo11.operations import train


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request.get("schema_version") != 1 or request.get("operation") != "train":
        raise ValueError("Unsupported YOLO distributed request")
    directory, _ = worker_context(request)
    result = train(
        Path(request["checkpoint"]) if request.get("checkpoint") else None,
        Path(request["data"]),
        TrainConfig.resolve(request["config"]),
        directory,
        resume=request.get("resume"),
        distributed=request,
    )
    if request["rank"] == 0:
        write_json(Path(sys.argv[2]), {"schema_version": 1, **result})


if __name__ == "__main__":
    with worker_lock(Path(sys.argv[1])):
        main()
