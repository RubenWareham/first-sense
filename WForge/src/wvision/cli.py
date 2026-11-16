"""Thin CLI over the same Vision application layer used from Python."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wvision.api import Vision
from wvision.core.contracts import (
    ComparisonResult,
    EvaluationResult,
    PredictionResult,
    PretrainResult,
    RunResult,
    TransferComparisonResult,
)
from wvision.core.registry import list_models
from wvision.datasets.budgets import prepare_label_budget
from wvision.evaluation.learning import compare_learning
from wvision.evaluation.pretraining import compare_pretraining
from wvision.training.pretraining import pretrain


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wvision", description="Existing-model vision workflows")
    root.add_argument("--version", action="version", version="wvision 0.1.0.dev0")
    commands = root.add_subparsers(dest="operation", required=True)
    commands.add_parser("models", help="List exact models and supported capability combinations")
    train = commands.add_parser("train", help="Train a registered detection model")
    train.add_argument(
        "--model", default="ultralytics/yolo11n", help="Registered ID or model YAML path"
    )
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--pretraining-run", type=Path)
    train.add_argument("--backbone-weights", type=Path)
    train.add_argument(
        "--detector-run", type=Path, help="Initialize Soft Teacher from a supervised run"
    )
    train.add_argument("--learning")
    for name in ("epochs", "max-iters", "imgsz", "batch", "seed"):
        train.add_argument(f"--{name}", type=int)
    train.add_argument("--device")
    train.add_argument("--backend-python", type=Path)
    train.add_argument("--output", default=Path("runs"), type=Path)
    for name in ("val", "test", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True, type=Path)
        command.add_argument("--backend-python", type=Path)
        command.add_argument("--device", help="Override the run device, e.g. cpu or cuda:0")
        command.add_argument("--output", default=Path("runs"), type=Path)
        if name == "predict":
            command.add_argument("--source", required=True, type=Path)
    budget = commands.add_parser(
        "prepare-budget", help="Create training-only labeled/unlabeled subsets"
    )
    budget.add_argument("--data", required=True, type=Path)
    budget.add_argument("--destination", required=True, type=Path)
    budget.add_argument("--labeled-count", required=True, type=int)
    budget.add_argument("--seed", default=0, type=int)
    compare = commands.add_parser(
        "compare", help="Compare supervised Faster R-CNN and Soft Teacher"
    )
    compare.add_argument("--data", required=True, type=Path)
    compare.add_argument("--backend-python", type=Path)
    compare.add_argument("--device")
    compare.add_argument(
        "--model", default="mmdetection/faster-rcnn-r50", help="Registered ID or model YAML path"
    )
    compare.add_argument("--output", default=Path("runs"), type=Path)
    for name in ("max-iters", "imgsz", "batch", "seed"):
        compare.add_argument(f"--{name}", type=int)
    ssl = commands.add_parser("pretrain", help="Self-supervised SimSiam on train_unlabeled only")
    ssl.add_argument(
        "--model", default="mmpretrain/simsiam-r50", help="Registered ID or model YAML path"
    )
    ssl.add_argument("--data", required=True, type=Path)
    ssl.add_argument("--backend-python", type=Path)
    ssl.add_argument("--output", default=Path("runs"), type=Path)
    for name in ("max-iters", "imgsz", "batch", "seed"):
        ssl.add_argument(f"--{name}", type=int)
    ssl.add_argument("--lr", type=float)
    ssl.add_argument("--device")
    transfer = commands.add_parser(
        "compare-pretraining", help="Compare random and SimSiam-initialized Faster R-CNN"
    )
    transfer.add_argument("--data", required=True, type=Path)
    transfer.add_argument("--pretraining-run", required=True, type=Path)
    transfer.add_argument("--backend-python", type=Path)
    transfer.add_argument("--device")
    transfer.add_argument(
        "--model", default="mmdetection/faster-rcnn-r50", help="Registered ID or model YAML path"
    )
    transfer.add_argument("--output", default=Path("runs"), type=Path)
    for name in ("max-iters", "imgsz", "batch", "seed"):
        transfer.add_argument(f"--{name}", type=int)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result: (
        RunResult
        | EvaluationResult
        | PredictionResult
        | ComparisonResult
        | PretrainResult
        | TransferComparisonResult
        | dict[str, str]
        | list[dict[str, Any]]
    )
    try:
        # Keep stdout machine-readable, even when upstream emits progress messages.
        with redirect_stdout(sys.stderr):
            if args.operation == "models":
                result = [asdict(spec) for spec in list_models()]
            elif args.operation == "compare-pretraining":
                overrides = {
                    key: getattr(args, key)
                    for key in ("max_iters", "imgsz", "batch", "seed", "device")
                    if getattr(args, key) is not None
                }
                result = compare_pretraining(
                    args.data,
                    model=args.model,
                    pretraining_run=args.pretraining_run,
                    backend_python=args.backend_python,
                    output=args.output,
                    **overrides,
                )
            elif args.operation == "pretrain":
                overrides = {
                    key: getattr(args, key)
                    for key in ("max_iters", "imgsz", "batch", "seed", "lr", "device")
                    if getattr(args, key) is not None
                }
                result = pretrain(
                    args.data,
                    model=args.model,
                    backend_python=args.backend_python,
                    output=args.output,
                    **overrides,
                )
            elif args.operation == "prepare-budget":
                manifest = prepare_label_budget(
                    args.data,
                    args.destination,
                    labeled_count=args.labeled_count,
                    seed=args.seed,
                )
                result = {"manifest": str(manifest)}
            elif args.operation == "compare":
                overrides = {
                    key: getattr(args, key)
                    for key in ("max_iters", "imgsz", "batch", "seed", "device")
                    if getattr(args, key) is not None
                }
                result = compare_learning(
                    args.data,
                    model=args.model,
                    backend_python=args.backend_python,
                    output=args.output,
                    **overrides,
                )
            elif args.operation == "train":
                overrides = {
                    key: getattr(args, key)
                    for key in (
                        "learning",
                        "epochs",
                        "max_iters",
                        "imgsz",
                        "batch",
                        "seed",
                        "device",
                    )
                    if getattr(args, key) is not None
                }
                result = Vision(args.model, backend_python=args.backend_python).train(
                    args.data,
                    checkpoint=args.checkpoint,
                    pretraining_run=args.pretraining_run,
                    backbone_weights=args.backbone_weights,
                    detector_run=args.detector_run,
                    output=args.output,
                    **overrides,
                )
            else:
                model = Vision.from_run(args.run, backend_python=args.backend_python)
                if args.operation == "predict":
                    result = model.predict(args.source, output=args.output, device=args.device)
                elif args.operation == "val":
                    result = model.val(output=args.output, device=args.device)
                else:
                    result = model.test(output=args.output, device=args.device)
        payload = result if isinstance(result, (dict, list)) else asdict(result)
        print(json.dumps(payload, default=str, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"wvision: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
