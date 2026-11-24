"""Sequential multi-model/parameter-grid experiments with held-out test isolation."""

import csv
import html
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from wvision.adapters.shared.process import stop_process_tree
from wvision.api import Vision
from wvision.core.config import load_yaml, reject_unknown
from wvision.core.contracts import ExperimentResult
from wvision.core.model_config import model_reference, training_values
from wvision.core.registry import get_model
from wvision.datasets.detection import load_dataset, sha256
from wvision.evaluation.detection import COCO_PROTOCOL, evaluate_predictions, require_evaluator
from wvision.experiments.artifacts import operation, write_json


def _reference(value: Any, base: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Experiment paths must be nonempty strings")
    return str((base / value).resolve(strict=True))


def _jobs(plan: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    reject_unknown(
        plan, {"schema_version", "data", "experiments", "require_groups"}, "experiment plan"
    )
    if type(plan.get("require_groups", False)) is not bool:
        raise ValueError("require_groups must be a boolean")
    if type(plan.get("schema_version")) is not int or plan["schema_version"] != 1:
        raise ValueError("Experiment schema_version must be 1")
    entries = plan.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("experiments must be a nonempty list")
    result = []
    names = set()
    generated = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each experiment must be a mapping")
        reject_unknown(
            entry,
            {
                "name",
                "model",
                "run",
                "python",
                "backend_python",
                "training",
                "grid",
                "initialization",
            },
            "experiment",
        )
        name = entry.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Experiment names must be unique letters/numbers/underscores/hyphens")
        names.add(name)
        if bool(entry.get("model")) == bool(entry.get("run")):
            raise ValueError("Each experiment must select model or an existing run")
        training = entry.get("training", {})
        grid = entry.get("grid", {})
        initial = entry.get("initialization", {})
        if not all(isinstance(value, dict) for value in (training, grid, initial)):
            raise ValueError("training, grid, and initialization must be mappings")
        reject_unknown(
            initial,
            {"checkpoint", "pretraining_run", "backbone_weights", "detector_run", "transfer_scope"},
            "initialization",
        )
        if entry.get("run") and (training or grid or initial):
            raise ValueError("An existing run cannot also specify training/grid/initialization")
        if any(not isinstance(k, str) or not isinstance(v, list) or not v for k, v in grid.items()):
            raise ValueError("Grid fields require nonempty lists of values")
        model = entry.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("Experiment model must be an identifier or YAML path")
        if model and Path(model).suffix.lower() in {".yaml", ".yml"}:
            model = _reference(model, base)
        run = _reference(entry["run"], base) if entry.get("run") else None
        if model:
            model_id = model_reference(model)[0]
        else:
            assert run is not None
            model_id = json.loads((Path(run) / "run.json").read_text(encoding="utf-8"))["model"]
        if get_model(model_id).task != "detect":
            raise ValueError("The shared COCO bbox experiments workflow only supports detection")
        backend = _reference(entry["backend_python"], base) if entry.get("backend_python") else None
        detector = Vision(model, backend_python=backend) if model else None
        defaults = training_values(detector._definition, training) if detector else training
        for index, values in enumerate(itertools.product(*grid.values())):
            settings = deepcopy(defaults)
            for key, value in zip(grid, values, strict=True):
                if key.startswith("backend_options.") and key.count(".") == 1:
                    settings.setdefault("backend_options", {})[key.split(".")[1]] = value
                elif "." not in key:
                    settings[key] = value
                else:
                    raise ValueError(f"Unsupported grid field: {key}")
            if detector:
                settings = detector._adapter.resolve(settings).to_dict()
            result.append(
                {
                    "name": f"{name}-{index + 1:03d}" if grid else name,
                    "model": model,
                    "run": run,
                    "python": _reference(entry["python"], base)
                    if entry.get("python")
                    else sys.executable,
                    "backend_python": backend,
                    "training": settings,
                    "initialization": {
                        key: (value if key == "transfer_scope" else _reference(value, base))
                        for key, value in initial.items()
                    },
                }
            )
            if result[-1]["name"] in generated:
                raise ValueError("Expanded experiment names must be unique")
            generated.add(result[-1]["name"])
    return result


def _report(directory: Path, results: list[dict[str, Any]], fingerprint: str) -> None:
    snapshot = directory / "dataset.json"
    grouping = (
        bool(json.loads(snapshot.read_text(encoding="utf-8")).get("groups"))
        if snapshot.exists()
        else False
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        identity = json.dumps(
            {
                "model": result["model"],
                "training": {k: v for k, v in result["training"].items() if k != "seed"},
                "initialization": result.get("initialization"),
            },
            sort_keys=True,
        )
        groups.setdefault(identity, []).append(result)
    aggregate = []
    for members in groups.values():
        keys = {
            key
            for member in members
            for key in member["validation"]["metrics"]
            if key.startswith("wforge/coco/")
        }
        summaries = {}
        for key in sorted(keys):
            values = [
                m["validation"]["metrics"][key]
                for m in members
                if m["validation"]["metrics"].get(key, -1) >= 0
            ]
            summaries[key] = {
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else None,
            }
        aggregate.append(
            {
                "runs": [m["name"] for m in members],
                "distinct_seeds": len({m["training"]["seed"] for m in members}),
                "metrics": summaries,
            }
        )
    write_json(
        directory / "comparison.json",
        {
            "schema_version": 1,
            "dataset_fingerprint": fingerprint,
            "runs": results,
            "aggregates": aggregate,
            "selection_split": "val",
            "test_evaluated": False,
            "group_isolation": "verified" if grouping else "unverified: groups not supplied",
            "limitations": [
                "Shared COCO metrics permit comparison; native metrics retain namespaces; "
                "no cross-protocol ranking",
                "Training-call wall time includes startup, I/O, validation and checkpointing",
                "Different settings/budgets are recorded, not assumed to have equal compute",
            ],
        },
    )
    metrics = sorted({key for result in results for key in result["validation"]["metrics"]})
    fields = [
        "name",
        "model",
        "evaluator",
        "max_detections",
        "seed",
        "epochs",
        "max_iters",
        "imgsz",
        "batch",
        "global_batch",
        "precision",
        "train_seconds",
        *metrics,
    ]
    rows = [
        {
            **{k: result.get(k, "") for k in fields},
            **result["training"],
            **result["validation"]["metrics"],
        }
        for result in results
    ]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    visible = [
        key
        for key in fields
        if key not in {"evaluator", "max_detections"}
        and (key not in metrics or key.startswith("wforge/coco/"))
    ]
    table = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in visible)
        + "</tr>"
        for row in rows
    )
    detail = "".join(
        f"<details><summary>{html.escape(r['name'])}: settings and provenance</summary>"
        f"<pre>{html.escape(json.dumps(r, indent=2))}</pre></details>"
        for r in results
    )
    heading = "".join(
        '<th onclick="sortRows(this.cellIndex)">'
        f"{html.escape(key.replace('wforge/coco/', ''))}</th>"
        for key in visible
    )
    bars = []
    for index, result in enumerate(results):
        value = result["validation"]["metrics"].get("wforge/coco/bbox_mAP")
        if value is None or value < 0:
            continue
        y = 22 + index * 30
        bars.append(
            f'<text x="0" y="{y + 14}">{html.escape(result["name"])}</text>'
            f'<rect x="200" y="{y}" width="{value * 400:.3f}" height="20" fill="#227c82"/>'
            f'<text x="{205 + value * 400:.3f}" y="{y + 14}">{value * 100:.4f}%</text>'
        )
    chart = (
        "<h2>Common COCO validation AP50:95</h2><p>Full scale: 0–100% AP.</p>"
        '<svg role="img" aria-label="Validation AP by experiment" '
        f'viewBox="0 0 720 {50 + 30 * len(results)}" '
        'style="max-width:900px;font:13px system-ui">' + "".join(bars) + "</svg>"
    )
    script = """<script>
function sortRows(column){
  const body=document.querySelector('tbody'), rows=Array.from(body.rows);
  const header=document.querySelectorAll('th')[column];
  const direction=header.dataset.direction==='-1'?1:-1;
  header.dataset.direction=String(direction);
  rows.sort((a,b)=>{
    const x=a.cells[column].textContent,y=b.cells[column].textContent;
    return direction*((x.trim()!==''&&y.trim()!==''&&Number.isFinite(Number(x))
      &&Number.isFinite(Number(y)))?Number(x)-Number(y):x.localeCompare(y));
  });
  rows.forEach(row=>body.appendChild(row));
}
</script>"""
    (directory / "comparison.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>WForge experiments</title>'
        "<style>body{font:15px system-ui;margin:32px;color:#17232c}"
        "table{border-collapse:collapse;font-size:13px}"
        "td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}"
        "th{background:#edf3f5;cursor:pointer}pre{white-space:pre-wrap}"
        "details{margin:18px 0}.scroll{overflow:auto}</style>"
        "<h1>WForge experiment comparison</h1><p>Validation results. Test data was not evaluated. "
        "Compare scores within the same evaluator and metric protocol; "
        "equal training steps do not imply equal compute.</p>"
        f"<p>Dataset: <code>{fingerprint}</code></p>"
        f"<p>Group isolation: {'verified' if grouping else 'unverified — groups not supplied'}.</p>"
        f"{chart}<p>Click a column to sort.</p>"
        '<div class="scroll"><table>'
        f"<thead><tr>{heading}</tr></thead><tbody>{table}</tbody></table></div>{detail}{script}</html>",
        encoding="utf-8",
    )


def run_experiments(plan: str | Path, *, output: str | Path = "runs") -> ExperimentResult:
    """Run explicit candidates/grids sequentially, retaining completed results on failure."""
    path = Path(plan).resolve(strict=True)
    require_evaluator()
    checksum = sha256(path)
    document = load_yaml(path)
    jobs = _jobs(document, path.parent)
    dataset = load_dataset(_reference(document.get("data"), path.parent), ("train_labeled", "val"))
    if document.get("require_groups") and not dataset.groups:
        raise ValueError("This experiment requires explicit, isolated dataset groups")
    if checksum != sha256(path):
        raise ValueError("Experiment plan changed during preflight")
    with operation(output, "experiments", {"plan": str(path), "sha256": checksum}) as directory:
        write_json(directory / "plan.json", document)
        write_json(directory / "dataset.json", dataset.snapshot())
        results: list[dict[str, Any]] = []
        _report(directory, results, dataset.fingerprint)
        for job in jobs:
            target = directory / job["name"]
            target.mkdir()
            job.update(
                data=str(dataset.manifest), output=str(target), result=str(target / "result.json")
            )
            request = target / "job.json"
            write_json(request, job)
            with (target / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [job["python"], "-m", "wvision.experiments.worker", str(request)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                )
                try:
                    process.wait()
                except BaseException:
                    stop_process_tree(process)
                    raise
            if process.returncode:
                raise RuntimeError(f"Experiment {job['name']} failed; see {target / 'worker.log'}")
            result = json.loads(Path(job["result"]).read_text(encoding="utf-8"))
            if result["dataset_fingerprint"] != dataset.fingerprint:
                raise ValueError("Experiment dataset identity changed")
            checkpoint = Path(result["checkpoint"])
            if sha256(checkpoint) != result["checkpoint_sha256"]:
                raise ValueError("Experiment checkpoint changed after prediction")
            evaluation = evaluate_predictions(
                dataset,
                Path(result["comparison_predictions"]),
                checkpoint=checkpoint,
                output=target,
            )
            result["native_evaluator"] = result["evaluator"]
            result["evaluator"] = COCO_PROTOCOL
            result["max_detections"] = 100
            result["common_validation"] = {
                "artifact_dir": str(evaluation.artifact_dir),
                "metrics": evaluation.metrics,
            }
            result["validation"]["metrics"].update(evaluation.metrics)
            results.append(result)
            _report(directory, results, dataset.fingerprint)
    return ExperimentResult(
        directory,
        directory / "comparison.json",
        directory / "comparison.csv",
        directory / "comparison.html",
    )
