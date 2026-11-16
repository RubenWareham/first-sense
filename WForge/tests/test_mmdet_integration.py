"""Real Soft Teacher execution and comparison, not a mocked backend contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from wvision import Vision, prepare_label_budget
from wvision.adapters.shared.process import stop_process_tree
from wvision.datasets.detection import load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.mmdetection]


def run_cli(arguments: list[str], timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "wvision", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        stop_process_tree(process)
        process.communicate()
        raise
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_windows_backend_cancellation_reaps_redirector_children() -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if os.name != "nt" or not backend:
        pytest.skip("Windows backend interpreter required for redirector cleanup verification")
    process = subprocess.Popen(
        [backend, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())
    finally:
        stop_process_tree(process)
    command = Path(os.environ["SystemRoot"]) / "System32/tasklist.exe"
    listing = subprocess.run(
        [str(command), "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f'"{child_pid}"' not in listing.stdout


def test_real_soft_teacher_comparison(
    manifest: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = os.environ.get("WVISION_MMDET_PYTHON")
    if not backend:
        pytest.skip("Set WVISION_MMDET_PYTHON to the pinned isolated backend executable")
    monkeypatch.setenv("WVISION_TEST_OFFLINE", "1")
    derived = prepare_label_budget(manifest, tmp_path / "budget", labeled_count=2, seed=0)
    # Exercise a catalog manifest outside the data root while retaining budget provenance.
    catalog = tmp_path / "dataset-catalog.yaml"
    catalog.write_text(
        yaml.safe_dump({**yaml.safe_load(derived.read_text()), "root": "budget"}), encoding="utf-8"
    )
    derived = catalog
    output = tmp_path / "runs"
    result = run_cli(
        [
            "compare",
            "--data",
            str(derived),
            "--backend-python",
            backend,
            "--model",
            "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
            "--output",
            str(output),
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    comparison = json.loads(result.stdout)
    report = json.loads(Path(comparison["report"]).read_text())
    assert report["dataset_fingerprint"] == load_dataset(derived, ("test",)).fingerprint
    assert report["label_budget"]["labeled_count"] == 2 and report["seed_count"] == 1
    assert report["compute"]["max_iters"] == 2
    assert report["test"]["supervised"]["mmdetection/coco/bbox_mAP"] >= 0
    assert report["test"]["semi_supervised"]["mmdetection/coco/bbox_mAP"] >= 0
    assert report["test_delta"]["mmdetection/coco/bbox_mAP_l"] is None
    semi_run = Path(comparison["semi_supervised_run"])
    logs = json.loads((semi_run / "training-scalars.json").read_text())
    assert logs["train/unsup_loss_cls"] > 0
    assert any(value > 0 for key, value in logs.items() if "unsup_loss" in key)
    provenance = json.loads((semi_run / "model-source.json").read_text())
    assert provenance["method"] == "SoftTeacher" and provenance["pretraining_data"] is None
    assert provenance["teacher"]["update"] == "upstream MeanTeacherHook"
    unlabeled = json.loads((semi_run / "train_unlabeled.coco.json").read_text())
    assert unlabeled["annotations"] == []
    assert not (semi_run / "test.coco.json").exists()
    assert not (Path(comparison["supervised_run"]) / "train_unlabeled.coco.json").exists()
    model = Vision.from_run(semi_run)
    assert model.test(output=output).split == "test"
    sample = load_dataset(derived, ("test",)).splits["test"][0].path
    prediction = model.predict(sample, output=output)
    predicted = json.loads(prediction.predictions.read_text())
    assert prediction.sample_count == 1
    assert predicted["classes"] == {"0": "rectangle"}
    assert predicted["width"] == predicted["height"] == 128
    assert len(predicted["boxes"]) == len(predicted["scores"]) == len(predicted["class_ids"])
    for operation, extra in (("val", []), ("test", []), ("predict", ["--source", str(sample)])):
        command = run_cli(
            [
                operation,
                "--run",
                str(semi_run),
                "--output",
                str(output),
                *extra,
            ],
        )
        assert command.returncode == 0, command.stdout + command.stderr
        assert Path(json.loads(command.stdout)["artifact_dir"]).is_dir()
    config_file = semi_run / "resolved-mmdet.json"
    config_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="model configuration content has changed"):
        model.test(output=output)
