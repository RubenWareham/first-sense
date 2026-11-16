# M1 verification evidence

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

Verified on 2026-11-07 in a newly created virtual environment without system-site packages. Installation used the documented CPU-wheel step and project extras with compatibility constraints. Environment: Windows, Python 3.12.3, Ultralytics 8.3.4, PyTorch 2.4.1+cpu, torchvision 0.19.1+cpu, NumPy 1.26.4, Pillow 10.3.0. The [full dependency snapshot](../../constraints/windows-cpu-py312.lock.txt) records the remaining packages and tools. No Git revision exists in this workspace.

## Commands and results

From the repository root:

```powershell
.wvision/clean-env/Scripts/python -m pip check
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pytest -q -o addopts= --basetemp .wvision/verified
.wvision/clean-env/Scripts/wvision --help
```

- Dependencies: no broken requirements.
- Ruff: all checks passed; all 15 Python files formatted.
- Mypy: no issues in 10 source files.
- Pytest: **28 passed**, 128.17 seconds; no skipped tests.
- Console entry point: help succeeds.
- Warnings: 13 upstream Matplotlib/Pyparsing deprecation warnings; no verification failure.

The test command overrides the default integration exclusion. Use a new/nonexistent `--basetemp` location if retaining evidence; pytest can remove an existing base-temp directory on later runs.

Real integration tests required an execution environment permitting Windows named pipes. The restricted sandbox denied those pipes during upstream label checking; the successful run used the necessary host permissions and blocked socket connections in the tests. No upstream algorithm was replaced to obtain a pass.

## Behaviors verified

Both Python and CLI execute one-epoch training, validation, explicit test evaluation, and local-image prediction. Python additionally initializes a new training run from a locally generated YOLO11n checkpoint. Tests check run reload, checkpoint integrity, changed dataset rejection, incompatible architectures and class maps, missing extras, unsupported learning modes, missing annotations, split overlap, duplicate image content, config precedence, immutable source data through run-local copies, and failed/interrupted status records.

The CLI tests execute the same module/main function used by the installed console entry point, preserving JSON stdout and backend progress on stderr. Core import/help checks confirm neither PyTorch nor Ultralytics is imported.

## Fixture and model identity

The original synthetic rectangle dataset contains four labeled training images, two validation images, and two test images at 128×128. Tests use one epoch, batch 2, seed 0, and CPU. The Python test disables mosaic; the CLI test exercises upstream defaults. These are independent execution checks, not an accuracy comparison.

- Dataset fingerprint: `4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.
- Upstream definition: `ultralytics/cfg/models/11/yolo11.yaml`, scale `n`, release `v8.3.4`.
- Definition SHA-256: `94cf4eb9ac55aec23513b447626a6d637300f7462244e413e59ea4b68885c4a5`.
- Initial weights: random, plus a separate fine-tuning check using its locally produced checkpoint; no pretrained weights downloaded.
- Observed AP50 and AP50–95: zero on this deliberately tiny one-epoch execution fixture. This establishes neither useful model quality nor state-of-the-art performance.

## Local artifacts and limits

The raw test report is `.wvision/verification.log`. Per-operation configs, environment, dataset hashes, model provenance, metrics, statuses, and upstream artifacts remain under `.wvision/verified/test_python_workflow0/runs/` and `.wvision/verified/test_cli_workflow0/runs/`. Some failure/interruption artifacts are intentional error-path tests. These directories are ignored generated output and are not required source files.

The environment is retained at `.wvision/clean-env`; the initial development environment is `.venv`. This is a Windows CPU verification only. Accelerator behavior, Linux execution, external pretrained weights, distributed training, export, resume, other tasks, and semi-/self-/unsupervised methods have not passed integration checks and remain unsupported. Distribution licensing remains a separate release decision described in [third-party notices](../../THIRD_PARTY_NOTICES.md).
