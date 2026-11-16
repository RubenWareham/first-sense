# Initialized semi-supervised detection verification

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

Verified **2026-11-08**, Windows CPU. Scope: explicit initialization compatibility and upstream Soft Teacher initialized from real WVision detector/representation runs. No Git revision is available because this workspace is not a Git repository. No dependencies were installed or upgraded.

## Environment and commands

Core: Python 3.12.3. Isolated backend: Python 3.10.20, torch 2.1.0+cpu, torchvision 0.16.0+cpu, MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, MMPreTrain 1.2.0. Existing dependency snapshots and setup remain in the [semi-supervised](../SEMI_SUPERVISED.md) and [self-supervised](../SELF_SUPERVISED.md) guides.

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest tests/test_initialized_semi_integration.py -m integration -q --basetemp=.wvision/initialized-semi-probe
.wvision/clean-env/Scripts/python -m pytest -m '' -q --basetemp=.wvision/initialization-verified
.wvision/clean-env/Scripts/python -m pytest -q
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pip check
.wvision/mmdet-env/Scripts/python -m pip check
.wvision/transformers-env/Scripts/python -m pip check
```

The targeted real test passed **three cases in 301.70 seconds**. The full regression suite passed **129 tests**, with **one expected skip** for unavailable official DINOv3 weights, in **662.23 seconds**. Thirteen Matplotlib/Pyparsing deprecation warnings came from YOLO's dependency stack. The default suite passed **119 tests**, deselecting eleven integration cases, in **18.47 seconds**. Ruff lint and formatting pass for 44 Python files; Mypy passes for all 26 source modules. All three dependency checks and local Markdown links pass.

Real ML checks ran with permission for Windows local subprocess communication. Tests block networking in workers; checkpoints are generated locally by actual upstream training. Logs are `.wvision/initialized-semi-probe.log`, `.wvision/initialization-core.log`, and `.wvision/initialization-verified.log`. Artifacts remain in the corresponding ignored directories and are not distributed weights.

## Actual transfer evidence

All source and target runs use the synthetic two-label budget with fingerprint `6f081c42490f25e74501d54e404867aeccc9b26970694effe6fc761151402a13`: two labeled training images, two unlabeled training images, two validation images, two test images. Source supervised training uses only the labeled pool; SimSiam uses only unlabeled images. Each source and target trains for two iterations. Detector images are 128 pixels; representation crops are 64 pixels. Fixture data is original CC0 content.

The targeted artifacts below are relative to `.wvision/initialized-semi-probe/`. In each target, `transfer.json` confirms exact tensor equality after upstream initialization hooks and before the first training step, in both teacher and student.

| Source → target | Target directory | Tensors checked per branch | Final unlabeled classification loss |
| --- | --- | --- | --- |
| Supervised Caffe-style Faster R-CNN → Soft Teacher | `test_real_initialized_soft_tea0/runs/train-afb75bbc502e4b41b1029a8094ad81d5` | 348, complete detector | 0.160804 |
| Supervised PyTorch-style Faster R-CNN → Soft Teacher | `test_real_initialized_soft_tea1/runs/train-700e627949ac4509884d6b0c9dd2d1af` | 348, complete detector | 2.766371 |
| SimSiam R50 → PyTorch-style Soft Teacher | `test_real_initialized_soft_tea2/runs/train-157c15c4e6d34557a9549a9a2089d235` | 318, backbone only | 2.771636 |

Source checkpoint SHA-256 values recorded by those targets:

- Caffe detector: `3a3f6755e25f796063eaf0410561e4b6eb1d80fac474e556bd9c9cc613cea62c`.
- PyTorch detector: `5faac20b572df9ba9c3938073b491514d9b68ba58545aa846193d3a75e611344`.
- SimSiam: `b440ad0205573983c731954ae3b17c57e37fa2b15fabc4667b8489e020f6371f`.

The complete detector transfers detection heads too. The representation transfer leaves heads randomly initialized. All runs start fresh optimizer/schedule state. The audit does not update weights: upstream loading, `MeanTeacherHook`, and training own that behavior.

## Public workflows and rejection checks

The Caffe-style target trains through CLI model-config selection and `--detector-run`; both PyTorch-style targets train through `Vision(model_config).train(...)`. Each reloads through `Vision.from_run` and performs actual validation, test, and prediction. Validation teacher AP selects the checkpoint; evaluation/prediction uses that teacher. Both inner detector and outer multi-branch preprocessing match the requested style. Unlabeled COCO annotations are empty, and test annotations never enter training.

The real test deliberately changes a source checkpoint's class metadata and then removes a state tensor, updating its recorded checksum each time. Both attempts fail in the worker and persist failed run status, demonstrating semantic/state rejection beyond a checksum guard. The already-trained target still predicts after source damage. Consequently, the Caffe source checkpoint retained in these test artifacts is deliberately damaged; its original hash above describes the successful transfer, not the final damaged bytes.

Framework-free checks cover source completion, model/method, dataset fingerprint, class map, style, version, checkpoint/config checksums, exclusive initialization inputs, and unsupported adapter/method combinations. A metadata fixture is used only for these rejection boundaries and is not counted as model execution evidence.

## Limits

These are single-seed execution checks with deliberately low pseudo-label thresholds. They establish neither accuracy improvement nor benchmark leadership. Same-dataset sources, matching classes/styles/structures, and trusted local WVision artifacts are required. Cross-dataset/raw external detector initialization, optimizer resume, GPU, YOLO semi-supervised training, and arbitrary backbone substitution remain unsupported. Official pretrained DINOv3 acceptance still requires licensed local weights. See the [working contract](../WEIGHT_COMPATIBILITY.md) and [example](../../examples/initialized_semi_supervised.py).
