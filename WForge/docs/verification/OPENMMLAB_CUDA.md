# Single-GPU OpenMMLab verification

This report records the M3g OpenMMLab slice. Later YOLO/DINOv3 CUDA support is recorded in [detection CUDA verification](YOLO_DINOV3_CUDA.md).

Verified **2026-11-14**, Windows, NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB
device memory, NVIDIA driver 610.60. Changes are an uncommitted working-tree slice
against `90b5caeaab8440d536734ee660c1c48ab8939e79`. No model architecture, training
loss, optimizer, or teacher-update algorithm was implemented here.

## Environment

Python 3.12.3 runs the core; the isolated backend is Python 3.10.20, torch
2.1.0+cu121, torchvision 0.16.0+cu121, MMCV 2.1.0, MMEngine 0.10.7, MMDetection
3.3.0, MMPreTrain 1.2.0, NumPy 1.26.4, Pillow 10.3.0, and setuptools 78.1.0.
The [full package snapshot](../../constraints/windows-mmdet-cuda-py310.lock.txt) and
[installation guide](../GPU.md) record explicit CUDA wheel sources. Existing CPU
environments were preserved. A real CUDA allocation and MMCV GPU NMS invocation
succeeded before training was tested.

## Commands and current results

```powershell
$env:WVISION_MMDET_CUDA_PYTHON = (Resolve-Path .wvision/mmdet-cuda-env/Scripts/python.exe).Path
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest tests/test_cuda_integration.py -m cuda -q --basetemp=.wvision/cuda-probe3
.wvision/clean-env/Scripts/python -m pytest tests/test_cuda_integration.py::test_cuda_pretraining_comparison -m cuda -q --basetemp=.wvision/cuda-transfer-comparison
.wvision/github-core/Scripts/python -m pytest -q
.wvision/github-core/Scripts/python -m ruff check src examples tests
.wvision/github-core/Scripts/python -m ruff format --check src examples tests
.wvision/github-core/Scripts/python -m mypy
.wvision/mmdet-cuda-env/Scripts/python -m pip check
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest -m 'not cuda' -q --basetemp=.wvision/cuda-cpu-regression
```

The initial **three-case real CUDA run passes in 226.58 seconds**. The additional
CUDA random-versus-SimSiam supervised transfer comparison passes in **105.09
seconds**, completing all four available CUDA cases across focused runs. The
core-only run passes **138 tests**, with fifteen integration cases deselected, in
**20.85 seconds**. Lint,
formatting, and typing are checked separately; the dependency check passes for the
CUDA backend. The broad CPU/backend regression passes **148 tests**, with one
expected official-DINOv3-weight skip, in **742.03 seconds**. CUDA cases were
excluded from that run and verified separately. It collected three CUDA cases
before the fourth comparison case was added; all four are covered by the focused
results above. Thirteen existing Matplotlib/Pyparsing deprecation warnings came
from YOLO's dependencies.

Logs and owned artifacts are `.wvision/cuda-probe3.log`, `.wvision/cuda-probe3/`,
`.wvision/cuda-core.log`, and `.wvision/cuda-cpu-regression.log`. Real tests allow
Windows local subprocess communication and block network access inside workers.
No model weights or training data were downloaded.

## Actual training evidence

The original CC0 synthetic fixture is reduced to two labeled and two unlabeled
training images, retaining two validation and two test images. Its fingerprint is
`6f081c42490f25e74501d54e404867aeccc9b26970694effe6fc761151402a13`.
Representation pretraining uses only unlabeled images. Both detection source and
target runs use the same labeled pool and validation selection; test follows
checkpoint selection.

Each training run uses two iterations and batch two. SimSiam crops are 64 pixels;
detector inputs are 128 pixels. Detection uses the existing low-threshold smoke
profiles and learning rate, with upstream FP32 training. These settings test
execution, not accuracy or throughput.

| Actual CUDA run | Initialization scope | Checked transfer tensors per branch | Final training loss / unlabeled classification loss |
| --- | --- | --- | --- |
| SimSiam R50 | Random representation | — | total loss -0.016990 |
| PyTorch-style Soft Teacher | SimSiam backbone | 318, teacher and student | unlabeled classification 2.774563 |
| Supervised Caffe-style Faster R-CNN | Random detector | — | total loss 2.548881 |
| Caffe-style Soft Teacher comparison | Random detector | — | unlabeled classification 2.919237 |
| Initialized Caffe-style Soft Teacher | Full supervised detector | 348, teacher and student | unlabeled classification 2.922963 |

Source hashes retained in the successful transfer audits:

- SimSiam: `ec2396da2f24d8cedcb38cce3e00052feef27d7f0d9981601cf1eff462e3b86d`.
- Supervised detector: `a7f37b5f7a6fd1eb4bbe7fc34f31adc6cb4bc91e87f800bdc11f2a9cba223822`.

The corresponding target directories under `.wvision/cuda-probe3/` are
`test_cuda_simsiam_to_soft_teac0/runs/train-2a9f95871a284108935c4e5024ebee73` and
`test_cuda_comparison_and_super0/runs/train-0ac49edcc4414ec6b14d016c4ada6db7`.
`transfer.json` checks exact loaded-weight equality before the first training step.
This initialization check is not a claim that independent CPU/GPU training runs
are bitwise identical.

`device.json` verifies actual CUDA tensor allocation, GPU identity, and placement
of all model parameters, recording teacher/student separately. It records the
CUDA allocator peak observed at the audit; upstream logging may reset these
counters, so this is not an end-to-end memory benchmark.

## Public operations and failures

The real CUDA SimSiam → Soft Teacher path uses the Python model-profile API.
Checkpoint reload, validation, test, and prediction run on CUDA. CLI prediction
with `--device cpu` uses the same CUDA-capable backend with CUDA explicitly hidden;
another prediction reloads the CUDA-trained detector in the original CPU-only
backend. Both produce valid output without changing the saved training device.

The CLI `compare --device cuda:0` runs the matched supervised and Soft Teacher
comparison, preserving the two-label budget and recording the CUDA device in its
report. Its supervised checkpoint then initializes both CUDA Soft Teacher branches.
Positive unlabeled losses verify that the upstream semi-supervised branch executes.
Unlabeled COCO annotations stay empty, and test annotations never enter training.

The additional Python `compare_pretraining(..., device="cuda:0")` test runs GPU
SimSiam, random PyTorch-style supervised Faster R-CNN, and the matching
backbone-initialized supervised detector. It verifies both downstream models'
CUDA placement, the 318-tensor source audit, label-budget preservation, and the
CUDA device in the comparison report. Its log/artifacts are
`.wvision/cuda-transfer-comparison.log` and `.wvision/cuda-transfer-comparison/`.

Real failure checks request nonexistent `cuda:99` and request CUDA in the CPU-only
backend. Both persist failed status and a diagnostic log before building the
training recipe; neither trains on CPU as a fallback. Core checks additionally
cover canonical device syntax, unsupported accelerators/multi-GPU requests,
capability combinations, child-process visibility masks, and non-mutating inference
overrides.

Initial environment/setup probes exposed missing setuptools in the fresh backend
and a stale installed wheel in a CLI test host. The environment recipe now pins
setuptools, and successful tests run against the editable development checkout.
The successful CUDA invocations pass all assertions; earlier failed probes
remain in separate ignored `cuda-probe*` directories.

## Limits

Verified support is single-device Windows CUDA for the OpenMMLab workflows.
YOLO/DINOv3 CUDA, Linux CUDA verification, distributed training, AMP, automatic
batch adjustment, resume, and official pretrained DINOv3 acceptance remain
separate gates. These synthetic results make no accuracy, label-efficiency,
speedup, or state-of-the-art claim.

Final static checks pass for all 48 Python files and 27 typed source modules.
All 31 public Markdown documents have valid local links; fourteen Python blocks
parse, 46 CLI examples parse, and 24 catalog selections validate. CUDA backend
dependencies are consistent. Distribution checks include the shared standalone
device helper next to all backend workers, and the clean source ZIP is refreshed.

The independently installed wheel also performs an actual CUDA prediction using
its packaged worker/helper while the Python 3.12 host imports no optional ML
frameworks. Its output is
`.wvision/cuda-installed-wheel/predict-a2312fd267fb4221bbb84a5118a86d31`.
The wheel has 35 files and the source distribution has 122. The final audited Git
upload list contains 114 public files, with local environments, model weights,
generated runs, and local instructions excluded. Existing CPU environments were
not modified; the unused task-created YOLO CUDA experiment environment was removed.
