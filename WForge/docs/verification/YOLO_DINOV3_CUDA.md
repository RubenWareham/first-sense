# YOLO and DINOv3 CUDA verification

Verified **2026-11-14** on Windows with an NVIDIA GeForce RTX 4050 Laptop GPU
(6 GB), driver 610.60. This extends the earlier [OpenMMLab CUDA slice](OPENMMLAB_CUDA.md).
Changes are an uncommitted working-tree slice against
`90b5caeaab8440d536734ee660c1c48ab8939e79`; no commit or push was performed.

## Environments

| Workflow | Python | Runtime |
| --- | --- | --- |
| YOLO11n | 3.12.3 | Ultralytics 8.3.4, torch 2.4.1+cu124, torchvision 0.19.1+cu124 |
| DINOv3 ConvNeXt Tiny / DETR | 3.12.3 | Transformers 5.22.2, torch 2.8.0+cu128, torchvision 0.23.0+cu128, Accelerate 1.10.1 |

The [YOLO snapshot](../../constraints/windows-yolo-cuda-py312.lock.txt) and
[DINOv3 snapshot](../../constraints/windows-transformers-cuda-py312.lock.txt) record
the complete tested package versions. CPU and OpenMMLab environments were
preserved. No model weights or datasets were downloaded. Tests block Python
network connections; local backbone weights are generated from the upstream
architecture and identified as random fixtures, never official pretrained SSL.

## Commands and results

```powershell
$env:WVISION_YOLO_CUDA_TEST = '1'
.wvision/yolo-cuda-env/Scripts/python -m pytest tests/test_cuda_detection_integration.py::test_cuda_yolo_train_reload_and_cpu_gpu_switching -m cuda -q --basetemp=.wvision/yolo-cuda-probe
$env:WVISION_YOLO_CPU_TEST = '1'
.wvision/clean-env/Scripts/python -m pytest tests/test_cuda_detection_integration.py::test_yolo_cuda_request_with_cpu_torch_fails -m cuda -q --basetemp=.wvision/yolo-cuda-missing
$env:WVISION_TRANSFORMERS_CUDA_PYTHON = (Resolve-Path .wvision/transformers-cuda-env/Scripts/python.exe).Path
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest tests/test_cuda_detection_integration.py -k dinov3 -m cuda -q --basetemp=.wvision/dinov3-cuda-probe
.wvision/clean-env/Scripts/python -m pytest tests/test_integration.py tests/test_dinov3_integration.py -m integration -q --basetemp=.wvision/detection-cuda-cpu-regression
$env:WVISION_MMDET_CUDA_PYTHON = (Resolve-Path .wvision/mmdet-cuda-env/Scripts/python.exe).Path
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest tests/test_cuda_integration.py::test_cuda_simsiam_to_soft_teacher -m cuda -q --basetemp=.wvision/all-cuda-openmmlab-regression
.wvision/github-core/Scripts/python -m pytest -q
```

- YOLO CUDA public workflow: **one passed in 31.62 seconds**, including fine-tuning.
- YOLO CUDA request in CPU-only torch: **one passed in 3.94 seconds**.
- DINOv3 CUDA workflows: **three passed in 117.79 seconds**.
- Existing CPU YOLO/DINOv3 workflows: **four passed, one skipped in 161.56 seconds**.
  The skip remains official pretrained DINOv3 acceptance without licensed weights.
- Existing CUDA SimSiam → Soft Teacher and CPU reload: **one passed in 124.99 seconds**.
- Final framework-free core suite: **145 passed**, with twenty integration cases
  deselected, in **17.48 seconds**.

YOLO's dependency stack emits thirteen existing Matplotlib/Pyparsing deprecation
warnings per relevant invocation. All 49 Python files pass lint and formatting;
Mypy passes all 27 source modules. Both new CUDA environments pass `pip check`.
Other unchanged OpenMMLab combinations retain the preceding verification; this
slice reruns the affected CPU backends and the shared CUDA transfer workflow,
not the entire previous full suite.

## Actual workflows

The original CC0 fixture has four labeled training images, two validation images,
and two test images. Its fingerprint is
`4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.
YOLO trains for one epoch at 128 pixels, batch two, from random weights and then
from its own selected checkpoint. DINOv3/DETR trains for two steps at 64 pixels,
batch two, with the full upstream 45,537,638-parameter architecture. Upstream
libraries own every architecture, loss, optimizer, and training loop.

Successful CUDA training artifacts:

- YOLO: `.wvision/yolo-cuda-probe/test_cuda_yolo_train_reload_an0/runs/train-31f550603803428a9b48d720be12a46b`
  and `train-ed4d4946024249e0a2633ba7ab2ef4cf` in the same `runs/` directory.
- Random DINOv3/DETR:
  `.wvision/dinov3-cuda-probe/test_cuda_dinov3_random_detect0/runs/train-1f7bf0a913c84d8a9a7d6ec0e4daf43c`.
- Local frozen-backbone DINOv3/DETR:
  `.wvision/dinov3-cuda-probe/test_cuda_dinov3_local_frozen_0/runs/train-3bece637e4df41f68c55e466b85ca51b`.

Each `device.json` confirms real CUDA allocation and GPU parameter placement.
YOLO training is audited at the upstream training-start callback and after
training; the post-training wrapper's reloaded best model is not mistaken for
the actual trainer model. Validation/prediction placement is audited too.
The adapter delegates to upstream `select_device(torch.device(...))` through
scoped selector bindings, avoiding string-based visibility remapping. CPU → GPU
switching in one process preserves the caller's CUDA visibility and current device.

DINOv3's isolated worker pins exactly one visible GPU before torch/Accelerate
imports, overrides conflicting Accelerate device hints, and records both original
request and worker-local execution device. This prevents automatic DataParallel
on multiple visible GPUs. Trainer owns model placement; image preprocessing and
postprocessing tensors are moved to the same device. CUDA runs use FP32 weights
with AMP disabled.

The local-backbone test transfers **178 tensors** and then compares the actual
saved frozen backbone against its source, verifying exact tensor equality. This
establishes loading/freeze behavior with an owned random fixture; it does not
verify official pretrained DINOv3 weights or any downstream accuracy claim.

CUDA train/val/test/predict and checkpoint reload pass through Python and CLI
entry points. DINOv3 additionally reloads into its original CPU-only backend;
YOLO switches explicitly to CPU and back inside one CUDA-capable process.
Nonexistent GPU indices and CUDA requests with CPU-only torch fail without
training on CPU. Distributed-launch hints and unsupported device syntax are
rejected in core policy checks. Validation still selects checkpoints, and test
labels remain outside training.

## Remaining limits

Single-device Windows CUDA is verified. Multi-GPU/DDP, AMP, resume, export, Linux
CUDA execution, and official pretrained DINOv3 acceptance remain separate gates.
GPU support does not add semi-supervised YOLO or DINOv3 SSL pretraining. Synthetic
smoke results establish execution, not accuracy, throughput, or benchmark leadership.

The built wheel was installed into a separate target directory and imported
with Python's isolated mode. YOLO CPU → CUDA prediction passed with caller CUDA
visibility/current-device preservation. DINOv3 CUDA prediction passed through the
packaged worker while the calling core process imported neither torch nor
Transformers. Both reused the GPU-trained smoke checkpoints above.

Logs live under the matching ignored `.wvision/*cuda*.log` paths. The publication
audit checked 120 public source files, 32 linked documents, 14 Python examples,
48 CLI examples, and 26 profiles. The wheel contains 35 files and the source
distribution 128 files; catalogs and worker helpers are included as appropriate.
Backend environments, weights, and generated runs remain excluded from the
refreshed source upload archive.
