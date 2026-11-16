# NVIDIA GPU training

Single-device CUDA execution is available for **YOLO11n**, **DINOv3 ConvNeXt Tiny /
DETR**, **MMDetection Faster R-CNN / Soft Teacher**, and **MMPreTrain SimSiam**,
including their checked initialization paths. CPU remains the default. WVision delegates
training, losses, teacher updates, and optimization to upstream code.

The initial CUDA environment is Windows, Python 3.10, torch 2.1.0+cu121,
torchvision 0.16.0+cu121, MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0, and
MMPreTrain 1.2.0. It uses prebuilt CUDA 12.1 wheels. The main WVision application
continues to use Python 3.12 in its existing environment.

Each backend keeps its own compatible environment:

| Backend | Python | torch / torchvision | CUDA wheel index |
| --- | --- | --- | --- |
| YOLO11n, in process | 3.12 | 2.4.1 / 0.19.1 | `cu124` |
| DINOv3/DETR, isolated worker | 3.12 | 2.8.0 / 0.23.0 | `cu128` |
| Faster R-CNN / Soft Teacher / SimSiam, isolated worker | 3.10 | 2.1.0 / 0.16.0 | `cu121` |

Do not replace one environment's torch packages to satisfy another backend.
Distributed training, mixed precision, automatic batch sizing, optimizer resume,
and arbitrary backbone substitution remain unsupported. DINOv3's official
pretrained-weight verification is still pending; CUDA checks use random weights
and owned local-backbone fixtures.

## YOLO11n

YOLO runs in the application process. Use its CUDA environment's Python for both
the API and CLI; `backend_python` does not select an interpreter for YOLO.

```powershell
py -3.12 -m venv .wvision/yolo-cuda-env
.wvision/yolo-cuda-env/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple torch==2.4.1 torchvision==0.19.1 numpy==1.26.4 Pillow==10.3.0
.wvision/yolo-cuda-env/Scripts/python -m pip install -c constraints/yolo-cuda-py312.txt -e ".[ultralytics]"
.wvision/yolo-cuda-env/Scripts/python -m pip check
.wvision/yolo-cuda-env/Scripts/python -m wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml --device cuda:0
```

Prepare the [example dataset](../datasets/README.md) first. Validation, test, and
prediction inherit the saved device and accept `--device cpu` overrides. The
adapter uses upstream explicit `torch.device` selection without changing CUDA
visibility, including final validation. CPU/GPU switching in one process is checked;
the caller's environment and current CUDA device are restored after each call.
The runnable example also accepts the device:
`.wvision/yolo-cuda-env/Scripts/python examples/detection.py datasets/examples/tiny-detection.yaml --device cuda:0`.

## DINOv3 ConvNeXt Tiny / DETR

Keep the Python 3.12 core installed and create a separate CUDA backend:

```powershell
py -3.12 -m venv .wvision/transformers-cuda-env
.wvision/transformers-cuda-env/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple torch==2.8.0 torchvision==0.23.0 numpy==1.26.4 Pillow==10.3.0
.wvision/transformers-cuda-env/Scripts/python -m pip install -c constraints/transformers-cuda-py312.txt transformers==5.22.2 accelerate==1.10.1 safetensors==0.8.0 pycocotools==2.0.11 scipy==1.15.3
.wvision/transformers-cuda-env/Scripts/python -m pip check
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-cuda-env/Scripts/python.exe).Path
python -m wvision train --model models/detection/dinov3-convnext-tiny-detr/smoke.yaml --data datasets/examples/tiny-detection.yaml --device cuda:0
```

Alternatively pass `backend_python=` / `--backend-python`. Local backbone loading
still requires explicit `backbone_weights` and compatible safetensors; see the
[DINOv3 guide](DINOV3.md). Frozen-backbone CUDA transfer is verified with a locally
generated random fixture, not official pretrained weights.

Before importing torch, this worker maps the requested visible GPU to its sole
visible device. Transformers therefore cannot start automatic multi-GPU
DataParallel. `device.json` records the original `requested` index and selected visibility mask;
its `execution_device` and parameter placement use the worker-local `cuda:0`.
Image inputs and postprocessing tensors are placed on that same device.
Inherited distributed-launch settings are rejected instead of starting extra processes.
`examples/dinov3_detection.py` accepts `--device` and `--backend-python` as well.
Create its `.wvision/dinov3-data` example dataset before running it.

## Install the OpenMMLab CUDA backend

Use an NVIDIA GPU and a compatible NVIDIA driver. Confirm that `nvidia-smi` can
see the device. From `WForge/` (the package root) in PowerShell, with 64-bit Python 3.10 installed:

```powershell
py -3.10 -m venv .wvision/mmdet-cuda-env
.wvision/mmdet-cuda-env/Scripts/python -m pip install --upgrade pip setuptools==78.1.0
.wvision/mmdet-cuda-env/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.1.0 torchvision==0.16.0 numpy==1.26.4 Pillow==10.3.0
.wvision/mmdet-cuda-env/Scripts/python -m pip install -c constraints/mmdet-cuda-py310.txt --only-binary=mmcv --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html mmcv==2.1.0 mmengine==0.10.7 mmdet==3.3.0 mmpretrain==1.2.0
.wvision/mmdet-cuda-env/Scripts/python -m pip check
.wvision/mmdet-cuda-env/Scripts/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-cuda-env/Scripts/python.exe).Path
```

An explicit Python 3.10 interpreter path can replace `py -3.10`. Keep this backend
separate from the Python 3.12 application and the existing CPU backend. Package
downloads happen only in these explicit installation commands. WVision does not
compile operators, install missing packages, or download model weights during a run.

Upstream binaries: [PyTorch 2.1.0 wheels](https://pytorch.org/get-started/previous-versions/)
and [MMCV CUDA 12.1 / torch 2.1 wheels](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html).
The [CUDA constraints](../constraints/mmdet-cuda-py310.txt) pin the integration's
direct compatibility dependencies; they are not a hash-verified lock file.
The [full verified Windows snapshot](../constraints/windows-mmdet-cuda-py310.lock.txt)
records all backend package versions and can be used as constraints with the same
wheel indexes. It also contains no artifact hashes.

## Train Soft Teacher

Generate the example dataset and reproducible two-label training budget if they
do not exist yet:

```powershell
python examples/make_tiny_dataset.py .wvision/tiny
python -m wvision prepare-budget --data datasets/examples/tiny-detection.yaml --destination .wvision/tiny-budget --labeled-count 2 --seed 0
python -m wvision train --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml --device cuda:0
```

Use the printed `artifact_dir` for evaluation and prediction. Operations inherit
the saved training device unless explicitly overridden:

```powershell
python -m wvision val --run runs/train-<id>
python -m wvision test --run runs/train-<id>
python -m wvision predict --run runs/train-<id> --source .wvision/tiny-budget/test/images --device cuda:0
python -m wvision predict --run runs/train-<id> --source .wvision/tiny-budget/test/images --device cpu
```

`cuda` is shorthand for `cuda:0`. `cuda:N` selects one index in the backend's
visible CUDA device list; an inherited `CUDA_VISIBLE_DEVICES` mask is respected.
Device lists, negative indices, automatic selection, and other accelerators are
rejected. Unavailable CUDA or an invalid index fails explicitly; no CPU fallback
occurs. Passing `--device cpu` hides CUDA in the isolated worker so upstream
automatic device detection cannot silently select a GPU.

## SimSiam pretraining and semi-supervised transfer

Run this script from `WForge/` (the package root) after preparing the budget:

```python
from wvision import Vision, pretrain


def main():
    data = "datasets/examples/tiny-semi-supervised.yaml"
    representation = pretrain(
        data,
        model="models/representation/simsiam-r50/smoke.yaml",
        device="cuda:0",
    )
    model = Vision("models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml")
    run = model.train(
        data, device="cuda:0", pretraining_run=representation.artifact_dir
    )
    print(run.artifact_dir)
    print(model.val().metrics)
    print(model.test().metrics)


if __name__ == "__main__":
    main()
```

The [complete example](../examples/gpu_semi_supervised.py) accepts a manifest and
backend interpreter. SimSiam transfers only backbone tensors; detection heads
start randomly. Full detector initialization uses `detector_run=` with the same
class/data/style compatibility rules as CPU execution. See [weight compatibility](WEIGHT_COMPATIBILITY.md).

## Compare and reload

```powershell
python -m wvision compare --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml --device cuda:0
python -m wvision pretrain --model models/representation/simsiam-r50/smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml --device cuda:0
python -m wvision compare-pretraining --model models/detection/faster-rcnn-r50/transfer-smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml --pretraining-run runs/pretrain-<id> --device cuda:0
```

These commands retain the CPU workflow's validation/test separation, label-budget
provenance, and additional-compute reporting. Equal iterations are not equal
compute, and synthetic smoke results do not establish accuracy or speed gains.

To use a CUDA-trained detector in a CPU-only backend, explicitly select both the
backend interpreter and execution device:

```python
model = Vision.from_run(
    "runs/train-<id>", backend_python=".wvision/mmdet-env/Scripts/python.exe"
)
predictions = model.predict("path/to/image.png", device="cpu")
```

An evaluation/prediction override does not change the saved training configuration
or the default device of later calls.

## Execution records and limits

Each supported backend operation adds `device.json`: requested device, torch/CUDA build,
visible-device mask, and, for CUDA, actual allocation device and GPU properties.
The worker verifies all model parameters are on the requested device and records
peak allocated CUDA memory. Soft Teacher records teacher and student separately.
The memory value is the allocator peak observed at the audit, not a complete
operation or driver-VRAM measurement: upstream logging can reset peak counters.
An initialization audit checks source tensor equality before training, including
when loaded weights and model parameters reside on different devices.

Training uses upstream optimizers with FP32 weights and AMP disabled. No custom training loop or precision policy is introduced.
Deterministic settings and CUDA BLAS workspace configuration are recorded/applied,
but CPU and GPU runs are not claimed to be bitwise identical. Memory requirements
depend on images, batch, proposals, and method. Out-of-memory errors preserve the
failed-run status and backend log; WVision does not silently reduce the batch or
change the training protocol. Start with smoke settings and tune on validation.

The [YOLO CUDA constraints](../constraints/yolo-cuda-py312.txt) and
[DINOv3 CUDA constraints](../constraints/transformers-cuda-py312.txt) pin their
compatibility baselines. Full tested snapshots are available for
[YOLO](../constraints/windows-yolo-cuda-py312.lock.txt) and
[DINOv3](../constraints/windows-transformers-cuda-py312.lock.txt).
See [CUDA detection verification](verification/YOLO_DINOV3_CUDA.md) for real execution
evidence and the outstanding official-weight gate.
