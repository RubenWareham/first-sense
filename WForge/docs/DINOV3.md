# DINOv3 ConvNeXt Tiny with DETR

Model ID: `transformers/dinov3-convnext-tiny-detr`. WVision integrates Transformers' existing DINOv3 ConvNeXt Tiny backbone and DETR detection implementation. Upstream owns the architecture, detection losses, matching, optimizer, training loop, and postprocessing. WVision translates data/configuration, invokes `Trainer`, and records results. This follows the upstream [backbone training pattern](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/training_vision_backbone.md); the [pinned implementation](https://github.com/huggingface/transformers/tree/v5.22.2/src/transformers/models/dinov3_convnext) is used rather than copying model source.

Model-file construction is available through the [model catalog](../models/README.md): `Vision("models/detection/dinov3-convnext-tiny-detr/smoke.yaml")` or the local-weight `frozen-backbone.yaml` / `fine-tune.yaml` profiles in that directory. Dataset manifests remain separate, and local backbone paths are still explicit `backbone_weights` inputs. Official pretrained verification is still pending.

**Verification scope:** CPU and single-device CUDA random-initialization detection and loading an owned, randomly initialized local backbone fixture. See [GPU setup](GPU.md) and [CUDA execution evidence](verification/YOLO_DINOV3_CUDA.md). Official pretrained DINOv3 weights are not available in this workspace, so their transfer and accuracy are an outstanding gate. A random-weight DINOv3 architecture is not a self-supervised model. The adapter never silently replaces missing weights with random ones.

The local-backbone workflow performs supervised downstream detection. If supplied weights were self-supervised, their representations are reused, but detection labels still train the head and validation labels measure selection loss. It does not run DINOv3's self-supervised pretraining algorithm. Meta's original released detector uses a different, 7B backbone; this integration does not reproduce that detector or its published metrics. [Upstream DINOv3 release](https://github.com/facebookresearch/dinov3)

## Independent backend environment

Keep the existing core, YOLO, and MMDetection environments. Use Python 3.12 to create a separate backend:

```powershell
python -m venv .wvision/transformers-env
.wvision/transformers-env/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu
.wvision/transformers-env/Scripts/python -m pip install -c constraints/transformers-cpu-py312.txt transformers==5.22.2 accelerate==1.10.1 safetensors==0.8.0 pycocotools==2.0.11 scipy==1.15.3 numpy==1.26.4 Pillow==10.3.0
.wvision/transformers-env/Scripts/python -m pip check
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
```

The [full Windows dependency snapshot](../constraints/windows-transformers-cpu-py312.lock.txt) can replace the constraints file for exact versions. The CPU wheel index is still required for torch/torchvision. It records versions without wheel hashes. The installed backend does not need WVision: the core calls a fixed packaged worker with versioned JSON and no shell. `backend_python=` / `--backend-python` can replace the environment variable and is recorded in the run.

The initial environment is Windows CPU, Python 3.12.3. The current Transformers backbone API requires a newer framework than the earlier integrations; do not upgrade those environments in place. Single-device CUDA is verified in a separate torch 2.8.0+cu128 environment; see [GPU setup](GPU.md). BF16 mixed precision, optimizer-state recovery, portable bundles, and the unverified DDP launcher are described in [training operations](TRAINING_OPERATIONS.md). ONNX export remains unsupported for this adapter.

## Explicit random-weight smoke test

In the core environment:

```powershell
python examples/make_tiny_dataset.py .wvision/dinov3-data
wvision models
wvision train --model models/detection/dinov3-convnext-tiny-detr/smoke.yaml --data .wvision/dinov3-data/dataset.yaml
```

This trains the full upstream backbone/head from random initialization for two steps. It prints a JSON result with the run directory, checkpoint references, and validation loss. It establishes execution, not useful pretrained representations or detector quality.

Use the printed run directory:

```text
wvision val --run runs/train-<id>
wvision test --run runs/train-<id>
wvision predict --run runs/train-<id> --source .wvision/dinov3-data/images/test-0.png
```

## Using a local DINOv3 backbone

Obtain access and download weights explicitly from Meta's [official model repository](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) under the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). WVision does not accept terms, log into accounts, or download weights on the user's behalf. Transformers' implementation has its own Apache-2.0 code license; that does not replace the weight license.

This first adapter accepts a local Hugging Face directory containing `config.json` and one `model.safetensors`. Raw Meta `.pth` files, sharded checkpoints, other backbone sizes, URLs, remote Hub model IDs, and executable custom configurations are rejected or unsupported. The directory must describe DINOv3 ConvNeXt Tiny (`hidden_sizes: [96,192,384,768]`, `depths: [3,3,9,3]`). Loading uses `local_files_only=True`, safetensors, and `trust_remote_code=False`.

Optionally add `wvision-source.json` to record owner-supplied provenance:

```json
{
  "source": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
  "revision": "the exact downloaded Hub commit",
  "supervision": "self_supervised",
  "pretraining_data": "LVD-1689M; distilled DINOv3 ConvNeXt Tiny",
  "license": "DINOv3 License"
}
```

This metadata is a declaration, not proof of model origin. Without it, supervision and pretraining data are recorded as unknown. WVision records hashes of the configuration, safetensors, and optional provenance. The worker verifies them again, rejects missing/mismatched backbone tensors, checks the complete detector-backbone state, and verifies loaded tensor equality. An unused final pooling normalization from a representation checkpoint is allowed and recorded. Detection heads start randomly.

```text
wvision train --model models/detection/dinov3-convnext-tiny-detr/frozen-backbone.yaml --data path/to/dataset.yaml --backbone-weights .wvision/weights/dinov3-convnext-tiny
```

The local-backbone recipe freezes the backbone by default. `freeze_backbone: false` fine-tunes it through upstream `Trainer`. Official weight compatibility is still pending verification; incompatible files fail with a retained backend log. No accuracy or state-of-the-art claim is made for the WVision integration.

## Python interface

```python
from wvision import Vision, list_models

model = Vision(
    "models/detection/dinov3-convnext-tiny-detr/frozen-backbone.yaml",
    backend_python=".wvision/transformers-env/Scripts/python.exe",
)
run = model.train(
    "path/to/dataset.yaml",
    backbone_weights=".wvision/weights/dinov3-convnext-tiny",
)
validation = model.val()
test = model.test()
predictions = model.predict("path/to/image.png")
restored = Vision.from_run(run.artifact_dir)
```

Use a `main()` guard on Windows; [the example](../examples/dinov3_detection.py) includes one. Original run references are absolute; use [portable bundles](TRAINING_OPERATIONS.md#portable-bundles) for relocation. After training, inference uses the complete detector checkpoint and no longer needs the source backbone directory.

## Settings, selection, and metrics

Settings: `schema_version` (1), `learning` (`supervised`), `max_iters`, `imgsz`, `batch`, `seed`, `device` (`cpu`), `lr`, `initialization` (`local_backbone` or explicit `random`), `freeze_backbone` (boolean), `precision`, `devices`, and `checkpoint_interval`. Unknown settings fail. Image size is at least 64 and divisible by 32; batch and iteration count are positive integers. Initializer paths are separate `backbone_weights` arguments. `checkpoint` and `pretraining_run` inputs are unsupported for this adapter.

The initial recipe uses square resizing and ImageNet RGB normalization, no stochastic augmentation, upstream default DETR dimensions (six encoder and six decoder layers, width 256, 100 queries), AdamW, and a constant learning rate. The full model has 45,537,638 parameters for the one-class fixture. These are integration settings, not a tuned reproduction of a published benchmark.

Validation loss is measured by upstream `Trainer` at checkpoint intervals and minimized for selection. `checkpoint_interval` controls intermediate validation/saves; recovery preserves optimizer/scheduler/scaler/RNG state. Selection uses loss, while standalone val/test report AP. The selected and final checkpoint can differ. The two-step smoke profile still evaluates at its final step with default intervals.

`val` and `test` compute official pycocotools bbox AP50:95, AP50, AP75, and area metrics with `maxDets=[1,10,100]`, preserving undefined values as -1. Metrics have a `transformers/coco/` prefix. This setting differs from earlier adapters; do not assume raw cross-backend metrics form a matched benchmark. DETR postprocessing keeps all 100 queries at score threshold zero; COCO ranks detections. Prediction returns the existing JSONL contract: original image size, absolute pixel `xyxy` boxes, scores, zero-based class IDs, and a class map.

COCO staging uses category IDs 1..N. The training adapter maps them to zero-based DETR labels; predictions map back for COCO evaluation. Human annotations for unlabeled training and test are not staged during training. Standard dataset, run, checkpoint-hash, and config-hash safeguards remain active.

## Artifacts and verification

The normal config/dataset/environment/status/run/metric files are saved alongside `backend-request.json`, `backend-result.json`, `backend-environment.json`, `backend.log`, `resolved-transformers.json`, `model-source.json`, and structured `training-scalars.json`. Local transfer adds `transfer.json`; evaluation retains COCO predictions. Framework logging is preserved rather than parsed for primary metrics.

Model extension now uses an explicit catalog and [typed adapter protocol](MODEL_ADAPTERS.md). Add a reviewed catalog entry and adapter factory without modifying `Vision` or CLI dispatch. New capabilities still need real tests. See [DINOv3 verification](verification/DINOV3.md) for actual execution and outstanding official-weight verification.
