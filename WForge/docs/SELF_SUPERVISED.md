# Self-supervised pretraining and detection transfer

WVision runs the existing **MMPreTrain 1.2.0 SimSiam ResNet-50** implementation on `train_unlabeled`, then transfers its backbone into **MMDetection 3.3.0 Faster R-CNN R50-FPN**, either supervised or inside upstream Soft Teacher. Upstream owns both training loops, the SimSiam model, losses, augmentations, and optimizer behavior. WVision does not implement a learning algorithm.

The [model catalog](../models/README.md) now supports `pretrain(data, model="models/representation/simsiam-r50/smoke.yaml")` and `compare_pretraining(..., model="models/detection/faster-rcnn-r50/transfer-smoke.yaml")`. Model documents carry defaults and provenance; dataset manifests remain independent. Select model documents through `model` / `--model`; registered IDs also remain supported.

Upstream references: [MMPreTrain v1.2.0 recipe](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/configs/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py), [SimSiam implementation](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/mmpretrain/models/selfsup/simsiam.py), and [upstream method guide](https://mmpretrain.readthedocs.io/en/latest/papers/simsiam.html). Runtime adaptation and initialization checks are described below.

Pretraining uses images without human labels. Downstream detection training uses `train_labeled` boxes and class IDs; validation labels select the detector checkpoint. This workflow is **self-supervised pretraining followed by supervised detection**, not fully unsupervised detection. See the separate [unsupervised assessment](UNSUPERVISED_ASSESSMENT.md).

## Environment

Use the Python 3.12 core and isolated Python 3.10 Windows CPU backend from the [semi-supervised setup](SEMI_SUPERVISED.md). Add the pinned package explicitly:

```powershell
.wvision/mmdet-env/Scripts/python -m pip install -c constraints/selfsup-cpu-py310.txt mmpretrain==1.2.0
.wvision/mmdet-env/Scripts/python -m pip check
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
```

The backend uses torch 2.1.0+cpu, torchvision 0.16.0+cpu, MMCV 2.1.0, and MMEngine 0.10.7. The complete [Windows dependency snapshot](../constraints/windows-selfsup-cpu-py310.lock.txt) includes both OpenMMLab packages. CPU/MMCV wheel indexes from the setup guide are still required for a fresh installation. This is a version snapshot, without wheel hashes. No packages or weights are downloaded by imports or training. Single-device CUDA is available in a separate [GPU backend environment](GPU.md), including SimSiam-to-Soft Teacher transfer. External raw pretrained weights remain unsupported. See [training operations](TRAINING_OPERATIONS.md) for mixed precision, optimizer-state recovery, portable bundles, and the unverified DDP boundary. Checked Soft Teacher transfer is described in [weight compatibility](WEIGHT_COMPATIBILITY.md).

## Commands

From the activated core environment, create the example and a training-only label budget:

```powershell
python examples/make_tiny_dataset.py .wvision/ssl-source
wvision prepare-budget --data .wvision/ssl-source/dataset.yaml --destination .wvision/ssl-budget --labeled-count 2 --seed 0
wvision pretrain --data .wvision/ssl-budget/dataset.yaml --model models/representation/simsiam-r50/smoke.yaml
```

`pretrain` prints JSON with `artifact_dir`, `checkpoint`, `dataset_fingerprint`, and namespaced training metrics. Use the printed pretraining directory below:

```text
wvision compare-pretraining --data .wvision/ssl-budget/dataset.yaml --pretraining-run runs/pretrain-<id> --model models/detection/faster-rcnn-r50/transfer-smoke.yaml
```

The comparison prints `random_run`, `pretrained_run`, and `report` paths. Both detector checkpoints are selected on validation COCO AP before either test evaluation starts. `comparison.json` records label counts, source pretraining identity, initialization policies, both validation/test metric tables, deltas, configuration, and single-seed/compute limitations. Undefined COCO values (`-1`) have a null delta.

For a single detector instead of a comparison:

```text
wvision train --model models/detection/faster-rcnn-r50/transfer-smoke.yaml --data .wvision/ssl-budget/dataset.yaml --pretraining-run runs/pretrain-<id>
wvision val --run runs/train-<id>
wvision test --run runs/train-<id>
wvision predict --run runs/train-<id> --source .wvision/ssl-budget/test/images
```

`--backend-python` is supported on these commands and overrides the environment variable. Original run paths contain absolute references; use [portable bundles](TRAINING_OPERATIONS.md#portable-bundles) for relocation.

## Python

```python
from wvision import Vision, pretrain, compare_pretraining

representation = pretrain(
    ".wvision/ssl-budget/dataset.yaml",
    backend_python=".wvision/mmdet-env/Scripts/python.exe",
    model="models/representation/simsiam-r50/smoke.yaml",
)
comparison = compare_pretraining(
    ".wvision/ssl-budget/dataset.yaml",
    pretraining_run=representation.artifact_dir,
    backend_python=".wvision/mmdet-env/Scripts/python.exe",
    model="models/detection/faster-rcnn-r50/transfer-smoke.yaml",
)
detector = Vision.from_run(comparison.pretrained_run)
predictions = detector.predict(".wvision/ssl-budget/test/images")
```

Use a `main()` guard on Windows; [the complete example](../examples/self_supervised.py) includes one. A pretraining run is a representation artifact and cannot be loaded with `Vision.from_run` for detection. Pass it as `pretraining_run=` to `Vision.train` instead.

## Configuration and transfer contract

Pretraining fields: `schema_version` (1), `max_iters`, `imgsz`, `batch`, `seed`, `device` (`cpu`, `cuda`, or `cuda:N`), `precision`, `devices`, `checkpoint_interval`, and `lr`. Unknown settings fail. Image size is at least 64 and divisible by 32; batch is at least two because SimSiam uses batch normalization. The unlabeled pool must contain at least one batch. The registered model is `mmpretrain/simsiam-r50`; other pretraining models are rejected.

The fixed installed recipe is `simsiam_resnet50_8xb32-coslr-200e_in1k.py`. Adaptations are single-device BatchNorm rather than distributed SyncBN, the requested crop size, an iteration budget, a constant learning rate with the upstream SimSiam predictor hook, an infinite sampler, zero loader workers, and a local image-only dataset. Random initialization uses no external pretraining. The final iteration checkpoint is selected without validation or test evaluation. The two-iteration, 64-pixel smoke settings verify execution and are not a recommended accuracy recipe.

Downstream transfer requires `backend_options.backbone_style=pytorch`. Both `learning=supervised` and `learning=semi_supervised` are supported; the comparison helper remains supervised. For Soft Teacher, pass `pretraining_run=` to `Vision("models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml").train(...)`. Both comparison arms use that style and RGB ImageNet normalization matching the upstream representation recipe. They use the same labeled samples, seed, detector architecture, augmentation, batch, image size, learning rate, and iteration budget. Detection heads initialize randomly. Same seed does not guarantee identical head tensors across different backbone initialization paths. Frozen stages and normalization behavior remain in the saved detector recipe. The earlier M2 comparison defaults to Caffe-style ResNet and is a different protocol.

Only a completed WVision SimSiam run on the exact same dataset fingerprint is accepted. Changing dataset membership or labels requires a new representation run in this initial contract. Checkpoint hashes are checked in the core and again before backend initialization. The adapter checks every backbone key, shape, and dtype, delegates prefix loading to MMEngine, and verifies tensor equality before training. No projection or predictor head is transferred. `transfer.json` records the result. Checkpoints use upstream Python deserialization; only use trusted local runs.

## Data isolation and artifacts

The core checks all declared splits for integrity, including held-out annotations, but only copies `train_unlabeled` images into pretraining. Labels and held-out paths are absent from the worker request. The audit-only `dataset.json` records split fingerprints; it is not supplied to the trainer. Unlabeled label files, if they exist beside source images, are ignored and never staged. Pretraining excludes `train_labeled` images deliberately, so its pool is unambiguous.

Pretraining saves the normal config/environment/status/dataset artifacts, an image-only list, resolved upstream Python and JSON configurations, recipe and inherited-file hashes, backend logs/environment, training scalars, and checkpoint SHA-256. Metrics are training loss and execution statistics, not detector AP. Negative SimSiam cosine loss is valid. No validation-based “best” representation checkpoint is invented.

Downstream runs retain source representation provenance and a complete detector checkpoint. Reloading that detector for evaluation/prediction does not require the pretraining checkpoint. Any new transfer does require the original intact representation artifact.

Equal downstream budgets do not account for additional SimSiam compute. Training-call wall times include startup, data checks, I/O, validation, and checkpointing; they are not latency benchmarks. Single-seed synthetic results establish execution only. See [M3 verification](verification/SIMSIAM.md) for measured results and actual checks.
