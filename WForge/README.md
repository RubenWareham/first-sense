# WForge

**Reproducible computer vision workflows with fewer labeled images.**

WForge is First Sense's training component, using the `wvision` package
and `Vision` API. It brings existing models into a consistent Python API and
command-line interface. Train detectors, learn representations from unlabeled
images, compare learning methods, and evaluate selected checkpoints with explicit data splits.

```python
from wvision import Vision

model = Vision("models/detection/yolo11n/smoke.yaml")
run = model.train("datasets/examples/tiny-detection.yaml")
```

WVision integrates upstream models and training methods. It provides configuration,
dataset validation, experiment tracking, and normalized results.

## Supported workflows

| Model / method | Workflow | Guide |
| --- | --- | --- |
| YOLO11n | Supervised detection; fine-tuning from a trusted local checkpoint | [Detection](docs/DEVELOPMENT.md) |
| Faster R-CNN R50-FPN + Soft Teacher | Supervised and semi-supervised detection using labeled and unlabeled images | [Semi-supervised detection](docs/SEMI_SUPERVISED.md) |
| SimSiam ResNet-50 | Self-supervised pretraining; backbone transfer into Faster R-CNN or Soft Teacher | [Self-supervised learning](docs/SELF_SUPERVISED.md) |
| DINOv3 ConvNeXt Tiny + DETR | Supervised detection with random or explicitly supplied local backbone weights | [DINOv3](docs/DINOV3.md) |

**Verified on Windows: CPU and single-NVIDIA-GPU workflows for all listed models.**
The main application uses Python 3.12. YOLO runs in that process; DINOv3 uses an
isolated Python 3.12 backend. Faster R-CNN, Soft Teacher, and SimSiam use separate
Python 3.10 backends. Each CUDA environment has its own verified torch build;
see [GPU setup](docs/GPU.md). Optimizer-state recovery, explicit mixed precision,
portable bundles, multi-model/parameter-grid comparison with shared COCO validation,
and YOLO ONNX export are described in [training operations](docs/TRAINING_OPERATIONS.md).
The managed DDP launcher still needs actual two-GPU verification. Fully unsupervised
detection, pose, and classification remain unsupported. Official
pretrained DINOv3 transfer remains unverified; random and local-fixture execution
have been checked.

## Install

Download or clone First Sense, open PowerShell in **`WForge/`** (the directory
containing `pyproject.toml`), and install **64-bit Python 3.12**. The following installs the verified YOLO CPU backend:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
python -m pip install -c constraints/cpu-py312.txt -e ".[ultralytics]"
python -m wvision --help
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` instead of `python`
in subsequent commands. Run the examples from `WForge/`: model profiles
and dataset templates are included in the source checkout.

For the lightweight application without an ML backend, use
`python -m pip install -e .` in your Python 3.12 environment. Training additionally
requires its backend. Install [Soft Teacher / SimSiam](docs/SEMI_SUPERVISED.md) or
[DINOv3](docs/DINOV3.md) in the separate environments described in their guides.
There is no automatic backend installation or weight download during training.

## Quick start

Create a small synthetic dataset, then run a one-epoch detection smoke test:

```powershell
python examples/make_tiny_dataset.py .wvision/tiny
python -m wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml
```

The command prints JSON containing `artifact_dir`, checkpoint paths, and metrics.
Use the printed training directory for evaluation and prediction:

```powershell
python -m wvision val --run runs/train-<id>
python -m wvision test --run runs/train-<id>
python -m wvision predict --run runs/train-<id> --source .wvision/tiny/images/test-0.png
```

Replace `runs/train-<id>` with the actual `artifact_dir`. This synthetic example
verifies the workflow; one epoch from random initialization is not an accuracy
benchmark. Validation selects the checkpoint; the test split evaluates it afterward.

### Python

After generating the example data, save this script in `WForge/`:

```python
from wvision import Vision


def main():
    model = Vision("models/detection/yolo11n/smoke.yaml")
    run = model.train("datasets/examples/tiny-detection.yaml")
    print(model.val().metrics)
    print(model.test().metrics)
    print(model.predict(".wvision/tiny/images/test-0.png").predictions)

    restored = Vision.from_run(run.artifact_dir)
    print(restored.model)


if __name__ == "__main__":
    main()
```

Keep the `main()` guard when running Windows training scripts.
[More examples](examples/) cover detection, pretraining, semi-supervised comparisons,
and initialized Soft Teacher.

## Use your own dataset

Copy a [dataset template](datasets/templates/), set its root and class names, and
provide separate training, validation, and test image lists:

```yaml
schema_version: 1
task: detect
format: yolo
root: ../data/my-dataset
classes:
  0: person
  1: bicycle
splits:
  train_labeled: splits/train_labeled.txt
  val: splits/val.txt
  test: splits/test.txt
```

Paths in split lists are relative to `root`, for example `images/train/photo.jpg`.
Annotations use the corresponding `labels/train/photo.txt` path, with one normalized
YOLO box per line: `class_id center_x center_y width height`. Empty annotation files
represent verified negative images. Missing annotations in labeled splits fail
validation. Add `train_unlabeled` for semi-supervised training or pretraining.
See the [dataset guide](datasets/README.md) for preparation and split requirements.

Select a model profile and override the settings you need:

```python
model = Vision("models/detection/yolo11n/supervised.yaml")
run = model.train("datasets/my-dataset.yaml", epochs=50, imgsz=640, batch=4)
```

Profiles are starting configurations, not benchmark-tuned accuracy promises.
Tune settings on validation data and keep test data out of that process.

## GPU training

Follow the [CUDA setup](docs/GPU.md) for your model, then select a single GPU
explicitly. For example, using YOLO's CUDA application environment:

```powershell
.wvision/yolo-cuda-env/Scripts/python -m wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml --device cuda:0
```

The saved device is reused for validation, test, and prediction. Override it with
`--device cpu` when needed. CUDA requests fail if the GPU/backend is unavailable;
WVision never silently changes to CPU. See the [GPU example](examples/gpu_semi_supervised.py)
for SimSiam pretraining followed by semi-supervised transfer.

## Run and compare experiments

After preparing data and the selected backend environments:

```powershell
python -m pip install -e ".[evaluation]"
python -m wvision experiments --plan examples/experiments.yaml
python -m wvision resume --run runs/train-<id> --epochs 50
python -m wvision bundle --run runs/train-<id> --destination bundles/detector
```

Plans support multiple models, parameter grids, explicit interpreters, and existing
runs on the same dataset. Reports include a sortable HTML table, common COCO
validation metrics, a chart, CSV, configurations, and seed aggregates. Test evaluation
stays separate. See [training operations](docs/TRAINING_OPERATIONS.md) for mixed
precision, recovery, optional portable optimizer state/data, ONNX, and backend CI.

## Learn from unlabeled images

After configuring the [OpenMMLab backend](docs/SEMI_SUPERVISED.md), prepare a
reproducible label budget from a fully annotated training pool:

```powershell
python -m wvision prepare-budget --data datasets/examples/tiny-detection.yaml --destination .wvision/tiny-budget --labeled-count 2 --seed 0
python -m wvision compare --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml
```

The comparison trains a supervised baseline and Soft Teacher on the same labeled
subset, evaluates both, and writes `comparison.json`. Withheld training annotations
are excluded from the derived unlabeled pool. Equal iteration counts do not imply
equal compute; the report records this distinction.

Other supported paths:

- [Pretrain SimSiam](docs/SELF_SUPERVISED.md) on `train_unlabeled`, then transfer its
  backbone into a detector.
- [Initialize Soft Teacher](docs/WEIGHT_COMPATIBILITY.md) from a completed supervised
  Faster R-CNN run or a compatible SimSiam backbone.
- [Load a local DINOv3 backbone](docs/DINOV3.md) for downstream supervised detection.

Weight transfer is explicit and checked for compatibility. A shared checkpoint
format does not make every model interchangeable. Use `python -m wvision models`
to inspect supported model IDs, operations, learning modes, and initialization rules.

## Results and reproducibility

Every operation creates its own directory under `runs/`. Training records resolved
settings, dataset identity, environment versions, source provenance, checkpoint
checksums, metrics, and status. Backend logs and upstream outputs are retained.
Predictions are JSONL records with image dimensions, pixel `xyxy` boxes, scores,
class IDs, and the class map.

Data checks reject overlapping samples, identical decoded images, invalid boxes,
and inconsistent annotations. Validation and test stay separate. Run-local data
copies protect source files from backend caches.

## Documentation

- [Installation and detection reference](docs/DEVELOPMENT.md)
- [Model profiles](models/README.md) and [dataset setup](datasets/README.md)
- [Semi-supervised detection](docs/SEMI_SUPERVISED.md)
- [Self-supervised pretraining](docs/SELF_SUPERVISED.md)
- [DINOv3 detection](docs/DINOV3.md)
- [GPU setup and execution](docs/GPU.md)
- [Training operations, experiment comparison, bundles, and export](docs/TRAINING_OPERATIONS.md)
- [Weight compatibility](docs/WEIGHT_COMPATIBILITY.md)
- [Architecture and code navigation](docs/ARCHITECTURE.md)
- [Roadmap](docs/MILESTONES.md) and [API migration notes](docs/MIGRATION.md)

## Contributing

Bug reports, documentation improvements, and verified upstream integrations are
welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and checks.

## License

WVision's original code is licensed under [Apache-2.0](LICENSE). Dependencies,
datasets, and model weights retain their own licenses. In particular, the optional
Ultralytics integration is subject to its upstream AGPL-3.0 or enterprise terms;
WVision's license does not replace those terms. See [third-party notices](THIRD_PARTY_NOTICES.md).

Historical backend/publication checks are grouped in the [verification index](docs/verification/README.md).
