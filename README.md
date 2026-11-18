# First Sense

**Giving AI its primary sense. Effortlessly.**

First Sense stands for making computer vision accessible: helping people give
their applications the ability to see, understand, and eventually react to the
world in real time. It brings model training, learning from unlabeled images,
and experiment comparison into one workflow.

## Why First Sense?

Trying a different model often means learning another framework, rewriting
configuration, and collecting results by hand. Labeling images takes time, and
comparing learning methods becomes harder when every experiment handles data and
evaluation differently.

First Sense aims to make that work easier through **one interface to train,
refine, and compare computer vision models**. Choose a supported model, reuse
your dataset, run several experiments, and inspect their results together. Spend
more time improving your data and testing ideas, with the settings and evidence
needed to understand what changed.

The training component is **[WForge](WForge/README.md)**. Its Python package and
command-line tool are named `wvision`; the main detection API is `Vision`.
WForge connects existing models and training methods through shared configuration,
data checks, experiment records, and evaluation. Training runs through the
underlying frameworks.

## What you can do today

- **Train through one interface:** use a common Python API or CLI across supported
  model families, with explicit model-specific settings.
- **Run and compare experiments:** train multiple models on the same dataset, or
  vary one model's parameters and seeds. Inspect shared validation metrics,
  configurations, and checkpoints in HTML, CSV, and JSON reports.
- **Learn from fewer labels:** use Soft Teacher for semi-supervised detection, or
  pretrain SimSiam on unlabeled images and transfer its backbone into a detector.
  Compare these workflows against supervised or random-initialization baselines.
- **Continue and move your work:** use checkpoint recovery, mixed precision on
  supported GPUs, portable run bundles, and verified YOLO ONNX export.

| Model / method | Current workflow |
| --- | --- |
| YOLO11n | Supervised object detection and local-checkpoint fine-tuning |
| Faster R-CNN / Soft Teacher | Supervised and semi-supervised object detection |
| SimSiam ResNet-50 | Self-supervised representation pretraining and checked detector transfer |
| DINOv3 ConvNeXt Tiny / DETR | Supervised detection with random or supplied local backbone weights |

Supported workflows have been checked on Windows with CPU and a single NVIDIA
GPU, using separate compatible backend environments. See the
[model and environment details](WForge/README.md#supported-workflows).

## Where First Sense is heading

The longer-term goal is to carry selected trained models into **real-time
applications**, including local and edge devices. That means optimizing and
measuring accuracy, memory use, and latency on the hardware where a model will
run, alongside making training easier.

Current work provides training, evaluation, portable artifacts, and YOLO ONNX
export. Real-time runtime integration and deployment optimization are planned,
along with pose estimation and classification. Follow the
[roadmap](WForge/docs/MILESTONES.md) and [project mission](PROJECT_MISSION.md).

## Get started

### 1. Install a training backend

For the verified Windows CPU example, use **64-bit Python 3.12**. Open PowerShell
in this repository and run:

```powershell
cd WForge
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
python -m pip install -c constraints/cpu-py312.txt -e ".[ultralytics,evaluation]"
python -m wvision models
```

If activation is blocked, use `.venv\Scripts\python.exe` instead of `python`.
**Run all examples below from `WForge/`.** For CUDA, follow the
[GPU setup](WForge/docs/GPU.md). OpenMMLab and DINOv3 each have their own
[backend setup guides](WForge/README.md#supported-workflows).

### 2. Train and evaluate a model

Generate a tiny synthetic dataset and run a one-epoch smoke test:

```powershell
python examples/make_tiny_dataset.py .wvision/tiny
python -m wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml
```

The command prints the run's `artifact_dir`, checkpoint paths, and metrics.
Replace `runs/train-<id>` below with that actual directory:

```powershell
python -m wvision val --run runs/train-<id>
python -m wvision test --run runs/train-<id>
python -m wvision predict --run runs/train-<id> --source .wvision/tiny/images/test-0.png
```

Use validation data to select models and parameters, then evaluate the chosen
checkpoint on the held-out test split. This tiny example checks execution;
use your own [dataset manifest](WForge/datasets/README.md) and an appropriate
training schedule for useful results.

The same workflow is available in Python. After generating the example data:

```python
from wvision import Vision


def main():
    model = Vision("models/detection/yolo11n/smoke.yaml")
    run = model.train("datasets/examples/tiny-detection.yaml")
    print(model.val().metrics)
    print(model.test().metrics)
    print(run.artifact_dir)


if __name__ == "__main__":
    main()
```

### 3. Run several trainings and compare them

Save this as `experiments.yaml` inside `WForge/`:

```yaml
schema_version: 1
data: datasets/examples/tiny-detection.yaml
experiments:
  - name: yolo-learning-rates
    model: models/detection/yolo11n/smoke.yaml
    training:
      backend_options:
        optimizer: SGD
        mosaic: 0.0
    grid:
      backend_options.lr0: [0.001, 0.01]
      seed: [0, 1]
```

```powershell
python -m wvision experiments --plan experiments.yaml
```

This runs four experiments sequentially: two learning rates across two seeds.
Open the reported `comparison.html` to compare validation results and settings;
JSON and CSV reports are saved beside it. Test evaluation stays separate.

Add entries for other supported models to compare model families on the same
dataset. The [multi-model example](WForge/examples/experiments.yaml) includes
Faster R-CNN and requires its OpenMMLab environment. See
[experiment plans](WForge/docs/TRAINING_OPERATIONS.md#multiple-models-and-parameter-grids)
for backend selection, initialization, and grid options.

### 4. Use unlabeled images

WForge supports both **semi-supervised learning** (labeled and unlabeled images
used together) and **self-supervised learning** (representation pretraining on
unlabeled images before detector training). Each method has explicit compatible
models and transfer paths.

After the [OpenMMLab setup](WForge/docs/SEMI_SUPERVISED.md), compare supervised
Faster R-CNN with Soft Teacher on a reproducible label budget:

```powershell
python -m wvision prepare-budget --data datasets/examples/tiny-detection.yaml --destination .wvision/tiny-budget --labeled-count 2 --seed 0
python -m wvision compare --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml
```

For SimSiam, follow the [self-supervised guide](WForge/docs/SELF_SUPERVISED.md)
to pretrain, transfer the backbone, and compare against random initialization.
The [GPU example](WForge/examples/gpu_semi_supervised.py) combines SimSiam
pretraining with Soft Teacher training.

### 5. Resume, bundle, or export a run

```powershell
python -m wvision resume --run runs/train-<id> --epochs 50
python -m wvision bundle --run runs/train-<id> --destination bundles/detector
python -m pip install -e ".[onnx]"
python -m wvision export --run runs/train-<id> --format onnx
```

For YOLO, `--epochs 50` means 50 total epochs. OpenMMLab and DINOv3 use
`--max-iters`. ONNX export currently covers YOLO; native bundles support the
other integrated families. See [training operations](WForge/docs/TRAINING_OPERATIONS.md)
for precision settings, portable data/optimizer state, and recovery requirements.

## Status and further reading

First Sense is an alpha project. Actual two-GPU execution, hosted backend
automation, and official pretrained DINOv3 acceptance remain verification gates.
Real-time deployment, pose, classification, and fully unsupervised detection
remain roadmap work. Current evidence is in the
[verification index](WForge/docs/verification/README.md).

- [WForge usage and supported workflows](WForge/README.md)
- [Dataset setup and split requirements](WForge/datasets/README.md)
- [Training operations and experiment comparison](WForge/docs/TRAINING_OPERATIONS.md)
- [Architecture and code navigation](WForge/docs/ARCHITECTURE.md)
- [Development checks](WForge/docs/DEVELOPMENT.md#verification-commands),
  [contributing](WForge/CONTRIBUTING.md), and [publication](WForge/docs/PUBLISHING.md)

Original code is licensed under [Apache-2.0](LICENSE). Dependencies and weights
retain their [upstream terms](WForge/THIRD_PARTY_NOTICES.md). Local datasets,
weights, environments, and generated runs stay outside source control.
