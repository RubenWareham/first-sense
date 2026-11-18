# First Sense

First Sense brings together computer vision training and future local/edge runtime
work. **WForge** contains the existing WVision training project; its Python API
and CLI retain the `wvision` name.

## Install

Use Python 3.12. From this directory:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ./WForge
.venv/Scripts/python -m wvision --help
.venv/Scripts/python -m wvision models
```

For development, install `"./WForge[dev]"`. The core installation does not install
an ML backend. Follow the [backend setup](WForge/docs/DEVELOPMENT.md) or
[GPU setup](WForge/docs/GPU.md) for the selected model. Separate environments
remain necessary for the tested upstream stacks.

## Use the existing API

From `WForge/`, after backend setup and dataset preparation:

```python
from wvision import Vision

model = Vision("models/detection/yolo11n/supervised.yaml")
run = model.train("datasets/my-dataset.yaml", epochs=100, device="cpu")
validation = model.val()
test = model.test()
predictions = model.predict("path/to/image.jpg")
restored = Vision.from_run(run.artifact_dir)
```

The dataset and image paths are supplied by you. See the
[training README](WForge/README.md) for an explicitly generated smoke dataset,
supported model profiles, CLI commands, and limitations. YOLO uses `epochs`;
OpenMMLab and DINOv3 use `max_iters`.

## Components

| Directory | Responsibility |
| --- | --- |
| [WForge](WForge/README.md) | Training, validation, testing, prediction, learning methods, and experiment records through WVision |

The imported training project supports YOLO11n, Faster R-CNN/Soft Teacher,
SimSiam, and DINOv3 ConvNeXt Tiny/DETR for their documented CPU/single-GPU
workflows. Official pretrained DINOv3 verification remains pending. Pose,
classification, runtime integration, and fully unsupervised detection are
future capabilities; see the [milestones](WForge/docs/MILESTONES.md).

## Development checks

After installing the development extra, run from `WForge/`:

```powershell
python -m pytest -q
python -m ruff check src examples tests
python -m ruff format --check src examples tests
python -m mypy
python -m build
```

Default tests run without ML backends. Real upstream/GPU tests require their
documented environments and explicit markers. GitHub Actions runs the component's
core checks from the First Sense root workflow.

Current source layout and supported model combinations are documented in the
[architecture map](WForge/docs/ARCHITECTURE.md). The repository contains only the
implemented WForge component; future workspace/runtime components remain roadmap
items, with no placeholder directories.

WForge adds optimizer-state resume, explicit mixed precision, parameter-grid/model
comparisons with common COCO validation, portable bundles, and verified YOLO ONNX
export. See [training operations](WForge/docs/TRAINING_OPERATIONS.md). Actual
two-GPU training remains unverified on the one-GPU development machine.

Original runs retain absolute references; use portable bundles when moving runs,
with optional dataset and optimizer-state copies.

First Sense's original code uses [Apache-2.0](LICENSE); WForge retains the same
license inside its buildable package.
Dependencies and weights retain their [upstream terms](WForge/THIRD_PARTY_NOTICES.md).
See [the project mission](PROJECT_MISSION.md) for scope and direction.

Find implementation owners in the [code structure and navigation map](WForge/docs/ARCHITECTURE.md#finding-and-changing-code).

## Contributing and publication

See [contribution setup](WForge/CONTRIBUTING.md), the
[first-commit guide](WForge/docs/PUBLISHING.md), and the
[verification index](WForge/docs/verification/README.md). Raw/local datasets,
generated runs, environments, weights, and editor state are excluded by the
repository-root `.gitignore`.
