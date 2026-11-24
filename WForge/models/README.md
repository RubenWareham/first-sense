# Model configs

Select a registered model and its training defaults with one readable YAML file:

```python
from wvision import Vision

model = Vision("models/detection/yolo11n/smoke.yaml")
run = model.train("datasets/examples/tiny-detection.yaml")
```

CLI equivalent:

```text
wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml
```

Prepare the example dataset explicitly using the [dataset catalog](../datasets/README.md). Paths are relative to the working directory when selecting a model file; its content contains no dataset or machine-specific paths. `Vision(Path(...))` also works. Registered IDs such as `Vision("ultralytics/yolo11n")` remain supported.

## Python model code versus YAML profiles

This directory holds reusable YAML documents. Every registered integration
has its Python code under `src/wvision/models/<framework>/<model>/`.
For example, YOLO11's Python integration lives
separately in [`src/wvision/models/ultralytics/yolo11/`](../src/wvision/models/ultralytics/yolo11/):
`adapter.py` connects the public workflow, `config.py` validates settings, and
`operations.py` invokes existing upstream training/evaluation/prediction. Shared
Ultralytics runtime utilities remain under `src/wvision/adapters/ultralytics/`.
The existing YAML paths and `Vision("ultralytics/yolo11n")` remain unchanged.

## Catalog

| Model | Configs | Purpose |
| --- | --- | --- |
| YOLO11n | [supervised](detection/yolo11n/supervised.yaml), [smoke](detection/yolo11n/smoke.yaml) | Supervised defaults or one-epoch execution check |
| YOLO11n-pose | [smoke](pose/yolo11n-pose/smoke.yaml) | One-epoch supervised pose execution check using the dataset's skeleton |
| RTMPose-t | [smoke](pose/rtmpose-t/smoke.yaml) | One-epoch top-down COCO-17 or Halpe-26 pose check |
| Faster R-CNN R50-FPN | [supervised](detection/faster-rcnn-r50/supervised.yaml), [supervised smoke](detection/faster-rcnn-r50/supervised-smoke.yaml) | Existing supervised detector |
| Soft Teacher / Faster R-CNN | [semi-supervised](detection/faster-rcnn-r50/soft-teacher.yaml), [smoke](detection/faster-rcnn-r50/soft-teacher-smoke.yaml), [PyTorch-style smoke](detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml) | Existing labeled/unlabeled method; optional checked detector/backbone initialization |
| Faster R-CNN / SimSiam transfer | [transfer smoke](detection/faster-rcnn-r50/transfer-smoke.yaml) | Supply a completed SimSiam run through `pretraining_run` |
| DINOv3 ConvNeXt Tiny / DETR | [frozen backbone](detection/dinov3-convnext-tiny-detr/frozen-backbone.yaml), [fine-tune](detection/dinov3-convnext-tiny-detr/fine-tune.yaml), [random smoke](detection/dinov3-convnext-tiny-detr/smoke.yaml) | Local backbone initialization or explicit random execution check |
| SimSiam R50 | [pretrain](representation/simsiam-r50/pretrain.yaml), [smoke](representation/simsiam-r50/smoke.yaml) | Existing image-only representation pretraining |

Profiles describe intent:

- `smoke`: small execution checks. Low Soft Teacher thresholds deliberately exercise unlabeled losses and are not accuracy recommendations.
- `baseline`: starting defaults for the implemented adapter. Longer schedules have not been benchmarked on your dataset.
- `local_weights`: requires an explicit local backbone argument. Official pretrained DINOv3 verification remains pending.

None of these profiles is labeled state of the art or benchmark-tuned. Upstream references identify the existing implementation; WVision's CPU schedules and other adaptations remain documented in the backend guides. There is no fallback from unsupported methods or missing weights to another model or learning mode.

## Schema and precedence

```yaml
schema_version: 1
model: ultralytics/yolo11n
task: detect
profile: smoke
description: One-epoch execution check, not an accuracy recipe.
upstream:
  repository: https://github.com/ultralytics/ultralytics
  version: 8.3.4
  reference: cfg/models/11/yolo11.yaml
  license: AGPL-3.0 or upstream enterprise terms
training:
  epochs: 1
  imgsz: 128
  batch: 2
  backend_options:
    mosaic: 0.0
```

`model` must be an exact registered ID and `task` must match its catalog entry. `training` uses that adapter's existing validated settings. Metadata references are recorded, not downloaded or executed. Imports remain independent of optional ML packages. Runtime adapter/version checks remain authoritative; descriptive metadata cannot select code or change the backend.

Resolution order: adapter defaults, model-file `training`, explicit call/CLI overrides. Each field replaces its earlier value. `backend_options` is replaced as a whole mapping. Shared profiles live only in this catalog; flat settings overlays are not accepted.

```python
model = Vision("models/detection/yolo11n/supervised.yaml")
run = model.train("datasets/my-dataset.yaml", epochs=50, batch=4)
```

Construction validates the document and adapter defaults without loading a model. Changes to the source YAML after construction are rejected before training; construct a new `Vision` instance to use edited defaults. Training saves `model-definition.json` with the source path, SHA-256, and parsed document. `run.json` also records it alongside the actual resolved settings. `Vision.from_run` uses the recorded model ID, checkpoint, and resolved settings and does not need the original YAML to remain present.

Datasets, checkpoints, backbone directories, output locations, and backend interpreters are explicit separate inputs. Initialization uses `checkpoint`, `pretraining_run`, `detector_run`, or `backbone_weights`, subject to each adapter's explicit rules; choose at most one. See [weight compatibility](../docs/WEIGHT_COMPATIBILITY.md). They are not embedded in shared model configs. Class counts and names come from the dataset manifest. No arbitrary config inheritance or executable YAML is supported.

## Pretraining and comparisons

Representation models retain their distinct public operation:

```python
from wvision import pretrain

representation = pretrain(
    "datasets/examples/tiny-semi-supervised.yaml",
    model="models/representation/simsiam-r50/smoke.yaml",
    backend_python=".wvision/mmdet-env/Scripts/python.exe",
)
```

CLI:

```text
wvision pretrain --model models/representation/simsiam-r50/smoke.yaml --data datasets/examples/tiny-semi-supervised.yaml
wvision compare --data datasets/examples/tiny-semi-supervised.yaml --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml
wvision compare-pretraining --data datasets/examples/tiny-semi-supervised.yaml --pretraining-run runs/pretrain-<id> --model models/detection/faster-rcnn-r50/transfer-smoke.yaml
```

Comparisons accept `model=` / `--model` with a registered ID or a model document for the expected detector. They preserve the model-definition snapshot and existing safeguards. Their supervised arms override learning mode/options explicitly; the saved resolved settings describe the actual run.

See backend setup and limits in [development](../docs/DEVELOPMENT.md), [semi-supervised detection](../docs/SEMI_SUPERVISED.md), [self-supervised pretraining](../docs/SELF_SUPERVISED.md), and [DINOv3](../docs/DINOV3.md). The [adapter strategy](../docs/MODEL_ADAPTERS.md) explains how to add a genuinely supported model before adding its configs. [Catalog verification](../docs/verification/CATALOGS.md) records actual public workflows and checks. These source-repository catalogs are not currently bundled as package resources.
