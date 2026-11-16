# Configuration cleanup

The post-M3d cleanup removes the duplicate `configs/` catalog and flat settings-file
loaders. Select a registered model ID or a full model document through `model` /
`--model`. Override individual training settings explicitly. There is no `config=`
or `--config` alias.

## Current entry points

```python
from wvision import Vision, compare_learning, compare_pretraining, pretrain

model = Vision("models/detection/yolo11n/smoke.yaml")
run = model.train("datasets/my-data.yaml", epochs=10, batch=4)

representation = pretrain(
    "datasets/my-budget.yaml",
    model="models/representation/simsiam-r50/smoke.yaml",
)
comparison = compare_pretraining(
    "datasets/my-budget.yaml",
    model="models/detection/faster-rcnn-r50/transfer-smoke.yaml",
    pretraining_run=representation.artifact_dir,
)
semi = compare_learning(
    "datasets/my-budget.yaml",
    model="models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
)
```

Use a Windows main guard in scripts and configure the isolated backend interpreter
as described in the [backend guide](SEMI_SUPERVISED.md). These are smoke profiles;
changing the epoch count alone does not establish useful accuracy.

```text
wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/my-data.yaml --epochs 10
wvision pretrain --model models/representation/simsiam-r50/smoke.yaml --data datasets/my-budget.yaml
wvision compare --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data datasets/my-budget.yaml
wvision compare-pretraining --model models/detection/faster-rcnn-r50/transfer-smoke.yaml --data datasets/my-budget.yaml --pretraining-run runs/pretrain-<id>
```

Settings resolve from adapter defaults, model `training`, then explicit overrides.
`backend_options` replaces the whole earlier mapping. For reusable custom settings,
copy a model document and edit its `training` section. Datasets, source weights,
backend interpreter, and output paths remain separate inputs.

## Removed profile paths

| Removed file under `configs/` | Canonical file under `models/` |
| --- | --- |
| `detect-smoke.yaml` | [YOLO smoke](../models/detection/yolo11n/smoke.yaml) |
| `soft-teacher-smoke.yaml` | [Soft Teacher smoke](../models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml) |
| `transfer-smoke.yaml` | [Detector transfer smoke](../models/detection/faster-rcnn-r50/transfer-smoke.yaml) |
| `simsiam-smoke.yaml` | [SimSiam smoke](../models/representation/simsiam-r50/smoke.yaml) |
| `dinov3-detr-smoke.yaml` | [DINOv3/DETR random smoke](../models/detection/dinov3-convnext-tiny-detr/smoke.yaml) |
| `dinov3-detr-local.yaml` | [DINOv3/DETR frozen backbone](../models/detection/dinov3-convnext-tiny-detr/frozen-backbone.yaml) |

Passing removed arguments or flat YAML as a model document fails before training.
Registered IDs remain the exact backend selectors; they are needed for saved-run
reload and are not deprecated.

## Artifact readers

Detection `config.json` uses `initialization_sources` for provenance, separate
from adapter settings such as DINOv3's `initialization` mode. `run.json` retains
`initialization` for source provenance and nests adapter settings under `config`.
Duplicate `initial_checkpoint`, `initial_checkpoint_sha256`, and top-level
`pretraining` copies were removed. Read `run.json`'s `initialization.checkpoint`,
`initialization.checkpoint_sha256`, or `initialization.pretraining` as appropriate.
Representation/comparison reports retain their own method-specific fields.

Completed schema-1 runs still load through `Vision.from_run`; original model YAML
and removed profile files are not needed for reload. Existing datasets, weights,
run artifacts, and backend environments have not been deleted. Historical
verification reports retain their original commands and counts as dated evidence.

Label-budget metadata belongs at `<dataset root>/budget.json`, where
`prepare_label_budget` writes it. The manifest-directory fallback was removed.
Move metadata there if maintaining a custom older layout; dataset fingerprint
validation remains mandatory. A manifest can still live anywhere.

## First Sense / WForge internal structure

The training project lives under `First Sense/WForge`. Its installed package and
CLI remain `wvision`; keep using `from wvision import Vision, pretrain,
prepare_label_budget, compare_learning, compare_pretraining, list_models`.
Public signatures, model IDs, YAML paths, and run schema are unchanged.

Internal modules are now grouped by responsibility:

| Previous internal location | Current location |
| --- | --- |
| `config`, `contracts`, `model_config`, `registry` | Same module names under `wvision.core` |
| `data` | `wvision.datasets.detection` |
| `budgets` | `wvision.datasets.budgets` |
| `pretraining` | `wvision.training.pretraining` |
| `comparison` | `wvision.evaluation.learning` |
| `transfer` | `wvision.evaluation.pretraining` |
| `artifacts` | `wvision.experiments.artifacts` |
| Flat YOLO integration | `wvision.adapters.ultralytics` |
| Flat MMDetection/SimSiam integration | `wvision.adapters.openmmlab` |
| Flat DINOv3 integration | `wvision.adapters.transformers` |

Update any direct internal imports; compatibility forwarding modules are not
provided for these private paths. Adapter factories now live in each backend's
`adapter.py`. The protocol stays in `adapters/base.py`; device and subprocess helpers live
under `adapters/shared/`. The [architecture map](ARCHITECTURE.md) identifies each owner.

This refactor does not relocate old run artifacts, backend environments, datasets,
or weights. Existing absolute references must remain valid or require a new run.

## YOLO11 model folder

The first model-oriented slice moves the YOLO11 wrapper from
`wvision.adapters.ultralytics.adapter` to
`wvision.models.ultralytics.yolo11.adapter`. Its `TrainConfig` moves from
`wvision.core.config` to `wvision.models.ultralytics.yolo11.config`.
Model loading, training, evaluation, and prediction live in
`wvision.models.ultralytics.yolo11.operations`. Shared Ultralytics dependency,
runtime/device configuration, and metric helpers stay in
`wvision.adapters.ultralytics.backend`. Update direct private imports accordingly;
there are no forwarding modules at the old locations.

Public `Vision` imports, signatures, YOLO11n ID, YAML profiles, and run schema are
unchanged. Validation and test remain separate public operations using the same
model evaluation implementation on distinct staged splits. Other model packages
are not migrated as part of this slice.

## Remaining model folders

All registered model integrations now have a Python home under `wvision.models`:

| Previous private owner | Current owner |
| --- | --- |
| `adapters.openmmlab.adapter` | `models.mmdetection.faster_rcnn_r50.adapter` |
| `adapters.openmmlab.backend.MMDetConfig` / `MODEL_ID` | `models.mmdetection.faster_rcnn_r50.config` |
| `adapters.openmmlab.weights` | `models.mmdetection.faster_rcnn_r50.weights` |
| `adapters.openmmlab.pretraining` | `models.mmpretrain.simsiam_r50.config` |
| `training.pretraining.load_pretraining` | `models.mmpretrain.simsiam_r50.adapter` |
| `adapters.transformers.adapter` | `models.transformers.dinov3_convnext_tiny_detr.adapter` and `.config` |

Each model's isolated worker is now `operations.py` beside its adapter/config.
The shared OpenMMLab backend only selects an explicit interpreter; it no longer
chooses detector versus pretraining workers through a boolean flag. Each model
adapter selects its fixed worker. Runtime preflight/dump helpers live in
`adapters/openmmlab/runtime.py` or `adapters/transformers/runtime.py`.
The common pretraining lifecycle retains the public `pretrain()` operation.
Private old modules are removed without forwarding shims. Public imports, model
IDs, YAML profiles, run schema, learning capabilities, and split rules remain
unchanged. Existing run and backend absolute paths must still remain valid.
