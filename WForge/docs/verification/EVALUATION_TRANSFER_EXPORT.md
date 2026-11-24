# Grouped evaluation, flexible transfer, and DINOv3 ONNX

Subsequent [MMDetection ONNX verification](MMDET_ONNX.md) adds Faster R-CNN and
Soft Teacher export; the unsupported-export statement below records the earlier
verification state.

Verified in the First Sense working tree on **2026-11-18**, Windows, Python
3.12.3 application and Python 3.10.20 OpenMMLab backend. One NVIDIA GeForce RTX
4050 Laptop GPU (6,141 MiB). No commits, pushes, external dataset uploads, source
annotation edits, or automatic weight downloads were performed.

## Implemented behavior

- `prepare_group_split` / `wvision prepare-groups` copies explicit match/video/
  subject assignments into a new dataset and retains the source mapping.
- Optional `groups` manifests require complete assignments and reject groups
  crossing train/validation/test. Training labeled/unlabeled pools may share groups.
- Group metadata participates in fingerprints and survives label budgets and
  relocated dataset-containing bundles. Existing manifests without groups keep
  their earlier fingerprint and are marked unverified for group isolation.
- Experiment plans can require groups. Shared COCO evaluation includes per-class
  AP50:95/AP50, area metrics, object counts, and existing seed mean/sample standard
  deviation. `val` and `test` expose the same explicit `coco` protocol independently.
- Completed supervised Faster R-CNN sources can initialize compatible supervised
  detectors or both Soft Teacher branches across datasets. Complete detector
  transfer requires identical classes; explicit backbone transfer permits new
  target classes and leaves the remaining modules newly initialized.
- SimSiam backbone transfer supports another target dataset/class set. Transfer
  checks source/target decoded content, explicit groups when available, and recorded
  initialization exposure. Source detector validation exposure is included in
  cross-dataset checks. Unknown near-duplicates/external pretraining remain a limit.
- DINOv3/DETR ONNX export verifies raw logits and normalized boxes against native
  FP32 CPU outputs. YOLO export is retained. MMDetection/MMDeploy export remains
  a separate, unsupported adapter rather than an advertised capability.

See [dataset groups](../../datasets/README.md#group-separated-evaluation),
[transfer rules](../WEIGHT_COMPATIBILITY.md), and
[training/export usage](../TRAINING_OPERATIONS.md).

## Checks

Core regression command: `python -m pytest -q` — **183 passed**, 33 optional
integration cases deselected. Ruff lint/format and mypy pass for 62 source modules.
Logs are component-local ignored files under `.wvision/`.

Real backend checks:

| Evidence | Behavior | Result |
| --- | --- | --- |
| `group-evaluation-tests.log` | Grouped exact-ground-truth COCO val/test, per-class AP=1, rejection of non-split predictions | 2 passed |
| `cross-dataset-final.log` | Full detector across datasets, new-class backbone transfer into supervised and both Soft Teacher branches, cross-dataset SimSiam, CPU reload, unchanged source hashes | 1 passed |
| `dino-export-integration.log` | Actual DINOv3 CPU train/resume/relocation/ONNX verification | 1 passed |
| `grouped-workflow.log` | Public group CLI followed by two real YOLO seed runs; verified grouping/per-class reporting/seed aggregation; test not evaluated | Passed |

The OpenMMLab check uses torch 2.1.0+cu121, MMCV 2.1.0, MMEngine 0.10.7,
MMDetection 3.3.0, and MMPretrain 1.2.0. Initialization audits compare 348 complete
detector tensors or 318 backbone tensors before the first training step. The
SimSiam case uses the existing two-step smoke profile with learning rate .0001.
An initial test with the full-training default learning rate was unstable; its
logs remain in `cross-dataset-tests.log`. A run without a validation-selected
checkpoint now reports an explicit error instead of attempting to construct a
path from a missing value. No checkpoint-selection fallback was introduced.

Core regressions also cover incomplete/overlapping group mappings, unlabeled
held-out rejection, changed classes, incompatible source state, direct and
recorded-ancestor held-out overlap, unverified-group experiment rejection, and
preservation of group fingerprints after budget/bundle relocation.

## Actual football artifacts

The existing 87 labeled / 785 unlabeled / 292 validation / 293 test dataset was
unchanged. Its fingerprint remains:

```text
ac8242b5c0a7b9a2b2b1fcdf2f09b3d155fd02aa44d71cf70c10a397d8ee2b48
```

New shared-protocol val/test reports for the existing YOLO checkpoint are in
`football-coco-val.json`, `football-coco-test.json`, and the corresponding logs.
Their directories under `reliable-football-evaluation/` contain `per-class.json`,
including Ball, Player, and Referee AP and ground-truth counts. **Group isolation
is explicitly unverified**: the current dataset lacks trusted match/video IDs and
has the previously documented video overlap. Source data was not regrouped by
guessing filenames. Accurate groups and useful multi-seed training budgets are
still needed before a real generalization/quality comparison.

The football-trained DINOv3 checkpoint exported with torch 2.8.0+cu128,
Transformers 5.22.2, ONNX 1.17.0, and ONNX Runtime 1.20.1. Local evidence:
`dino-onnx-checks/export-478f3f84345a49568847b0af09ae11ce/export.json`.
At batch one, 64 pixels, FP32 CPU, opset 17, two deterministic input tensors pass
`rtol=.001`, `atol=.0001`. Maximum absolute errors are **1.430511474609375e-6**
for logits and **5.960464477539063e-8** for boxes. Fixed-shape tracing is deliberate;
dynamic shapes, variable padding masks, and deployment latency are not verified.

## Packaging and remaining gates

The wheel/sdist contain the new group/transfer modules, grouped example plan,
and documentation, excluding data, weights, environments, and generated outputs.
The extracted wheel imports the framework-free catalog/API, runs the CLI, reloads
representations, and predicts through the existing isolated detector backends.
The backend workflow installs DINOv3 export dependencies and retains per-class
and group-selection reports. Local syntax checks do not establish hosted success.

Actual two-GPU execution, hosted backend automation, official licensed DINOv3
weight acceptance, and a verified MMDeploy export adapter remain separate gates.
No Pose, classification, TensorRT/C++, or real-time runtime capability is added.
