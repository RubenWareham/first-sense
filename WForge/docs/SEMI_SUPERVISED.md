# Semi-supervised detection

WVision integrates **MMDetection 3.3.0 Soft Teacher with Faster R-CNN R50-FPN**, alongside a supervised version of the same detector. Upstream supplies the models, losses, teacher updates, pseudo-label generation, samplers, and training loops. WVision supplies data validation, configuration translation, process isolation, provenance, and comparison reports.

The [model catalog](../models/README.md) provides `models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml` for `Vision(model_config)` and for the comparison command's `--model` argument. Independent [dataset examples](../datasets/README.md) can live outside their data root while preserving label-budget metadata. See [weight compatibility](WEIGHT_COMPATIBILITY.md) for checked initialization from supervised detector runs or SimSiam backbones.

This integration supports Windows CPU and explicit single-NVIDIA-GPU execution. Follow the [CUDA setup](GPU.md) for `device="cuda:0"`; CPU remains the default. External raw pretrained initialization remains unsupported. Mixed precision, optimizer-state recovery, and the unverified managed DDP path are described in [training operations](TRAINING_OPERATIONS.md). A separate [self-supervised workflow](SELF_SUPERVISED.md) supports checked local SimSiam backbone transfer into supervised Faster R-CNN and both Soft Teacher branches. Soft Teacher also accepts a completed supervised detector through `detector_run=` / `--detector-run`, with the same dataset and compatible detector structure. The smoke recipe is an execution check, not a reproduction of published accuracy.

## Separate environment

Keep the Python 3.12 core environment from [development setup](DEVELOPMENT.md). The compiled backend uses a separate Python 3.10 environment: torch 2.1.0+cpu, torchvision 0.16.0+cpu, MMCV 2.1.0, MMEngine 0.10.7, and MMDetection 3.3.0. A prebuilt [upstream MMCV CPU wheel](https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html) avoids compiling operators.

Supply an existing Python 3.10 interpreter in the first command; do not alter another project's environment. PowerShell, from `WForge/` (the package root):

```powershell
$backendBootstrapPython = 'path/to/python3.10.exe'
& $backendBootstrapPython -m venv .wvision/mmdet-env
.wvision/mmdet-env/Scripts/python -m pip install --upgrade pip
.wvision/mmdet-env/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch==2.1.0+cpu torchvision==0.16.0+cpu numpy==1.26.4 Pillow==10.3.0
.wvision/mmdet-env/Scripts/python -m pip install -c constraints/mmdet-cpu-py310.txt --only-binary=mmcv --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html mmcv==2.1.0 mmengine==0.10.7 mmdet==3.3.0
.wvision/mmdet-env/Scripts/python -m pip check
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
```

The backend does not install WVision itself; the core invokes a packaged standalone worker by an argument vector, without shell evaluation. `--backend-python` or Python's `backend_python=` can replace the environment variable. Its location is saved in each run and can be overridden when loading that run. The existing YOLO environment is unchanged.

For the exact runtime dependency snapshot, replace the compatibility constraints in the second installation step with [windows-mmdet-cpu-py310.lock.txt](../constraints/windows-mmdet-cpu-py310.lock.txt). The CPU/MMCV wheel indexes remain necessary. This pins versions for this platform; it is not a hash-verified artifact lock.

## Reproducible label budgets

From the activated core environment:

```powershell
python examples/make_tiny_dataset.py .wvision/semi-source
wvision prepare-budget --data .wvision/semi-source/dataset.yaml --destination .wvision/semi-budget --labeled-count 2 --seed 0
```

The source fixture has four labeled training samples, two validation samples, and two test samples. The derived data has two labeled and two unlabeled training samples, retaining the original held-out samples. The destination must be new. Withheld training annotations are not copied into it; they are never read by the backend. Existing source unlabeled samples, if any, remain unlabeled and join the derived pool.

`budget.json` records the selection policy (`seeded_sample_v1`), seed, source/derived fingerprints, exact sample lists, labelable pool size, total training pool size, labeled fraction, and labeled class counts. The fraction's denominator includes both labeled and unlabeled training samples. The selection is sample-level, without automatic stratification or subject/video grouping. Source splits must already enforce the required held-out group boundaries. This helper must not be used to disguise incomplete annotations as fully labeled data.

Python equivalent:

```python
from wvision import prepare_label_budget

manifest = prepare_label_budget(
    ".wvision/semi-source/dataset.yaml",
    ".wvision/semi-budget",
    labeled_count=2,
    seed=0,
)
```

## Train and use Soft Teacher

```powershell
wvision train --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data .wvision/semi-budget/dataset.yaml
wvision val --run runs/train-<id>
wvision test --run runs/train-<id>
wvision predict --run runs/train-<id> --source .wvision/semi-budget/test/images
```

Or use Python, with a main guard in Windows scripts:

```python
from wvision import Vision

model = Vision(
    "models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml",
    backend_python=".wvision/mmdet-env/Scripts/python.exe",
)
run = model.train(".wvision/semi-budget/dataset.yaml")
validation = model.val()
test = model.test()
predictions = model.predict(".wvision/semi-budget/test/images")
```

For this model, explicit `learning=supervised` selects the underlying Faster R-CNN baseline, while `learning=semi_supervised` selects the upstream Soft Teacher wrapper. It never silently substitutes one mode for another. Other learning modes are rejected. This backend uses `max_iters`, not `epochs`. Passing `epochs` is an error.

MMDetection evaluation and prediction use the selected checkpoint of a completed run. Replacing it with an unrelated checkpoint is rejected before upstream loading. Raw external initialization remains unsupported. Local SimSiam transfer uses `pretraining_run`; a completed supervised detector uses `detector_run`. Both have separate checked source contracts and cannot be combined.

Configuration fields: `schema_version` (1), `learning`, `max_iters`, `imgsz`, `batch`, `seed`, `device`, `precision`, `devices`, `checkpoint_interval`, `backend_options`. Image size is a multiple of 32, at least 64; batch is even and at least 2. The supervised run uses the full batch as labeled images. Soft Teacher uses a 1:1 labeled/unlabeled source ratio, with weak/strong views from upstream pipelines.

Backend options expose `lr` and upstream semi-supervised settings `pseudo_label_initial_score_thr`, `rpn_pseudo_thr`, `cls_pseudo_thr`, and `reg_pseudo_thr`. The latter settings require semi-supervised mode. Defaults retain upstream pseudo-label thresholds. The smoke recipe lowers score thresholds to exercise pseudo-label losses with random weights, uses a very small learning rate, two iterations, and 128-pixel inputs. These thresholds and schedule are not accuracy recommendations.

`backbone_style` defaults to `caffe`, retaining the M2 protocol. `pytorch` switches both ResNet stride placement and RGB normalization, including Soft Teacher's outer preprocessor. It supports supervised training and Soft Teacher; SimSiam requires this style. Use `models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml` for the initialized semi-supervised smoke path. Full-detector transfer requires matching source/target styles.

## Compare against the same-subset baseline

```powershell
wvision compare --data .wvision/semi-budget/dataset.yaml --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml
```

Or call `compare_learning(data, model="models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml", backend_python=..., output=...)`. This trains the supervised detector and Soft Teacher against the same derived manifest, seed, image size, iteration budget, and learning rate. Both checkpoints are selected on validation AP before test evaluation begins. Comparison configuration must use semi-supervised mode; the baseline receives the same common settings and no pseudo-label-specific options.

The resulting `comparison.json` includes both run paths, label budget, validation/test metrics, test deltas, seed count, wall times, and compute differences. COCO's undefined metrics (`-1`) receive no numeric delta. Equal iterations are not equal compute: the baseline sees more labeled images per iteration, while Soft Teacher also performs teacher and box-jitter computations. Both use random initialization with the same seed; constructing a teacher consumes additional random numbers, so initial parameter tensors need not be identical. The report records this limitation.

The built-in command is a single-seed comparison. Use multiple explicit training seeds and aggregate results before making quality claims. Do not tune against its test reports. The synthetic result is not evidence of improvement or generalization.

The [recorded smoke comparison](verification/SOFT_TEACHER.md) publishes the actual results, including the lower Soft Teacher test score. Reported training-call wall times include startup, preflight, I/O, and checkpointing, so cold/warm caches and host load prevent speed claims.

## Artifacts and provenance

The core saves the standard dataset/config/status/environment/checkpoint artifacts. Each backend call adds `backend-request.json`, `backend-result.json`, `backend.log`, and `backend-environment.json`. Logs preserve upstream messages without scraping them as the metrics API.

Standalone evaluation calls upstream inference and `CocoMetric` directly, preserving the selected COCO image IDs without constructing a training runner. Training still uses upstream `Runner` and `TeacherStudentValLoop`. New recovery checkpoints retain optimizer/scheduler state for explicit resume. Worker cancellation terminates the owned Windows process tree, including virtual-environment redirector children.

Operations sharing an output root reuse `.mmdet-matplotlib-cache` for font discovery. This cache contains no samples, annotations, or metrics. Dataset snapshots and upstream annotation caches remain isolated per operation.

Training writes `training-scalars.json` from MMEngine's structured message hub, `model-source.json`, and both a readable `resolved-mmdet.py` and a reloadable `resolved-mmdet.json`. Only the fixed, installed upstream recipe is initially loaded as Python. Saved-run reload uses JSON, restoring tuple-valued resize settings for upstream compatibility. The generated Python snapshot is not executed during reload. JSON/checkpoint hashes are verified before use; saved checkpoints must still be trusted because their upstream format uses Python deserialization.

The selected semi-supervised checkpoint uses teacher validation COCO AP; prediction/test uses the teacher. Supervised selection uses the detector's validation COCO AP. COCO conversion maps zero-based source class IDs to one-based category IDs and preserves a sorted class map. Unlabeled COCO files contain images/categories and no annotations. Training receives no test annotation file. Test uses a separate selected-split evaluator request.

Pseudo-labels are produced online by upstream Soft Teacher. They are not persisted per image by this integration. Provenance records the full pseudo-label settings, source recipe/hash, initialization policy and source hashes where supplied, upstream `MeanTeacherHook` momentum, and teacher/student checkpoints. It does not claim per-image pseudo-label visibility. No pretrained checkpoint is downloaded. The recipe uses random initialization by default or an explicitly supplied compatible source, fixed image scale, a 1:1 source ratio, a user iteration budget, and no warmup/decay; this differs from the published pretrained 180k-iteration recipe.

Metrics are namespaced `mmdetection/coco/...` and use upstream `CocoMetric`/pycocotools. The detector emits at most 100 predictions per image; upstream recall proposal limits are 100/300/1000. Keep this protocol separate from the YOLO evaluator. Very short random runs can produce empty student predictions and upstream evaluator warnings; missing metrics are not fabricated. Teacher/student training logs and available metrics remain in the run.

## Tests

```powershell
python -m pytest -q
python -m pytest -m mmdetection -q
```

The second command requires `WVISION_MMDET_PYTHON`. It executes real training/comparison, verifies positive unlabeled losses, tests checkpoint reload/prediction through Python and CLI, and checks that withheld labels and test inputs never enter training. The test sets `WVISION_TEST_OFFLINE=1` to deny socket connections inside the worker. Without a configured backend, the optional real test reports a skip; do not count that as backend verification. Restricted Windows runners must permit local named pipes for upstream data processing.

Upstream references: [official recipe](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py), [semi-supervised guide](https://mmdetection.readthedocs.io/en/latest/user_guides/semi_det.html), and [teacher-update implementation](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/engine/hooks/mean_teacher_hook.py).
