# Development and usage

## Environment and installation

The initial target is Windows CPU, Python 3.12, Ultralytics 8.3.4, YOLO11n detection, PyTorch 2.4.1 CPU, and torchvision 0.19.1. This baseline was selected for reproducibility with available upstream code, not as a latest-model recommendation. YOLO also supports verified single-device CUDA in a separate Python 3.12 application environment; see [GPU setup](GPU.md). Linux model execution and other accelerators still need separate verification.

This page describes the YOLO backend. The [semi-supervised guide](SEMI_SUPERVISED.md) covers the separate Python 3.10 MMDetection environment, `max_iters` configuration, label budgets, and comparison commands. Do not mix those backend dependencies into this environment.

From `WForge/` (the package root), PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1
.venv/Scripts/python -m pip install -c constraints/cpu-py312.txt -e ".[dev,ultralytics]"
```

Activate the environment or use `.venv/Scripts/python -m wvision` in place of `wvision`. On Linux, environment executables are under `.venv/bin/`. The constraints pin key compatibility dependencies, not the complete dependency tree. Run artifacts record the actual environment.

For the exact clean Windows verification dependencies, replace `constraints/cpu-py312.txt` in the second command with [constraints/windows-cpu-py312.lock.txt](../constraints/windows-cpu-py312.lock.txt). It records the full dependency snapshot, including developer tools; it is specific to this Python/platform baseline and contains no artifact hashes. The first CPU-wheel installation command is still required.

For the core alone, install `-e ".[dev]"`. Import and help require no PyTorch or Ultralytics. Backend execution reports missing/mismatched dependencies. Installation downloads packages explicitly; the initial training path uses the bundled architecture with random initialization and downloads no weights.

## Verification commands

```powershell
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check src examples tests
.venv/Scripts/python -m ruff format --check src examples tests
.venv/Scripts/python -m mypy
.venv/Scripts/python -m pytest -m integration -q
```

Default tests exclude real training. Integration tests exercise Python and CLI train/val/test/predict, block socket connections, and check incompatible checkpoints and changed datasets. The fixture generator creates eight original images: four train, two validation, two test. Execution on this fixture is not a quality benchmark.

Restricted Windows sandboxes may deny named pipes used by upstream label checking despite `workers=0`. Run real backend checks in an environment that permits those local pipes; do not replace upstream training code to bypass that restriction.

## Configuration and public operations

This section describes the Ultralytics adapter. The [semi-supervised guide](SEMI_SUPERVISED.md) describes MMDetection; the [self-supervised guide](SELF_SUPERVISED.md) describes SimSiam pretraining and detection transfer; the [DINOv3 guide](DINOV3.md) describes Transformers' downstream detector and pending official-weight verification. Each documents its environment and limits. `wvision models` / `list_models()` expose the catalog without loading ML frameworks. New integrations follow the [adapter strategy](MODEL_ADAPTERS.md).

`Vision(model).train(data, **overrides)` resolves adapter defaults, selected model-document settings, then explicit overrides. The CLI uses the same path. The manifest is a separate `data` argument. Unknown fields are errors.

`Vision("models/detection/yolo11n/smoke.yaml")` selects a registered model plus its defaults through the [model-config schema](../models/README.md). On that instance, resolution is adapter defaults → model `training` section → explicit overrides. `backend_options` is replaced as a whole at each layer. The [dataset catalog](../datasets/README.md) supplies independent manifests and preparation steps. Registered-ID selection remains supported; flat settings files and `config=` / `--config` have been removed. See [migration notes](MIGRATION.md).

Settings: `schema_version` (1), `learning`, `epochs`, `imgsz`, `batch`, `seed`, `device`, `precision`, `devices`, `checkpoint_interval`, `backend_options`. See [training operations](TRAINING_OPERATIONS.md) for recovery, mixed precision, and the unverified multi-GPU boundary. `learning=supervised` is supported with `device=cpu`, `cuda`, or `cuda:N`. CUDA requires the separately documented GPU environment; there is no automatic CPU fallback. Image size must be at least 64 and divisible by 32; training batch must be at least 2. See the [smoke recipe](../models/detection/yolo11n/smoke.yaml).

Backend options currently exposed: `lr0`, `patience`, `mosaic`, `mixup`, `degrees`, `fliplr`, `optimizer`. Other fields fail so they cannot override split selection or output ownership. Upstream semantics still apply; for example, its automatic optimizer can choose its own learning rate. The upstream `args.yaml` records its settings.

Evaluation and prediction use the run's image size; evaluation also uses its batch size. Standalone Python evaluation with explicit `data` and `checkpoint` defaults to image size 640 and batch 16. The CLI evaluates completed runs only. Runs refuse changed data or checkpoints. Intentional cross-dataset evaluation uses a new `Vision()` instance with explicit data/checkpoint.

The CLI prints one JSON result to stdout and routes upstream progress to stderr. Errors return exit code 2. Python returns typed result records and propagates exceptions.

## Artifacts and data isolation

Each operation creates a unique directory with `config.json`, `environment.json`, and `status.json`. Data-based operations add `dataset.json`, generated split lists, `upstream-data.yaml`, and selected image/label copies. Copies consume disk space proportional to selected splits and are checked against preflight hashes. Source annotations/caches are not reused or modified.

Training adds `model-source.json`, the bundled `upstream-model.yaml`, `metrics.json`, and `run.json`. The upstream directory retains arguments, CSV logs, and checkpoints. Best-checkpoint selection maximizes the pinned upstream rule `0.1 * AP50 + 0.9 * AP50_95`; it is not configurable yet.

Validation and test add namespaced upstream metrics. Test inputs are passed as `val` in a separate upstream evaluation config; WVision records the operation as `test`. The training config contains neither test nor unlabeled inputs.

Prediction writes input hashes and `predictions.jsonl`. Inputs are local PNG/JPEG files or a non-recursive directory. URLs, video, and streams are not exposed. Verified YOLO, DINOv3 and MMDetection ONNX export is described in [training operations](TRAINING_OPERATIONS.md). Records are written incrementally, but the upstream image-list loader may hold images in memory; this is not a bounded-memory guarantee.

Failed/interrupted operations retain their files and record the error type. Recovery is explicit through `resume`; original runs retain absolute references. Use [portable bundles](TRAINING_OPERATIONS.md#portable-bundles) for relocation. There is no automatic cleanup.

## Backend boundary

The adapter lazily imports upstream code and disables automatic dependency installation, update checks, plot-font downloads, telemetry, and external experiment integrations for its calls. Scoped overrides are needed for this pinned version and do not implement training algorithms or modify upstream source. Calls are serialized; applications concurrently using Ultralytics directly should isolate WVision in another process.

The first upstream import may probe connectivity and create `.wvision/ultralytics` settings. Exposed execution does not fetch model weights or datasets. Local `.pt` inputs must be trusted, with pretraining sources and terms supplied by their owner. Local checkpoint fine-tuning is not optimizer-state resume and has not yet been benchmarked.
