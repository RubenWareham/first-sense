# Architecture

Status: detection workflows implement this boundary through a typed adapter protocol and explicit model registry. The implementation groups framework-free responsibilities into `core/`, `datasets/`, `training/`, `evaluation/`, and `experiments/`, with model-owned adapters/settings/operations under `models/<framework>/<model>/` and reusable runtimes under `adapters/`. Future task packages and operations below remain design targets. See [development instructions](DEVELOPMENT.md), [semi-supervised usage](SEMI_SUPERVISED.md), and [DINOv3 usage](DINOV3.md) for actual behavior and limits.

## Boundaries

```text
Python API / CLI
       |
validated configuration + capability checks
       |
experiment orchestration ---- dataset preflight / run artifacts
       |
backend adapter
       |
existing upstream training / inference / evaluation implementation
```

The core does not import a training framework. Adapter loading is lazy and explicit. Installing multiple backend environments may require separate processes or virtual environments; do not assume all ML dependencies can coexist.

`Vision` accepts either an exact registered model ID or a model-config YAML path. `core/model_config.py` safely parses its model/task/profile/upstream metadata and training defaults; adapters still validate actual settings and capabilities. [Model configs](../models/README.md) live under `models/<task>/<model>/`; independent [dataset manifests](../datasets/README.md) live under `datasets/`. Config source snapshots are recorded with actual resolved training settings; completed runs do not need the original model YAML for reload.

Device capability entries identify `cpu` or the single-device `cuda` family. Validated requests select a concrete `cuda:N` index. The framework-free adapter device helper also serves standalone workers; runtime CUDA checks stay inside backend execution. Worker CPU requests hide CUDA before framework imports, and CUDA requests verify allocation and model placement. Evaluation/prediction can explicitly override a saved training device without changing it. See [GPU execution](GPU.md).

`core/registry.py` contains reviewed `ModelSpec` entries and lazy factory references. Capabilities are tuples of learning mode, operation, and device. Explicit initialization rules describe source model/kind, transfer scope, target learning mode, and required backbone style; they do not imply arbitrary checkpoint compatibility. `adapters/base.py` defines the small `DetectionAdapter` protocol, shared execution context, initialization inputs, and typed training output. `Vision` calls that protocol without model-ID branches. Dataset staging and framework translation live in adapters; core run lifecycle and integrity checks remain shared. See the [extension strategy](MODEL_ADAPTERS.md).

Implemented process boundary: the Python 3.12 core writes a versioned JSON request, invokes the packaged `models/mmdetection/faster_rcnn_r50/operations.py` using an explicit Python 3.10 executable and no shell, and reads a versioned JSON result. Upstream logs and backend environment are retained. Training starts from a fixed installed upstream recipe; future evaluation reloads its generated JSON snapshot instead of executing user configuration code. The core terminates/waits for its worker on interruption. The [comparison workflow](SEMI_SUPERVISED.md) orchestrates existing upstream runs without implementing a training loop.

The [self-supervised workflow](SELF_SUPERVISED.md) shares this process boundary through a separate fixed `models/mmpretrain/simsiam_r50/operations.py`. `pretrain` returns a representation checkpoint selected at the final iteration, with no detector or held-out metrics. `Vision.train(pretraining_run=...)` validates its source identity and transfers only the compatible backbone through upstream initialization. `compare_pretraining` orchestrates two detector runs and evaluations; neither the core nor the workers implement a learning loop. `Vision.train(detector_run=...)` validates a completed supervised MMDetection run before initializing both upstream Soft Teacher branches. A framework-free source validator checks data/class/version/hash provenance; the isolated worker checks resolved structure and full tensor state, delegates loading upstream, and audits both branches before training. See [weight compatibility](WEIGHT_COMPATIBILITY.md). Fully unsupervised detection is a separate, currently unsupported contract.

`adapters/shared/process.py` shares versioned transport and owned-worker cancellation between OpenMMLab and the independent DINOv3/DETR worker. DINOv3 source backbone files are explicitly local; no remote Hub IDs are resolved. Existing model IDs and schema-1 training runs remain loadable. No runtime entry-point scanning or executable adapter imports from configuration are enabled.

## Current layout

```text
WForge/
  src/wvision/
    __init__.py            # stable public imports: Vision, pretrain, comparisons
    api.py, cli.py         # Python facade and CLI entry points
    core/                  # config, model documents, result contracts, registry
    datasets/              # detection manifests, staging, integrity, label budgets
    training/              # representation pretraining orchestration
    evaluation/            # learning-method and pretraining comparison workflows
    experiments/           # lifecycle, environment, status, artifact writing
    models/
      ultralytics/yolo11/  # YOLO11n adapter, config, operations
      mmdetection/faster_rcnn_r50/  # Faster R-CNN/Soft Teacher + source weights
      mmpretrain/simsiam_r50/      # SimSiam settings, source checks, pretraining worker
      transformers/dinov3_convnext_tiny_detr/  # DETR adapter, local source checks, worker
    adapters/
      base.py              # typed detection adapter contract
      shared/              # device policy/audits and worker transport/cancellation
      ultralytics/         # shared Ultralytics runtime and metric normalization
      openmmlab/           # shared interpreter and standalone environment runtime
      transformers/        # shared standalone environment/file-integrity runtime
  models/                  # model profiles grouped by task/model
  datasets/                # independent manifest examples/templates
  examples/                # runnable public workflows
  tests/                   # core and explicitly marked real integration checks
  constraints/             # verified environment pins and snapshots
  docs/                    # usage, engineering rules, milestones, evidence
```

Introduce task packages only when another implemented task needs them. Keep reviewed
adapter registration explicit; no runtime plugin discovery or empty task scaffolds.

## Finding and changing code

| Change | Owner |
| --- | --- |
| Public detection workflow | `api.py` |
| CLI arguments or output | `cli.py` |
| Model registration/capabilities | `core/registry.py` |
| Model YAML validation/precedence | `core/model_config.py` |
| Public result records | `core/contracts.py` |
| Dataset checks/conversion | `datasets/detection.py` |
| Labeled/unlabeled budget generation | `datasets/budgets.py` |
| Representation pretraining | `training/pretraining.py` |
| Optimizer-state recovery and continuation | `training/recovery.py` |
| Multi-model / parameter-grid experiments | `experiments/comparison.py` |
| Portable run bundles | `experiments/portable.py` |
| Shared cross-model COCO validation | `evaluation/detection.py` |
| Supervised/semi-supervised comparison | `evaluation/learning.py` |
| Random/pretrained comparison | `evaluation/pretraining.py` |
| Status, environment, artifact writing | `experiments/artifacts.py` |
| YOLO11 wrapper/settings | `models/ultralytics/yolo11/adapter.py`, `config.py` |
| YOLO11 train/evaluate/predict | `models/ultralytics/yolo11/operations.py` |
| Shared Ultralytics runtime | `adapters/ultralytics/backend.py` |
| Faster R-CNN/Soft Teacher | `models/mmdetection/faster_rcnn_r50/` |
| SimSiam settings, source integrity, worker | `models/mmpretrain/simsiam_r50/` |
| DINOv3/DETR | `models/transformers/dinov3_convnext_tiny_detr/` |
| Shared OpenMMLab interpreter/preflight | `adapters/openmmlab/backend.py`, `runtime.py` |
| Shared Transformers preflight | `adapters/transformers/runtime.py` |

The Python `src/wvision/models/` tree owns model-specific integration code.
It is distinct from the repository's top-level `models/` YAML profile catalog.
All registered families now have model homes. This organization does not add
new architectures, variants, learning modes, or checkpoint compatibility.
Soft Teacher shares the Faster R-CNN model folder because it wraps that
detector; no duplicate detector implementation is introduced.

Within `models/ultralytics/yolo11/`, `adapter.py` implements the common protocol,
`config.py` validates model settings, and `operations.py` owns architecture checks,
model provenance, and calls to upstream train/val/predict. Validation and test use
the same evaluation function with separately staged splits. The shared
`adapters/ultralytics/backend.py` handles the pinned dependency, scoped runtime
settings, device selection, and metric normalization. No learning loop is copied.

Standalone workers locate the shared framework-free `adapters/shared/devices.py`
relative to their packaged model file. Model-owned launchers choose a fixed
neighboring `operations.py`; shared transport never chooses a model or method.
Workers also load the matching standalone backend runtime. They do not import the application's package
initializer or require WVision to be installed in the backend environment.
Only the helper folder is added to the worker import path, so backend folders
named `transformers` or `ultralytics` cannot shadow upstream libraries.

Top-level public imports and the CLI remain stable. Internal module paths changed
in the First Sense reorganization; see [migration notes](MIGRATION.md). No legacy
forwarding modules or duplicate implementations are retained.

## Public operations

| Operation | Responsibility |
| --- | --- |
| `train` | Validate request, select adapter, start an upstream training run or explicit checked continuation |
| `val` | Evaluate the validation split for development and selection |
| `test` | Evaluate the explicit test split using an already selected checkpoint |
| `predict` | Return task-specific predictions and optional saved outputs |
| `export` | Verified YOLO ONNX export; other formats/adapters fail clearly |

Use `val` consistently in Python and CLI. A CLI command should only parse arguments, invoke application logic, and format the result. It must not implement its own training behavior.

## Shared contracts

- **Model specification:** stable registered identifier, backend, task, upstream model/config reference, and optional weights reference.
- **Capabilities:** supported combinations of task, learning mode, operation, device constraints, and format. Separate task, method, and operation lists are insufficient if their combinations are invalid.
- **Training request:** model specification, data manifest, learning mode, seed, device, output directory, common training settings, and backend options.
- **Run result:** run identifier, status, artifact directory, last checkpoint, optional best checkpoint, and metric summary. Define selection metric and direction before calling a checkpoint "best."
- **Evaluation result:** task, split, checkpoint identity, dataset identity, metric names/definitions, values, and upstream evaluator details.
- **Detection prediction:** sample identity, original image dimensions, absolute pixel `xyxy` boxes, scores, and class IDs with a class map. Convert upstream coordinate conventions explicitly.

Use immutable typed records where useful. Return paths to large outputs rather than loading all predictions or tensors into memory. A streaming prediction option should be introduced only when supported and tested.

## Configuration behavior

Resolve settings in this order: library defaults, model-config training defaults if selected, explicit call or CLI overrides. Fields replace earlier values; backend option mappings are replaced as a whole. Save final resolved settings and the model-config source snapshot. Reject unknown common fields and invalid combinations. Keep advanced options in a namespaced backend section and validate them using upstream schemas where possible; fail clearly when upstream rejects an option.

Use safe YAML parsing with a versioned schema. Resolve relative dataset paths against the manifest location and config references against their containing file. Record resolved locations and portable source references. Do not evaluate Python expressions embedded in user YAML. If a backend requires executable Python configuration, document that trust boundary and require an explicit input rather than hiding execution behind a generic manifest.

## Run lifecycle

Validate configuration and splits before expensive imports or GPU allocation. Create a unique run directory and write a running status. Record success, failure, or interruption honestly, preserving available upstream logs and checkpoint references. Never overwrite another run implicitly. Resume must verify backend, configuration, data identity, and checkpoint compatibility, or require an explicit documented override for supported differences.

Do not parse human-readable logs as the primary metrics API when upstream exposes structured results. Preserve raw upstream artifacts alongside normalized summaries for auditability.
