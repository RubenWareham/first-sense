# Model extension strategy

This work follows the user's request to prioritize DINOv3 and easier model integration before the planned pose milestone. The implementation adds an explicit catalog and a small detection adapter protocol. Training algorithms stay upstream.

## One application flow, explicit capabilities

`Vision` owns configuration resolution, capability checks, split validation, run lifecycle, fingerprints, checkpoint integrity, and normalized public results. Each adapter translates validated data/settings and calls its upstream framework. The facade must not branch on model identifiers or infer capabilities from a framework's model catalog.

The catalog records a stable model ID, task, backend, and supported tuples of learning mode, operation, and device family (`cpu` or single-device `cuda`). A CUDA request specifies its visible index as `cuda:N`; only verified adapters expose that family. Representation pretraining and detection are distinct contracts. Adding a model does not automatically enable every operation or learning mode. Unsupported combinations fail before data staging or model loading.

The [model-config catalog](../models/README.md) selects those registered IDs and adds named training profiles plus upstream metadata. `Vision("models/.../profile.yaml")` uses the same adapter factory as an ID. Adding a config does not register a new implementation or grant an unsupported learning capability. Dataset manifests stay in the separate [dataset catalog](../datasets/README.md). Configs are selected by explicit file path, without plugin scanning or executable inheritance.

The detection protocol has configuration resolution, checkpoint/initialization validation, training, evaluation, and prediction methods, plus selection/evaluator metadata. Training returns typed checkpoint/metric/config references. Register explicit `InitializationRule` entries for each tested source/kind/scope/learning/style combination, then validate provenance in the adapter and actual state in the worker. See [weight compatibility](WEIGHT_COMPATIBILITY.md); adding a rule alone never establishes support. Framework imports stay inside adapter execution or a fixed subprocess worker. Heavy backend versions remain isolated where incompatible.

Implemented references: [protocol and typed records](../src/wvision/adapters/base.py), [catalog and lazy factories](../src/wvision/core/registry.py), [YOLO adapter](../src/wvision/models/ultralytics/yolo11/adapter.py), [OpenMMLab adapter](../src/wvision/models/mmdetection/faster_rcnn_r50/adapter.py), and [DINOv3 adapter](../src/wvision/models/transformers/dinov3_convnext_tiny_detr/adapter.py). A factory receives `ModelSpec` and the optional backend interpreter and returns `DetectionAdapter`. The catalog is an immutable tuple of reviewed entries; it does not mutate at runtime or scan the environment.

## Model folders

The YOLO11 integration lives in
[`src/wvision/models/ultralytics/yolo11/`](../src/wvision/models/ultralytics/yolo11/):

- `adapter.py`: supported operation contract and dataset/result translation.
- `config.py`: validated YOLO11 training settings, formerly mixed into core config.
- `operations.py`: existing YOLO11n architecture/loading checks, provenance, and
  upstream train/evaluate/predict calls.

Reusable runtime/version/device/network settings remain in
[`adapters/ultralytics/backend.py`](../src/wvision/adapters/ultralytics/backend.py).
Framework imports occur only during execution in model integration modules or
shared adapters/workers. Package/core imports still work without ML dependencies.
The registry selects the model's factory directly; no forwarding adapter remains
at the old location. Separate `train.py`, `val.py`, and `test.py` files would only
be introduced for substantial independent implementations, not thin forwards.

Only the registered models and modes are supported; folder organization does
not add architectures or learning methods. The remaining integrations follow
the same ownership rule:

| Model folder | Responsibilities |
| --- | --- |
| `models/mmdetection/faster_rcnn_r50/` | Detector/Soft Teacher adapter, config, checked detector sources, operations worker |
| `models/mmpretrain/simsiam_r50/` | Config, source-run validation/worker launcher, operations worker |
| `models/transformers/dinov3_convnext_tiny_detr/` | Adapter, config/local backbone validation, operations worker |

The shared representation run lifecycle stays in `training/pretraining.py`.
SimSiam's adapter owns its fixed worker and source model identity. Detection
adapters own their fixed workers as well. Standalone OpenMMLab/Transformers
version checks and serialization helpers live in matching `adapters/` runtime
modules and remain importable without the WVision application in backend
environments. Only helper directories are added to worker search paths, so
model/framework folders cannot shadow installed upstream libraries.
Keep YAML profiles in the top-level `models/` catalog; no duplicate profiles are
introduced inside the Python model package.

## Adding a model

1. Select an exact upstream implementation, model configuration, environment, and weight source/license. Confirm the intended combination uses upstream-supported entry points.
2. Reuse an adapter for a new variant only if its settings, data translation, checkpoint validation, and outputs are compatible. Otherwise implement the small protocol in a model-specific adapter under `src/wvision/models/<framework>/<model>/`, reusing shared backend execution where it actually fits.
3. Add one explicit catalog entry and factory reference. This is reviewed Python source, not executable import paths from user YAML or automatic plugin scanning.
4. Add an offline core test for unsupported combinations and a real upstream public workflow test. Verify class IDs, box coordinates, split isolation, checkpoint identity, and provenance.
5. Add a minimal config/example, a support-status entry, and execution evidence. Only claim pretrained or accelerator support after the corresponding run succeeds.

No inheritance hierarchy, universal backend option dictionary, custom training loop, or automatic installation is required. Existing YOLO/MMDetection IDs and run artifacts retain compatibility. Task-specific protocols should be added when pose/classification need their own prediction contracts.

## DINOv3 slice

The first candidate is Transformers' upstream DETR with its DINOv3 ConvNeXt Tiny backbone API. It follows the existing upstream backbone/head integration, rather than creating a new detector or loss. This is downstream supervised detection using a self-supervised foundation backbone when official weights are supplied. Random initialization is an explicit execution-baseline option and cannot be presented as SSL.

Local Hugging Face configuration and safetensors files are the initialization boundary. No remote IDs, arbitrary Python code, hidden downloads, or implicit fallback to random weights are accepted. The adapter records source file checksums and known supervision. User-approved access to Meta's official weights is separate from code/package installation. Official pretrained transfer is not verified until those weights are available; source-compatible local fixture loading establishes only the loading mechanism.

Single-device DINOv3/DETR CUDA execution is verified; see [GPU evidence](verification/YOLO_DINOV3_CUDA.md). Meta's original released 7B detector, DINOv3 self-supervised teacher/student pretraining, semi-supervised DETR, and additional DINOv3 model variants remain separate capabilities. This slice must not mark them supported merely because DINOv3 feature inference works.
