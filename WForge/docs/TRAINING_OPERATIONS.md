# Training operations and experiment comparison

These workflows extend the existing `Vision` API and CLI before Pose. Training
algorithms, optimizers, AMP, and distributed model wrappers remain upstream.
Use the separate backend environments in [GPU setup](GPU.md). No command installs
packages, downloads weights, or uploads datasets during execution.

## Multiple models and parameter grids

Install the shared evaluator in the application environment:

```powershell
python -m pip install -e ".[evaluation]"
python -m wvision experiments --plan examples/experiments.yaml
```

Prepare the [tiny example](../datasets/README.md) first, or copy the
[plan](../examples/experiments.yaml) and point `data` at your own WVision manifest.
All paths in a plan resolve relative to the plan. Model YAML defaults apply first,
then `training`, then the Cartesian product of `grid` values. A dotted grid key
such as `backend_options.lr0` changes that option while preserving other settings.

```yaml
schema_version: 1
data: dataset.yaml
experiments:
  - name: yolo
    model: ../models/detection/yolo11n/smoke.yaml
    training:
      device: cuda:0
      precision: fp16
      backend_options:
        optimizer: SGD
        mosaic: 0.0
    grid:
      backend_options.lr0: [0.001, 0.01]
      seed: [0, 1]
  - name: detector
    model: ../models/detection/faster-rcnn-r50/supervised-smoke.yaml
```

Use `python` on an entry when its application interpreter differs (particularly
in-process YOLO), and `backend_python` for an isolated MMDetection/DINOv3 backend.
Their values are explicit local executable paths. `initialization` accepts the
same `checkpoint`, `pretraining_run`, `backbone_weights`, or `detector_run` source
as `Vision.train`; compatibility checks still apply. An entry containing `run`
instead of `model` evaluates an existing completed run, with no grid/training fields.

For compatible detector-run sources, `initialization.transfer_scope` selects
`detector` (default, identical classes) or `backbone` (new target heads/classes).
See the [cross-dataset transfer contract](WEIGHT_COMPATIBILITY.md).

Jobs run sequentially to avoid accidental GPU oversubscription. Every job retains
its configuration, checkpoints, log, and validation artifacts. The collection
updates `comparison.json`, `comparison.csv`, and `comparison.html` after each
successful candidate. Failure stops the plan and retains completed results.

The HTML report has sortable columns, an AP chart, and expandable settings and
provenance. Identical model/settings/initialization groups aggregate common COCO
metrics across seeds, recording counts, distinct seeds, means, and sample standard
deviation (undefined for a single measurement). Undefined COCO area metrics remain
undefined and are excluded from numeric aggregates.

Every candidate uses the same dataset fingerprint. Comparisons use a shared
pycocotools bbox evaluator: IoU .50:.95, `maxDets=[1,10,100]`, score threshold .001,
and only validation images. `wforge/coco/*` metrics are comparable under this
protocol; native namespaced metrics and original checkpoint-selection policies
are retained separately. Image size, precision, training budget, initialization,
and preprocessing differences remain visible. Equal steps are not equal compute.
Training-call wall time includes startup, preflight, I/O, validation, and checkpointing;
it is not inference latency.

Test evaluation is a separate explicit operation after selecting a candidate:

```powershell
python -m wvision test --run runs/train-<id>
```

It retains that backend's native test protocol. The experiment command does not
evaluate or rank candidates on test metrics. Dataset integrity preflight still
checks every declared split. Video/match grouping remains the dataset owner's
responsibility; this workflow does not change your splits or establish accuracy.

To require group evidence, add `require_groups: true` at the plan's top level.
The manifest must contain complete, isolated group assignments; use
[group preparation](../datasets/README.md#group-separated-evaluation) before training.
[The grouped comparison plan](../examples/grouped-experiments.yaml) illustrates
multiple models and three seeds on one prepared label budget. Its profiles are
starting points; budgets are not compute-matched or benchmark-tuned.

Shared COCO reports include per-class AP50:95/AP50 (for example Ball, Player,
and Referee), plus small/medium/large area metrics. Ground-truth object counts
are saved in `per-class.json`. HTML/JSON report whether the supplied group
assignments were verified; missing groups are explicitly unverified. Seed
means/sample standard deviations still use the same target protocol.

The same shared protocol is available for explicit final evaluation:

```powershell
python -m wvision val --run runs/train-<id> --protocol coco
python -m wvision test --run runs/train-<id> --protocol coco
```

Python uses `model.val(protocol="coco")` / `model.test(protocol="coco")`.
The native evaluator remains the default. A test report never feeds back into
the experiment command's selection or seed aggregates.

Python entry point:

```python
from wvision import run_experiments

comparison = run_experiments("experiments.yaml", output="runs")
print(comparison.html)
```

## Periodic recovery and longer training

New training/pretraining runs write `training.json` before backend execution.
`checkpoint_interval` controls recovery intervals: epochs for YOLO, iterations
for OpenMMLab/DINOv3. Defaults are one YOLO epoch and 100 iterations otherwise;
shorter schedules still save their final checkpoint. OpenMMLab/DINOv3 also perform
validation at checkpoint intervals. Best validation selection is retained across
continuation, including a copied previous best checkpoint.

```powershell
python -m wvision train --model models/detection/yolo11n/smoke.yaml --data dataset.yaml --epochs 20 --checkpoint-interval 1
python -m wvision resume --run runs/train-<id> --epochs 50
python -m wvision resume --run runs/pretrain-<id> --max-iters 2000
```

Budgets are **total** epochs/iterations, not additional steps. `resume` creates a
new run and copies the selected recovery state. It restores upstream optimizer
state and training position, plus scaler/scheduler/RNG state where the upstream
checkpoint supplies them. YOLO preserves an unstripped recovery checkpoint and
its scaler state separately from upstream's final inference weights. Upstream
sampling/EMA/serialization behavior remains applicable; bitwise equivalence to an
uninterrupted run is not claimed.

Resume rejects changed model/data/settings, tampered checkpoints, active application
or backend writers, and budgets that have no remaining steps. Only the total budget
may change. Backend executable overrides remain possible through `--backend-python`.
An interrupted run can recover only after a complete recovery checkpoint was saved.
The old run is preserved. Runs predating `training.json` cannot gain optimizer state
retroactively; weight initialization remains a separate operation.

Continuation records retain the source model provenance, including declared
pretraining/weight terms, so that portable continued runs retain their lineage.
Recovery records point at immutable, checksummed copies. The current and previous
committed snapshots are retained; a failed new snapshot does not overwrite the
previous checkpoint, even when upstream rewrites or rotates its native files.

```python
from wvision import resume

continued = resume("runs/train-<id>", epochs=50)
```

## Mixed precision

FP32 remains the default. CPU training uses FP32. On a supported CUDA device,
select precision explicitly through `precision=` or `--precision`:

| Backend | CUDA precision settings |
| --- | --- |
| YOLO11n | `fp32`, `fp16` |
| Faster R-CNN / Soft Teacher | `fp32`, `fp16` |
| SimSiam | `fp32`, `fp16`, `bf16` |
| DINOv3 / DETR | `fp32`, `bf16` |

```powershell
python -m wvision train --model models/detection/yolo11n/smoke.yaml --data dataset.yaml --device cuda:0 --precision fp16
python -m wvision pretrain --model models/representation/simsiam-r50/smoke.yaml --data dataset.yaml --device cuda:0 --precision fp16
```

BF16 requires hardware support, checked before execution. DINOv3 FP16 is rejected:
the verified short random-initialization case produced overflowing gradients;
the BF16 path passed. MMDetection BF16 is rejected because the pinned MMCV CUDA
ROIAlign operator does not implement it. Requested precision never silently changes to another format.
Inference/evaluation and verified ONNX export use FP32. Reduced memory use and
speed gains depend on the workload; no performance improvement is promised.

## Multiple GPUs: implementation and verification boundary

`devices=[0,1]` / `--devices 0 1` launches one upstream process per GPU with
`torch.distributed.run`. `device` selects the first GPU in that list. Indices refer
to the backend's visible devices; an inherited visibility mask is respected.
`batch` is per GPU. This is single-machine DDP, not multi-node training.

```powershell
python -m wvision train --model models/detection/yolo11n/smoke.yaml --data dataset.yaml --device cuda:0 --devices 0 1
```

The launcher checks GPU availability before training and manages the owned process
tree. Upstream DDP/Trainer/Runner handle synchronization; rank-local device records
and logs are retained. Checkpoint/report writes belong to rank zero. External
`torchrun` launches are rejected so WVision owns cancellation and output paths.
Linux selects NCCL; Windows selects Gloo, since NCCL is unavailable on Windows.
See [PyTorch's backend guidance](https://docs.pytorch.org/docs/stable/distributed.html).

**Actual two-GPU training/resume remains unverified on the current one-GPU laptop.**
Rank isolation, visibility validation, configuration, and unavailable-GPU failure
are covered. Do not treat the launcher or a skipped test as evidence of multi-GPU
support. The dedicated real tests require two devices and explicit
`WVISION_TEST_DISTRIBUTED=1`; there is no single-GPU fallback.

## Portable bundles

```powershell
python -m wvision bundle --run runs/train-<id> --destination bundles/detector
python -m wvision bundle --run runs/train-<id> --destination bundles/experiment --include-data --include-training
```

Destinations must be new. Bundles use relative runtime references and checksums.
Default bundles include inference/representation weights, model configuration,
metrics, and provenance. Optional data copies preserve original split membership,
relative sample paths, annotations, and dataset fingerprint. Original source paths
in audit records are historical provenance, not runtime dependencies.

```python
from wvision import Vision, bundle_run, resume

bundle = bundle_run("runs/train-<id>", "bundles/detector", include_data=True)
restored = Vision.from_run(bundle, backend_python="path/to/compatible/python.exe")
prediction = restored.predict("image.jpg", device="cpu")
```

Install the compatible backend on the destination machine and select it explicitly
or through the existing backend environment variables. The bundle does not package
Python, drivers, or backend libraries. YOLO uses the current application interpreter.
Representation bundles load through the existing checked transfer workflow.

With `include_training=True`, optimizer recovery is included. After relocation:

```powershell
python -m wvision resume --run bundles/experiment --epochs 50
python -m wvision resume --run bundles/experiment --data dataset.yaml --epochs 50
```

`--data` is required when recovery was bundled without data; its fingerprint must
match. `Vision.from_run(..., data=...)` similarly remaps evaluation data for an
inference-only bundle. Dataset-containing bundles can be large. On Windows, keep
the destination shallow when preserving long legacy source filenames.

## ONNX export

Verified ONNX export covers YOLO11n, DINOv3 ConvNeXt Tiny/DETR, and MMDetection
Faster R-CNN / Soft Teacher through pinned MMDeploy. Native portable bundles
remain available for all registered training workflows.
Export does not add TensorRT, C++, or video/streaming runtime integration.

```powershell
python -m pip install -e ".[onnx]"
python -m wvision export --run runs/train-<id> --format onnx
```

All exporters use the selected checkpoint, fixed input size, batch one, FP32,
opset 17, and no simplifier. For YOLO and DINOv3, ONNX checker and CPU ONNX Runtime
verify raw outputs against the native FP32 model on two deterministic inputs
(`rtol=1e-3`, `atol=1e-4`). These two exporters reject `verification_images`.
`export.json` records hashes, class map, tensor shapes, preprocessing, tolerances,
and observed error. The model-specific contracts are:

| Model | Preprocessing | Raw output |
| --- | --- | --- |
| YOLO11n | RGB/NCHW, letterbox, normalize to 0–1 | Input-pixel `xywh` boxes and class scores; no NMS |
| DINOv3/DETR | RGB/NCHW, fixed square PIL bilinear resize, divide by 255, ImageNet mean/std; int64 all-ones pixel mask | Query class logits including the final no-object class, and normalized `cxcywh` boxes; no NMS |

For DINOv3, install export dependencies **in its compatible backend environment**:

```powershell
.wvision/transformers-env/Scripts/python -m pip install onnx==1.17.0 onnxruntime==1.20.1
python -m wvision export --run runs/train-<id> --backend-python .wvision/transformers-env/Scripts/python
```

Replace the backend executable with your CPU/CUDA environment from the DINOv3
guide. The DINOv3 manifest describes softmax/no-object handling and conversion of
normalized boxes back to original image dimensions. Inputs/shapes are fixed;
dynamic shapes and varying padding masks are not verified. Consumers must follow
the exported model's own manifest rather than sharing YOLO postprocessing.

### Faster R-CNN and Soft Teacher

Prepare a separate Python 3.10 MMDetection environment using the verified
[CPU setup](SEMI_SUPERVISED.md) or [CUDA setup](GPU.md), then add its export pins.
Use `.wvision/mmdeploy-env` as the environment directory in those setup commands.
Do not mix these ONNX/protobuf versions with the DINOv3 environment:

```powershell
.wvision/mmdeploy-env/Scripts/python -m pip install -c constraints/mmdeploy-py310.txt mmdeploy onnx onnxruntime protobuf
python -m wvision export --run runs/train-<id> --backend-python .wvision/mmdeploy-env/Scripts/python --verification-images validation/frame1.jpg validation/frame2.jpg
```

Python accepts `model.export(verification_images=["frame1.jpg", "frame2.jpg"])`.
Supply at least two distinct representative PNG/JPEG images, preferably covering
every class and relevant scene type. Verification never trains or selects weights;
use training/validation images so test data stays reserved for final evaluation.
At least one image must produce native detections. Every supplied image must pass;
there is no option to label a failed export as verified.

The selected detector is exported unchanged. Soft Teacher uses the configured
inference branch (normally `teacher`), with an exact tensor-state check after
extracting the plain Faster R-CNN. Both Caffe-style and PyTorch-style R50 backbones
are covered. MMDeploy conversion hooks preserve native per-pyramid-level RPN NMS
and class-specific regression boxes; see [verification](verification/MMDET_ONNX.md).

Preprocessing resizes to a fixed square using OpenCV bilinear interpolation,
without preserving aspect ratio, then applies the manifest's RGB/BGR ordering,
mean and standard deviation to 0–255 pixels. The graph takes float32 NCHW input
and includes decoding and NMS. Outputs are `dets` (xyxy input-pixel boxes plus
scores) and zero-based `labels`; ignore padded rows with score zero. Multiply x
by original-width/input-width and y by original-height/input-height to rescale.
Read `export.json` for the exact model's contract and thresholds.

Verification uses ordinary CPU ONNX Runtime without custom operator registration.
It compares complete detection sets within each class, allowing ordering changes;
box tolerances are `rtol=1e-4`, `atol=1e-3` pixels, and score tolerances are
`rtol=1e-3`, `atol=1e-4`. Different classes/counts, nonfinite outputs, or custom
ONNX domains fail. Floating-point proposal/NMS ties can change discrete decisions
on other inputs: calibration evidence covers the supplied images, not every
possible image. A failed comparison retains diagnostic artifacts with failed run
status and no verified `export.json`. Dynamic shapes, larger batches, other
precisions and inference accelerators remain unverified.

## Backend automation

The [real-backend workflow](../../.github/workflows/backends.yml) runs scheduled
Windows CPU checks weekly and can be dispatched manually. It creates separate
environments with the documented constraints and runs actual upstream training,
recovery, relocation, comparison, and export checks. Existing core CI remains
independent of ML backends.

Manual CUDA jobs require a self-hosted Windows/X64 runner labeled `gpu`, compatible
drivers, and BF16 support for DINOv3. `distributed=true` additionally requires two
GPUs. A missing device fails; it does not become a CPU job. Reports, metrics,
device records, and backend failure logs are retained; datasets and weights are
not uploaded by the workflow. Hosted workflow execution requires committing and
running it on GitHub and is separate from local verification.
