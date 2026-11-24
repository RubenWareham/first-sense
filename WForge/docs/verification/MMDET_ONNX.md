# Faster R-CNN and Soft Teacher ONNX verification

Verified **2026-11-18**, Windows, Python 3.12.3 application / Python 3.10.20
backend. The optional exporter uses MMDeploy 1.3.1, ONNX 1.13.1, ONNX Runtime
1.20.1, protobuf 3.20.2, numpy 1.26.4, torch 2.1.0+cu121, torchvision 0.16.0+cu121,
MMCV 2.1.0, MMEngine 0.10.7 and MMDetection 3.3.0. Export and parity run on CPU,
FP32, batch one, fixed 128×128 input in these checks. Training uses the existing
single RTX 4050 Laptop GPU. This report makes no speed or model-quality claim.

## Behavior and conversion boundary

Public Python/CLI export selects the original inference checkpoint. Soft Teacher
extracts its configured teacher/student branch into a plain Faster R-CNN and
checks every state tensor after reload. Source files remain unchanged. Artifacts
include `model.onnx`, the extracted `detector.pth`, model/deployment configuration,
and a checksummed `export.json` with preprocessing, classes, output coordinates,
versions, image hashes, and measured errors. See the
[usage/setup contract](../TRAINING_OPERATIONS.md#faster-r-cnn-and-soft-teacher).

Generic MMDeploy 1.3.1 conversion changed two native inference semantics in the
initial probe: proposal NMS grouped all feature levels together, and class-specific
regression boxes were replaced by the winning class's boxes. Scoped conversion
hooks now retain separate feature-level NMS channels and class-specific box/score
rows while calling upstream decoding/NMS/export APIs. Padded invalid proposals
cannot enter final detection NMS. No model, loss, or training method is copied.
Hook ownership, upstream references and removal conditions are recorded in
[third-party notices](../../THIRD_PARTY_NOTICES.md).

The graph uses standard ONNX operators only. Verification opens it in an ordinary
ONNX Runtime CPU session without MMDeploy custom DLL/operator registration.
At least two representative images are mandatory; every supplied image must pass
complete per-class one-to-one detection matching. Ordering alone is irrelevant.
Box tolerances are `rtol=1e-4`, `atol=1e-3` pixels; score tolerances are
`rtol=1e-3`, `atol=1e-4`. No all-negative calibration success is accepted.

## Real checks

```powershell
$env:WVISION_MMDEPLOY_PYTHON = (Resolve-Path .wvision/mmdeploy-env/Scripts/python.exe).Path
$env:WVISION_TEST_DEVICE = "cuda:0"
$env:WVISION_TEST_OFFLINE = "1"
python -m pytest -q -m integration tests/test_mmdet_export_integration.py
```

The four train/export cases passed, covering supervised Faster R-CNN and
semi-supervised Soft Teacher, each with Caffe-style and PyTorch-style backbones.
They use the owned two-class fixture, two training iterations and two validation
images; source hashes, positive detections, class maps and branch identity are
checked. A separate backend regression passed for reordered valid detections and
rejection of changed boxes, scores, classes, counts and NaN in padded output.
Core validation passed **184 tests**; Ruff lint/format and mypy passed.

Local ignored evidence: `.wvision/mmdeploy-integration.log` (4 passed),
`.wvision/mmdeploy-parity-test.log` (1 passed), `.wvision/mmdeploy-core.log`.
The scheduled/manual backend workflow now installs these conversion pins and
selects these actual tests; YAML and all 48 rendered PowerShell snippets parse.
Hosted workflow execution remains a separate open gate.

Wheel and source distribution build successfully. All 63 packaged Python modules
match the source bytes; the archives include the new worker, constraints, tests
and documentation, and exclude datasets, weights and generated environments.
The extracted wheel imports/catalogs without ML frameworks, exposes the CLI export
arguments, and passed the same real-football public export check. Evidence is
`.wvision/mmdeploy-package.log`; build output is `.wvision/mmdeploy-dist/`.

The existing football Soft Teacher checkpoint also passed public `Vision.export`
verification on three validation images, with 100 detections per image and
Ball/Player/Referee class metadata. All 348 source tensors matched after branch
extraction. Maximum box error was **0.000015259 pixels**, maximum score error
**0.000000089407**. The source checkpoint SHA256 is
`215501429aa29c00a0a70d8b0c9a875c6f8d0ee4c4f651875d4ddffd9d05b61f`.
Source-run checkpoint hashes remained unchanged. Local artifacts are under
`.wvision/mmdet-export-verified/export-68ca8065e28f489badc25ed79b0ca65e/`,
with the check log at `.wvision/mmdet-export-verified.log`.

The dataset retains 87 labeled / 785 unlabeled / 292 validation / 293 test images,
fingerprint `ac8242b5c0a7b9a2b2b1fcdf2f09b3d155fd02aa44d71cf70c10a397d8ee2b48`.
No labels or source splits were changed. Known video overlap and unverified group
IDs prevent a held-out football generalization claim. No test images were used
for this export check.

## Limits and postponed work

A deterministic synthetic-noise probe exposed different proposal/NMS decisions
from floating-point ranking ties; the strict checker rejected it. Those logs
remain at `.wvision/mmdeploy-parity.log`. Tolerances were not loosened to admit
different detection sets. Passing representative images establishes parity on
those images; it does not prove all-input equivalence. Failures keep diagnostic
artifacts and failed status without a verified export manifest.

Only fixed-shape FP32 batch-one standard ONNX Runtime CPU inference is verified.
Dynamic input, other runtimes, reduced precision and larger batches are separate
gates. No TensorRT/C++/streaming runtime was added. No data/weights are distributed.

Official licensed DINOv3 ConvNeXt Tiny verification was **postponed by the user**
until local weights are available. No authenticated model download was performed.
Random/local-fixture DINOv3 evidence remains distinct from official pretrained
verification. Actual two-GPU execution and longer grouped football benchmarks
also remain open.
