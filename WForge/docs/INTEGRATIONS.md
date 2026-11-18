# Model integration policy

## Selection principle

Integrate existing implementations with documented entry points, accessible weights/configurations, acceptable terms, a viable dependency environment, and a reproducible evaluation path. An adapter is a translation layer, not a new training framework.

Ultralytics, MMDetection Soft Teacher/Faster R-CNN, MMPreTrain SimSiam-to-detection transfer, and Transformers DINOv3/DETR execution are implemented as recorded below. Official pretrained DINOv3 weights remain an open gate. Other candidates remain **unimplemented and unsupported**. Check maintenance, compatibility, and terms again before adding or upgrading a backend.

## Initial candidates

| Priority | Candidate | Intended role | Gate before adoption |
| --- | --- | --- | --- |
| First | Ultralytics | Accessible supervised detection baseline; later assess pose/classification reuse | Verify exact release, model/weight choice, environment, evaluation behavior, and license suitability |
| Second | MMDetection with an existing semi-supervised recipe such as Soft Teacher | Labeled/unlabeled teacher-student detection through upstream training | Reproduce the official recipe in an isolated compatible environment; validate configuration trust boundary |
| Research gate | Existing self-supervised representation and unsupervised detection implementations | Learn from unlabeled data without inventing an algorithm | Select a concrete task, maintained implementation, supervision definition, transfer protocol, and reproducible weights/configuration |

Ultralytics documents simple task/mode commands in its [official CLI guide](https://docs.ultralytics.com/usage/cli/). Its [licensing page](https://www.ultralytics.com/license) describes AGPL and enterprise options. Record the applicable package and weights terms before distributing an integration; do not assume that optional dependencies remove licensing obligations.

MMDetection documents labeled/unlabeled pipelines and an existing teacher-student workflow in its [semi-supervised detection guide](https://mmdetection.readthedocs.io/en/latest/user_guides/semi_det.html). This makes it a candidate for integration, not evidence that it shares the first backend's dependency environment or covers every learning mode.

Source references reviewed during the documentation foundation on 2026-11-07. These are selection starting points, not claims that a particular algorithm currently leads benchmarks.

## Acceptance checklist for every adapter

- [ ] Record upstream repository, exact release/commit, package name, model config, and weight source/checksum.
- [ ] Record code, weight, and example dataset licenses, attribution, and redistribution constraints.
- [ ] Provide a verified install path and Python/framework/accelerator compatibility matrix.
- [ ] Declare exact supported task/learning-mode/operation combinations; reject everything else.
- [ ] Translate configuration without silently dropping unsupported common settings.
- [ ] Validate class maps, split assignments, annotation conventions, and output coordinates.
- [ ] Execute a real tiny train/validation/test/prediction workflow for the advertised operations.
- [ ] Verify failure behavior for missing dependencies, invalid input, unsupported mode, and incompatible checkpoint.
- [ ] Save normalized results and preserve upstream artifacts and provenance.
- [ ] Document deterministic limitations, resume support, device support, and explicit download behavior.
- [ ] Publish a short runnable example and the actual verification evidence.

Export, distributed execution, mixed precision, resume, and CPU training are separate capabilities. Do not advertise them merely because the upstream framework exposes an option.

## Dependency isolation

Keep each backend in its own optional extra and documented environment. If compatible packages cannot share an interpreter, use separate environments and introduce a subprocess boundary only when required by a real integration. Exchange versioned config/artifact files across such a boundary; report failures and environment identity clearly.

## Support ledger

| Backend | Model/release | Learning modes | Operations verified | Status |
| --- | --- | --- | --- | --- |
| Ultralytics | YOLO11n / 8.3.4 | Supervised detection | Python and CLI train, val, test, predict; Python local-checkpoint fine-tuning | Verified Windows CPU and single-device CUDA, Python 3.12.3; see [CUDA evidence](verification/YOLO_DINOV3_CUDA.md) and [CPU evidence](verification/BASELINE.md) |
| MMDetection | Faster R-CNN R50-FPN / 3.3.0; upstream Soft Teacher in semi-supervised mode | Supervised and semi-supervised detection | Train, val, test, predict; label budgets and same-detector comparison | Verified isolated Windows CPU and single-NVIDIA-GPU CUDA, Python 3.10.20 backend / 3.12.3 core; see [GPU guide](GPU.md) and [evidence](verification/OPENMMLAB_CUDA.md) |
| MMPreTrain + MMDetection | SimSiam ResNet-50 / 1.2.0; Faster R-CNN / 3.3.0 | Self-supervised pretraining with supervised or semi-supervised detection transfer | Pretrain; checked backbone transfer; downstream train/val/test/predict and random-vs-transfer comparison | Verified isolated Windows CPU and CUDA backends; see [GPU evidence](verification/OPENMMLAB_CUDA.md) and [M3 evidence](verification/SIMSIAM.md) |
| Transformers | DINOv3 ConvNeXt Tiny + DETR / 5.22.2 | Supervised downstream detection; optional local backbone initialization | Train, val, test, predict; explicit random and owned local-fixture paths | Windows CPU/CUDA execution verified; **official pretrained DINOv3 transfer pending**; see [status](DINOV3.md) and [evidence](verification/DINOV3.md) |

Add entries only after real upstream execution succeeds. YOLO supports random initialization or an explicitly supplied trusted local YOLO11n checkpoint. MMDetection supports random initialization, checked SimSiam backbone transfer into supervised or semi-supervised PyTorch-style Faster R-CNN, and complete supervised detector initialization of both Soft Teacher branches. See [weight compatibility](WEIGHT_COMPATIBILITY.md) and [initialization evidence](verification/INITIALIZATION.md). Evaluation/prediction uses its selected detector checkpoint. Single-GPU CUDA is supported for all registered workflows in their own verified environments; see [GPU setup](GPU.md). Explicit recovery, mixed precision, shared-metric comparison, portable bundles, and verified YOLO ONNX export are documented in [training operations](TRAINING_OPERATIONS.md). Actual two-GPU DDP verification remains open; fully unsupervised detection remains unsupported. Upstream identities, licensing decisions, dependency snapshots, and limitations are recorded in the [M1 report](verification/BASELINE.md), [M2 report](verification/SOFT_TEACHER.md), [M3 report](verification/SIMSIAM.md), [development guide](DEVELOPMENT.md), [semi-supervised guide](SEMI_SUPERVISED.md), [self-supervised guide](SELF_SUPERVISED.md), [unsupervised assessment](UNSUPERVISED_ASSESSMENT.md), and [third-party notices](../THIRD_PARTY_NOTICES.md).

The DINOv3 adapter uses upstream Transformers backbone/DETR entry points and `Trainer`, not Meta's 7B detector or an implementation of DINOv3's SSL training method. It accepts an explicitly supplied local Hugging Face safetensors backbone with recorded provenance. Owner-supplied metadata does not verify official origin. No licensed official weights were available during this integration; do not claim pretrained support or published accuracy from the random-fixture checks. The [model strategy](MODEL_ADAPTERS.md) specifies the protocol/catalog extension gate.
