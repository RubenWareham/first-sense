# Verification evidence

[YOLO11 and RTMPose sizes and RTMDet detection](../MILESTONES.md#m4b--yolo11-and-rtmpose-sizes-and-rtmdet-detection)
records checks for the expanded model families.

[Supervised 2D pose](POSE.md) records local CPU and CUDA train, evaluation and
prediction checks on 2026-11-19.

[MMDetection ONNX export](MMDET_ONNX.md) records Faster R-CNN/Soft Teacher
conversion, real-image parity, strict failure checks, and the postponed official
DINOv3 weight gate on 2026-11-18.

[Grouped evaluation, cross-dataset transfer, and DINOv3 ONNX](EVALUATION_TRANSFER_EXPORT.md)
records the 2026-11-18 extension and its real backend checks.

[Training operations and real football checks](TRAINING_OPERATIONS.md) records
recovery, precision, relocation, YOLO ONNX export, and four-candidate comparison,
including the remaining multi-GPU and hosted-workflow gates.

[First Sense first-commit preparation](FIRST_COMMIT.md) records the current
source cleanup, package checks, ignore audit, and remaining publication actions.

Current capabilities, remaining gates, and later model-folder verification are
recorded in [milestones](../MILESTONES.md). Use current
[setup](../DEVELOPMENT.md), [GPU instructions](../GPU.md), and
[migration notes](../MIGRATION.md) rather than treating old commands as current API.

The reports below are **dated historical evidence**, retained for dependency,
recipe, license, split-isolation, and execution provenance. Their paths, counts,
local environments, and Git revisions describe the original check. Some refer
to the earlier standalone WVision repository; those revisions/artifacts are not
part of this First Sense checkout. Generated datasets, logs, checkpoints, and
environments are excluded. Tiny fixtures establish execution, not accuracy or
state-of-the-art results. Hosted CI results require an actual push.

| Report | Scope |
| --- | --- |
| [Baseline](BASELINE.md) | Initial YOLO CPU workflow |
| [Soft Teacher](SOFT_TEACHER.md) | Initial semi-supervised comparison |
| [Simsiam](SIMSIAM.md) | Representation pretraining and detection transfer |
| [Dinov3](DINOV3.md) | DINOv3 random/local-backbone execution; official weights pending |
| [Initialization](INITIALIZATION.md) | Checked backbone/full-detector initialization |
| [Openmmlab Cuda](OPENMMLAB_CUDA.md) | SimSiam / Faster R-CNN / Soft Teacher CUDA |
| [Yolo Dinov3 Cuda](YOLO_DINOV3_CUDA.md) | YOLO and DINOv3 CUDA |
| [Catalogs](CATALOGS.md) | Model/dataset catalog workflow |
| [Cleanup](CLEANUP.md) | Earlier legacy cleanup |
| [Github Preparation](GITHUB_PREPARATION.md) | Earlier standalone WVision publication preparation |
