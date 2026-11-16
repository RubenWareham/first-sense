# Verification evidence

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
