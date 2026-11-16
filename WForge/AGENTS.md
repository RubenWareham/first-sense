# Agent instructions

## Purpose and current state

WVision integrates existing computer vision models behind a small, consistent Python API and CLI. Priorities are object detection, learning from limited labels, reproducible evaluation, and ease of use. Next come 2D pose estimation, image classification, and other tasks.

This repository implements supervised YOLO11n, an isolated MMDetection Soft Teacher/Faster R-CNN workflow, and upstream SimSiam pretraining with supervised or semi-supervised detection transfer. All registered workflows also support verified single-device CUDA in their own environments; read [GPU setup and device rules](docs/GPU.md) before changing device handling. Read [weight compatibility](docs/WEIGHT_COMPATIBILITY.md) before changing initialization contracts. Read [development instructions](docs/DEVELOPMENT.md), [semi-supervised instructions](docs/SEMI_SUPERVISED.md), and [self-supervised instructions](docs/SELF_SUPERVISED.md) for actual usage and limits. Fully unsupervised detection remains unsupported; see the [assessment](docs/UNSUPERVISED_ASSESSMENT.md). Architecture documents also include future design targets. Check [milestones](docs/MILESTONES.md) before implementation; never report a planned capability as implemented.

## Read before changing code

1. Read this file and [engineering rules](docs/ENGINEERING.md).
2. Read [architecture](docs/ARCHITECTURE.md) and [data and evaluation rules](docs/DATA_AND_EVALUATION.md).
3. For backend work, read [integration rules](docs/INTEGRATIONS.md).
4. Find the active milestone in [milestones](docs/MILESTONES.md); implement its smallest complete slice.
5. Check for more specific `AGENTS.md` instructions in the directories being changed.

## Non-negotiable rules

Registered model integrations live under `src/wvision/models/<framework>/<model>/`: YOLO11, Faster R-CNN/Soft Teacher, SimSiam, and DINOv3/DETR. Their settings, source checks, adapters, and actual upstream operation calls belong there. Shared version/runtime/interpreter utilities stay under `adapters/`; worker transport/device helpers stay under `adapters/shared/`. The root `models/` directory remains the independent YAML catalog.

For new model work, read [model adapter strategy](docs/MODEL_ADAPTERS.md) and the relevant backend guide. Detection additions use the explicit registry and typed adapter protocol; do not reintroduce model-ID branches into `Vision`. The [DINOv3 guide](docs/DINOV3.md) distinguishes verified random/local-fixture execution from pending official pretrained-weight verification.

Shared model configs belong in [models](models/README.md), grouped by task/model, and dataset manifests belong in [datasets](datasets/README.md). Keep those concerns independent. Use model documents and explicit overrides; do not restore flat-config loaders or duplicate profiles under `configs/`. Label smoke/default/local-weight profiles accurately; do not present unbenchmarked settings as state of the art.

- Integrate upstream models and training methods. Do not implement new architectures, losses, teacher updates, optimizers, or research methods here.
- Keep framework imports and behavior inside model integration execution modules, shared adapters, or isolated workers. The core must import without optional ML dependencies.
- Keep the first public workflow simple: configure a dataset, select a supported model, train, validate, test, predict.
- Support only combinations that a tested backend actually implements. Reject unsupported task, method, and operation combinations before starting work. Never silently switch to supervised training or another model.
- Keep labeled training, unlabeled training, validation, and test data separate. Test labels must never influence training, pseudo-labeling, threshold selection, or checkpoint selection.
- Delegate training to upstream code. WVision owns configuration, orchestration, validation, provenance, and result normalization.
- Use explicit typed contracts and small modules. Avoid speculative frameworks and unnecessary abstractions.
- No hidden downloads, cloud uploads, arbitrary config execution, or automatic installation during imports. Make downloads explicit in the user workflow.
- Document upstream versions, model/weight sources, licenses, and compatibility evidence. Do not claim a backend is supported from a paper or successful import alone.
- Do not call a model state of the art without a dated, reproducible comparison on the relevant dataset, metric, and label budget.
- Preserve user changes. Do not commit datasets, weights, generated runs, credentials, or machine-specific paths.

## Definition of done

- The requested behavior works through its public entry point.
- Relevant success, failure, and unsupported cases are checked.
- Configuration and output artifacts are documented, with a minimal example.
- Relevant tests and formatting/type checks pass; report skipped or unavailable checks honestly.
- Documentation, capability status, and milestone evidence match the implemented behavior.
- No unrelated refactors or new heavyweight dependency is introduced without a concrete need.

## Working style

Inspect existing code before proposing changes. State assumptions when requirements are incomplete. Prefer one vertical slice over many stubs. Explain what changed, how it was verified, and any remaining limitations. Do not mark a milestone complete based on scaffolding, mocks, or documentation alone.
