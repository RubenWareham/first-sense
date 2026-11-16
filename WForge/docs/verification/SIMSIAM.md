# M3 verification: self-supervised pretraining and detection transfer

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

Verified **2026-11-08** on Windows CPU. M3 is complete for upstream SimSiam representation training and supervised detection transfer, with a separate decision to leave fully unsupervised detection unsupported. This is execution evidence on original synthetic data, not a quality benchmark. No Git revision is available because this workspace is not a Git repository.

## Environment and installation

Core: Python 3.12.3 in the existing `.wvision/clean-env` from M1. Backend: Python 3.10.20 in the isolated `.wvision/mmdet-env` from M2, extended with MMPreTrain 1.2.0. Existing backend pins remain torch 2.1.0+cpu, torchvision 0.16.0+cpu, MMCV 2.1.0, MMEngine 0.10.7, and MMDetection 3.3.0. Both environments pass `pip check`.

Actual installation: `.wvision/mmdet-env/Scripts/python -m pip install mmpretrain==1.2.0`. New package versions were captured in [constraints](../../constraints/selfsup-cpu-py310.txt) and the [complete runtime snapshot](../../constraints/windows-selfsup-cpu-py310.lock.txt). The documented constrained install was also resolved offline with `--dry-run --no-index` against the installed environment. This verifies the additive install, not a new empty-environment rebuild. See [setup and usage](../SELF_SUPERVISED.md) for the upstream CPU wheel installation path.

Package installation and real Windows backend checks required execution beyond the restricted process sandbox. The first probe exposed a Python module-name collision, which was fixed by naming the configuration adapter `self_supervised.py`. A subsequent restricted worker stalled before training; its identified process tree was stopped, and the offline verification succeeded outside that restriction. No fallback training implementation or algorithm patch was used. Failed probe artifacts remain under ignored `.wvision/m3-probe` directories.

## Commands and checks

From the repository root, the full suite was run with:

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest -m '' -q --basetemp=.wvision/m3-verified
.wvision/clean-env/Scripts/python -m pytest -q
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pip check
.wvision/mmdet-env/Scripts/python -m pip check
```

The full suite passed **62 tests in 349.14 seconds**, without skips. This includes 57 core/contract tests and five real integration tests: two YOLO tests, Soft Teacher comparison, Windows cancellation cleanup, and SimSiam/detection transfer. Thirteen existing Matplotlib/Pyparsing deprecation warnings came from YOLO's dependency stack. Backend logs also retain upstream deprecation and image-augmentation warnings. The final default core run passed **57 tests**, deselecting the five integration tests. A supervised-only style guard and comparison-provenance wording were finalized during the full run; the additional core run checks the final source. No training/evaluation algorithm changed during verification.

Ruff lint and formatting pass; Mypy checks all 18 source modules. Local Markdown links are checked separately. Tests block socket connections inside the actual training/inference workers. A successful import or a boundary mock is not counted as model support.

Raw local evidence: `.wvision/m3-verification.log` and `.wvision/m3-core-final.log`. These files and generated runs are ignored, local artifacts rather than hosted reproducibility assets.

## Public execution and artifacts

The real integration test uses CLI `pretrain` and `compare-pretraining`, which invoke the shared Python application layer. The comparison performs actual upstream detector training, validation, and test evaluation. Reloaded detector prediction is checked through Python; CLI test evaluation is checked after the source representation checkpoint is deliberately corrupted. A new transfer correctly rejects that corruption before starting another run.

The test confirms that only unlabeled training images are staged for SimSiam, with no human annotations or held-out inputs in its worker request, and that its final checkpoint is selected without held-out evaluation. Both detector arms receive the same labeled subset and no unlabeled or test annotations during training. Backbone transfer checks all **318** state tensors for complete key/shape compatibility and exact equality after upstream initialization. Detector heads are not transferred. Run reload and the existing checkpoint/data/config integrity tests also pass.

Artifact root:

```text
.wvision/m3-verified/test_real_simsiam_and_detectio0/runs/
  pretrain-bafc53779dcf43a880c9833144bca173/
  compare-pretraining-ac12f33e6bea4ae9b399b33480339ae3/
    comparison.json
    train-6245426ebd514805a25c912143de34d9/   # random initialization
    train-813f751cedca4fbc8430567deec93570/   # SimSiam backbone transfer
```

The integrity test intentionally corrupts the representation checkpoint at its end; these test artifacts are audit evidence, not reusable released weights. Generate a fresh run for use. The separate successful manual probe remains under `.wvision/m3-probe/runs/compare-pretraining-31e1373496b24eba98e0d4a2b4e936d4`.

## Identities and supervision

Original fixture: eight distinct CC0 synthetic rectangle images, four training/two validation/two test. Training-only seed-0 selection uses two labeled samples and two unlabeled samples; held-out membership is unchanged.

- Source dataset SHA-256 identity: `4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.
- Derived dataset identity: `6f081c42490f25e74501d54e404867aeccc9b26970694effe6fc761151402a13`.
- Installed SimSiam recipe SHA-256: `32badbe20d0ff74fd76cb9b96c619992223cc3d5792ccdaaef5cd7f122766f06`.
- Original representation checkpoint SHA-256 before the deliberate integrity test: `dbee3bea38d029841864bb30f8f5e05542304afe6b794a9effcf85d1139dde5f`.

`model-source.json` records the upstream recipe and all three inherited configuration hashes, random initialization, no external weights, CPU BatchNorm adaptation, requested crop size, constant learning rate, and iteration budget. `dataset.json` preserves the complete audit snapshot; only the unlabeled image list is given to the representation trainer.

SimSiam: two iterations, 64-pixel crops, batch two, seed zero, learning rate 0.0001, upstream two-view augmentation. Its final structured training loss is **-0.0138272457**; negative cosine loss is valid and is not a detector accuracy metric.

Both downstream arms: supervised Faster R-CNN R50-FPN, PyTorch-style ResNet with matching RGB normalization, two iterations, image size 128, batch two, seed zero, learning rate 0.000001, identical labeled subset/augmentation/selection settings. One arm starts randomly; the other loads only the SimSiam backbone. Same seed does not guarantee identical random head tensors across initialization paths. Unlike M2's Caffe-style recipe, this is a separate transfer protocol. Additional representation training compute is disclosed.

## Measured results

Values are upstream MMDetection/pycocotools COCO AP in the 0–1 range. Both detector checkpoints maximize validation `coco/bbox_mAP`. Test evaluation occurs after both checkpoints are selected.

| Split / metric | Random initialization | SimSiam backbone | Transfer minus random |
| --- | ---: | ---: | ---: |
| Validation AP50:95 | 0.005 | 0.003 | -0.002 |
| Validation AP50 | 0.027 | 0.009 | -0.018 |
| Test AP50:95 | 0.002 | 0.003 | +0.001 |
| Test AP50 | 0.008 | 0.023 | +0.015 |
| Test AP75 | 0.000 | 0.000 | 0.000 |
| Test small AP | 0.004 | 0.011 | +0.007 |
| Test medium AP | 0.000 | 0.002 | +0.002 |
| Test large AP | Undefined (-1) | Undefined (-1) | Null |

The comparison report preserves all raw values, run paths, detector settings, label budget, pretraining provenance, and training-call wall times. Those times include startup, preflight, training, validation, I/O, and checkpointing and are not a speed benchmark. A two-step synthetic single-seed comparison cannot establish accuracy gains or generalization.

## Unsupported capabilities and separate decision

GPU, distributed execution, external pretrained checkpoints, resume, and SimSiam-to-Soft-Teacher transfer are unverified and rejected. Semantic fully unsupervised detection remains unsupported. The separate [LOST/TokenCut assessment](../UNSUPERVISED_ASSESSMENT.md) records concrete environment, local-weight, class-mapping, and evaluation blockers without claiming those implementations were executed. Downstream human labels are explicitly retained in the implemented workflow.
