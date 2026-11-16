# M2 verification evidence

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

The semi-supervised detection integration was exercised on 2026-11-07 with the Python 3.12.3 core and a separate Python 3.10.20 environment. The backend uses MMDetection 3.3.0, MMEngine 0.10.7, MMCV 2.1.0, torch 2.1.0+cpu, torchvision 0.16.0+cpu, and pycocotools 2.0.11 on Windows. Backend installation passed `pip check`. Exact runtime dependencies are in the [dependency snapshot](../../constraints/windows-mmdet-cpu-py310.lock.txt).

## Actual same-subset comparison

Both methods used the existing Faster R-CNN R50-FPN detector from the installed Soft Teacher recipe, with random initialization and no downloaded pretrained weights. Upstream Soft Teacher supplied the pseudo-labeling, uncertainty handling, and mean-teacher updates. The supervised baseline used the same detector and upstream supervised augmentation branch.

The source fixture contains four labeled training images and two images in each held-out split. `seeded_sample_v1`, selection seed 0, retained training images 1 and 3 as labeled and withheld labels for training images 0 and 2. Both methods used that exact derived dataset, training seed 0, two iterations, image size 128, total batch 2, and learning rate 0.000001.

| Metric (fractions, not percentages) | Supervised | Soft Teacher | Semi minus supervised |
| --- | ---: | ---: | ---: |
| Validation AP50:95 | 0.000 | 0.000 | 0.000 |
| Validation AP50 | 0.000 | 0.000 | 0.000 |
| Test AP50:95 | 0.003 | 0.000 | -0.003 |
| Test AP50 | 0.027 | 0.000 | -0.027 |
| Test AP75 | 0.000 | 0.000 | 0.000 |

Large-object AP is undefined (`-1`) for this fixture; its delta is `null` in the machine-readable report. Both selected checkpoints were frozen before test evaluation began. No test results were used to change the recipe or select a checkpoint.

**This is a single-seed synthetic execution check. Soft Teacher did not improve the measured scores. These tiny runs establish neither useful accuracy nor statistical significance or benchmark leadership.**

Wall time around the full training calls was 339.86 seconds for the baseline and 203.22 seconds for Soft Teacher on this run. These include data preflight, process/model startup, filesystem I/O, validation, and checkpoint writing; cold/warm cache effects and host load prevent any speed comparison. The baseline sees two labeled images per iteration, while Soft Teacher sees one labeled and one unlabeled image plus teacher/jitter work. Equal iteration counts do not equalize compute or labeled exposure. Random initialization uses the same seed/policy but teacher construction can consume additional RNG, so initial tensors need not be identical.

## Identities and provenance

- Source dataset fingerprint: `4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.
- Derived dataset fingerprint: `6f081c42490f25e74501d54e404867aeccc9b26970694effe6fc761151402a13`.
- Original upstream recipe: `soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py`, release `v3.3.0`.
- Recipe SHA-256: `de1e369e362a66fae99fac43d9d774a0c43779b34c6d01d183cfb67f58efcff6`.
- Upstream `MeanTeacherHook` momentum: 0.001; pseudo-label settings are saved in the resolved configuration and provenance.
- Final-iteration unlabeled classification loss was approximately 3.08 and unlabeled box-regression loss approximately 3.19. These positive upstream scalars establish execution of the unlabeled loss path, not pseudo-label quality.

The smoke recipe uses lowered pseudo-label score thresholds, fixed image scale, a 1:1 source ratio, random initialization, no warmup/decay, and a two-iteration budget. It is not a reproduction of the published pretrained 180k-iteration accuracy recipe. Online pseudo-labels are not serialized per image; provenance captures settings and teacher/student checkpoints without claiming that visibility.

## Reproduction and artifacts

Verification was split into groups so that passing heavyweight YOLO tests did not need repeating after changes confined to the MMDetection worker:

| Group | Result |
| --- | --- |
| Core/data/config/contracts | 38 passed in the core batch; the additional foreign-checkpoint guard test passed separately (39 current core checks) |
| Existing real YOLO integration | Both tests passed in the initial full run |
| Real MMDetection integration | 2 passed, no skips, in 1164.97 seconds; comparison, reload/prediction, and Windows process-tree cancellation covered |
| Current collected test set | 43 checks, all covered by the passing groups above |
| Ruff / formatting / Mypy | Passed; 22 Python files formatted, 14 source files type checked |
| Dependency checks | Both backend environments passed `pip check` |

The new foreign-checkpoint unit check confirms rejection before upstream's permissive checkpoint loading can occur. Core import/help still loads neither ML framework. The targeted MMDetection report has no warnings/skips; the earlier full run contained 13 upstream Matplotlib/Pyparsing deprecation warnings from the YOLO group.

Follow [setup and usage](../SEMI_SUPERVISED.md). The real integration command, from the configured core environment, is:

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
python -m pytest -m mmdetection -q --basetemp .wvision/m2-verified-final
```

The test sets the offline worker flag and blocks socket connections. A Windows host with local named-pipe access is required. Use a new base-temp path to preserve earlier evidence; pytest may remove an existing base-temp directory on later runs.

The actual comparison is under `.wvision/m2-verified-final/test_real_soft_teacher_compari0/runs/compare-8f2e60cb86e6404fb47f69f10094355f/comparison.json`. The parent folder contains both training runs, validation/test operations, structured training scalars, backend logs, resolved configurations, selected checkpoints, and full environment records. The raw targeted test report is `.wvision/m2-verification-final.log`. These are ignored local verification artifacts, not source files or distributed weights. Integrity-error tests may deliberately alter fixture artifacts.

An initial full-suite attempt passed 40 checks and hit the comparison harness's former 300-second limit. That exposed missing Windows process-tree cleanup in the test harness. Cleanup now covers virtual-environment redirector children, and standalone evaluation uses upstream inference plus `CocoMetric` without rebuilding a training runner. Font discovery is also shared within each output root. Earlier incomplete runs are retained separately; they are not the published comparison above.

The workspace is not a Git repository, so no revision identifier was available. GPU execution, Linux backend verification, external pretrained initialization, resume, distributed training, and fully unsupervised/self-supervised learning remain outside this milestone.
