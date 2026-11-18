# Training operations and real football checks

Verified locally on **2026-11-17**, with final recovery/package checks on
**2026-11-18**, in the First Sense working tree. No commit,
push, hosted CI execution, or external dataset upload was performed. Generated
datasets, environments, weights, logs, and reports remain ignored local artifacts.
See [usage and support boundaries](../TRAINING_OPERATIONS.md).

## Environment

Windows; Python 3.12.3 application, Python 3.10.20 OpenMMLab worker. One NVIDIA
GeForce RTX 4050 Laptop GPU (6,141 MiB). Existing isolated backends were reused:

| Backend | Actual versions | Precision checked |
| --- | --- | --- |
| YOLO | Ultralytics 8.3.4, torch 2.4.1+cu124 | FP32 CPU, FP16 CUDA |
| OpenMMLab | MMDetection 3.3.0, MMPretrain 1.2.0, MMCV 2.1.0, MMEngine 0.10.7, torch 2.1.0+cu121 | FP16 detector/SimSiam, BF16 SimSiam |
| DINOv3/DETR | Transformers 5.22.2, torch 2.8.0+cu128 | BF16 CUDA, local/random weights |
| Export | ONNX 1.17.0, ONNX Runtime 1.20.1, torch 2.4.1+cpu | FP32 CPU |

MMDetection BF16 fails in the pinned ROIAlign CUDA kernel; the public config
rejects it. DINO FP16 produced gradient overflow in the short random-weight case;
the public config rejects it. No claimed speedup follows from these checks.
Official licensed DINOv3 pretrained weights retain their earlier verification gate.

## Actual football data

Manifest: component-local `.wvision/football-ssl-10pct/dataset.yaml`.
Classes: Ball, Player, Referee. Splits: **87 labeled training, 785 unlabeled
training, 292 validation, 293 test**. Dataset fingerprint:

```text
ac8242b5c0a7b9a2b2b1fcdf2f09b3d155fd02aa44d71cf70c10a397d8ee2b48
```

The local acceptance script `.wvision/check-football-operations.py` completed:

| Workflow | Smoke settings | Result |
| --- | --- | --- |
| SimSiam pretraining | FP16, 64 pixels, batch 4, 2 iterations then resume to 3 | Passed |
| YOLO grid | FP16, 128 pixels, batch 4, SGD, learning rates .001/.01, 1 epoch | Both passed |
| Soft Teacher from SimSiam | FP16, 128 pixels, batch 2, 2 iterations | Passed |
| DINOv3/DETR | BF16, 64 pixels, batch 2, random initialization, 2 iterations | Passed |

All four detector candidates used the same fingerprint and the same shared COCO
validation protocol (IoU .50:.95, maxDets 1/10/100, score threshold .001). No test
metrics were used by the experiment command. Subsequently, one run per detector
family resumed to 2 epochs / 3 iterations, evaluated the 293 test images, bundled,
moved, reloaded, and predicted on CPU. YOLO's data-containing bundle retained its
fingerprint. Source runs and dataset hashes were unchanged.

YOLO exported at fixed batch one, 128 pixels, FP32, opset 17. ONNX checker passed;
ONNX Runtime matched native raw outputs on two deterministic tensors with
`rtol=1e-3`, `atol=1e-4`; maximum absolute error was **9.1552734375e-5**.
Other model families explicitly reject ONNX export and support native bundles.

Local evidence from the WForge directory:

- `.wvision/football-operations.log`
- `.wvision/football-operations-107104b1/acceptance.json`
- `.wvision/football-operations-107104b1/experiments-ee17e6a83ca9449790ba1406f62b08db/comparison.html`
- The adjacent JSON/CSV reports, plan, checkpoint hashes, and backend logs.

These are execution checks, not a useful accuracy comparison: schedules are tiny,
weights are random/local fixtures, budgets and input sizes differ, and the existing
football splits contain video overlap. Reliable generalization measurement requires
video/match-separated held-out data. No original annotations or split lists changed.

## Recovery and regression checks

The local `.wvision/check-football-interruption.py` interrupts YOLO after a complete
first-epoch recovery save, continues the planned three epochs, moves a bundle
containing dataset and optimizer state, then resumes that bundle to four epochs.
The source runs/bundle remain byte-for-byte unchanged and the fingerprint matches.
The initial evidence is `.wvision/football-interruption-72adc3cd/acceptance.json`;
the final immutable-snapshot repetition is recorded in
`.wvision/football-snapshot-interruption.log`.
Its acceptance record is
`.wvision/football-interruption-a558ad05/acceptance.json`.

Recovery uses atomic manifests and immutable checkpoint copies, retaining current
and previous committed snapshots. A failure-injection regression verifies that an
interrupted new copy cannot invalidate the prior checkpoint and a later retry works.
Active application and orphan backend writers block recovery. Tampered snapshots,
changed datasets/settings, and unsupported precision/device/export requests fail.

Core command: `python -m pytest -q` — **175 passed**, 31 optional integration cases
deselected. Ruff checks/formatting pass (120 source/example/document files);
mypy passes for 60 source modules. Local Markdown links and 18 Python documentation
examples were checked.

Real integration cases live in `tests/test_training_operations_integration.py`.
With the matching application interpreter and documented backend environment
variables, explicitly select `-m integration` and the desired tests. Local logs:

- `.wvision/operations-real-yolo.log`: 2 passed; includes resume, grid, bundle,
  ONNX, exact-ground-truth COCO AP=1 and non-validation prediction rejection.
- `.wvision/operations-real-mm.log`: FP16 SimSiam/detector recovery and relocation.
- `.wvision/operations-real-mm-bf16-confirmed.log`: BF16 SimSiam recovery and
  explicit BF16 detector rejection.
- `.wvision/operations-real-dino.log`: frozen/unfrozen BF16 recovery and relocation.
- `.wvision/operations-snapshot-backends.log`: repeats the four OpenMMLab/DINO
  cases after introducing immutable recovery copies — **4 passed**. The first
  repetition exposed a Windows path-length limit in temporary snapshot names;
  shortening those names at their owner resolved it.
- `.wvision/operations-cpu-interrupt.log`: actual CPU YOLO interruption/recovery.

Transformers' redundant Trainer weight reload emits legacy-key warnings after
the model has already been restored through `from_pretrained`. A separate check
against a real football checkpoint verifies full initial loading, compares all
**443 tensors** before/after that redundant reload, and finds **zero changes**
(`.wvision/operations-dino-resume-state.log`). Optimizer/scheduler/RNG recovery
continues through the upstream Trainer.

No multi-GPU test is counted as passed: this laptop has one GPU. A real request for
two GPUs fails before training (`.wvision/operations-ddp-failure.log`). Managed DDP
launching, rank isolation, and configuration checks do not establish synchronization
correctness. Run the explicit `distributed` integration cases on two GPUs.

## Packaging and automation

The wheel/sdist include the new operations modules, standalone workers, examples,
and notices; source distributions include the experiment YAML. Local data, weights,
environments, and generated runs are excluded. Package/import evidence is retained
in `.wvision/operations-build.log` and `.wvision/operations-package.log`.
All 60 packaged source modules match the checkout. The extracted wheel imports
its catalog/adapters without optional ML frameworks, runs the CLI, reloads SimSiam,
and performs actual CPU prediction for all three detector families.

The scheduled/manual [backend workflow](../../../.github/workflows/backends.yml)
configures separate Windows CPU environments and opt-in self-hosted CUDA/DDP
jobs. YAML/matrix validation and all 48 rendered PowerShell snippets pass local
syntax checks. **Hosted execution remains
unverified until the workflow runs on GitHub.** Failure evidence excludes datasets
and weights. Actual two-GPU execution and hosted automation are the remaining M3m
verification gates before calling the whole milestone complete.
