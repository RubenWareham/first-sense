# DINOv3 and adapter-interface verification

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

Date: **2026-11-08**. Scope: existing Transformers DINOv3 ConvNeXt Tiny / DETR downstream detection and a shared detection adapter interface. Official pretrained weights were unavailable; this report does not claim their compatibility, pretrained quality, or DINOv3 SSL training support. The workspace has no Git repository, so no commit identity is available.

## Environment

Core: existing Python 3.12.3 `.wvision/clean-env`, retaining YOLO's torch 2.4.1 CPU environment. OpenMMLab remains in its existing Python 3.10 `.wvision/mmdet-env`. A new Python 3.12.3 `.wvision/transformers-env` has `include-system-site-packages=false` and does not install WVision itself.

The new backend uses torch 2.8.0+cpu, torchvision 0.23.0+cpu, Transformers 5.22.2, Accelerate 1.10.1, safetensors 0.8.0, pycocotools 2.0.11, SciPy 1.15.3, NumPy 1.26.4, and Pillow 10.3.0. See [installation](../DINOV3.md), [compatibility constraints](../../constraints/transformers-cpu-py312.txt), and the [full dependency snapshot](../../constraints/windows-transformers-cpu-py312.lock.txt). These are version pins, not wheel hashes. CPU wheels came from the official PyTorch index. The backend passes `pip check`.

Earlier Transformers 4.57.1 was rejected during selection because its AutoBackbone mapping did not support this DINOv3 backbone. The selected current release requires PyTorch >=2.6, so a separate environment was used. Other backend environments were not upgraded. Installation and actual Windows ML tests used execution outside the restricted process sandbox; tests block worker socket connections and set Hub offline mode.

## Checks and commands

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest -m '' -q --basetemp=.wvision/dinov3-verified-final
.wvision/clean-env/Scripts/python -m pytest -q
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pip check
.wvision/mmdet-env/Scripts/python -m pip check
.wvision/transformers-env/Scripts/python -m pip check
```

The full combined regression passed **77 tests**, with **one skipped** official-weight acceptance test, in **374.84 seconds**. It includes 70 core/contract checks and seven real integration checks across all supported backends. Thirteen existing Matplotlib/Pyparsing deprecation warnings came from YOLO's dependency stack. The final default run passed **70 core/contract tests**, deselecting eight integration cases, in 12.66 seconds. Raw logs are `.wvision/dinov3-core-final.log` and `.wvision/dinov3-verification-final.log`.

Targeted real checks already passed: the random-initialization public train/val/test/predict/reload test, and the local-backbone/frozen-state/rejection test. The latter passed in 36.35 seconds after adapter fixes. Logs from unsuccessful probes are retained under ignored `.wvision/dinov3-*` directories; these are not support evidence.

Core checks cover framework-free import/catalog/help, joint learning/operation/device capability rejection, configuration validation, explicit initialization with no fallback, local weight schema checks, and custom executable configuration rejection. A catalog-extension boundary test reuses an adapter through a new catalog entry and exercises the existing facade without model-ID branches; that mock is not counted as model support.

Ruff lint and formatting pass for 38 Python files; Mypy passes for all 24 source modules. All three environment dependency checks pass. Local Markdown verification found 20 documents and 112 working local links. Real regressions cover YOLO, Soft Teacher, SimSiam transfer, and Windows worker cleanup through the shared process transport. A pre-existing M3 schema-1 Faster R-CNN artifact also loaded successfully through the new facade, separately from generated test runs.

## Real DINOv3 execution

The detection model is the full upstream ConvNeXt Tiny backbone plus standard DETR, with **45,537,638 parameters** for the fixture's one class. No smaller substitute architecture or mock model is used. The fixture is the existing eight original CC0 rectangle images: four train, two validation, two test. Its dataset identity is `4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.

The random run trains all model parameters for two upstream Trainer steps, batch two, 64-pixel square inputs, seed zero, learning rate 0.0001. It selects the final checkpoint using validation loss. Recorded validation loss is **5.8058300018**. Explicit validation and test COCO AP50:95/AP50/AP75 are **0.000**; large-object AP is undefined (-1). These results establish execution only.

The local-backbone test creates a real upstream DINOv3 ConvNeXt Tiny representation checkpoint from random initialization with seed 17. It labels its provenance as random, loads it locally, and trains only the detector head with a frozen backbone. It verifies complete compatibility and exact equality of all **178 backbone tensors**, and separately loads the saved detector to confirm that those tensors remained unchanged during training. There are **17,719,046 trainable parameters** in this frozen configuration. Validation loss is **5.8547978401**. A pooling-only final layer normalization is the only allowed unused representation state.

The test deliberately replaces the source safetensors file with an incomplete checkpoint and verifies that a new training operation fails with a durable failed status/log instead of randomly filling missing weights. Separately, CLI prediction from an already trained detector succeeded after its source backbone was corrupted, proving that inference uses the complete detector checkpoint. No source model directory is required for detector reload.

The public random workflow checks both Python and CLI val/test/predict, original image dimensions, 100 DETR queries, class mapping, normalized output records, checkpoint reload, and changed-model-config rejection. Training receives neither test nor unlabeled annotations. COCO category IDs are translated between 1..N staging and zero-based upstream DETR labels. Test uses pycocotools with `maxDets=[1,10,100]` after training/validation selection.

## Identities and retained artifacts

The new backend records installed upstream source hashes:

- DETR modeling file: `3650e9f804d0f513087c4319ff65b809308feadb9dd6c296c728a2929adf8978`.
- DINOv3 ConvNeXt modeling file: `5a28b491b9409aa94e967c11e223cd0cb486e7b74ee8079b1aaddcf0a8a5a480`.
- Owned local fixture `config.json`: `06a6b3c75f615f3177143aa4ea196a4ae3a255f94d44ac6daf8664c43467172c`.
- Owned local fixture weights before deliberate corruption: `36106a2cd40df990c02999e3276d5f2bfe5aa3ac041cc6d14ec9616181cc1628`.

Combined-run artifact locations:

```text
.wvision/dinov3-verified-final/
  test_real_dinov3_train_evaluat0/runs/train-7def0265a9284711a6956d5f089f5c69/
  test_real_local_dinov3_backbon0/runs/train-bca1543c79a14e59b7366aa2104d7d71/
```

Checkpoints, resolved model configuration, dependency snapshots, source/initialization metadata, worker requests/results/logs, training scalars, and evaluation predictions remain local ignored artifacts. Integrity tests intentionally alter some configuration/source files; generate fresh runs for practical use. No pretrained weights or datasets are distributed with the repository.

## Upstream compatibility decisions

The pinned `AutoBackbone.from_pretrained` probes Hub `repo_exists` even for local paths. The adapter uses the concrete upstream `DINOv3ConvNextBackbone.from_pretrained` class with offline/local-only settings instead; no upstream code is patched. Model construction still uses upstream DETR's backbone interface.

Transformers saves compatibility-format checkpoint keys. Direct safetensors state loading and Trainer's unnecessary final best-model reload do not apply all required key conversions in this configuration. Detector inference therefore uses upstream `DetrForObjectDetection.from_pretrained` with the independently hashed resolved configuration, local safetensors, and strict loading diagnostics. Training validates/saves once at the final step, so it does not need a second best-model reload. Actual saved checkpoint loading is tested end to end. No training loop, architecture, matching, or loss code was reimplemented.

## Outstanding official-weight gate

`test_official_pretrained_dinov3_when_supplied` is a distinct integration case. It skips when `WVISION_DINOV3_WEIGHTS` is absent; no random fixture is substituted. After obtaining official licensed ConvNeXt Tiny weights and recording their revision/provenance, run:

```powershell
$env:WVISION_DINOV3_WEIGHTS = (Resolve-Path .wvision/weights/dinov3-convnext-tiny).Path
.wvision/clean-env/Scripts/python -m pytest tests/test_dinov3_integration.py -m integration -k official -q
```

Only actual execution against those weights can close the pending pretrained verification gate. Even that smoke test would not establish accuracy or state-of-the-art performance. GPU, additional DINOv3 sizes, Meta's 7B detector, DINOv3 SSL pretraining, semi-supervised DETR, export, resume, and quantitative pretrained comparisons remain unsupported. See [usage](../DINOV3.md) and [model extension strategy](../MODEL_ADAPTERS.md).
