# Model-config and dataset-catalog verification

Historical execution evidence. Commands and compatibility statements describe the version tested here. See [migration notes](../MIGRATION.md) for the current API after legacy cleanup.

Verified **2026-11-08** on the existing Windows CPU environments. Scope: the repository structure and its public configuration entry points. No new model, learning algorithm, weight download, or semi-supervised backbone-transfer capability was added. No Git revision is available because this workspace is not a Git repository.

## Delivered behavior

- Twelve model documents under `models/`, grouped by task and model, with profile/description/upstream metadata and training defaults.
- `Vision(model_yaml_path)` and CLI `train --model path.yaml`, retaining registered model IDs and legacy flat `config=` settings.
- SimSiam `pretrain(..., model=path.yaml)` and comparison commands accepting the corresponding detector model documents.
- Independent dataset example/template manifests under `datasets/` with paths resolved against the manifest and data root.
- Model-definition snapshots and source hashes saved with resolved settings; completed runs reload without the original model YAML.
- Data-root label-budget metadata lookup, preserving reproducible selections when catalog manifests live elsewhere.

See [model config usage](../../models/README.md) and [dataset preparation](../../datasets/README.md). Smoke, baseline, and local-weight profiles describe intent; baseline schedules are not benchmarked quality presets. Official pretrained DINOv3 verification remains open.

## Commands and results

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest -m '' -q --basetemp=.wvision/catalog-verified
.wvision/clean-env/Scripts/python -m pytest -q
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pip check
.wvision/mmdet-env/Scripts/python -m pip check
.wvision/transformers-env/Scripts/python -m pip check
```

The full suite passed **111 tests**, with **one skipped** official pretrained DINOv3 acceptance case, in **377.04 seconds**. Thirteen existing Matplotlib/Pyparsing deprecation warnings came from YOLO's dependency stack. The default core run passed **104 tests**, deselecting eight integration cases, in 14.72 seconds. Ruff lint and formatting pass for 40 Python files; Mypy passes for all 25 source modules. All three environment dependency checks pass. Local Markdown links are verified separately.

Raw logs are `.wvision/catalog-verification.log` and `.wvision/catalog-core.log`. Real Windows ML checks ran outside the restricted process sandbox with networking blocked in workers/tests. No package installation or framework upgrade was needed for this milestone.

## What the checks establish

Real YOLO training, validation, test, prediction, and checkpoint reload run through the new Python constructor and CLI model-path selection. Legacy registered-ID fine-tuning and invalid checkpoint/data tests remain active. The real DINOv3 random-weight workflow also selects its model document, preserves actual initialization settings, and passes its existing public-operation and integrity checks.

The real Soft Teacher comparison accepts its catalog model config and a dataset manifest stored outside the generated budget's data root. Its report retains the exact labeled count, seed policy, and fingerprint from `budget.json`; supervised and semi-supervised runs still use their explicit distinct settings. Actual upstream unlabeled losses remain exercised. Real SimSiam pretraining selects its representation model file, and the downstream transfer comparison accepts its corresponding detector config. Existing frozen-backbone checks and Windows worker cancellation also pass.

Core checks validate all twelve documents against registered adapters/settings without importing optional ML frameworks. They verify six legacy/profile parity mappings, field precedence, whole-map replacement of `backend_options`, invalid/unknown fields, safe YAML rejection, missing files, changed-source rejection, task compatibility, representation entry-point errors, saved source snapshots, and run reload after the original model document is removed.

Dataset checks create actual fixture data through a copied catalog manifest, compare its identity against the generated original manifest, prepare a derived budget, load that budget through the independent semi-supervised catalog, and reject mismatched budget metadata. Only paths and audit metadata change; source/derived data fingerprints remain the same as the earlier fixture identities:

- Original: `4d5ae40bb038e6ead93e13b76bd0676b7d6e0d5e5cd73202435066f10cd0e11c`.
- Two-label budget: `6f081c42490f25e74501d54e404867aeccc9b26970694effe6fc761151402a13`.

The original generated images/annotations remain CC0. Source data, run snapshots, checkpoints, and test outputs remain under ignored `.wvision/catalog-verified/`. Some integrity tests deliberately alter their generated files; these are audit artifacts rather than released usable weights. No accuracy improvement or state-of-the-art claim follows from these smoke executions.

## Compatibility and limits

Model files select existing reviewed registry entries and do not register executable adapters. Metadata references are not evaluated or downloaded. Settings resolve from adapter defaults, model training defaults, an optional flat settings overlay, and explicit overrides. Runtime class maps come from the dataset. All current backends retain their previously documented CPU/device/learning constraints.

The source catalogs currently ship with the repository, not as installed package resources. Templates require real local data; baseline schedules still need dataset-specific validation and tuning. This change does not fetch Hugging Face assets, accept licenses, expose DINOv3 SSL pretraining, or make arbitrary backbones work with Soft Teacher. Those are separate integrations with their own acceptance gates.
