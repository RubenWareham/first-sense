# Legacy cleanup verification

Maintenance slice after M3d, **2026-11-08**. No Git revision is available because
this workspace is not a Git repository. Models, upstream algorithms, dependency
versions, CPU-only support, and the outstanding official-DINOv3-weight gate are
unchanged. See [migration notes](../MIGRATION.md) for intentional API changes.

## Delivered cleanup

- Removed six duplicate flat YAML profiles and their `configs/README.md`.
- Removed flat-file parsing from the four backend settings resolvers and removed
  their path parameter from the detection protocol.
- Removed training/pretraining `config=` and all CLI `--config` options.
  Comparisons now use `model=` / `--model`, matching the other workflows.
- Simplified shared model-default/override resolution, preserving whole-map
  replacement of `backend_options` and safe model YAML validation.
- Removed unused `Initialization.to_dict` and duplicate detection initialization
  provenance fields. Configuration provenance stays in `initialization_sources`,
  separate from adapter initialization settings; `run.json` uses `initialization`.
- Removed manifest-directory budget metadata fallback. The data root owns
  `budget.json`, even when the manifest lives elsewhere.
- Removed six compatibility-only parity tests, added four public argument-removal
  cases and an initialization-metadata collision regression, and retained behavior checks for defaults, overrides, source snapshots,
  failed runs, data isolation, and saved-run reload.
- Updated current documentation/examples/agent rules and replaced the speculative
  architecture tree with the actual module layout. Historical execution reports
  retain original commands and results with an explicit migration notice.

Existing generated runs, datasets, weights, environments, and checkpoints were
preserved. A completed pre-cleanup schema-1 Soft Teacher run from
`.wvision/initialization-verified/test_real_initialized_soft_tea1/runs/` also reloads
through the current `Vision.from_run`, without importing ML frameworks.

## Commands and checks

Existing verified Windows CPU environments are reused. Core Python is 3.12.3;
MMDetection/MMPreTrain use the isolated Python 3.10.20 backend, and DINOv3 uses its
separate Python 3.12.3 backend. Dependency snapshots remain under `constraints/`.

```powershell
$env:WVISION_MMDET_PYTHON = (Resolve-Path .wvision/mmdet-env/Scripts/python.exe).Path
$env:WVISION_TRANSFORMERS_PYTHON = (Resolve-Path .wvision/transformers-env/Scripts/python.exe).Path
.wvision/clean-env/Scripts/python -m pytest -q
.wvision/clean-env/Scripts/python -m pytest -m '' -q --basetemp=.wvision/cleanup-verified
.wvision/clean-env/Scripts/python -m pytest tests/test_dinov3_integration.py::test_real_dinov3_train_evaluate_predict_and_reload -m integration -q --basetemp=.wvision/cleanup-dinov3-fixed
.wvision/clean-env/Scripts/python -m ruff check src examples tests
.wvision/clean-env/Scripts/python -m ruff format --check src examples tests
.wvision/clean-env/Scripts/python -m mypy
.wvision/clean-env/Scripts/python -m pip check
.wvision/mmdet-env/Scripts/python -m pip check
.wvision/transformers-env/Scripts/python -m pip check
```

The final default suite passed **118 tests**, deselecting eleven integration cases,
in **18.44 seconds**. All **ten available real integration cases** passed across
the broad run and focused rerun; official pretrained DINOv3 acceptance remains
skipped because licensed local weights are unavailable. Ruff lint and formatting pass for 44 Python files; Mypy passes
for all 26 source modules. All three dependency checks pass. Local Markdown links
and the updated CLI help were also checked. All ten current Python documentation
blocks parse; all 34 current CLI examples parse, including 18 valid catalog model
selections. All 27 local Markdown documents have valid local links.

The initial broad run finished with **126 passed, one failed, one skipped** in
**619.22 seconds**, plus thirteen existing Matplotlib/Pyparsing deprecation
warnings. The failure exposed a metadata collision: source provenance overwrote
DINOv3's `initialization` setting in `config.json`. The fix keeps provenance under
`initialization_sources`. A new core test verifies both fields survive serialization.
The affected real DINOv3 train/val/test/predict/reload workflow then passed in
**52.36 seconds**. The final core suite was also rerun after this fix; the other
nine real integration cases had passed in the broad run. This is verification
across runs, rather than a claim that the original full invocation was green.

Logs: `.wvision/cleanup-core.log`, `.wvision/cleanup-verified.log`, and
`.wvision/cleanup-dinov3-fixed.log`. Real upstream
tests run with Windows local subprocess communication allowed and networking
blocked. Artifacts remain under ignored `.wvision/cleanup-verified/`. Some failure
tests intentionally damage their owned files; these are execution evidence, not
released weights or evidence of detection quality.
