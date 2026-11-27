# Milestones

This is the ordered implementation plan. M0 through M3, the M3c catalog structure, and M3d initialized semi-supervised detection are complete for their documented environments and decisions. M3b's code integration is verified with its official pretrained-weight gate still open. These slices precede M4 and further model integrations. Checkboxes represent deliverables; each milestone's exit gate must also pass. Record evidence here as work lands. Do not invent completion dates or mark quality claims from smoke tests.

## M0 — Documentation foundation

- [x] Root agent rules and contribution workflow.
- [x] Project scope, learning terminology, architecture, and engineering rules.
- [x] Data integrity, label budgets, evaluation, and provenance requirements.
- [x] Integration candidates, acceptance checklist, and ordered milestones.

Exit gate: documents have consistent links and distinguish proposed behavior from working features.

Evidence: Markdown foundation added; local Markdown links verified. No runtime capabilities delivered.

## M1 — Package foundation and supervised detection

Depends on M0. Completed for the initial Windows CPU baseline; broader platform and accelerator support remain separate work.

- [x] Review and select an exact first backend release and existing detection model; record compatibility and licensing decision.
- [x] Create `pyproject.toml`, `src/wvision`, optional backend dependency, and verified developer setup.
- [x] Implement typed configuration/result contracts and capability checks needed by this backend.
- [x] Add manifest parsing, dataset preflight checks, and an original synthetic detection fixture with distinct splits.
- [x] Implement the first lazy-loaded adapter using upstream train/evaluate/predict entry points.
- [x] Expose matching Python and CLI train/val/test/predict workflows.
- [x] Save resolved config, environment, split identity, metrics, status, and checkpoint references.
- [x] Add offline core/contract checks and real upstream integration tests.
- [x] Publish working examples, tested environment, supported combinations, and actual outputs.

Exit gate: in a clean documented environment, one existing model trains on the tiny fixture, validates, evaluates the explicit test split, and predicts through both public interfaces. Core import/help works without the backend. Missing extras and unsupported learning modes fail clearly. No model internals are reimplemented.

Suggested slices: select environment → package/config → data/run artifacts → adapter → API/CLI → real smoke test and example. Keep each slice small without calling the entire workflow supported early.

Evidence (2026-11-07): clean Python 3.12.3 Windows environment, Ultralytics 8.3.4, PyTorch 2.4.1+cpu, torchvision 0.19.1+cpu. All 28 tests passed (26 core/contract, 2 real integration), including both public interfaces and local-checkpoint fine-tuning; lint, formatting, typing, and dependency checks passed. See [full verification](verification/BASELINE.md) for commands, artifact locations, identities, warnings, and limitations. Synthetic AP was zero; no model-quality claim is made. No Git revision was available because the workspace is not yet a Git repository.

## M2 — Existing semi-supervised detection method

Depends on M1. Completed for the initial Windows CPU integration; see [usage](SEMI_SUPERVISED.md) and [verification evidence](verification/SOFT_TEACHER.md).

- [x] Select existing MMDetection 3.3.0 Soft Teacher with a pinned, fully resolved recipe adaptation; no algorithm reimplementation.
- [x] Verify an isolated Python 3.10/MMCV 2.1.0/torch 2.1.0 CPU environment and subprocess adapter.
- [x] Add reproducible training-only label-budget generation, class counts, withheld-label removal, and overlap checks.
- [x] Save recipe/hash, thresholds, mean-teacher settings, teacher/student checkpoints, and explicit online pseudo-label visibility limits.
- [x] Run real semi-supervised training, validation, test evaluation, and prediction through upstream APIs.
- [x] Compare with the same-detector supervised baseline on the identical labeled subset and held-out splits.
- [x] Report random initialization, recipe changes, compute/exposure differences, wall-time scope, and single-seed limitations.

Exit gate: a documented command invokes an upstream semi-supervised method using both training pools, produces a valid checkpoint, and evaluates it without leakage. Publish comparison results whether or not they improve over the baseline. Do not label ordinary supervised fine-tuning as semi-supervised training.

Evidence (2026-11-07): the current 43 checks passed across core and backend verification groups, including the real comparison and Windows cancellation cleanup. Test AP50:95 was 0.003 supervised versus 0.000 Soft Teacher on the two-iteration synthetic fixture. Positive upstream unsupervised losses verified execution of the unlabeled branch; this is not an accuracy-improvement claim. Full recipes, fingerprints, result tables, commands, checks, and limitations are in [M2 verification](verification/SOFT_TEACHER.md). GPU, pretrained MMDetection initialization, resume, and fully unsupervised methods remain unsupported.

## M3 — Learning without target training labels

Depends on M2. Completed for Windows CPU self-supervised pretraining and supervised detection transfer, with a documented decision to leave fully unsupervised detection unsupported. See [usage](SELF_SUPERVISED.md) and [verification](verification/SIMSIAM.md).

- [x] Select MMPreTrain 1.2.0 SimSiam ResNet-50 and a matched supervised Faster R-CNN transfer protocol.
- [x] Integrate upstream image-only pretraining and checked backbone transfer; record downstream human labels separately.
- [x] Assess LOST and TokenCut separately, including semantic-label mapping and evaluation assumptions.
- [x] Record concrete environment/weight/evaluation blockers and leave fully unsupervised detection unsupported.
- [x] Publish a matched random-initialization comparison with exact label/compute budgets and single-seed limits.

Exit gate: the self-supervised workflow is reproducible and accurately named; the separate fully unsupervised detection decision has documented evidence. Claim fully unsupervised detection support only after an actual integration passes its own execution and evaluation checks.

Evidence (2026-11-08): all 62 tests passed in one full invocation, including real pretraining, checked transfer of 318 backbone tensors, detector validation/test/prediction, all earlier backend regressions, and Windows cancellation cleanup. Default core tests, lint, formatting, typing, and dependency checks also passed. Test AP50:95 was 0.002 random versus 0.003 after two-step SimSiam pretraining on the synthetic fixture; no quality-improvement claim is made. [M3 verification](verification/SIMSIAM.md) records environment, hashes, commands, artifacts, and limitations. [The separate assessment](UNSUPERVISED_ASSESSMENT.md) leaves fully unsupervised detection unsupported.

## M3b — DINOv3 and model extension interface

Requested after M3. Strategy: [model adapters](MODEL_ADAPTERS.md). Code integration and regressions verified; official pretrained-weight verification remains open.

- [x] Introduce an explicit model/capability catalog and typed detection adapter protocol.
- [x] Migrate existing detection integrations without changing their public workflows.
- [x] Integrate upstream DINOv3 ConvNeXt Tiny / DETR downstream training, validation, test, and prediction.
- [x] Verify random-initialization execution and strict local checkpoint loading without hidden downloads.
- [ ] Verify official pretrained backbone transfer when a licensed local checkpoint is available; otherwise record this outstanding gate explicitly.
- [x] Document exact environments, extension steps, capabilities, and real execution evidence.

Exit gate: real upstream operations succeed through the shared facade; existing integrations pass regression checks. Random-weight runs are not called self-supervised learning. Official pretrained verification remains open until actual weights are exercised. DINOv3 SSL pretraining is a separate, unsupported operation.

Evidence (2026-11-08): 77 passed and one skipped in the complete regression suite; the skip is the separate official-weight gate because no licensed local DINOv3 checkpoint was available. The real DINOv3 tests use the full upstream architecture, verify 178 transferred backbone tensors, confirm frozen tensors remain unchanged during training, and reject incomplete weights. All earlier backend regressions, 70 default core checks, lint, formatting, typing, dependency checks, and local documentation links pass. See [DINOv3 verification](verification/DINOV3.md) for source/data hashes, environments, metrics, artifact locations, and outstanding pretrained verification.

## M3c — Model configs and dataset catalogs

Requested before further semi-supervised backbone integration. Completed; see [model configs](../models/README.md), [dataset catalog](../datasets/README.md), and [verification](verification/CATALOGS.md).

- [x] Add model configs organized by task/model with profile and upstream metadata.
- [x] Accept `Vision(model_config_path)` while preserving registered IDs and existing flat configs.
- [x] Add independent dataset examples/templates with explicit splits and preparation steps.
- [x] Preserve config precedence, source identity, run reload, and training-only budget metadata.
- [x] Verify real public workflows, core failure cases, and catalog documentation.

No new learning method, backbone-transfer capability, weight download, or benchmark claim is part of this milestone. M3b's official pretrained DINOv3 verification gate remains open.

Evidence (2026-11-08): 111 passed and one expected official-weight skip in the full regression suite. This includes real model-config paths for YOLO, Soft Teacher comparison, SimSiam/transfer, and DINOv3, with separate dataset-manifest placement and retained label-budget provenance. All 104 default core checks, lint, formatting, typing, dependency checks, and local documentation links pass. Twelve model configs and four dataset examples/templates are present; baseline defaults remain unbenchmarked. Details are in [catalog verification](verification/CATALOGS.md).

## M3d — Weight compatibility and initialized semi-supervised detection

Requested after the catalog work. Completed for the documented Windows CPU source/target combinations. Strategy: [weight compatibility](WEIGHT_COMPATIBILITY.md).

- [x] Describe explicit source/scope/method/style initialization rules through the existing registry.
- [x] Validate supervised detector source artifacts and preserve provenance without arbitrary checkpoint imports.
- [x] Delegate supervised Faster R-CNN → Soft Teacher loading to upstream code and verify both branches.
- [x] Verify SimSiam ResNet-50 backbone transfer into both Soft Teacher branches.
- [x] Run real initialized semi-supervised training, evaluation, prediction, and failure checks.
- [x] Publish working configs, examples, compatibility limits, and execution evidence.

Evidence (2026-11-08): all three real initialized workflows pass, including full-detector transfer in both styles and backbone-only SimSiam transfer. Before-training audits match 348 detector tensors or 318 backbone tensors in each branch; actual unlabeled losses are positive. The full regression suite passes 129 tests with one expected official-DINOv3-weight skip. Validation, test, reload/prediction, source-damage rejection, 119 core checks, lint, typing, dependency checks, and local links pass. The [initialization verification report](verification/INITIALIZATION.md) records commands, hashes, artifacts, regression results, and limits. No accuracy-improvement claim is made.

Scope excludes YOLO semi-supervised algorithms, DINOv3-to-ResNet conversion, external weight downloads, optimizer resume, and cross-dataset initialization. These need separate upstream integrations or transfer protocols.

## M3e — Remove legacy configuration paths

Requested maintenance slice before adding further features. Completed; see
[cleanup evidence](verification/CLEANUP.md).

- [x] Remove duplicate flat profiles and compatibility-only loaders/parity tests.
- [x] Use `model` / `--model` consistently across training, pretraining, and comparisons.
- [x] Simplify settings resolution and consolidate initialization provenance.
- [x] Remove unused serialization and legacy budget-metadata lookup.
- [x] Update agent rules, current examples, architecture, and [migration notes](MIGRATION.md).
- [x] Verify all public backend workflows and publish cleanup evidence.

Evidence (2026-11-08): 118 final core checks and all ten available real integration
cases pass across the broad run and focused DINOv3 rerun. One official-weight
acceptance case remains skipped. The broad run caught a configuration provenance
collision; it was fixed and covered by a new core regression before the affected
real workflow passed again. Lint, formatting, typing, dependency checks, current
documentation examples, and local links pass. Existing schema-1 run reload was
also verified against a pre-cleanup artifact.

Earlier milestone reports describe the APIs and behavior verified at that time;
the migration guide describes intentional breaking changes from this cleanup.
No model, learning algorithm, dependency, or support claim is added.

## M3f — First GitHub publication preparation

Requested documentation and distribution slice before adding more models.

- [x] Rewrite the README around user installation, quick start, datasets, and supported workflows.
- [x] Add Apache-2.0 licensing for original code and preserve upstream terms separately.
- [x] Add human-facing contribution guidance, issue/PR templates, and core CI configuration.
- [x] Exclude local tooling, environments, credentials, weights, and generated artifacts from Git.
- [x] Verify fresh installation, source/wheel builds, public examples, and first-upload contents.

Evidence (2026-11-08): a fresh core-only source install passes 119 tests, lint,
formatting, typing, and dependency checks. Both real YOLO public workflows pass.
Source/wheel builds, standalone wheel installation, licensing metadata, all
public documentation examples, and the 106-file Git upload manifest are verified.
See [publication verification](verification/GITHUB_PREPARATION.md) and the
[first-upload guide](PUBLISHING.md). Original-code licensing is now Apache-2.0;
upstream terms remain separate.

No remote repository, commit, tag, release, or package publication is created by
this slice. Hosted CI results remain unverified until the first GitHub run.

## M3g — Single-GPU semi-supervised and representation learning

Requested GPU training, prioritizing Soft Teacher and SimSiam. Completed for the
documented single-device Windows CUDA environment; see [GPU usage](GPU.md).

- [x] Add explicit single-CUDA-device settings and capability checks without ML imports in the core.
- [x] Propagate saved/overridden devices through training, comparisons, evaluation, and prediction.
- [x] Isolate and pin a CUDA OpenMMLab environment with verified compiled GPU operators.
- [x] Reject unavailable CUDA and invalid devices; enforce explicit CPU placement in workers.
- [x] Verify real CUDA SimSiam, initialized Soft Teacher, GPU comparison, reload, and CPU override.
- [x] Run CPU/backend regressions and publish GPU execution evidence.

Evidence (2026-11-14): four actual CUDA cases pass across focused invocations on
an RTX 4050 Laptop GPU, including SimSiam, backbone/full-detector initialization,
both comparison workflows, CUDA evaluation/prediction, CPU-only backend reload,
and no-fallback failures. All 148 CPU/backend regression cases pass with one
expected official-weight skip; all 138 core checks, static checks, dependency
checks, and public documentation checks pass. See [GPU verification](verification/OPENMMLAB_CUDA.md).

Scope excludes YOLO/DINOv3 CUDA, multi-GPU/distributed training, automatic fallback,
AMP, optimizer resume, and accuracy/performance claims from synthetic smoke data.

## M3h — YOLO and DINOv3 CUDA detection

Requested after the OpenMMLab GPU slice. Completed for the documented Windows
single-device CUDA environments; see [CUDA detection evidence](verification/YOLO_DINOV3_CUDA.md).

- [x] Add explicit CUDA device settings/capabilities to YOLO11n and DINOv3/DETR.
- [x] Preserve requested single-device selection through upstream training and inference.
- [x] Verify real YOLO CUDA fine-tuning, reload, evaluation, prediction, and CPU/GPU switching.
- [x] Verify DINOv3 CUDA random training, frozen local-backbone transfer, and CPU-only reload.
- [x] Verify invalid indices and CUDA requests in CPU-only environments fail without fallback.
- [x] Complete CPU/OpenMMLab regressions, public setup documentation, and distribution checks.

Evidence (2026-11-14): five focused CUDA detection checks pass across the GPU and
CPU-only environments. The affected CPU backend regression passes four cases with
one expected official-weight skip, and the OpenMMLab CUDA transfer regression
passes. All 145 framework-free core checks, static checks, CUDA dependency checks,
and documentation checks pass. Package/source archives include the shared worker
device helper, with generated assets excluded.

Official pretrained DINOv3 weights remain a separate acceptance gate. CUDA support
does not imply DINOv3 SSL pretraining, YOLO semi-supervised training, AMP, or DDP.

## M3i — First Sense integration and code navigation

Completed at the user's request, retaining the existing `wvision` distribution,
public API/CLI, model IDs, and schema-1 run contracts inside First Sense/WForge.

- [x] Import public source, profiles, tests, docs, and licenses without copying
  local environments, weights, datasets, run artifacts, or Git history.
- [x] Group existing internals into `core`, `datasets`, `training`, `evaluation`,
  and `experiments`; keep Python/CLI entry points at the package root.
- [x] Put each backend's adapter/execution code together, separate the combined
  YOLO/MMDetection module, and retain shared device/process helpers separately.
- [x] Fix standalone worker import paths so local backend folders cannot shadow
  upstream packages; add regressions covering all three workers.
- [x] Update the First Sense mission, CI working directory, packaging, navigation
  map, source links, and internal-import migration notes.
- [x] Verify core, CPU workflows, OpenMMLab CUDA transfer, and packaged inference.

Evidence (2026-11-15): all 148 core checks pass (20 integration checks deselected).
Real YOLO/DINOv3 CPU regression: four passes and one expected official-weight skip.
The CUDA SimSiam-to-Soft-Teacher train/transfer/val/test/predict/CPU-reload regression
passes, including 318 transferred tensors and equality checks. The built wheel
passes YOLO CPU/CUDA switching and DINOv3 CUDA inference using existing trusted
smoke checkpoints. Core wheel import/catalog/profile loading work without ML
framework imports; removed flat modules are absent from the distributions.
Ruff checks/formatting pass for 59 Python files and mypy passes for 37 source
modules. Checks cover 35 Markdown documents, 16 Python examples, source links,
unchanged public API/CLI ASTs, and unchanged separated adapter class ASTs.

See [code ownership/navigation](ARCHITECTURE.md#finding-and-changing-code) and
[internal import migration](MIGRATION.md#first-sense--wforge-internal-structure).
Existing absolute run/backend references still need to remain valid; source
reorganization does not relocate them. Official pretrained DINOv3 verification,
new learning methods/tasks, and accuracy benchmarks remain separate gates.

## M3j — YOLO11 model-oriented folder, first slice

Completed for YOLO11 only, before applying the layout to other model families.

- [x] Move the existing YOLO11 adapter, validated settings, and actual upstream
  train/evaluate/predict operations into `src/wvision/models/ultralytics/yolo11/`.
- [x] Keep reusable Ultralytics dependency/runtime/device/metric utilities in
  `adapters/ultralytics/backend.py`; remove model settings from common core config.
- [x] Point the explicit registry at the model-owned factory, without changing
  public imports, signatures, YAML profiles, IDs, capabilities, or run schema.
- [x] Remove the old adapter location without compatibility forwarding files.
- [x] Update the mission, model catalog explanation, navigation, and migration
  notes to distinguish Python model code from the YAML catalog.
- [x] Verify real Python/CLI CPU workflows, CUDA train/reload/fine-tuning and
  CPU/GPU switching, plus isolated installed-wheel loading and CUDA prediction.

Evidence (2026-11-16): all 148 core checks pass (20 integration checks deselected),
including constructing YOLO through the model factory without ML imports. The
two real CPU YOLO cases and the focused real CUDA YOLO case pass. Wheel checks
verify all new model files, exclusion of the old adapter, and actual GPU reload/
prediction using a freshly trained smoke checkpoint. Ruff checks/formatting pass
for 64 Python files; mypy passes for 42 source modules. Source review verifies
identical moved class/function ASTs and unchanged public API/CLI implementations.
All 35 Markdown documents, local source links, and 16 Python examples are checked.

Only YOLO11n detection is supported by this family folder. Other YOLO sizes,
learning methods, and model-folder migrations are separate work. OpenMMLab and
DINOv3 execution code is unchanged and their heavyweight integrations were not
rerun for this YOLO-only slice. No model-quality claim follows from these checks.

## M3k — Remaining model folders and local dataset assessment

Completed for the existing Faster R-CNN/Soft Teacher, SimSiam ResNet-50, and
DINOv3 ConvNeXt Tiny/DETR integrations, following the reviewed YOLO11 layout.

- [x] Move model settings, adapters/source checks, and actual worker operations
  into `src/wvision/models/<framework>/<model>/`; remove old model files.
- [x] Extract shared OpenMMLab/Transformers environment/serialization runtimes;
  keep interpreter selection and process/device helpers independent of models.
- [x] Let each model adapter choose its fixed neighboring operations worker;
  remove the shared detector/pretraining boolean dispatch.
- [x] Preserve public API/CLI, IDs, YAML profile paths, run schema, capabilities,
  initialization rules, and held-out data isolation.
- [x] Update source links, navigation, mission, private-import migration notes,
  and standalone-worker namespace regression probes.
- [x] Audit the provided dataset read-only, document export/class/split issues,
  and ignore the local First Sense `Data/` folder in source control.
- [x] Verify core, real OpenMMLab CUDA workflows, DINOv3 CPU workflows, and the
  installed wheel's model workers and DINOv3 CUDA inference.

Evidence (2026-11-16): all 148 core checks pass (20 integration checks deselected).
The two real OpenMMLab CUDA checks pass: SimSiam-to-Soft-Teacher transfer including
CPU reload, and supervised/semi-supervised comparison with supervised detector
initialization. Two real DINOv3 CPU checks pass, including loading/freezing local
backbone weights and rejecting a damaged source; official licensed-weight testing
is the expected skip. The wheel loads all model factories and three standalone
workers without importing optional ML frameworks in the calling process, excludes
old model paths, and performs actual DINOv3 CUDA reload/prediction. All 33 moved
worker/runtime function ASTs, model configuration/source-validator ASTs, public
API/CLI implementations, and YOLO operations are unchanged. Ruff checks/formatting
pass for 75 Python files; mypy passes for 53 source modules. Local links and 16
Python examples across 35 Markdown documents are checked.

The local player/ball export has 1,250/359/178 train/validation/test images, matching
label files, no invalid box rows, and no exact decoded-image duplicates. Class
names are corrupt. All 30 identifiable video groups cross splits; 691 filenames
cannot be assigned to a source video by the current filename heuristic. This is
an audit, not a verified class mapping or near-duplicate detector. Source images
and annotations are unchanged, and no training uses this private dataset. See
[dataset/export guidance](../datasets/README.md#roboflow-exports-for-local-testing).
Correct class metadata and video/match-separated held-out data are required
before a reliable real-data semi-supervised comparison. The audit lives in the
ignored component-local `.wvision/player-ball-audit.json`.

No architecture, research method, pretrained-weight gate, or model-quality claim
is added by reorganizing existing integrations.

## M3l — First Sense first-commit cleanup

Completed as source preparation, without creating a Git commit or remote.

- [x] Remove verified empty future-component directories and retain roadmap docs.
- [x] Consolidate root ignore rules and verify local data/fixtures/weights/runs
  remain outside prospective source files.
- [x] Add root Apache-2.0 license discovery while preserving build-time notices.
- [x] Group dated verification reports; correct links, package-root setup, and
  current-versus-planned component descriptions.
- [x] Run core/static/package checks and a separate-index source/whitespace audit.

Evidence (2026-11-16): 148 core checks pass, 20 integrations deselected; Ruff
checks/formatting and mypy pass. No heavyweight backend retraining is needed for
this cleanup. The prospective source list, root ignore policy, source links,
examples, and distributions are checked. See the
[first-commit report](verification/FIRST_COMMIT.md). Actual Git initialization,
commit/remote/push and first hosted CI result remain publication actions.

## M3m — Training operations and experiment comparison

Requested before Pose. Single-device features verified locally on 2026-11-17;
actual distributed execution and hosted automation remain open gates.

- [x] Periodic recoverable checkpoints and explicit optimizer-state resume.
- [x] Explicit per-backend mixed precision with unsupported combinations rejected.
- [ ] Verify actual distributed multi-GPU training/resume on a two-GPU host.
- [x] Portable inference/training bundles and verified YOLO ONNX inference export.
- [x] Configure automated real-backend checks with retained failure artifacts.
- [ ] Execute the workflow on GitHub and inspect its artifacts.
- [x] Multi-model and parameter-grid experiments on identical dataset splits,
  with validation comparison, reproducible settings, and separate final testing.

Export target: ONNX. Multi-GPU execution requires a multi-GPU verification host;
the current development laptop has one GPU. Preserve the prior model, data, and
initialization contracts while extending each backend through upstream APIs.
See the [training-operations verification](verification/TRAINING_OPERATIONS.md)
and [usage guide](TRAINING_OPERATIONS.md). Real football checks preserve the
87 labeled / 785 unlabeled / 292 validation / 293 test samples. Short schedules
verify execution; existing video overlap prevents a generalization claim.

## M3n — Grouped evaluation, flexible transfer, and broader ONNX export

Requested before Pose. The implementation adds explicit group-based dataset
preparation, shared per-class COCO val/test, held-out exposure checks for
cross-dataset initialization, and verified DINOv3/DETR ONNX export. See
[verification and remaining limits](verification/EVALUATION_TRANSFER_EXPORT.md).

- [x] Copy explicit video/match groups into new, isolated datasets; preserve sources.
- [x] Preserve group identity through label budgets and portable dataset bundles.
- [x] Require grouping in experiment plans and report per-class AP/seed summaries.
- [x] Transfer compatible detector weights across datasets or only the backbone
  to new class sets, with source/target exposure audits.
- [x] Verify DINOv3/DETR ONNX outputs against native FP32 inference.

Real football quality benchmarking still requires accurate group IDs and longer,
comparable multi-seed training. Actual two-GPU execution, hosted automation,
and official pretrained DINOv3 acceptance remain open. MMDetection export is
covered by M3o below.

## M3o — MMDetection ONNX export and official DINOv3 gate

- [x] Export Faster R-CNN and the selected Soft Teacher inference branch through
  MMDeploy, retaining per-level proposal NMS and class-specific box regression.
- [x] Verify Caffe/PyTorch-style backbones, source tensor identity, ordinary ONNX
  Runtime inference, and complete detection-set parity on representative images.
- [x] Check a real football checkpoint on three validation images; preserve all
  source checkpoints and data.
- [ ] Verify licensed official DINOv3 ConvNeXt Tiny weights. **Postponed by the
  user on 2026-11-18** until local weights are supplied; no authenticated download
  or official pretrained accuracy claim.

See [MMDetection ONNX evidence](verification/MMDET_ONNX.md) and the
[export contract/setup](TRAINING_OPERATIONS.md#faster-r-cnn-and-soft-teacher).

## M4 — 2D pose estimation

Depends on the detection foundation, earlier learning-mode assessments, M3m, and M3n.

YOLO11n-pose supports the supervised FP32 workflow on CPU and single-device CUDA.
[RTMPose-t](RTMPOSE.md) adds top-down COCO-17 and Halpe-26 through MMPose with
explicit prediction boxes.
See the [implementation plan](POSE_PLAN.md), [usage](POSE.md), and
[verification](verification/POSE.md).

- [x] Select YOLO11n-pose and reuse the existing YOLO11 workflow.
- [x] Add skeleton, keypoint order, visibility, and coordinate contracts.
- [x] Implement dataset validation and normalized pose predictions.
- [x] Verify train/val/test/predict and document pose metrics.
- [x] Add a real upstream smoke test and a short example.

Exit gate: one existing 2D pose model completes the public workflow with correct keypoint semantics. Semi-supervised or unsupervised pose remains unsupported until separately verified.

## M4b — YOLO11 and RTMPose sizes and RTMDet detection

- [x] Add YOLO11 s/m/l/x detection and pose alongside n, with task and scale
  checks when loading checkpoints.
- [x] Add RTMPose s/m/l/x alongside t for COCO-17 and Halpe-26. Use the installed
  192×256 COCO recipes for t/s/m/l and the 288×384 Body8 recipe for x.
- [x] Add supervised RTMDet tiny/s/m/l/x through MMDetection, with separate
  validation and test metrics and normalized pixel-box predictions.
- [x] Add registered IDs, smoke profiles and an RTMDet-to-RTMPose example using
  the existing person-box input.

The [model catalog](../models/README.md#catalog), [RTMDet guide](RTMDET.md), and
[RTMPose guide](RTMPOSE.md) describe profiles, model sources, and setup.
The RTM workflows start from random weights and use supervised single-device FP32.

Verification uses the synthetic detection and keypoint fixtures. From WForge
with the pinned YOLO application environment and the isolated interpreters set
in `WVISION_MMDET_PYTHON` and `WVISION_MMPOSE_PYTHON`:

```powershell
$env:WVISION_TEST_OFFLINE = "1"
$env:WVISION_TEST_DEVICE = "cpu"
python -m pytest -q -m integration tests/test_detection_variants.py tests/test_rtmpose.py
$env:WVISION_TEST_DEVICE = "cuda:0"
python -m pytest -q -m integration tests/test_detection_variants.py tests/test_rtmpose.py -k "detect-x or pose-x or tiny or rtmdet_workflow[x] or x-halpe26"
```

Evidence (2026-11-25): 20 CPU cases completed train, validation, test, and
prediction on Windows. These cover all eight added YOLO11 variants, all five
RTMDet sizes, and seven RTMPose size/skeleton combinations. Five cases completed
the same workflow on one NVIDIA RTX 4050 Laptop GPU. RTMDet-x also trained with
batch size one through the CLI. YOLO11 s/m/l/x detection exports matched native
FP32 outputs at 64×64 with ONNX Runtime 1.20.1.

The checks used Python 3.12.3 with Ultralytics 8.3.4 and PyTorch 2.4.1
(CPU or cu124), plus an isolated Python 3.10.20 stack with MMDetection 3.3.0,
MMPose 1.3.2, and PyTorch 2.1.0+cu121. All 207 core tests and both existing
MMDetection integration regressions passed. Ruff, mypy, and source/wheel builds
passed. The installed wheel ran the RTMDet CLI prediction and passed three
synthetic RTMDet boxes to RTMPose, which returned 26 joints per box.
These checks establish workflow execution on synthetic fixtures.

## M5 — Image classification

Depends on M4.

- [ ] Integrate an existing classifier with explicit class mapping and score semantics.
- [ ] Add classification data validation, metrics, and train/val/test/predict outputs.
- [ ] Prioritize an existing semi-supervised or self-supervised classification method after the supervised contract is verified.
- [ ] Add runnable examples, capability entries, and real upstream checks.

Exit gate: one existing classifier completes the public workflow; each additional learning mode has its own evidence.

## M6 — Expansion and release hardening

Depends on completed supported workflows. Select features from demonstrated need.

- [ ] Test selected export formats against reference predictions and declared tolerances.
- [ ] Verify resume and interrupted-run recovery per backend.
- [ ] Add supported-environment CI and optional accelerator regression jobs.
- [ ] Publish reproducible label-efficiency benchmarks with dated recipes and results.
- [ ] Establish release/versioning policy, migration notes, and distribution licensing/attribution.
- [ ] Add another task only with a concrete upstream implementation and task-specific contract.

Exit gate: a clean installation reproduces published examples and declared support matches tested combinations. A release does not require implementing every possible task or learning mode.

## Evidence format

For each completed milestone, record commit/revision if available, environment, exact commands, fixture/dataset identity, generated artifact locations, check results, and known limitations. Keep large artifacts outside the repository and link to their durable location when one exists.
