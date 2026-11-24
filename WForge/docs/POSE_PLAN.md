# M4 implementation plan — supervised 2D human pose

**Status: implemented and verified, 2026-11-19.** See [usage](POSE.md) and
[CPU and CUDA evidence](verification/POSE.md). The plan below records the design
and the first milestone's completion criteria.

## First model and public interface

Use the registered ID `ultralytics/yolo11n-pose` and a smoke profile at `models/pose/yolo11n-pose/smoke.yaml`. The installed Ultralytics 8.3.4 already includes the nano scale of `cfg/models/11/yolo11-pose.yaml`, a pose trainer, validator and predictor. This model estimates all people and their joints directly from each image. It fits the existing YOLO environment without introducing a person-crop detector or a second framework. Start with random initialization from the bundled definition and reload WForge's selected local checkpoint. Record package version, definition hash, scale, initialization and the existing Ultralytics license terms.

Target usage after implementation, from `WForge/`:

```python
from wvision import Vision

model = Vision("models/pose/yolo11n-pose/smoke.yaml")
run = model.train("dataset.yaml", device="cpu")
pose = Vision.from_run(run.artifact_dir)
validation = pose.val()
test = pose.test()
predictions = pose.predict("frame.jpg")
```

Use the existing CLI verbs and arguments: `wvision train --model ... --data ...`, then `wvision val --run ...`, `wvision test --run ...` and `wvision predict --run ... --source ...`. The model profile selects the task. Keep the existing result objects and run layout.

## Data and output contracts

- **Manifest.** Reuse the schema-v1 YOLO manifest, class map, split lists and optional groups. Set `task: pose`, `format: yolo`, `classes: {0: person}` and `skeleton: coco17`. The first implementation owns one versioned skeleton definition. Derive the 17 joint names, display edges, flip permutation, and upstream `kpt_shape: [17, 3]` from it. Persist the resolved metadata and index convention with the run. The joint order is nose, left/right eye, left/right ear, left/right shoulder, left/right elbow, left/right wrist, left/right hip, left/right knee, left/right ankle. Each pair lists left first. Edges refer to zero-based joint indices.
- **Labels.** Each person has one row of 56 values: class ID, normalized box center x/y and width/height, then 17 triples of normalized x, y and visibility. Use COCO visibility 0 for unlabeled, 1 for labeled but occluded, and 2 for visible. Require zero coordinates for unlabeled joints, finite in-range coordinates for labeled joints, valid boxes and class IDs, and the exact column count. An empty label file is a verified negative. A missing file on a labeled split is an error. The existing football box labels cannot supply joint supervision.
- **Staging and identity.** Preserve original label bytes. Retain split isolation, decoded-image duplicate detection, group checks and copied-file hash verification. Include task and resolved skeleton identity in pose fingerprints. Keep existing detection fingerprints reproducible. Stage only labeled training and validation for training. Evaluate test explicitly after selection.
- **Predictions.** Write one JSONL record per image, including empty frames. Include source identity, original width/height, skeleton ID, and a `persons` array. Each person has a class ID, box in original-image pixel xyxy, instance confidence and exactly 17 ordered joints with pixel x/y and joint confidence. Keep joint and person scores separate. In 8.3.4 the results object masks coordinates below joint confidence 0.5. Serialize those coordinates as null and retain the score. Record that threshold. Training visibility is an annotation field, not a prediction score.
- **Evaluation.** Use the native pose validator. Preserve its named pose precision, recall, AP50 and AP50–95 keys marked `(P)` and its separate box keys marked `(B)`. Persist the evaluator version, skeleton, OKS sigmas, thresholds, area definition and split fingerprint. In the installed version the native OKS computation uses box area × 0.53 and COCO-17 sigmas. Upstream selects the best checkpoint by maximizing the sum of pose and box fitness. Record this selection rule and load that exact checkpoint on reload.

## Implementation sequence

Complete each step with its check before expanding the change. Paths below are relative to `WForge/`.

1. **Start with the data boundary.** Read `datasets/detection.py` under `src/wvision/`, particularly `load_dataset`, `_annotation`, `Dataset.snapshot` and `write_upstream`. Put the COCO-17 definition and pose-row validation together in `src/wvision/datasets/pose.py`. Extend the existing loader with an expected-task argument defaulting to detection so existing callers retain their checks. Reuse its file traversal, group/fingerprint logic and staging. Add pose metadata to the staged YAML. Keep the bbox-only COCO writer restricted to detection. **Check:** parse multi-person, occluded, missing-joint and negative samples, verify left/right flip indices, and reject a box-only label or mismatched model/manifest before training. Compare an existing detection fixture's fingerprint before and after.
2. **Route the model through the current facade.** Inspect `core/registry.py`, `adapters/base.py`, `core/model_config.py`, `api.py` and `cli.py`. Extend the reviewed adapter factory to resolve supervised vision tasks by model spec and pass the expected task into training and evaluation data loading. Keep representation pretraining on its own entry point. Use the existing orchestration, operation directories, settings, result types and checkpoint integrity checks. Save task and skeleton with pose runs, and resolve older detection runs from their registered model ID. Keep keypoint prediction data out of the box schema. **Check:** framework-free model listing, model YAML resolution, old run reload, and early rejection of task mismatches.
3. **Adapt the existing YOLO family in place.** In `models/ultralytics/yolo11/` under `src/wvision/`, replace the hard-coded training definition, task, model-source entry and selection description with values from the registered variant. Keep one upstream train call, backend context and evaluation path. Give the pose-specific prediction serializer its own small function. Validate the loaded task, nano variant, pose head and joint shape when restoring weights. Reuse the namespaced metric writer in `adapters/ultralytics/backend.py`. Audit hard-coded detection descriptions in run/protocol metadata. **Check:** train for one epoch on CPU, inspect actual selection metadata and pose metric keys, then reload and execute both evaluations and prediction.
4. **Close accidental entry points.** Audit `Vision._evaluate`, `experiments/comparison.py` and its worker before passing pose results to code expecting bounding boxes. The initial milestone supports supervised FP32 train/val/test/predict and native pose evaluation. Keep bbox-only experiment ranking and shared COCO protocols detection-only. Validate unsupported pose requests before staging or launching a job, including SSL/transfer, external initialization, export, AMP, DDP and optimizer resume until independently covered. Check bundle callers for detection-specific manifest reconstruction. A pose run must either round-trip with its skeleton or reject the unsupported bundle operation explicitly. Existing detection behavior is the regression baseline.
5. **Prove the public workflow and document it.** Generate a tiny owned fixture with distinct PNGs, multiple people, asymmetric left/right joints, occlusion, an unlabeled joint, an empty frame, and separate labeled train/val/test splits. Use a non-square image to catch coordinate scaling errors. Keep fixture generation alongside the existing examples. Verify API training, CLI reload/evaluation/prediction, and compare WForge output to upstream on the same checkpoint and image. A random smoke model can return zero people. Exercise the non-empty serializer separately with known upstream pose results and assert joint ordering, null handling, confidence and box/person association. Compare recorded metric values to upstream, require finite pose metrics, and check that training never staged test data. Repeat the real smoke on CUDA before registering CUDA support. Update the capability table, example, M4 checklist and dated evidence with the exact environment, commands and observed results.

## Verification and handoff

From `WForge/`, run the normal core checks in the development environment:

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Mark the real pose test with the existing `integration` and `ultralytics` markers and run it explicitly in the pinned YOLO environment. Run the existing YOLO detection smoke alongside it because both tasks share the training code. Inspect `run.json`, `dataset.json`, `model-source.json`, native metric files and `predictions.jsonl` from the public run. Verify old detection outputs and offline imports. Record skips and unavailable hardware separately from passes.

**Completion criterion.** A fresh local fixture goes from labels to training to a reloaded pose checkpoint through documented commands. The records identify the skeleton and selection metric, and the tests would fail for swapped joints, box-only evaluation, wrong coordinates or a detector checkpoint. Pose accuracy on football needs a separately labeled pose dataset. Multi-run pose comparison with a shared OKS protocol, alternative skeletons, pose SSL, export and deployment can each follow with their own evidence.

Implementation references: [model adapter strategy](MODEL_ADAPTERS.md), [data rules](DATA_AND_EVALUATION.md), [backend acceptance](INTEGRATIONS.md), [Ultralytics v8.3.4 pose model](https://github.com/ultralytics/ultralytics/blob/v8.3.4/ultralytics/cfg/models/11/yolo11-pose.yaml), [native pose evaluation](https://github.com/ultralytics/ultralytics/blob/v8.3.4/ultralytics/models/yolo/pose/val.py).
