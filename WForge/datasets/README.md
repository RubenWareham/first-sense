# Dataset catalog

Dataset manifests are independent of model configs. The same detection manifest can be used with any compatible registered detection adapter. Files here contain paths, class maps, and split assignments; images, labels, weights, and generated runs stay outside source control.

## Runnable examples

Prepare the original synthetic CC0 example explicitly from the repository root:

```powershell
python examples/make_tiny_dataset.py .wvision/tiny
wvision train --model models/detection/yolo11n/smoke.yaml --data datasets/examples/tiny-detection.yaml
```

[tiny-detection.yaml](examples/tiny-detection.yaml) points to `.wvision/tiny` and its generated split lists. It describes four labeled training images, two validation images, and two test images. This example checks execution, not real-world accuracy.

Create a derived training-only label budget:

```text
wvision prepare-budget --data datasets/examples/tiny-detection.yaml --destination .wvision/tiny-budget --labeled-count 2 --seed 0
```

[tiny-semi-supervised.yaml](examples/tiny-semi-supervised.yaml) points to that derived data: two labeled and two unlabeled training images, preserving validation/test membership. It works for Soft Teacher or SimSiam pretraining, which only consumes the unlabeled training pool.

`budget.json` remains in the data root with its exact selections, seed, counts, and source/derived fingerprints. Comparison workflows find it there even though the catalog manifest is elsewhere, and reject a mismatched fingerprint. Metadata is read only from the data root.

No example downloads external datasets, copies withheld training labels into the unlabeled pool, or generates data during import or model construction. Destinations must be new; existing data is never silently overwritten.

Generated data copies use numbered filenames with the original image extension
to keep long source names from exceeding Windows path limits. Source files remain
unchanged; dataset snapshots and label-budget metadata retain source identities.

## Your own dataset

Copy either [the supervised template](templates/yolo-detection.yaml) or [the semi-supervised template](templates/semi-supervised-detection.yaml). Set its `root` relative to the copied manifest's location, replace class names, and supply the listed split files. Templates are not runnable until those files exist.

```yaml
schema_version: 1
task: detect
format: yolo
root: ../my-data
classes:
  0: person
  1: bicycle
splits:
  train_labeled: splits/train_labeled.txt
  train_unlabeled: splits/train_unlabeled.txt
  val: splits/val.txt
  test: splits/test.txt
```

Split lists contain image paths relative to `root`, with images under an `images/` directory. Labeled annotations use the mirrored `labels/` path and normalized YOLO boxes. An empty annotation file means a verified negative; missing annotations fail for labeled splits. `train_unlabeled` is optional for supervised detection and required by the integrated semi-supervised method/pretraining workflow.

Record dataset origin, license, annotation policy, and grouping decisions with your dataset documentation. Separate related video/subject/site groups before writing the split lists. Dataset integrity checks reject path overlap and exact decoded-image duplicates; they do not identify every near-duplicate or correlation.

Changing a catalog manifest's location or relative root does not change dataset identity if the same relative sample paths, class map, images, and annotations are retained. The actual manifest and resolved root are recorded in run snapshots. No dataset fetch service, public benchmark download, or new annotation format is added by this structure change. See [data and evaluation rules](../docs/DATA_AND_EVALUATION.md).

## Roboflow exports for local testing

Choose **YOLOv8 PyTorch TXT object detection**, with ordinary bounding-box labels,
images, and a `data.yaml` listing the actual class names. Do not substitute pose,
segmentation, or oriented-box exports for this detection contract. WVision uses
its own manifest/template above rather than loading arbitrary Roboflow YAML
directly; its adapters stage YOLO or COCO as required by the backend.

Verify every label ID against the source project's classes. A corrupt class-name
list cannot be fixed merely by switching export format. Keep source files intact
and create explicit split lists/derived data. For video datasets, assign whole
videos or matches before splitting; exact-image hashing cannot detect every
correlated neighboring frame. Validation/test stay labeled and held out. Only the
training pool may be divided into a labeled budget and an unlabeled remainder.

For a pipeline-only smoke test, you may explicitly retain existing frame-level
splits and create a training-only labeled/unlabeled budget. Record that choice
and do not interpret its held-out scores as evidence of generalization to unseen
videos or matches. Local audits and derived datasets stay outside source control.
