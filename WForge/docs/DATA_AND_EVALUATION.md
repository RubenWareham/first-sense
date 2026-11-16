# Data and evaluation rules

## Manifest contract

Define a versioned manifest with task, class map, annotation format, dataset root, and explicit split references. The first detection adapter should accept the upstream YOLO detection layout through a WVision manifest and validate it before handing it off. Add COCO conversion only when a backend requires it. Preserve source annotations; write converted data into a separate generated directory.

Implemented detection manifest schema:

```yaml
schema_version: 1
task: detect
format: yolo
root: ./example
classes:
  0: person
  1: bicycle
splits:
  train_labeled: splits/train_labeled.txt
  train_unlabeled: splits/train_unlabeled.txt
  val: splits/val.txt
  test: splits/test.txt
```

Resolve `root` relative to this manifest; resolve split references relative to `root`. Each split file lists one image path per line, relative to `root`. Images must remain inside the dataset root and under an `images` directory. Replace the last `images` component with `labels` and change the extension to `.txt` for annotation discovery. Each annotation row is `class_id center_x center_y width height` with normalized coordinates and a box inside the image. An empty file means a verified negative. PNG/JPEG are accepted. The [fixture generator](../examples/make_tiny_dataset.py) creates a complete example.

Preflight checks every declared split and fingerprints images, decoded RGB pixels, and labels. Identical decoded-content duplicates are rejected even within a split; approximate visual duplicates are not detected. Selected samples are copied into each run and verified before upstream execution so source data stays separate from caches.

The [dataset catalog](../datasets/README.md) provides separate runnable example manifests and templates. A manifest can live outside its data root; paths still resolve against the manifest/root contract, and the root is recorded in snapshots without changing dataset identity. Comparison budget metadata is read only from `budget.json` in the data root, regardless of where the manifest is stored. Model configs contain no dataset paths or class-count overrides.

`train_unlabeled` is optional for supervised learning. Validation and test annotations are required for labeled task metrics; prediction may operate on unlabeled images. A missing test split is an error for `test`, not a reason to substitute validation data.

## Integrity checks

- Check paths, supported image formats, readable samples, unique sample identities, and annotation schema.
- Check class ID consistency, positive image dimensions, finite coordinates, valid box ordering, and annotation bounds.
- Distinguish an explicitly verified negative image from an image whose annotations are missing. Unlabeled images must never become negative supervised samples accidentally.
- Reject sample/path overlap and exact-content duplicates across train, validation, and test splits. Record content hashes and grouping rules. Flag possible near-duplicates where detection is available; do not claim hash checks catch them.
- Split video, subjects, sessions, and capture sites by group when those correlations matter. Record the grouping policy before training.
- For partially annotated images, require upstream support and an explicit policy. Do not present partial labels as complete ground truth.

Never silently repair or delete source annotations. Report problems with sample references. Any conversion/filtering operation must produce an auditable derived dataset and summary.

## Label budgets and pseudo-labels

Generate labeled subsets from training data only using a saved seed and explicit selection policy. Store exact labeled and unlabeled sample lists, budget denominator, counts, class distribution, and selection version. Compare methods using identical subsets, splits, initialization policy, and declared compute budgets. Current checked detector/backbone source runs must share the exact target dataset fingerprint. A supervised source must have trained on that same labeled pool, selected only on validation; cross-dataset and external raw detector transfer require a separate protocol. Initialization adds source compute and must not be compared as equal total compute to a random baseline.

Implemented: `prepare_label_budget` / `wvision prepare-budget` creates a new derived dataset with deterministic sample-level selection, unchanged held-out membership, no withheld annotation copies, and a `budget.json` audit record. MMDetection receives run-local COCO conversions, with no ground-truth annotations in its unlabeled file. See [semi-supervised usage](SEMI_SUPERVISED.md) for denominator, grouping, and comparison limitations.

Pseudo-labels may only be generated for the allowed training pool. Preserve source model/checkpoint, upstream method version, score and filtering rules, generation configuration, coordinates, and dataset identity. Store them separately from human labels. Never use validation or test labels to generate pseudo-labels; use validation only for declared hyperparameter selection.

Pretrained models can carry prior supervision. Record known pretraining datasets and overlap limitations. Do not describe target-data label efficiency as training without any labels when labeled pretraining was used.

Implemented [SimSiam pretraining](SELF_SUPERVISED.md) stages only `train_unlabeled` images, passes no annotations or held-out paths to its worker, and selects the final iteration without validation. Later Faster R-CNN transfer uses human-labeled training boxes and validation checkpoint selection. The comparison reports both stages' supervision and additional pretraining compute. Full-manifest integrity checks still read held-out annotations in the core; those annotations are never pretraining inputs. The [fully unsupervised assessment](UNSUPERVISED_ASSESSMENT.md) defines the outstanding semantic-mapping/evaluation gate.

## Evaluation protocol

Detection starts with a declared COCO-style AP protocol: AP averaged over IoU thresholds 0.50:0.95, AP50, AP75, per-class AP where available, evaluator version, and settings including maximum detections. Map backend metrics only when their definitions match. Preserve namespaced metrics when they differ. Never compare COCO AP with a different AP convention under one generic score.

The first adapter preserves structured metrics with an `ultralytics/` prefix. It uses upstream AP, not an official `pycocotools` COCO benchmark. AP75 and per-class tables are not yet exposed in normalized summaries. Evaluation records the evaluator version, image size, batch size, and maximum detections (300), retaining upstream arguments. Full benchmark reporting remains future work.

The [DINOv3/DETR adapter](DINOV3.md) uses `transformers/coco/` AP metrics from pycocotools with `maxDets=[1,10,100]`. Its initial detector checkpoint selection minimizes validation loss, with one event at the final training step; later val/test AP reports do not select a checkpoint. Pretrained-backbone supervision is recorded separately from downstream detection labels. Official pretrained transfer remains unverified until real licensed weights are supplied.

Declare the checkpoint selection metric and direction before training. Tune thresholds and select checkpoints on validation data. Test uses the selected frozen checkpoint and configuration; repeated test-driven model selection invalidates the held-out claim. Keep validation and test reports distinct.

For each comparison, report dataset/split identity, label budget, pretraining source, model/backbone, image size, augmentation recipe, training steps/epochs, compute/device, seeds, evaluator, and metric settings. Report mean and spread over multiple seeds for quality claims where practical; single-seed runs must say so. Latency results need batch size, precision, hardware, warm-up, synchronization, and whether preprocessing is included.

## Required run artifacts

The implementation should save a resolved config, versioned dataset/split manifest and fingerprints, run status, upstream model/weights identity, dependency/environment information, seed/determinism settings, metric definitions and values, checkpoint references, and raw upstream log locations. Record the WVision version and commit when available. Large data and checkpoints stay outside source control.

Future pose metrics must identify skeleton, visibility semantics, and OKS settings. Classification metrics must identify class mapping and averaging policy. Add their exact contracts when those milestones begin.
