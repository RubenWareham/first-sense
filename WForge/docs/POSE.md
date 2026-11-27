# Supervised 2D human pose

For top-down models that take person boxes and support COCO-17 or Halpe-26,
see the [RTMPose guide](RTMPOSE.md).

YOLO11 n/s/m/l/x pose variants predict a person box and the joints declared by
your dataset. They use Ultralytics 8.3.4 and the same `Vision` API and CLI as
detection. The workflow supports supervised FP32 training from random initialization, native
validation and test, and prediction from the selected WForge checkpoint. It is
verified on Windows CPU and one NVIDIA CUDA GPU.

Use the [YOLO CPU setup](DEVELOPMENT.md) or [CUDA setup](GPU.md). From `WForge/`,
generate a tiny local fixture and train the smoke profile:

```powershell
python examples/make_tiny_pose_dataset.py .wvision/tiny-pose
python -m wvision train --model models/pose/yolo11n-pose/smoke.yaml --data .wvision/tiny-pose/dataset.yaml
```

Swap `yolo11n-pose` in that path for `yolo11s-pose`, `yolo11m-pose`,
`yolo11l-pose`, or `yolo11x-pose` to choose a larger model.

For a single NVIDIA GPU, use the CUDA environment and add `--device cuda:0`
to the training command. Evaluation and prediction default to the run's device.

Use the training result's `artifact_dir` for `--run`:

```powershell
$poseRun = "runs/train-<id>"
python -m wvision val --run $poseRun
python -m wvision test --run $poseRun
python -m wvision predict --run $poseRun --source .wvision/tiny-pose/images/test-0.png
```

```python
from wvision import Vision

model = Vision("models/pose/yolo11n-pose/smoke.yaml")
run = model.train(".wvision/tiny-pose/dataset.yaml")
model = Vision.from_run(run.artifact_dir)
validation = model.val()
final_test = model.test()
predictions = model.predict(".wvision/tiny-pose/images/test-0.png")
```

The smoke profile uses one epoch, 128-pixel inputs, and batch size two. It checks
execution. For your own training, provide a pose manifest and set the training
budget and image size in a model profile or through the usual overrides.

## Dataset and results

Use a schema-v1 WForge manifest with `task: pose`, `format: yolo`,
`classes: {0: person}`, and a skeleton definition. Paths and split lists follow
the [dataset catalog](../datasets/README.md). Use `skeleton: coco17` for the
original 17-joint layout or `skeleton: halpe26` for the 26-joint layout. Add
`--skeleton halpe26` to the fixture generator to try the larger layout with the
same training command. The built-in joint order follows the
[OpenMMLab Halpe-26 definition](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/_base_/datasets/halpe26.py).

For another layout, put its ordered joint names and flip mapping in the manifest.
For example, a three-point layout uses:

```yaml
skeleton:
  id: hip-and-heels
  joints: [left_heel, right_heel, hip]
  flip_idx: [1, 0, 2]
  edges: [[0, 2], [1, 2]]
```

Indices are zero-based. Each entry in `flip_idx` gives the mirrored joint's
index, or the joint's own index if it has no mirror partner. Edges are optional.
The number of names determines the pose head width. A 17-joint layout must use
COCO joint order with the pinned YOLO backend, which applies COCO weights by index.

Each annotation row contains a person class ID, four normalized YOLO box
values, and one normalized `x y visibility` triple per declared joint.
Visibility is 0 for
an unlabeled joint, 1 for labeled but occluded, and 2 for visible. Use `0 0 0`
for an unlabeled joint. Supply an empty label file for a verified negative frame.
Football box annotations need corresponding per-person joint labels to train pose.

The training and evaluation dataset records and prediction `config.json`
include the resolved joint names, edges and flip order. The selected run carries
its skeleton so prediction can reload without the training dataset.
`predictions.jsonl` has one record per image with the skeleton ID, joint count and a
`persons` list ordered as returned by Ultralytics. Each person has a class ID,
box, instance confidence and ordered joint coordinates and scores. Coordinates
use original-image pixels. Joints below upstream's 0.5 keypoint confidence
threshold have null coordinates and retain their confidence scores.
An image with zero detections produces an empty `persons` list.

Use the native evaluator for pose AP50 and AP50–95. These use Object Keypoint
Similarity (OKS) to compare joints, while box AP uses bounding-box overlap. The
metric keys end in `(P)` and `(B)`, respectively. `pose-protocol.json` records
the Ultralytics version, skeleton, OKS thresholds, sigmas and the box area scaling.
Ultralytics 8.3.4 uses COCO sigmas for COCO-17 and uniform `1 / joint_count`
sigmas for other layouts, including Halpe-26. Compare AP under the same skeleton
and evaluation protocol.
The best checkpoint maximizes the sum of native pose and box fitness.
Reserve the test split for final evaluation.

Optimizer resume, mixed precision, distributed training, transfer, portable
bundles, ONNX export and shared bbox experiment reports are outside the first
pose workflow. See the [M4 plan](POSE_PLAN.md) for the architecture and scope.
