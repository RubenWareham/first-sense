# RTMPose through MMPose

Choose `mmpose/rtmpose-t`, `mmpose/rtmpose-s`, `mmpose/rtmpose-m`,
`mmpose/rtmpose-l`, or `mmpose/rtmpose-x`. Each trains a top-down pose model
from local COCO-17 or Halpe-26 annotations in the
[pose dataset format](POSE.md#dataset-and-results). WForge converts the selected
splits to run-local COCO keypoint files and uses MMPose's installed model and
training loop with supervised FP32, random initialization, single-device
BatchNorm, and validation each epoch.

The t/s/m/l variants use 192×256 crops (`imgsz: 256`). The x variant uses the
upstream Body8 recipe's 288×384 crop (`imgsz: 384`). WForge configures its head
for the dataset's skeleton and trains only on the supplied local splits.

## Setup and train

Install WVision in the Python 3.12 application environment. Set up a separate
Python 3.10 OpenMMLab environment using the [CPU setup](SEMI_SUPERVISED.md)
or [CUDA setup](GPU.md). The RTMPose worker uses MMDetection 3.3.0 for
CSPNeXt, MMCV 2.1.0, MMEngine 0.10.7, and the same PyTorch 2.1.0 builds.
Install MMPose and its metric dependencies in that backend environment. Use
the backend's interpreter path in the examples:

```powershell
$posePython = "path/to/python310/backend/Scripts/python.exe"
& $posePython -m pip install wheel
& $posePython -m pip install --no-build-isolation chumpy==0.70
& $posePython -m pip install -c constraints/rtmpose-py310.txt mmpose xtcocotools
$env:WVISION_MMPOSE_PYTHON = $posePython

python examples/make_tiny_pose_dataset.py .wvision/tiny-rtmpose --skeleton halpe26
python -m wvision train --model models/pose/rtmpose-t/smoke.yaml --data .wvision/tiny-rtmpose/dataset.yaml
```

Use `--backend-python` or `backend_python=` to select an interpreter explicitly.
For CUDA, use the CUDA-enabled backend and pass `--device cuda:0`. The profile
is a one-epoch execution check. The worker uses upstream AdamW at a fixed
learning rate and removes the long-schedule hooks, optional Albumentations
step, and remote initialization from the upstream recipe.

## Evaluate

```powershell
$poseRun = "runs/train-<id>"
python -m wvision val --run $poseRun
python -m wvision test --run $poseRun
```

MMPose reports native keypoint AP with the selected skeleton's OKS sigmas.
The best checkpoint comes from validation AP. Both splits use ground-truth
person boxes, so the score measures pose given a supplied box. Native COCO
keypoint AP uses up to 20 predictions per image. Evaluating a detector and
RTMPose together requires a separate end-to-end benchmark.

## Predict from person boxes

Save a JSON object mapping each image path to a list of person boxes. Paths
are absolute or relative to the box file. Each box is `[x1, y1, x2, y2]` in
original-image pixels. Use an empty list for a frame with no people. For example,
save this as `.wvision/tiny-rtmpose/persons.json`:

```json
{
  "images/test-0.png": [[8, 4, 48, 82], [85, 4, 124, 82]]
}
```

```powershell
python -m wvision predict --run $poseRun --source .wvision/tiny-rtmpose/images/test-0.png --person-boxes .wvision/tiny-rtmpose/persons.json
```

```python
from wvision import Vision

model = Vision.from_run("runs/train-<id>")
result = model.predict(
    ".wvision/tiny-rtmpose/images/test-0.png",
    person_boxes=".wvision/tiny-rtmpose/persons.json",
)
```

For detector output, use the absolute `sample` and the `boxes` from each
WForge detection JSONL record after filtering to the person classes you want.
The prediction records preserve box order, skeleton and joint count. Each
joint uses original-image pixels and the native MMPose score. Instance
confidence is null because the input only supplies coordinates.

For example, use an [RTMDet](RTMDET.md) run trained to detect people:

```python
import json
from pathlib import Path

from wvision import Vision

detector = Vision.from_run("runs/train-detector")
pose = Vision.from_run("runs/train-pose")
detection = detector.predict("frame.png", score_threshold=0.25)
frame = json.loads(detection.predictions.read_text(encoding="utf-8").splitlines()[0])
boxes = [
    box
    for box, category in zip(frame["boxes"], frame["class_ids"], strict=True)
    if frame["classes"][str(category)] == "person"
]
box_file = Path("person-boxes.json")
box_file.write_text(json.dumps({frame["sample"]: boxes}), encoding="utf-8")
result = pose.predict(frame["sample"], person_boxes=box_file)
```

RTMPose currently accepts the two named skeletons. The training workflow
uses WForge checkpoints from random initialization. Other MMPose families,
external pretrained checkpoints, optimizer resume, and export are separate
integrations.

## RTMPose-t local verification

The offline fixture contains multiple people, an occluded joint, an unlabeled
joint, and a negative frame. The integration tests train from scratch, reload
the selected checkpoint, evaluate both splits separately, and check that
prediction preserves box order and joint count. An empty box list produces an
empty frame result. The test images are absent from the training workspace.

```powershell
python -m pytest -q -m integration tests/test_rtmpose.py -k "t-coco17 or t-halpe26"
$env:WVISION_TEST_DEVICE = "cuda:0"
python -m pytest -q -m integration tests/test_rtmpose.py -k t-halpe26
```

The local run on 2026-11-21 used Python 3.10.20, MMPose 1.3.2,
xtcocotools 1.14.3, and PyTorch 2.1.0+cu121. CPU runs used CPU execution
with the same CUDA-capable installation. The CUDA run used one NVIDIA GeForce
RTX 4050 Laptop GPU. These smoke checks exercise the workflow and do not
measure pose accuracy on football footage.

The CLI was also checked with two supplied boxes. Its result contained
two people with 26 joints each. Local logs are under `.wvision/rtmpose-*.log`.
