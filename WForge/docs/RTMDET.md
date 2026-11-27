# RTMDet object detection

Use `mmdetection/rtmdet-tiny`, `mmdetection/rtmdet-s`, `mmdetection/rtmdet-m`,
`mmdetection/rtmdet-l`, or `mmdetection/rtmdet-x` for supervised bounding-box
detection. WForge loads the installed [MMDetection 3.3.0 RTMDet recipes](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/rtmdet)
and trains with the upstream model, loss, assigner, AdamW optimizer, and runner.
The dataset's class names and count set the box head and prediction labels.

## Setup and run

Use the isolated Python 3.10 MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.7,
and PyTorch 2.1.0 stack. Follow the [CPU setup](SEMI_SUPERVISED.md) or
[CUDA setup](GPU.md), then set `WVISION_MMDET_PYTHON` or pass `--backend-python`.
Run these commands from `WForge/` with the Python 3.12 application environment:

```powershell
python examples/make_tiny_dataset.py .wvision/tiny
python -m wvision train --model models/detection/rtmdet-tiny/smoke.yaml --data .wvision/tiny/dataset.yaml
```

Use the returned `artifact_dir` in place of `runs/train-<id>`:

```powershell
$detectorRun = "runs/train-<id>"
python -m wvision val --run $detectorRun
python -m wvision test --run $detectorRun
python -m wvision predict --run $detectorRun --source .wvision/tiny/images/test-0.png
```

The smoke profiles run one iteration at 64 pixels. For your own dataset, use a
[detection manifest](../datasets/README.md), choose an iteration budget with
`--max-iters`, and set `--imgsz` to a multiple of 32, at least 64.
Use `--device cuda:0` with the CUDA stack. In Python,
pass the same model ID or profile to `Vision`. Use `Vision.from_run` to restore
the validation-selected checkpoint and resolved configuration.

This workflow uses random initialization, single-device FP32, the upstream
stage-two training transforms at a fixed size, and a fixed learning rate
(override `backend_options.lr` in Python or the profile). It validates with
native COCO bbox AP and writes the best and final checkpoints. Prediction
writes image dimensions, pixel `xyxy` boxes, scores, and class IDs. Pretrained
initialization, optimizer-state resume, AMP, distributed training, and RTMDet
ONNX export require separate integrations.

For a detector trained with a person class, you can pass its predicted person
boxes to [RTMPose](RTMPOSE.md#predict-from-person-boxes).
