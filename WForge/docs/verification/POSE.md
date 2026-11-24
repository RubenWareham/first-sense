# YOLO11n-pose verification

Verified on Windows with Python 3.12, Ultralytics 8.3.4, torch 2.4.1+cpu and
torch 2.4.1+cu124 on one NVIDIA GeForce RTX 4050 Laptop GPU. Both runs used FP32,
batch size two, and one epoch at 128 pixels. The generated fixture contains four
training, two validation and two test images. It covers multiple people, a
negative image, occlusion and an unlabeled joint. No dataset or model weights
were downloaded.

From `WForge/`, use the matching [backend environment](../POSE.md):

```powershell
python -m pytest -q -m integration tests/test_pose.py tests/test_integration.py::test_python_workflow
$env:WVISION_POSE_TEST_DEVICE = "cuda:0"
python -m pytest -q -m integration tests/test_pose.py
```

The CPU run passed the pose workflow and the YOLO11n detection regression,
including detector fine-tuning. The CUDA run passed the pose workflow and recorded
CUDA model placement. Both pose tests exercised API training and run reload,
CLI validation/test, original image dimensions, prediction parity with upstream,
and native pose AP parity. Known upstream result objects checked ordered joints,
person association and low-confidence null coordinates. The training workspace
contained only staged training and validation images. These fixture runs verify
execution and schema behavior. A football pose accuracy benchmark requires
separate keypoint annotations and dataset splits.

Local ignored logs are `.wvision/pose-cpu-final.log` and
`.wvision/pose-cuda-final.log`. Training artifacts include `run.json`,
`device.json`, `dataset.json`, `model-source.json`, and `upstream/results.csv`.

## Variable skeletons, 2026-11-21

The pose integration test now exercises COCO-17 and Halpe-26. It checks the
resolved skeleton on reload, native evaluation on both splits, rejection of
a mismatched checkpoint head, and ordered output with the declared joint count.
The Halpe-26 case also passed on the RTX 4050 using `WVISION_POSE_TEST_DEVICE=cuda:0`.
An existing COCO-17 run retained its original dataset fingerprint and resolved
skeleton on reload.

A separate synthetic CPU smoke check trained, reloaded, tested, and predicted
with 133 custom joints. The first 17 slots had fixture labels and the remaining
slots were unlabeled. The check verified the output shape and uniform OKS
sigmas. COCO-WholeBody accuracy and its benchmark protocol are separate checks.
Local logs are under `.wvision/variable-pose-*.log`.
