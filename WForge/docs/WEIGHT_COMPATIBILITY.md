# Weight compatibility and semi-supervised initialization

Implemented paths: supervised Faster R-CNN initialization of upstream Soft Teacher, and SimSiam ResNet-50 backbone initialization of the same method. This extends existing upstream methods; no teacher update, loss, optimizer, or architecture is implemented by WVision. See [execution evidence](verification/INITIALIZATION.md).

## Contract

Weights are compatible only when their source model, transfer scope, target learning method, architecture, preprocessing, class semantics, and provenance meet the target adapter's requirements. A shared file extension or matching tensor dimensions is insufficient. The registry describes accepted initialization rules; adapters validate source artifacts and the worker checks actual tensors.

The first contract deliberately accepts completed WVision source runs on the exact same dataset fingerprint. Supervised sources use only the declared labeled training pool and validation selection. This prevents initializing a label-budget experiment with a source trained on withheld labels or differently assigned held-out data. Other datasets or external raw checkpoints need a separate audited transfer contract and are not silently accepted.

| Source | Target | Scope | Required checks |
| --- | --- | --- | --- |
| WVision SimSiam ResNet-50 | Faster R-CNN supervised or Soft Teacher | Backbone only | Same dataset; PyTorch-style ResNet and RGB normalization; complete backbone state |
| WVision supervised Faster R-CNN R50-FPN | Soft Teacher | Complete detector into teacher and student | Same dataset/classes; same resolved detector structure, stride style, normalization, and source version; complete detector state |
| Trusted local YOLO11n checkpoint | YOLO11n supervised | Complete model | Existing YOLO architecture checks; external supervision supplied by owner |
| Local DINOv3 ConvNeXt Tiny safetensors | DETR supervised | Backbone | Existing strict local loader; official pretrained verification remains open |

The table is an integration boundary, not a statement that every model can use every source. A DINOv3 ConvNeXt checkpoint cannot initialize the current ResNet-50 Soft Teacher detector. A YOLO checkpoint cannot initialize Faster R-CNN. Optimizer-state resume is separate from initialization.

## Public workflow

Use `detector_run=` / `--detector-run`, separate from existing `pretraining_run` and `backbone_weights`. Select at most one initialization source. Source references and hashes are saved, and a failed check never falls back to random weights. `wvision models` exposes each model's accepted `initializations` rules, including source kind, scope, learning mode, and required backbone style.

```python
from wvision import Vision

data = ".wvision/semi-budget/dataset.yaml"
source = Vision("models/detection/faster-rcnn-r50/supervised-smoke.yaml").train(data)
model = Vision("models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml")
run = model.train(data, detector_run=source.artifact_dir)
validation = model.val()
```

Set `WVISION_MMDET_PYTHON` as described in the [backend setup](SEMI_SUPERVISED.md), or pass `backend_python=` to both constructors. The [complete example](../examples/initialized_semi_supervised.py) has a Windows main guard. These two-step smoke settings check execution only.

CLI equivalent, using a completed supervised source:

```text
wvision train --model models/detection/faster-rcnn-r50/soft-teacher-smoke.yaml --data .wvision/semi-budget/dataset.yaml --detector-run runs/train-<supervised-id>
```

For SimSiam, use `Vision("models/detection/faster-rcnn-r50/soft-teacher-pytorch-smoke.yaml").train(data, pretraining_run=representation.artifact_dir)`. The [pretraining guide](SELF_SUPERVISED.md) explains how to create that representation. A full detector source can also use the PyTorch-style target, provided the source has the same style.

For complete detector initialization, WVision verifies the completed supervised run, data identity, class map, checkpoint/config hashes, and backend version. The worker compares the resolved source detector against its target detector, excluding initialization-only configuration. It verifies complete state keys/shapes/dtypes and checkpoint class metadata, delegates loading to upstream MMEngine/MMDetection, and checks tensor equality in both teacher and student before training. Only trusted local checkpoints are accepted: upstream loading uses Python deserialization.

For SimSiam, only backbone tensors transfer; detector heads initialize randomly. Both teacher and student backbones must match the source before upstream training begins. The upstream MeanTeacherHook still controls initial teacher synchronization and later moving-average updates.

`transfer.json` records scope, source checksum, tensor count, both branch audits, and that optimizer state was not loaded. The read-only audit runs after upstream initialization hooks and before the first training step. `run.json` preserves the compatibility rule and source provenance; `model-source.json` records the actual recipe adaptation. Both the wrapper's and detector's input preprocessing match the selected style. Training starts a fresh schedule; this is initialization, not resume. The completed target has its own full checkpoint and can predict without the source artifact. OpenMMLab initialization also works in the [single-device CUDA workflow](GPU.md); pre-training audits compare tensors on CPU without altering the model weights. A CUDA-trained detector can be reloaded with an explicitly selected CPU backend/device.

Official upstream references: [MMDetection 3.3.0 supervised-to-teacher/student loading](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/detectors/semi_base.py) and [MeanTeacherHook](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/engine/hooks/mean_teacher_hook.py).

Real validation must train from actual upstream checkpoints, exercise unlabeled losses, evaluate frozen selected checkpoints, reload and predict, check tensor equality, and reject incompatible or damaged sources. Synthetic results establish execution and must not be reported as accuracy gains.
