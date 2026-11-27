# Third-party notices

WVision integrates optional upstream packages without copying their model/training implementations. This inventory does not replace upstream license texts.

| Component | Initial version | License source |
| --- | --- | --- |
| Ultralytics | 8.3.4 | [Release license](https://github.com/ultralytics/ultralytics/blob/v8.3.4/LICENSE), AGPL-3.0; enterprise licensing is also offered upstream |
| PyTorch | 2.4.1 | [Release license](https://github.com/pytorch/pytorch/blob/v2.4.1/LICENSE) |
| torchvision | 0.19.1 | [Release license](https://github.com/pytorch/vision/blob/v0.19.1/LICENSE) |
| MMDetection | 3.3.0 | [Release license](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/LICENSE), Apache-2.0 |
| MMDeploy (optional MMDetection export) | 1.3.1 | [Release license](https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/LICENSE), Apache-2.0 |
| MMCV | 2.1.0 | [Release license](https://github.com/open-mmlab/mmcv/blob/v2.1.0/LICENSE), Apache-2.0 |
| MMEngine | 0.10.7 | [Release license](https://github.com/open-mmlab/mmengine/blob/v0.10.7/LICENSE), Apache-2.0 |
| MMPose | 1.3.2 | [Release license](https://github.com/open-mmlab/mmpose/blob/v1.3.2/LICENSE), Apache-2.0 |
| MMPreTrain | 1.2.0 | [Release license](https://github.com/open-mmlab/mmpretrain/blob/v1.2.0/LICENSE), Apache-2.0 |
| OpenMMLab PyTorch / torchvision | 2.1.0 / 0.16.0 CPU or CUDA 12.1 | [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE), [torchvision license](https://github.com/pytorch/vision/blob/v0.16.0/LICENSE) |
| Transformers | 5.22.2 | [Release license](https://github.com/huggingface/transformers/blob/v5.22.2/LICENSE), Apache-2.0 |
| Accelerate | 1.10.1 | [Release license](https://github.com/huggingface/accelerate/blob/v1.10.1/LICENSE), Apache-2.0 |
| ONNX (optional export) | 1.17.0 | [Release license](https://github.com/onnx/onnx/blob/v1.17.0/LICENSE), Apache-2.0 |
| ONNX (MMDeploy environment) | 1.13.1 | [Release license](https://github.com/onnx/onnx/blob/v1.13.1/LICENSE), Apache-2.0 |
| ONNX Runtime (optional verification) | 1.20.1 | [Release license](https://github.com/microsoft/onnxruntime/blob/v1.20.1/LICENSE), MIT |
| pycocotools (optional shared evaluation) | 2.0.11 | [COCO API license](https://github.com/cocodataset/cocoapi/blob/master/license.txt) |
| DINOv3 backend PyTorch / torchvision | 2.8.0 / 0.23.0 CPU | [PyTorch license](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE), [torchvision license](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE) |
| Optional official DINOv3 weights | User-supplied local ConvNeXt Tiny | [Meta DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md); not downloaded, bundled, or verified here |

The initial integration uses the bundled YOLO11n configuration and random initialization. It distributes no pretrained weights or third-party datasets. User-provided checkpoint terms and supervision sources must be recorded. Generated checkpoints retain upstream license metadata.

The MMDetection integration loads the installed upstream Soft Teacher recipe and reuses upstream Faster R-CNN and teacher-student algorithms. It starts randomly or from explicitly supplied compatible WVision source runs, and distributes no pretrained assets. Runtime recipe adaptation and provenance are documented in the [semi-supervised guide](docs/SEMI_SUPERVISED.md). Upstream source is not vendored into WVision.

[RTMDet](docs/RTMDET.md) and [RTMPose](docs/RTMPOSE.md) load installed upstream
recipes from MMDetection 3.3.0 and MMPose 1.3.2, respectively. Both start from
random weights and use the caller's dataset.

Optional ONNX conversion calls MMDeploy's installed exporter, decoding and NMS.
Small conversion hooks in `faster_rcnn_r50/deployment.py` preserve MMDetection's
per-pyramid-level proposal NMS and class-specific regression boxes, which the
generic MMDeploy 1.3.1 rewrites change. Relevant upstream owners are
[RPN conversion](https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/dense_heads/rpn_head.py)
and [box-head conversion](https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/codebase/mmdet/models/roi_heads/bbox_heads/bbox_head.py),
under MMDeploy's Apache-2.0 license. No upstream model or algorithm source is
copied. Remove the hooks when an upstream upgrade passes the native-parity
regressions unchanged. See [export evidence](docs/verification/MMDET_ONNX.md).

The MMPreTrain integration loads its installed SimSiam recipe and trains from random initialization on local unlabeled training images. It optionally supplies that run's backbone to supervised Faster R-CNN using upstream MMEngine initialization. No upstream model, loss, or optimizer implementation is copied, and no external weights are distributed. See [self-supervised usage](docs/SELF_SUPERVISED.md). LOST and TokenCut were assessed but are not integrated or bundled.

The DINOv3/DETR adapter calls Transformers' existing DINOv3 ConvNeXt backbone, DETR model, and Trainer. Its execution tests use random initialization and a locally generated random backbone fixture. Transformers' code terms do not grant access to or replace the terms of Meta's official model weights. Those remain an explicit user download and separate verification gate. See [DINOv3 usage](docs/DINOV3.md). No upstream architecture, loss, or training-loop source is vendored into WVision.

Original synthetic fixture images and annotations from `examples/make_tiny_dataset.py` are dedicated to the public domain under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).

WVision's original code is licensed under [Apache-2.0](LICENSE), with attribution in
[NOTICE](NOTICE). This license does not relicense optional upstream packages,
datasets, or model weights. In particular, Ultralytics retains its AGPL-3.0 or
upstream enterprise terms, and official DINOv3 weights retain Meta's separate
license. Use and distribution of a combined application must respect the terms
of the components it includes.
