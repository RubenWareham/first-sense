"""Pinned MMDeploy ONNX conversion, with native detection semantics and parity checks.

MMDeploy 1.3.1's generic RPN rewrite suppresses across pyramid levels, and its
box-head rewrite shares the winning class's box across classes. The hooks below
retain MMDetection 3.3.0's grouping/class-specific boxes using upstream decoding
and NMS. No model, training algorithm, or upstream source is copied. Remove these
hooks when an upstream upgrade passes the native-parity regressions unchanged.
"""

import copy
from importlib import metadata
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch
from checkpoints import checkpoint_files  # type: ignore[import-not-found]
from mmdeploy.apis import build_task_processor, torch2onnx
from mmdeploy.codebase.mmdet.models.dense_heads import rpn_head as rpn_export
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import multiclass_nms
from mmengine import Config
from PIL import Image
from runtime import dump  # type: ignore[import-not-found]
from scipy.optimize import linear_sum_assignment


def digest(path: Path) -> str:
    return str(checkpoint_files(path)[path.name])


@FUNCTION_REWRITER.register_rewriter(  # type: ignore[untyped-decorator]
    "mmdet.models.roi_heads.bbox_heads.bbox_head.BBoxHead.predict_by_feat",
    backend="onnxruntime",
)
def class_specific_boxes(
    self: Any,
    rois: Any,
    cls_scores: Any,
    bbox_preds: Any,
    batch_img_metas: Any,
    rcnn_test_cfg: Any = None,
    rescale: bool = False,
) -> Any:
    if rescale or rcnn_test_cfg is None or self.custom_cls_channels:
        raise ValueError(
            "Export requires standard Faster R-CNN class channels and input-pixel boxes"
        )
    scores = cls_scores.softmax(-1)[..., : self.num_classes]
    boxes = self.bbox_coder.decode(
        rois[..., 1:], bbox_preds, max_shape=batch_img_metas[0]["img_shape"]
    )
    if self.reg_class_agnostic:
        boxes = boxes.unsqueeze(2).expand(-1, -1, self.num_classes, -1)
    boxes = boxes.reshape(1, -1, 4)
    # Standard ONNX NMS has one box per row. Flatten class-specific rows, retaining
    # each score only in its own class channel; other channels cannot enter NMS.
    diagonal = torch.eye(self.num_classes, dtype=torch.bool, device=scores.device)
    diagonal = diagonal.view(1, 1, self.num_classes, self.num_classes)
    matrix = torch.where(diagonal, scores.unsqueeze(-1), scores.new_full((), -2))
    # MMDeploy may pad the RPN output. Native RPN min_bbox_size=0 excludes these.
    valid = (rois[..., 3] > rois[..., 1]) & (rois[..., 4] > rois[..., 2])
    matrix = torch.where(valid[:, :, None, None], matrix, matrix.new_full((), -2))
    return multiclass_nms(
        boxes,
        matrix.reshape(1, -1, self.num_classes),
        max_output_boxes_per_class=rcnn_test_cfg.max_per_img,
        iou_threshold=rcnn_test_cfg.nms.iou_threshold,
        score_threshold=rcnn_test_cfg.score_thr,
        pre_top_k=-1,
        keep_top_k=rcnn_test_cfg.max_per_img,
    )


def compare_detections(expected: Any, dets: Any, labels: Any) -> dict[str, Any]:
    """Compare complete detection sets without requiring arbitrary tie ordering."""
    boxes = expected.bboxes.detach().cpu().numpy()
    scores = expected.scores.detach().cpu().numpy()
    classes = expected.labels.detach().cpu().numpy()
    if not np.isfinite(dets).all() or not np.isfinite(labels).all():
        raise ValueError("ONNX parity failed: non-finite output")
    keep = dets[:, 4] > 0  # MMDeploy pads unused output rows with score zero.
    dets, labels = dets[keep], labels[keep]
    if not np.isfinite(dets).all() or len(dets) != len(boxes):
        raise ValueError("ONNX parity failed: non-finite output or different detection count")
    if set(labels.tolist()) != set(classes.tolist()):
        raise ValueError("ONNX parity failed: different detected classes")
    box_error, score_error = 0.0, 0.0
    for category in np.unique(classes):
        native_rows, exported_rows = (
            np.flatnonzero(classes == category),
            np.flatnonzero(labels == category),
        )
        if len(native_rows) != len(exported_rows):
            raise ValueError("ONNX parity failed: different per-class detection counts")
        reference = boxes[native_rows]
        observed = dets[exported_rows, :4]
        costs = np.max(
            np.abs(reference[:, None] - observed[None])
            / (1e-3 + 1e-4 * np.abs(reference[:, None])),
            axis=2,
        )
        left, right = linear_sum_assignment(costs)
        actual_boxes = observed[right]
        actual_scores = dets[exported_rows[right], 4]
        target_scores = scores[native_rows[left]]
        if not np.allclose(actual_boxes, reference[left], rtol=1e-4, atol=1e-3) or not np.allclose(
            actual_scores, target_scores, rtol=1e-3, atol=1e-4
        ):
            raise ValueError(
                "ONNX parity failed: boxes/scores differ; inspect representative images "
                "and proposal/NMS ties"
            )
        box_error = max(box_error, float(np.max(np.abs(actual_boxes - reference[left]))))
        score_error = max(score_error, float(np.max(np.abs(actual_scores - target_scores))))
    return {
        "detections": len(boxes),
        "max_box_error_pixels": box_error,
        "max_score_error": score_error,
    }


def export_detector(
    model: Any, cfg: Any, request: dict[str, Any], directory: Path
) -> dict[str, str]:
    expected_versions = {
        "mmdeploy": "1.3.1",
        "onnx": "1.13.1",
        "onnxruntime": "1.20.1",
        "protobuf": "3.20.2",
    }
    for name, version in expected_versions.items():
        if metadata.version(name) != version:
            raise RuntimeError(f"MMDetection export requires {name}=={version}")
    paths = [Path(path).resolve(strict=True) for path in request["verification_images"]]
    if len(set(paths)) < 2:
        raise ValueError("Supply at least two distinct representative verification images")
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
    hashes = [digest(path) for path in paths]
    branch = "detector"
    if cfg.model.type == "SoftTeacher":
        branch = cfg.model.semi_test_cfg.get("predict_on", "teacher")
        if branch not in {"teacher", "student"}:
            raise ValueError("Soft Teacher export supports its teacher or student inference branch")
        detector = getattr(model, branch)
        cfg.model = copy.deepcopy(cfg.model.detector)
    else:
        detector = model
    if cfg.model.type != "FasterRCNN" or cfg.model.test_cfg.rpn.min_bbox_size != 0:
        raise ValueError("Export requires the registered FasterRCNN with RPN min_bbox_size=0")
    cfg.model.backbone.init_cfg = None
    cfg.test_pipeline = [item for item in cfg.test_pipeline if item["type"] != "LoadAnnotations"]
    checkpoint = directory / "detector.pth"
    torch.save(
        {"state_dict": detector.state_dict(), "meta": {"dataset_meta": model.dataset_meta}},
        checkpoint,
    )
    size = request["imgsz"]
    rcnn, rpn = cfg.model.test_cfg.rcnn, cfg.model.test_cfg.rpn
    deploy = Config(
        dict(
            onnx_config=dict(
                type="onnx",
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=17,
                save_file="model.onnx",
                input_names=["input"],
                output_names=["dets", "labels"],
                input_shape=[size, size],
                optimize=False,
            ),
            backend_config=dict(type="onnxruntime"),
            codebase_config=dict(
                type="mmdet",
                task="ObjectDetection",
                model_type="end2end",
                post_processing=dict(
                    score_threshold=rcnn.score_thr,
                    confidence_threshold=0.005,
                    iou_threshold=rcnn.nms.iou_threshold,
                    max_output_boxes_per_class=rcnn.max_per_img,
                    pre_top_k=-1,
                    keep_top_k=rcnn.max_per_img,
                    background_label_id=-1,
                ),
            ),
        )
    )
    task = build_task_processor(cfg, deploy, "cpu")
    native = task.build_pytorch_model(str(checkpoint))
    source_weights, restored_weights = detector.state_dict(), native.state_dict()
    if source_weights.keys() != restored_weights.keys() or not all(
        torch.equal(value.detach().cpu(), restored_weights[name].detach().cpu())
        for name, value in source_weights.items()
    ):
        raise ValueError("Export detector state differs from the selected source branch")
    _, pixels = task.create_input(
        images[0], [size, size], data_preprocessor=native.data_preprocessor
    )
    with torch.inference_mode():
        features = native.extract_feat(pixels)
    if rpn.nms_pre <= 0:
        raise ValueError("Export requires a positive fixed RPN nms_pre")
    counts = [
        min(int(feature.shape[2] * feature.shape[3]) * priors, rpn.nms_pre)
        for feature, priors in zip(
            features, native.rpn_head.prior_generator.num_base_priors, strict=True
        )
    ]
    level_ids = torch.cat(
        [torch.full((count,), index, dtype=torch.long) for index, count in enumerate(counts)]
    )

    def level_nms(boxes: Any, scores: Any, *args: Any, **kwargs: Any) -> Any:
        valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
        levels = level_ids.to(boxes.device)
        matrix = torch.where(
            (torch.arange(len(counts), device=boxes.device)[None, None] == levels[None, :, None])
            & valid[:, :, None],
            scores.expand(-1, -1, len(counts)),
            scores.new_full((), -2),
        )
        return multiclass_nms(
            boxes,
            matrix,
            max_output_boxes_per_class=rpn.max_per_img,
            iou_threshold=rpn.nms.iou_threshold,
            score_threshold=-1,
            pre_top_k=-1,
            keep_top_k=rpn.max_per_img,
        )

    dump(directory / "deployment-config.json", deploy.to_dict())
    dump(directory / "detector-config.json", cfg.to_dict())
    target = directory / "model.onnx"
    with patch.object(rpn_export, "multiclass_nms", level_nms):
        torch2onnx(
            images[0], str(directory), target.name, deploy, cfg, str(checkpoint), device="cpu"
        )
    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)
    domains = sorted({node.domain for node in graph.graph.node} - {"", "ai.onnx"})
    if domains:
        raise ValueError(f"Export contains unverified custom runtime operators: {domains}")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    cases = []
    for path, image, checksum in zip(paths, images, hashes, strict=True):
        data, pixels = task.create_input(
            image, [size, size], data_preprocessor=native.data_preprocessor
        )
        with torch.inference_mode():
            expected = native.predict(pixels, data["data_samples"], rescale=False)[0].pred_instances
        dets, labels = session.run(None, {"input": pixels.cpu().numpy()})
        result = compare_detections(expected, dets[0], labels[0])
        if digest(path) != checksum:
            raise ValueError("Verification image changed during export")
        cases.append({"path": str(path), "sha256": checksum, **result})
    if not any(case["detections"] for case in cases):
        raise ValueError(
            "Verification has no native detections; supply representative positive images"
        )
    preprocess = cfg.model.data_preprocessor
    dump(
        directory / "export.json",
        {
            "schema_version": 1,
            "format": "onnx",
            "model": target.name,
            "sha256": digest(target),
            "source_checkpoint_sha256": digest(Path(request["checkpoint"])),
            "source_branch": branch,
            "source_state_verified": True,
            "tensor_count": len(source_weights),
            "classes": dict(enumerate(model.dataset_meta["classes"])),
            "precision": "fp32",
            "opset": 17,
            "exporter": expected_versions,
            "input": {
                "name": "input",
                "shape": [1, 3, size, size],
                "dtype": "float32",
                "preprocessing": {
                    "resize": [size, size],
                    "preserve_aspect_ratio": False,
                    "interpolation": "OpenCV bilinear",
                    "layout": "NCHW RGB" if preprocess.get("bgr_to_rgb") else "NCHW BGR",
                    "pixel_range_before_normalization": [0, 255],
                    "mean": list(preprocess.mean),
                    "std": list(preprocess.std),
                },
            },
            "output": {
                "dets": "batch, padded detections, xyxy in input pixels plus score",
                "labels": "batch, padded detections; zero-based class IDs",
                "padding": "ignore rows with score zero",
                "nms_included": True,
                "score_threshold": rcnn.score_thr,
                "iou_threshold": rcnn.nms.iou_threshold,
                "max_detections": rcnn.max_per_img,
                "rescaling": "multiply x by original_width/input_width "
                "and y by original_height/input_height",
            },
            "custom_op_domains": domains,
            "verification": {
                "runtime": "onnxruntime CPUExecutionProvider",
                "samples": len(cases),
                "box_rtol": 1e-4,
                "box_atol": 1e-3,
                "score_rtol": 1e-3,
                "score_atol": 1e-4,
                "matching": "complete one-to-one sets within each class",
                "cases": cases,
                "limitations": "Proposal/NMS score ties can alter decisions on other inputs; "
                "fixed batch and input size only",
            },
        },
    )
    return {"model": str(target)}
