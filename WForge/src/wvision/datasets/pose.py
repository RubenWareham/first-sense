"""Skeleton definitions and annotation validation for 2D pose."""

import json
import math
from pathlib import Path
from typing import TypedDict

from wvision.core.config import reject_unknown


class PoseSkeleton(TypedDict):
    id: str
    schema_version: int
    index_base: int
    joints: list[str]
    edges: list[list[int]]
    kpt_shape: list[int]
    flip_idx: list[int]


JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def coco17() -> PoseSkeleton:
    return {
        "id": "coco17",
        "schema_version": 1,
        "joints": list(JOINTS),
        "edges": [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ],
        "index_base": 0,
        "kpt_shape": [len(JOINTS), 3],
        "flip_idx": [0, *(i + 1 if i % 2 else i - 1 for i in range(1, len(JOINTS)))],
    }


def halpe26() -> PoseSkeleton:
    skeleton = coco17()
    # OpenMMLab Halpe-26 joint order. Its first 17 joints follow COCO.
    skeleton["id"] = "halpe26"
    skeleton["joints"] += [
        "head",
        "neck",
        "hip",
        "left_big_toe",
        "right_big_toe",
        "left_small_toe",
        "right_small_toe",
        "left_heel",
        "right_heel",
    ]
    torso_edges = ([11, 12], [5, 11], [6, 12], [5, 6])
    skeleton["edges"] = [edge for edge in skeleton["edges"] if edge not in torso_edges] + [
        [17, 18],
        [18, 5],
        [18, 6],
        [18, 19],
        [19, 11],
        [19, 12],
        [15, 20],
        [16, 21],
        [15, 22],
        [16, 23],
        [15, 24],
        [16, 25],
    ]
    skeleton["flip_idx"] += [17, 18, 19, 21, 20, 23, 22, 25, 24]
    skeleton["kpt_shape"] = [len(skeleton["joints"]), 3]
    return skeleton


def resolve_skeleton(value: object) -> PoseSkeleton:
    if value == "coco17":
        return coco17()
    if value == "halpe26":
        return halpe26()
    if not isinstance(value, dict):
        raise ValueError("Pose skeleton must be coco17, halpe26, or a skeleton mapping")
    reject_unknown(value, {"id", "joints", "flip_idx", "edges"}, "pose skeleton")
    name, joints = value.get("id"), value.get("joints")
    if not isinstance(name, str) or not name.strip() or name in {"coco17", "halpe26"}:
        raise ValueError("A custom pose skeleton requires a nonempty, distinct ID")
    if (
        not isinstance(joints, list)
        or not joints
        or any(not isinstance(joint, str) or not joint.strip() for joint in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("Pose skeleton joints must be a nonempty list of unique names")
    count = len(joints)
    flip = value.get("flip_idx")
    if (
        not isinstance(flip, list)
        or any(type(index) is not int for index in flip)
        or sorted(flip) != list(range(count))
        or any(flip[index] != joint for joint, index in enumerate(flip))
    ):
        raise ValueError("Pose skeleton flip_idx must pair each joint with its mirror or itself")
    edges = value.get("edges", [])
    if not isinstance(edges, list) or any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(type(index) is not int or not 0 <= index < count for index in edge)
        or edge[0] == edge[1]
        for edge in edges
    ):
        raise ValueError("Pose skeleton edges must contain pairs of distinct joint indices")
    return {
        "id": name,
        "schema_version": 1,
        "index_base": 0,
        "joints": joints,
        "edges": edges,
        "flip_idx": flip,
        "kpt_shape": [count, 3],
    }


def restore_skeleton(value: object) -> PoseSkeleton:
    if not isinstance(value, dict):
        raise ValueError("Pose run is missing its resolved skeleton")
    name = value.get("id")
    definition = (
        name
        if name in ("coco17", "halpe26")
        else {key: value[key] for key in ("id", "joints", "flip_idx", "edges") if key in value}
    )
    skeleton = resolve_skeleton(definition)
    if skeleton != value:
        raise ValueError("Pose run has inconsistent skeleton metadata")
    return skeleton


def valid_joints(fields: list[str], count: int) -> bool:
    if len(fields) != 3 * count:
        return False
    values = iter(map(float, fields))
    return all(
        (x == y == 0 if visibility == 0 else visibility in (1, 2) and 0 <= x <= 1 and 0 <= y <= 1)
        for x, y, visibility in zip(values, values, values, strict=True)
    )


def read_person_boxes(
    path: str | Path, images: dict[Path, tuple[int, int]]
) -> dict[Path, list[list[float]]]:
    """Load explicit top-down crops in original-image pixel xyxy."""
    source = Path(path).resolve(strict=True)
    inputs = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(inputs, dict):
        raise ValueError("person_boxes must map image paths to lists of xyxy boxes")
    boxes: dict[Path, list[list[float]]] = {}
    for image, rows in inputs.items():
        sample = (source.parent / image).resolve()
        if sample not in images or sample in boxes or not isinstance(rows, list):
            raise ValueError(f"Unexpected or repeated image in person_boxes: {image}")
        width, height = images[sample]
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                or not (0 <= row[0] < row[2] <= width and 0 <= row[1] < row[3] <= height)
            ):
                raise ValueError(f"Invalid xyxy person box for {image}: {row!r}")
        boxes[sample] = rows
    if set(boxes) != set(images):
        raise ValueError("person_boxes must cover every prediction image; use [] for empty frames")
    return boxes
