# Project scope

## Goal

Make reliable computer vision experiments easy to run and compare, especially when labeled data is scarce. Provide an approachable API inspired by the simplicity of Ultralytics, while keeping backend choice explicit and preserving advanced upstream features.

"State of the art" is an ambition for selecting and evaluating integrations, not a permanent property of the repository. Prefer reproducible, maintained implementations over unsupported leaderboard claims. Revisit candidate selection when an integration begins.

## Task order

1. Object detection: boxes, classes, confidence scores, training, evaluation, inference.
2. 2D pose estimation: keypoints, visibility, skeleton metadata, task-specific metrics.
3. Image classification: image-level labels, scores, classification metrics.
4. Additional tasks through the same capability mechanism when a concrete integration needs them.

Do not force keypoints, classifications, masks, or anomaly maps into a bounding-box result format.

## Learning terminology

| Learning mode | Meaning | Evaluation requirement |
| --- | --- | --- |
| `supervised` | Train with labeled training samples | Held-out labels for task metrics |
| `semi_supervised` | Use labeled and unlabeled training samples with an existing upstream method | Fixed label budget and held-out labeled evaluation |
| `self_supervised` | Learn representations from a pretext objective on unlabeled data | Declared downstream evaluation protocol |
| `unsupervised` | Learn the target structure without target training annotations | A declared task and protocol; labels may be used only for held-out evaluation |

Self-supervised pretraining followed by a labeled detection head is not fully unsupervised detection. Pseudo-labeling using a supervised pretrained detector must disclose that source of supervision. Inference from a pretrained model alone is not evidence of unsupervised training support.

## Scope boundaries

WVision owns user-facing configuration, dataset preflight checks, backend selection, experiment metadata, normalized outputs, and reproducible comparisons. Existing upstream implementations own model definitions, training loops, algorithm logic, and task evaluators where available.

Initial scope excludes custom architectures, research algorithm reimplementation, annotation interfaces, a model hosting service, distributed scheduling infrastructure, and deployment servers. These can be reconsidered only for a concrete user need.

## Success criteria

- A new user can run the first supported workflow from a short documented example.
- Switching supported models does not require rewriting experiment orchestration.
- A failed or unsupported configuration stops with an actionable message before training.
- A result identifies its model, data split, label budget, environment, and metric definition.
- A semi-supervised comparison uses the same labeled subset as its supervised baseline.
