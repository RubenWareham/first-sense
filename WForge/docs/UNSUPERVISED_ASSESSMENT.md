# Fully unsupervised detection assessment

Decision reviewed **2026-11-08**: leave fully unsupervised detection unsupported. This is a selection decision for the current Windows CPU repository, not a claim that unsupervised detection is impossible. [SimSiam transfer](SELF_SUPERVISED.md) is a separate capability and uses downstream detection labels.

## Candidates and concrete gaps

| Upstream implementation | What it provides | Gap against the current acceptance gate |
| --- | --- | --- |
| [LOST](https://github.com/valeoai/LOST) | Self-supervised object discovery, plus separate class-agnostic and clustered detector workflows | Older Python 3.7 / torch 1.7.1 / CUDA 10.2 environment; detector stage uses Detectron2 0.5 and upstream multi-GPU recipes. No verified Windows CPU environment or WVision train/val/test/predict adapter. |
| [TokenCut](https://github.com/YangtaoWANG95/TokenCut) | Unsupervised object discovery and salient-region segmentation from self-supervised transformer features | Published setup uses Python 3.7 / torch 1.7.1; discovery evaluation reports CorLoc. It does not establish a verified semantic multi-class detection workflow under WVision's COCO AP contract. |

These assessments use upstream documentation and source inspection, not a failed attempt to run every method. Neither candidate was installed or executed here. No benchmark ranking or maintenance guarantee is inferred from repository popularity.

LOST's [README](https://github.com/valeoai/LOST/blob/master/README.md) separates object discovery, class-agnostic detector training, and class-aware clustering. Its [network loader](https://github.com/valeoai/LOST/blob/master/networks.py) automatically obtains reference DINO weights on the default path. An adapter would need explicit local-weight handling and recorded pretraining/overlap provenance. Its [license](https://github.com/valeoai/LOST/blob/master/LICENSE) is Apache-2.0. These are integration requirements, not reasons to silently use supervised weights instead.

TokenCut's [README](https://github.com/YangtaoWANG95/TokenCut/blob/master/README.md) distinguishes unsupervised discovery from weakly supervised localization and uses DINO features. Its repository declares an [MIT license](https://github.com/YangtaoWANG95/TokenCut/blob/master/LICENSE). Discovery or foreground segmentation alone cannot supply semantic class IDs for the existing detection API.

## Evaluation and semantic mapping gate

An implementation must declare whether outputs are class-agnostic boxes, anonymous clusters, or semantic classes. Class-agnostic AP/CorLoc must have separate names and ground-truth definitions. Anonymous cluster IDs must not be presented as named categories. Any cluster-to-category mapping must record its source supervision, mapping procedure, and split: test labels cannot choose that mapping or a checkpoint. Post-hoc test-based matching, if reproduced as an upstream research protocol, must be reported separately and cannot be called held-out semantic evaluation.

Before adoption, pin exact upstream commits and weight checksums/terms, verify an isolated environment, supply local weights without hidden downloads, validate training-only pseudo-label generation, preserve source-label isolation, and run real public operations on owned fixtures. A Linux/GPU research environment and a class-agnostic result contract may be a practical next assessment. Until these gates pass, `learning=unsupervised` remains rejected; there is no automatic fallback.
