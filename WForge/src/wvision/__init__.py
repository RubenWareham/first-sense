"""Simple workflows around existing vision models; no ML imports at package import."""

from wvision.api import Vision
from wvision.core.registry import list_models
from wvision.datasets.budgets import prepare_label_budget
from wvision.datasets.groups import prepare_group_split
from wvision.evaluation.learning import compare_learning
from wvision.evaluation.pretraining import compare_pretraining
from wvision.experiments.comparison import run_experiments
from wvision.experiments.portable import bundle_run
from wvision.training.pretraining import pretrain
from wvision.training.recovery import resume

__all__ = [
    "Vision",
    "prepare_label_budget",
    "prepare_group_split",
    "compare_learning",
    "pretrain",
    "compare_pretraining",
    "list_models",
    "resume",
    "run_experiments",
    "bundle_run",
]
__version__ = "0.1.0.dev0"
