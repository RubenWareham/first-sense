"""Simple workflows around existing vision models; no ML imports at package import."""

from wvision.api import Vision
from wvision.core.registry import list_models
from wvision.datasets.budgets import prepare_label_budget
from wvision.evaluation.learning import compare_learning
from wvision.evaluation.pretraining import compare_pretraining
from wvision.training.pretraining import pretrain

__all__ = [
    "Vision",
    "prepare_label_budget",
    "compare_learning",
    "pretrain",
    "compare_pretraining",
    "list_models",
]
__version__ = "0.1.0.dev0"
