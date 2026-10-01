"""The names every learner package re-exports.

Each learner package does ``from rieszreg.user_api import *``, so
``from rieszboost import ATE, KLLoss`` and ``from krrr import ATE, KLLoss``
give the same objects as ``from rieszreg import ATE, KLLoss``. Add a name
here (not in each package) to make it available everywhere.
"""

from .diagnostics import Diagnostics, diagnose
from .estimands import (
    ATE,
    ATT,
    AdditiveShift,
    Estimand,
    FiniteEvalEstimand,
    LinearForm,
    LocalShift,
    OutcomeRegNormSq,
    TSM,
    Tracer,
)
from .estimator import RieszEstimator
from .losses import BernoulliLoss, BoundedSquaredLoss, KLLoss, Loss, SquaredLoss
from .scoring import riesz_scorer

__all__ = [
    # Estimands: what to estimate
    "ATE",
    "ATT",
    "AdditiveShift",
    "LocalShift",
    "OutcomeRegNormSq",
    "TSM",
    "Estimand",
    "FiniteEvalEstimand",
    "LinearForm",
    "Tracer",
    # Losses
    "SquaredLoss",
    "KLLoss",
    "BernoulliLoss",
    "BoundedSquaredLoss",
    "Loss",
    # Estimator, scoring, diagnostics
    "RieszEstimator",
    "riesz_scorer",
    "diagnose",
    "Diagnostics",
]
