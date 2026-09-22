"""sklearn scorer factory for Bregman-Riesz losses.

`RieszEstimator.score()` uses the canonical squared yardstick for cross-
estimator comparability (analog of R² for regressors). When a different
yardstick is wanted — e.g. KL on a density-ratio problem — pass
`scoring=riesz_scorer(loss=KLLoss())` to any sklearn CV utility.
"""

from __future__ import annotations

from .losses import Loss, SquaredLoss


def riesz_scorer(loss: Loss | None = None):
    """Return an sklearn-compatible scorer (`(estimator, Z, y=None) -> float`).

    Parameters
    ----------
    loss : Loss or None, default=None
        Yardstick loss to evaluate on the held-out fold. If `None`, defaults
        to `SquaredLoss()` (matches `RieszEstimator.score`).

    Notes
    -----
    The fitted estimator's own link maps backend output η to α; the yardstick
    `loss` is then evaluated on that α with the held-out augmented (a, b)
    coefficients. The yardstick must accept the estimator's α: `SquaredLoss`
    has unrestricted α-domain, while `KLLoss` requires α > 0,
    `BernoulliLoss` requires α ∈ (0, 1), and `BoundedSquaredLoss(lo, hi)`
    requires α ∈ (lo, hi).
    """
    yardstick = loss if loss is not None else SquaredLoss()

    def _scorer(estimator, Z, y=None) -> float:
        return -estimator._loss_on(Z, y, yardstick)

    return _scorer
