"""Per-leaf decoupling: the closed-form leaf optimum matches a numerical
minimum of the leaf's augmented Riesz loss."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from rieszreg import (
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    SquaredLoss,
)
from riesztree.splitter import make_leaf_solvers


@pytest.mark.parametrize(
    "loss, linear_offset, domain",
    [
        (SquaredLoss(), 0.0, (-50.0, 50.0)),
        (BoundedSquaredLoss(lo=-0.5, hi=0.8), 0.0, (-0.5, 0.8)),
        # The KL kernel uses h(t) = t log t - t; rieszreg.KLLoss uses t log t.
        # The leaf losses differ by C, which cancels in every split gain.
        (KLLoss(), 1.0, (1e-9, 50.0)),
        (BernoulliLoss(), 0.0, (1e-9, 1 - 1e-9)),
    ],
)
@pytest.mark.parametrize(
    "D, C",
    [
        (10.0, -3.0),   # interior optimum for every loss
        (7.0, -6.5),    # α* = 0.93: above BoundedSquared's hi
        (4.0, 3.0),     # α* = -0.75: below BoundedSquared's lo; KL / Bernoulli infeasible
        (3.0, -0.2),    # small α*
        (5.0, 0.0),     # C = 0: KL and Bernoulli optimum sits on the boundary α = 0
        (3.0, -3.0),    # C = -D: Bernoulli optimum α* = 1 with loss 0 (once wrongly infeasible)
        (3.0, -3.03),   # C < -D: Bernoulli infeasible
    ],
)
def test_leaf_loss_is_the_minimum_of_the_rieszreg_loss(loss, linear_offset, domain, D, C):
    """The compiled leaf loss equals min over α in the loss's domain of
    D·h̃(α) + C·h'(α), built from rieszreg's own potentials, and α* is the
    minimizer. Infeasible (D, C) — the infimum is -∞ — give +∞."""
    leaf_loss, alpha_at = make_leaf_solvers(loss)
    objective = lambda a: float(D * loss.tilde_potential(a) + C * loss.potential_deriv(a))
    lo, hi = domain
    infeasible = (isinstance(loss, KLLoss) and C > 0) or (
        isinstance(loss, BernoulliLoss) and not (-D <= C <= 0))
    if infeasible:
        assert leaf_loss(D, C) == float("inf")
        return
    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    assert np.isclose(leaf_loss(D, C) + linear_offset * C, res.fun, rtol=1e-7, atol=1e-7)
    assert np.isclose(alpha_at(D, C), res.x, atol=1e-5)
