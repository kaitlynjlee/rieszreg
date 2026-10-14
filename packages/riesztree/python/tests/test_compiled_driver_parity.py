"""Parity tests for the Cython whole-tree drivers.

With no categorical features, no per-split feature subsampling and no early
stopping, a depthwise tree is grown by a Cython worklist driver:
``_grow_c`` for ``splitter='hist'`` (parent-minus-sibling histograms) and
``_grow_exact_c`` for ``splitter='exact'`` (columns presorted once, sorted
orders partitioned at each split). Other configurations use the Python
grower, which calls the Cython per-node kernels. Both must grow the same
tree, down to every stored node field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riesztree import (
    ATE,
    BernoulliLoss,
    BoundedSquaredLoss,
    KLLoss,
    RieszTreeRegressor,
    SquaredLoss,
    TSM,
)
from riesztree.tree import to_dict
from rieszreg.testing.dgps import scaled_tsm


def _make_df(n=600, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, p))
    logit = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    pi = 1.0 / (1.0 + np.exp(-logit))
    a = (rng.uniform(0, 1, size=n) < pi).astype(float)
    cols = {f"x{j}": X[:, j] for j in range(p)}
    cols["a"] = a
    return pd.DataFrame(cols)


def _ate(p):
    return ATE(treatment="a", covariates=tuple(f"x{j}" for j in range(p)))


def _assert_same_nodes(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        if key in ("left", "right"):
            _assert_same_nodes(a[key], b[key])
        elif isinstance(a[key], float):
            assert np.isclose(a[key], b[key], rtol=1e-10, atol=1e-10), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("splitter", ["exact", "hist"])
@pytest.mark.parametrize(
    "loss_factory, estimand_factory",
    [
        (lambda: SquaredLoss(), lambda p: _ate(p)),
        (
            lambda: KLLoss(),
            lambda p: TSM(treatment="a", covariates=tuple(f"x{j}" for j in range(p)), level=1.0),
        ),
        (lambda: BernoulliLoss(), lambda p: scaled_tsm(covariates=tuple(f"x{j}" for j in range(p)))),
        (lambda: BoundedSquaredLoss(lo=-3.0, hi=3.0), lambda p: _ate(p)),
    ],
    ids=["squared-ate", "kl-tsm", "bernoulli-scaled-tsm", "bounded-squared-ate"],
)
def test_compiled_driver_matches_python_grower(splitter, loss_factory, estimand_factory, monkeypatch):
    """The Cython driver and the Python grower pick identical splits and store
    the same per-node fields (counts, sums, gains, leaf losses), so
    predictions, diagnostics, pruning and saved models don't depend on which
    path grew the tree."""
    import riesztree.grow as grow

    df = _make_df(n=800, p=5)
    kwargs = dict(max_depth=5, splitter=splitter, random_state=0)
    driver = RieszTreeRegressor(estimand=estimand_factory(5), loss=loss_factory(), **kwargs).fit(df)
    monkeypatch.setattr(grow._Grower, "compiled_eligible", property(lambda self: False))
    python = RieszTreeRegressor(estimand=estimand_factory(5), loss=loss_factory(), **kwargs).fit(df)
    assert driver.get_n_leaves() >= 8  # a stump would make the comparison vacuous
    _assert_same_nodes(to_dict(driver.predictor_.tree), to_dict(python.predictor_.tree))
