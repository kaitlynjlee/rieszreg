"""Early stopping via held-out augmented loss."""
from __future__ import annotations

from riesztree import ATE, RieszTreeRegressor


def test_early_stopping_reduces_tree_size(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)

    # Without early stopping (deep tree).
    est_no_es = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10,
    ).fit(df)

    # With early stopping (small patience).
    est_es = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10,
        early_stopping_rounds=3,
        validation_fraction=0.2,
    ).fit(df)

    assert est_es.get_n_leaves() < est_no_es.get_n_leaves()


def test_early_stopping_works_with_leafwise(linear_gaussian_ate, covariate_keys):
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        growth_policy="leafwise",
        max_leaf_nodes=200,
        early_stopping_rounds=3,
        validation_fraction=0.2,
    ).fit(df)
    # Should stop well before 200 leaves on this DGP.
    assert est.get_n_leaves() < 200


import pytest


@pytest.mark.parametrize("growth_policy", ["depthwise", "leafwise"])
def test_early_stopping_keeps_best_partial_tree(
    linear_gaussian_ate, covariate_keys, growth_policy, monkeypatch
):
    """After stopping, the tree is rolled back to the partial tree with the
    lowest held-out loss seen during growth, so its held-out loss equals
    that minimum (not the loss after the trailing non-improving splits)."""
    import riesztree.grow as grow

    seen = []
    real = grow._EarlyStopping._mean_loss

    def recording(self):
        seen.append(real(self))
        return seen[-1]

    monkeypatch.setattr(grow._EarlyStopping, "_mean_loss", recording)
    make, _ = linear_gaussian_ate
    df = make(1500, seed=0)
    est = RieszTreeRegressor(
        estimand=ATE(treatment="a", covariates=covariate_keys),
        max_depth=10, max_leaf_nodes=200, growth_policy=growth_policy,
        early_stopping_rounds=3, validation_fraction=0.2,
    ).fit(df)
    assert len(seen) > 4
    assert est.best_score_ == pytest.approx(min(seen), rel=1e-9)


@pytest.mark.parametrize("categorical", [False, True])
def test_incremental_holdout_loss_matches_full_recompute(categorical, monkeypatch):
    """The held-out loss early stopping tracks split by split (re-scoring only
    the validation rows of the split leaf) equals the loss of the whole
    partial tree predicted from scratch, after every split. The tracker sees
    one split at a time whatever the growth policy, so leafwise growth (many
    splits before stopping) covers both policies."""
    import numpy as np
    import pandas as pd
    import riesztree.grow as grow
    from riesztree.fast import flat_tree_from_node, predict_alpha

    roots, checked = [], []
    real_start, real_mean = grow._EarlyStopping.start, grow._EarlyStopping._mean_loss

    def start(self, root):
        roots.append(root)
        real_start(self, root)

    def mean_loss(self):
        incremental = real_mean(self)
        alpha = predict_alpha(flat_tree_from_node(roots[-1]), self.valid.features)
        assert incremental == pytest.approx(self.valid.mean_loss(self.loss, alpha), rel=1e-12)
        checked.append(incremental)
        return incremental

    monkeypatch.setattr(grow._EarlyStopping, "start", start)
    monkeypatch.setattr(grow._EarlyStopping, "_mean_loss", mean_loss)
    rng = np.random.default_rng(1)
    n = 1200
    x = rng.normal(size=(n, 3))
    if categorical:
        x[:, 1] = rng.integers(0, 6, size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-x[:, 0])))
    Z = pd.DataFrame({"a": a, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2]})
    RieszTreeRegressor(
        estimand=ATE(treatment="a"), growth_policy="leafwise", max_depth=None,
        max_leaf_nodes=None, min_samples_leaf=5, early_stopping_rounds=20,
        categorical_features=["x1"] if categorical else None,
    ).fit(Z)
    assert len(checked) > 10
