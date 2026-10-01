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
