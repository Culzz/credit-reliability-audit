"""Synthetic tests for the T5.5 search-space sampler (no data, no results)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.harness import make_estimator
from src.models.spaces import (
    load_spaces,
    sample_all_configs,
    sample_config,
)

FAMILIES = list(load_spaces())


class TestSampler:
    def test_deterministic(self):
        for fam in FAMILIES:
            a = sample_config(fam, 7, 20180101)
            b = sample_config(fam, 7, 20180101)
            assert a == b, fam

    def test_all_families_covered_and_breadth_included(self):
        assert FAMILIES == ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    def test_stochastic_params_strictly_below_one(self):
        # the T5.5 invariant across 200 draws per booster
        for fam in ("xgboost", "lightgbm"):
            for i in range(200):
                params = sample_config(fam, i, 77)
                for key in ("subsample", "colsample_bytree", "colsample_bylevel"):
                    if key in params:
                        assert 0.5 <= params[key] < 1.0, (fam, i, key, params[key])
                assert 0.5 <= params["subsample"] < 0.95
                assert 0.5 <= params["colsample_bytree"] < 0.95

    def test_lr_spans_weak_and_overfitted_ends(self):
        cs = [sample_config("logistic_regression", i, 20261106)["C"] for i in range(300)]
        assert min(cs) < 0.02      # weak end: strong regularisation
        assert max(cs) > 200.0     # over-fitted end: near-unpenalised

    def test_integers_and_choices_in_range(self):
        for i in range(50):
            p = sample_config("xgboost", i, 5)
            assert 2 <= p["max_depth"] <= 10
            assert 1 <= p["min_child_weight"] <= 50
            assert 0.0 <= p["gamma"] <= 5.0
            assert 0.01 <= p["learning_rate"] <= 0.3
        for i in range(50):
            p = sample_config("random_forest", i, 5)
            assert p["max_depth"] in [4, 6, 8, 10, 12, None]
            assert 1 <= p["min_samples_leaf"] <= 100
            assert 0.3 <= p["max_features"] <= 1.0
            assert p["n_estimators"] == 400

    def test_lightgbm_fixed_freq_and_leaves(self):
        for i in range(30):
            p = sample_config("lightgbm", i, 11)
            assert p["subsample_freq"] == 1
            assert 8 <= p["num_leaves"] <= 128
            assert 5 <= p["min_child_samples"] <= 100

    def test_n_configs_batch(self):
        batch = sample_all_configs("lightgbm", 30, 99)
        assert len(batch) == 30
        assert batch[0] == sample_config("lightgbm", 0, 99)
        # breadth: the 30 draws must not collapse to clones
        sig = {tuple(sorted((k, round(v, 6) if isinstance(v, float) else v)
                           for k, v in p.items())) for p in batch}
        assert len(sig) >= 25  # allow rare collisions, not annihilation

    def test_unknown_family_and_validation(self):
        with pytest.raises(ValueError, match="unknown family"):
            sample_config("gradient_boosting", 0, 1)
        with pytest.raises(ValueError):
            sample_all_configs("xgboost", 0, 1)


@pytest.mark.parametrize("family", FAMILIES)
def test_sampled_params_feed_the_estimator_factory(family):
    x = np.random.default_rng(3).normal(size=(600, 5))
    truth = 1.0 / (1.0 + np.exp(-(1.2 * x[:, 0] - 0.7 * x[:, 2])))
    y = (np.random.default_rng(4).uniform(size=600) < truth).astype(float)
    params = sample_config(family, 3, 20261106)
    est = make_estimator(family, params, seed=20261106)
    est.fit(x, y)
    p = est.predict_proba(x)[:, 1]
    assert np.all((p > 0) & (p < 1))
