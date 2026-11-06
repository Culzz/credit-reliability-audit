"""Synthetic-only tests for grouped importance (PLAN.md T8.5)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src.explain.groups import (
    group_attributions,
    grouped_permutation_importance,
)

RNG = np.random.default_rng(4242)


def _dataset(n: int = 2_000):
    x = RNG.normal(size=(n, 6))
    truth = 1.0 / (1.0 + np.exp(1.6 * x[:, 0] - 0.1 * x[:, 1] + 1.2 * x[:, 4]))
    y = (RNG.uniform(size=n) < truth).astype(float)
    return x, y


GROUPS = {
    "repayment": [0, 1],       # carries the informative signal (x0)
    "capacity": [2, 3],        # inert
    "history": [4, 5],         # carries signal (x4)
}


class TestGroupAttributions:
    def test_group_sum_abs_vs_col_abs(self):
        V = np.array([[1.0, -2.0, 3.0, -4.0], [0.5, 1.0, -1.0, 2.0]])
        g = {"a": [0, 1], "b": [2, 3]}
        out = group_attributions(V, g)
        # |row sums|: a: |1-2|=1, |0.5+1|=1.5 -> mean 1.25; b: |3-4|=1, |-1+2|=1 -> 1
        assert out["a"] == pytest.approx(1.25)
        assert out["b"] == pytest.approx(1.0)

    def test_signal_groups_outweigh_inert_groups(self):
        n = 20_000
        V = np.zeros((n, 6))
        V[:, 0] += RNG.normal(0.15, 0.02, size=n)
        V[:, 4] += RNG.normal(0.12, 0.02, size=n)
        # remaining columns keep a tiny noise level
        V += RNG.normal(0.0, 0.001, size=(n, 6))
        imp = group_attributions(V, GROUPS)
        assert imp["repayment"] > imp["capacity"] * 50
        assert imp["history"] > imp["capacity"] * 50

    def test_partition_discipline(self):
        V = np.zeros((3, 4))
        with pytest.raises(ValueError, match="overlap"):
            group_attributions(V, {"a": [0, 1], "b": [1, 2]})
        with pytest.raises(ValueError, match="missing"):
            group_attributions(V, {"a": [0, 1]})
        with pytest.raises(ValueError):
            group_attributions(V, {})
        with pytest.raises(ValueError, match="outside"):
            group_attributions(V, {"a": [0, 9]})


class TestGroupedPermutation:
    @pytest.fixture(scope="class")
    def fitted_pipeline(self):
        x, y = _dataset()
        model = LogisticRegression(C=1.0, max_iter=2000).fit(x, y)
        return lambda Xb: model.predict_proba(Xb)[:, 1], x, y

    def test_signal_groups_matter_inert_group_is_near_zero(self, fitted_pipeline):
        predict, x, y = fitted_pipeline
        out = grouped_permutation_importance(
            predict, y, x, GROUPS,
            metric=lambda yt, p: log_loss(yt, p, labels=[0, 1]),
            metric_is_loss=True, repeats=10, seed=12,
        )
        assert out["repayment"]["mean"] > 0.10          # destroying x0 hurts the loss
        assert out["history"]["mean"] > 0.05            # destroying x4 hurts
        assert abs(out["capacity"]["mean"]) < 0.01      # inert block: ~no change
        for stats in out.values():
            assert set(stats) == {"mean", "sd"}

    def test_sign_convention_loss_vs_auc(self, fitted_pipeline):
        predict, x, y = fitted_pipeline
        loss_imp = grouped_permutation_importance(
            predict, y, x, GROUPS,
            metric=lambda yt, p: log_loss(yt, p, labels=[0, 1]),
            metric_is_loss=True, repeats=8, seed=3,
        )
        auc_imp = grouped_permutation_importance(
            predict, y, x, GROUPS,
            metric=roc_auc_score, metric_is_loss=False,
            repeats=8, seed=3,
        )
        assert loss_imp["repayment"]["mean"] > 0        # loss up when destroyed
        assert auc_imp["repayment"]["mean"] > 0

    # both conventions normalised positive = informative (the module's
    # documented sign rule); the inert group stays near zero on the AUC axis
        assert abs(auc_imp["capacity"]["mean"]) < 0.02

    def test_same_seed_same_numbers(self, fitted_pipeline):
        predict, x, y = fitted_pipeline
        kwargs = {
            "metric": lambda yt, p: log_loss(yt, p, labels=[0, 1]),
            "metric_is_loss": True, "repeats": 6,
        }
        a = grouped_permutation_importance(predict, y, x, GROUPS, seed=21, **kwargs)
        b = grouped_permutation_importance(predict, y, x, GROUPS, seed=21, **kwargs)
        assert a == b

    def test_validation(self, fitted_pipeline):
        predict, x, y = fitted_pipeline
        with pytest.raises(ValueError, match="repeats"):
            grouped_permutation_importance(
                predict, y, x, GROUPS,
                metric=lambda yt, p: log_loss(yt, p, labels=[0, 1]),
                metric_is_loss=True, repeats=1, seed=1,
            )
        with pytest.raises(ValueError, match="overlap"):
            grouped_permutation_importance(
                predict, y, x, {"a": [0, 1], "b": [1, 2], "c": [2, 3], "d": [4, 5]},
                metric=lambda yt, p: log_loss(yt, p, labels=[0, 1]),
                metric_is_loss=True, repeats=4, seed=1,
            )
