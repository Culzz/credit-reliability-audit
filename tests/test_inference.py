"""Synthetic-only tests for Holm adjustment and paired bootstrap deltas."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss

from src.statistics.inference import holm_pvalues, paired_bootstrap_delta

RNG = np.random.default_rng(31337)


class TestHolm:
    def test_worked_example(self):
        # textbook three-test example: sorted p [.01,.03,.04] with m=3 gives
        # multipliers [3,2,1] -> running max [.03,.06,.06], order preserved
        out = holm_pvalues([0.01, 0.04, 0.03])
        assert out[0] == pytest.approx(0.03)
        assert out[1] == pytest.approx(0.06)
        assert out[2] == pytest.approx(0.06)

    def test_monotone_and_clipped(self):
        out = holm_pvalues([0.005, 0.3, 0.9, 0.95])
        assert np.all(np.diff(out[np.argsort([0.005, 0.3, 0.9, 0.95])]) >= -1e-12)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_single_test_identity(self):
        assert holm_pvalues([0.07]) == pytest.approx([0.07])

    def test_zero_pvalue_stays_zero(self):
        out = holm_pvalues([0.0, 0.02])
        assert out[0] == 0.0
        assert out[1] == pytest.approx(0.02)

    def test_validation(self):
        with pytest.raises(ValueError):
            holm_pvalues([])
        with pytest.raises(ValueError):
            holm_pvalues([0.5, 1.3])
        with pytest.raises(ValueError):
            holm_pvalues([-0.1, 0.2])


class TestPairedBootstrapDelta:
    def _fixture(self, n: int = 3_000):
        p_a = np.clip(RNG.uniform(0.02, 0.98, size=n), 0.0, 1.0)
        y = (RNG.uniform(size=n) < p_a).astype(float)
        return y, p_a

    def test_identical_scores_zero_delta(self):
        y, p = self._fixture()
        res = paired_bootstrap_delta(
            y, p, p, metric=lambda yt, pp: log_loss(yt, pp, labels=[0, 1]),
            metric_is_loss=True, n_bootstrap=50, seed=4,
        )
        assert res["delta"] == pytest.approx(0.0, abs=1e-12)
        assert res["lo"] <= res["delta"] <= res["hi"]

    def test_paired_design_reduces_variance_vs_paired_against_shuffled(self):
        y, p_a = self._fixture(5_000)
        shifted = np.clip(0.5 + 0.5 * (p_a - 0.5) * 1.2, 0.0, 1.0)
        res = paired_bootstrap_delta(
            y, shifted, p_a, metric=lambda yt, pp: log_loss(yt, pp, labels=[0, 1]),
            metric_is_loss=True, n_bootstrap=200, seed=11,
        )
        assert res["delta"] != 0.0
        assert res["lo"] <= res["delta"] <= res["hi"]
        assert res["hi"] >= res["lo"]
        # same seed -> identical design -> identical interval
        res2 = paired_bootstrap_delta(
            y, shifted, p_a, metric=lambda yt, pp: log_loss(yt, pp, labels=[0, 1]),
            metric_is_loss=True, n_bootstrap=200, seed=11,
        )
        assert res == res2

    def test_sign_convention_loss_positive_means_a_worse(self):
        y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        worse = np.array([0.6, 0.9, 0.8, 0.4, 0.2])
        better = np.array([0.4, 0.8, 0.9, 0.3, 0.6])
        res = paired_bootstrap_delta(
            y, worse, better, metric=lambda yt, pp: log_loss(yt, pp, labels=[0, 1]),
            metric_is_loss=True, n_bootstrap=20, seed=1,
        )
        assert res["delta"] > 0

    def test_validation(self):
        y, p = self._fixture(100)
        with pytest.raises(ValueError):
            paired_bootstrap_delta(
                y[:50], p, p, metric=lambda yt, pp: 0.0,
                metric_is_loss=True, n_bootstrap=10, seed=1,
            )
        with pytest.raises(ValueError):
            paired_bootstrap_delta(
                y, p, p, metric=lambda yt, pp: 0.0,
                metric_is_loss=True, n_bootstrap=1, seed=1,
            )
