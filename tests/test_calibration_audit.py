"""Synthetic-only tests for the calibration audit metrics (no data files)."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.audit import (
    brier_decomposition,
    citl_and_slope,
    decision_flips,
    ece,
    mean_within_cohort_spearman,
)

RNG = np.random.default_rng(20261103)


# --------------------------------------------------------------------------
# Brier decomposition
# --------------------------------------------------------------------------
class TestBrierDecomposition:
    def test_perfectly_calibrated_scores_have_low_reliability(self):
        n = 200_000
        p = RNG.uniform(0.0, 1.0, size=n)
        y = (RNG.uniform(size=n) < p).astype(float)
        dec = brier_decomposition(y, p, n_bins=10)
        assert dec.reliability < 0.005
        # the partition closes exactly on the binned forecast
        assert dec.binned_brier == pytest.approx(
            dec.uncertainty - dec.resolution + dec.reliability, abs=1e-10
        )
        # and the binned Brier overstates the raw one by the within-bin spread
        assert dec.binned_brier >= dec.brier

    def test_constant_scores_match_base_rate_gives_low_reliability(self):
        n = 20_000
        base = 0.3
        y = (RNG.uniform(size=n) < base).astype(float)
        p = np.full(n, base)
        dec = brier_decomposition(y, p, n_bins=10)
        assert dec.reliability == pytest.approx(0.0, abs=1e-3)
        assert dec.resolution == pytest.approx(0.0, abs=1e-12)
        assert dec.uncertainty == pytest.approx(y.mean() * (1 - y.mean()))
        assert dec.binned_brier == pytest.approx(
            dec.uncertainty - dec.resolution + dec.reliability, abs=1e-10
        )

    def test_constant_scores_off_base_rate_gives_high_reliability(self):
        n = 20_000
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        p = np.full(n, 0.7)
        dec = brier_decomposition(y, p, n_bins=10)
        # a single bin holds everything, so reliability is (p - y_bar)^2
        assert dec.reliability == pytest.approx((0.7 - y.mean()) ** 2, abs=1e-10)

    def test_partition_identity_holds_on_random_scores(self):
        n = 50_000
        p = RNG.uniform(size=n)
        y = (RNG.uniform(size=n) < 0.25 + 0.5 * p).astype(float)
        dec = brier_decomposition(y, p, n_bins=8)
        assert dec.binned_brier == pytest.approx(
            dec.uncertainty - dec.resolution + dec.reliability, abs=1e-10
        )
        assert dec.n_bins_used == 8

    def test_uncertainty_matches_base_rate_variance(self):
        n = 10_000
        y = (RNG.uniform(size=n) < 0.4).astype(float)
        dec = brier_decomposition(y, np.full(n, 0.4))
        base = y.mean()
        assert dec.uncertainty == pytest.approx(base * (1 - base))

    def test_explicit_edges_are_respected(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        p = np.array([0.05, 0.06, 0.95, 0.94])
        dec = brier_decomposition(y, p, edges=np.array([0.1, 0.9]))
        # the middle bin (0.1, 0.9) receives no data, so only 2 bins are used
        assert dec.n_bins_used == 2

    def test_quantile_edges_option_runs(self):
        n = 5_000
        p = RNG.beta(2.0, 5.0, size=n)
        y = (RNG.uniform(size=n) < p).astype(float)
        dec = brier_decomposition(y, p, quantile_edges=True, n_bins=5)
        assert dec.binned_brier == pytest.approx(
            dec.uncertainty - dec.resolution + dec.reliability, abs=1e-9
        )

    def test_input_validation(self):
        with pytest.raises(ValueError):
            brier_decomposition(np.array([0.5]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            brier_decomposition(np.array([0.5, 2.0]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            brier_decomposition(np.array([0.5, 0.5]), np.array([0.5, 1.2]))


# --------------------------------------------------------------------------
# ECE
# --------------------------------------------------------------------------
class TestEce:
    def test_perfectly_calibrated_simulator_near_zero(self):
        n = 200_000
        p = RNG.uniform(0.0, 1.0, size=n)
        y = (RNG.uniform(size=n) < p).astype(float)
        assert ece(y, p, n_bins=15) < 0.005

    def test_constant_score_matching_base_rate_near_zero(self):
        n = 40_000
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        assert ece(y, np.full(n, 0.3), n_bins=15) < 0.005

    def test_constant_score_off_base_rate_equals_gap(self):
        n = 40_000
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        assert ece(y, np.full(n, 0.7), n_bins=15) == pytest.approx(0.7 - y.mean(), abs=1e-9)

    def test_bin_count_sensitivity_is_bounded_and_reportable(self):
        n = 100_000
        p = RNG.uniform(size=n)
        y = (RNG.uniform(size=n) < p).astype(float)
        e10, e15, e20, e30 = (ece(y, p, n_bins=b) for b in (10, 15, 20, 30))
        assert max(abs(e10), abs(e15), abs(e20), abs(e30)) < 0.01

    def test_validation(self):
        with pytest.raises(ValueError):
            ece(np.array([0.5]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            ece(np.array([0.5, 1.5]), np.array([0.5, 0.5]))


# --------------------------------------------------------------------------
# CITL / slope
# --------------------------------------------------------------------------
class TestCitlAndSlope:
    def test_true_probabilities_recover_slope_one_intercept_zero(self):
        n = 60_000
        p = RNG.uniform(0.02, 0.98, size=n)
        y = (RNG.uniform(size=n) < p).astype(int)
        res = citl_and_slope(y, p)
        assert res.slope == pytest.approx(1.0, abs=0.05)
        assert res.citl == pytest.approx(0.0, abs=0.05)
        assert res.n == n
        assert res.n_events == int(y.sum())

    def test_miscalibrated_scores_get_slope_below_one(self):
        # over-confident score: p_hat = p^2 (too extreme on the logit scale)
        n = 80_000
        p = RNG.uniform(0.05, 0.95, size=n)
        y = (RNG.uniform(size=n) < p).astype(int)
        p_hat = p**2
        res = citl_and_slope(y, p_hat)
        assert res.slope < 0.8

    def test_shifted_scores_get_nonzero_citl(self):
        # offset forecast: p_hat = sigmoid(logit(p) + 1) is over-confident;
        # the calibration refit must UNDO the offset, so the required
        # intercept is approximately -1 (citl < 0)
        n = 80_000
        p = RNG.uniform(0.05, 0.95, size=n)
        y = (RNG.uniform(size=n) < p).astype(int)
        logits = np.log(p / (1 - p)) + 1.0
        p_hat = 1.0 / (1.0 + np.exp(-logits))
        res = citl_and_slope(y, p_hat)
        assert res.slope == pytest.approx(1.0, abs=0.05)
        assert res.citl == pytest.approx(-1.0, abs=0.05)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            citl_and_slope(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))


# --------------------------------------------------------------------------
# Decision flips
# --------------------------------------------------------------------------
class TestDecisionFlips:
    def test_counts_and_directions(self):
        before = np.array([0.9, 0.4, 0.6, 0.2])
        after = np.array([0.4, 0.9, 0.6, 0.1])
        rep = decision_flips(before, after, threshold=0.5)
        assert rep.n == 4
        assert rep.n_flips == 2
        assert rep.flip_rate == pytest.approx(0.5)
        assert rep.flips_to_decline == 1  # 0.9 -> 0.4
        assert rep.flips_to_approve == 1  # 0.4 -> 0.9

    def test_identical_scores_never_flip(self):
        p = RNG.uniform(size=1_000)
        rep = decision_flips(p, p, threshold=0.5)
        assert rep.n_flips == 0 and rep.flip_rate == 0.0

    def test_threshold_on_the_boundary_belongs_to_approve(self):
        rep = decision_flips(
            np.array([0.5]), np.array([0.49]), threshold=0.5
        )
        assert rep.flips_to_decline == 1

    def test_input_validation(self):
        with pytest.raises(ValueError):
            decision_flips(np.array([0.5, 0.5]), np.array([0.5]))
        with pytest.raises(ValueError):
            decision_flips(np.array([0.5]), np.array([0.5]), threshold=1.0)


# --------------------------------------------------------------------------
# Within-cohort Spearman aggregator
# --------------------------------------------------------------------------
def test_mean_within_cohort_spearman_ignores_nan():
    vals = [0.9, float("nan"), 0.7, 0.8]
    assert mean_within_cohort_spearman(vals) == pytest.approx(0.8)
    assert np.isnan(mean_within_cohort_spearman([float("nan")]))
