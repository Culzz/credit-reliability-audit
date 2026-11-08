"""Synthetic-only tests for the calibration audit metrics (no data files)."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.audit import (
    ae_ratio,
    brier_decomposition,
    citl_and_slope,
    decision_flips,
    ece,
    mean_within_cohort_spearman,
    reliability_bins,
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


class TestAeRatio:
    def test_calibrated_simulator_near_one(self):
        n = 200_000
        p = RNG.uniform(0.05, 0.95, size=n)
        y = (RNG.uniform(size=n) < p).astype(float)
        assert ae_ratio(y, p) == pytest.approx(1.0, abs=0.01)

    def test_underprediction_above_one(self):
        # model predicts p = 0.3 but realised rate is 0.3 for scores 0.7:
        # mean(y) = 0.3/0.7 of mean(p) -> A/E = 0.3/0.7
        n = 50_000
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        assert ae_ratio(y, np.full(n, 0.7)) == pytest.approx(0.3 / 0.7, rel=0.02)

    def test_overprediction_below_one(self):
        n = 50_000
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        assert ae_ratio(y, np.full(n, 0.1)) == pytest.approx(3.0, rel=0.02)

    def test_undefined_mean_prediction(self):
        with pytest.raises(ValueError, match="non-positive mean"):
            ae_ratio(np.array([0, 1]), np.array([0.0, 0.0]))


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

    def test_offset_intercept_differs_from_free_fit_intercept(self):
        # CONTEXT 7 defines CITL via an OFFSET model (slope pinned to 1)
        # and the slope via a separate free fit: the two intercepts differ
        # whenever the slope is not 1. With a skewed score distribution
        # under a symmetric logit stretch, the free-fit (Platt) intercept
        # stays at ~0 (slope ~2/3) while the offset-model CITL must lift
        # the mean prediction up to the observed base rate.
        from src.calibration.recalibration import fit_recalibrator

        n = 200_000
        u = RNG.uniform(-4.0, 2.0, size=n)      # skewed: mean(y) > mean(expit(1.5u))
        y = (RNG.uniform(size=n) < 1.0 / (1.0 + np.exp(-u))).astype(int)
        p_hat = 1.0 / (1.0 + np.exp(-(1.5 * u)))
        res = citl_and_slope(y, p_hat)
        platt = fit_recalibrator("platt", y, p_hat).params
        assert abs(platt["intercept"]) < 0.02   # free fit: intercept ~0, slope absorbs
        assert platt["slope"] == pytest.approx(2 / 3, abs=0.05)
        assert res.citl > 0.02                  # offset model: positive correction
        assert (res.citl - abs(platt["intercept"])) > 0.01

    def test_citl_matches_intercept_arm_offset(self):
        # single source of truth: the audit's CITL must equal the fitted
        # offset of the "intercept" recalibration arm on the same data
        from src.calibration.recalibration import fit_offset_intercept

        n = 40_000
        p = RNG.uniform(0.01, 0.99, size=n)
        y = (RNG.uniform(size=n) < p ** 1.5).astype(int)
        res = citl_and_slope(y, p)
        assert res.citl == pytest.approx(fit_offset_intercept(y, p), abs=1e-9)

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


# --------------------------------------------------------------------------
# Reliability-diagram bin segments
# --------------------------------------------------------------------------
def test_reliability_bins_cover_and_match_binning():
    # dissipate deterministic scores across [0,1]; quantile edges make
    # equal-mass bins so shares must partition the cohort
    n = 1500
    p = np.linspace(0.0, 1.0, n) * 0.9999
    rng = np.random.default_rng(20261108)
    y = (rng.random(n) < p).astype(float)
    segs = reliability_bins(y, p, n_bins=15, quantile_edges=True)
    assert sum(s["n"] for s in segs) == n
    # shares are rounded to 6 decimals, so the identity holds only to
    # rounding: <= 5e-7 x one per non-empty bin
    assert abs(sum(s["share"] for s in segs) - 1.0) <= 5e-7 * len(segs)
    # equal-mass bins: every segment carries ~n/15 rows
    assert all(abs(s["n"] - n / 15) <= 1 for s in segs)
    # segments are ordered and p_bar increases with bin index
    bars = [s["p_bar"] for s in segs]
    assert bars == sorted(bars)


def test_reliability_bins_match_primary_ece_binning():
    # same machinery must reproduce the same |p_bar - y_bar| weighting
    # that ece_b15 reports: sum(share * |p_bar - y_bar|) == ece
    n = 900
    rng = np.random.default_rng(19)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)
    segs = reliability_bins(y, p, n_bins=15, quantile_edges=True)
    manual = sum(s["share"] * abs(s["p_bar"] - s["y_bar"]) for s in segs)
    assert manual == pytest.approx(ece(y, p, n_bins=15, quantile_edges=True), abs=5e-6)


def test_quantile_edge_path_delivers_exactly_n_bins():
    # regression pin: the quantile path once delivered n_bins+2 masks
    # (off-by-two in the inner-cut count); B must mean B, matching the
    # fixed-edge path's mask count for the same n_bins
    n = 1600
    rng = np.random.default_rng(7)
    p = rng.random(n)  # continuous scores, no duplicate quantile edges
    y = (rng.random(n) < 0.2).astype(float)
    for b in (10, 15, 20, 30):
        segs = reliability_bins(y, p, n_bins=b, quantile_edges=True)
        assert len(segs) == b, f"quantile binning delivered {len(segs)} bins for B={b}"
        edges = np.linspace(0.0, 1.0, b + 1)
        fixed = reliability_bins(y, p, n_bins=b, quantile_edges=False, edges=edges)
        assert len(fixed) == b
