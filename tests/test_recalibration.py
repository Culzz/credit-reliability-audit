"""Synthetic-only tests for the recalibration arms (PLAN.md T7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.recalibration import KINDS, fit_recalibrator

RNG = np.random.default_rng(77)


def _simulated_scores(n: int = 20_000, overconfidence: float = 0.0):
    """Calibration scores from a known true probability model.

    With overconfidence=0 the scores ARE the true probabilities; with
    overconfidence>0 they are stretched on the logit scale (too extreme).
    """
    p_true = RNG.uniform(0.02, 0.98, size=n)
    y = (RNG.uniform(size=n) < p_true).astype(float)
    logits = np.log(p_true / (1 - p_true)) + overconfidence * np.log(p_true / (1 - p_true))
    p_hat = 1.0 / (1.0 + np.exp(-logits))
    return y, p_hat, p_true


class TestFitRecalibrator:
    def test_kinds_whitelist(self):
        assert KINDS == ("none", "intercept", "platt", "isotonic")

    def test_none_arm_is_identity(self):
        _, p, _ = _simulated_scores(1_000)
        rec = fit_recalibrator("none", np.array([0.0, 1.0]), np.array([0.3, 0.7]))
        out = rec.apply(p)
        assert np.array_equal(out, p)
        assert rec.n_calibrated == 2 and rec.n_events == 1

    def test_well_calibrated_scores_stay_put(self):
        y, p_hat, _ = _simulated_scores(30_000, overconfidence=0.0)
        for kind in ("intercept", "platt"):
            rec = fit_recalibrator(kind, y, p_hat)
            out = rec.apply(p_hat)
            # already calibrated: fitted offsets near zero, mapping near identity
            assert np.mean(np.abs(out - p_hat)) < 0.02, kind

    def test_calibration_fixes_overconfidence(self):
        y, p_hat, p_true = _simulated_scores(60_000, overconfidence=0.5)
        base_brier = np.mean((p_hat - y) ** 2)
        # the true-probability Brier for this simulator: E[p_true(1-p_true)]
        oracle = float(np.mean(p_true * (1 - p_true)))
        for kind in ("platt", "isotonic"):
            rec = fit_recalibrator(kind, y, p_hat)
            out = rec.apply(p_hat)
            fixed = np.mean((out - y) ** 2)
            assert fixed < base_brier, (kind, fixed, base_brier)
            # and it approaches the oracle within sampling noise
            assert fixed < oracle + 0.01, (kind, fixed, oracle)
        # intercept-only cannot fix a pure slope (symmetric) stretch, and it
        # must not make things materially worse either
        rec = fit_recalibrator("intercept", y, p_hat)
        assert np.mean((rec.apply(p_hat) - y) ** 2) < 1.05 * base_brier

    def test_platt_recovers_the_true_mapping(self):
        y, p_hat, _p_true = _simulated_scores(60_000, overconfidence=0.5)
        rec = fit_recalibrator("platt", y, p_hat)
        # forecast logits are 1.5x the true logits -> the fitted slope must
        # SHRINK to 1/1.5 to undo the stretch (slope < 1 = overconfident)
        assert rec.params["slope"] == pytest.approx(1.0 / 1.5, abs=0.05)
        assert rec.params["intercept"] == pytest.approx(0.0, abs=0.05)

    def test_intercept_only_moves_offset_not_slope(self):
        # mean-shift case: every score is too high -> fitted offset negative
        y, p_hat, _p_true = _simulated_scores(30_000)
        shifted = 1.0 / (1.0 + np.exp(-((np.log(p_hat / (1 - p_hat))) - 1.0)))
        rec = fit_recalibrator("intercept", y, shifted)
        assert rec.params["offset"] == pytest.approx(1.0, abs=0.05)
        out = rec.apply(shifted)
        assert np.mean(np.abs(out - p_hat)) < 0.01

    def test_isotonic_is_monotone_and_in_range(self):
        y, p_hat, _ = _simulated_scores(20_000, overconfidence=0.3)
        rec = fit_recalibrator("isotonic", y, p_hat)
        grid = np.linspace(0.0, 1.0, 101)
        out = rec.apply(grid)
        assert np.all(np.diff(out) >= -1e-12)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_step_data_recovered_by_both_arms(self):
        # step-true probabilities with constant per-group forecasts: both
        # arms recover the group base rates (isotonic and Platt saturate a
        # single binary logit covariate identically)
        n = 20_000
        x = RNG.uniform(size=n)
        p_true = np.where(x < 0.5, 0.25, 0.75)
        y = (RNG.uniform(size=n) < p_true).astype(float)
        p_hat = np.where(x < 0.5, 0.1, 0.9)
        for kind in ("platt", "isotonic"):
            out = fit_recalibrator(kind, y, p_hat).apply(p_hat)
            lo_rate = out[x < 0.5].mean()
            hi_rate = out[x >= 0.5].mean()
            assert lo_rate == pytest.approx(0.25, abs=0.02), (kind, lo_rate)
            assert hi_rate == pytest.approx(0.75, abs=0.02), (kind, hi_rate)

    def test_holds_on_sample_sizes_like_the_feasibility_cells(self):
        # smallest feasible cohort cells are the 200/200 rule area
        n = 500
        y, p_hat, _ = _simulated_scores(n, overconfidence=0.4)
        for kind in ("platt", "isotonic"):
            rec = fit_recalibrator(kind, y, p_hat)
            out = rec.apply(p_hat)
            assert np.mean((out - y) ** 2) < np.mean((p_hat - y) ** 2), kind

    def test_single_class_raises_and_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="both outcome classes"):
            fit_recalibrator("platt", np.array([1.0, 1.0, 1.0]), np.array([0.9, 0.8, 0.7]))
        with pytest.raises(ValueError, match="unknown recalibrator kind"):
            fit_recalibrator("unknown", np.array([0.0, 1.0]), np.array([0.3, 0.7]))
        with pytest.raises(ValueError):
            fit_recalibrator("none", np.array([0.5]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            fit_recalibrator("none", np.array([0.5, 1.5]), np.array([0.5, 0.5]))
        rec = fit_recalibrator("intercept", np.array([0.0, 1.0]), np.array([0.3, 0.7]))
        with pytest.raises(ValueError):
            rec.apply(np.array([0.5, 1.5]))
        with pytest.raises(ValueError):
            rec.apply(np.array([]))
