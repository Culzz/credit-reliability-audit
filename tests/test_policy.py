"""Synthetic-only tests for the threshold decision policies (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from src.decisions.policy import (
    APPROVE,
    DECLINE,
    REFER,
    breakeven_threshold,
    fixed_rate_policy,
    stylised_utility,
    threshold_policy,
)

RNG = np.random.default_rng(88)


class TestBreakeven:
    def test_formula_over_the_configured_ratios(self):
        for ratio, expected in [(3.0, 0.25), (5.0, 1.0 / 6.0), (9.0, 0.10)]:
            assert breakeven_threshold(ratio) == pytest.approx(expected)

    def test_rejects_bad_ratios(self):
        for bad in (0.0, -2.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                breakeven_threshold(bad)


class TestThresholdPolicy:
    def test_three_band_codes(self):
        p = np.array([0.10, 0.15, 0.16, 0.19, 0.25])
        codes = threshold_policy(p, t_low=0.15, t_high=0.20)
        assert codes.tolist() == [
            APPROVE,
            REFER,
            REFER,
            REFER,
            DECLINE,
        ]

    def test_band_edges_are_refer(self):
        # policy wording: approve if p < t_low; refer if t_low <= p < t_high
        p = np.array([0.149, 0.150, 0.1500001, 0.199, 0.200])
        codes = threshold_policy(p, t_low=0.15, t_high=0.20)
        assert codes.tolist() == [APPROVE, REFER, REFER, REFER, DECLINE]

    def test_validation(self):
        with pytest.raises(ValueError):
            threshold_policy(np.array([0.5]), t_low=0.5, t_high=0.2)
        with pytest.raises(ValueError):
            threshold_policy(np.array([0.5, 1.1]), t_low=0.1, t_high=0.2)


class TestFixedRatePolicy:
    def test_approved_share_matches_request(self):
        p = np.sort(RNG.uniform(size=10_000))
        rate = 0.3
        codes, threshold = fixed_rate_policy(p, approval_rate=rate)
        share = float((codes == APPROVE).mean())
        # ties belong to NOT-approved, so the realised share is <= requested
        assert share == pytest.approx(rate, abs=0.01)
        assert share <= rate + 1e-12
        assert float(threshold) == pytest.approx(float(np.quantile(p, rate)))

    def test_strict_less_than_keeps_ties_out(self):
        p = np.array([0.1, 0.1, 0.2, 0.3])
        codes, threshold = fixed_rate_policy(p, approval_rate=0.5)
        assert all(codes[p >= threshold] == DECLINE)
        # only the strictly-below entries are approved
        assert all(codes[p < threshold] == APPROVE)

    def test_validation(self):
        with pytest.raises(ValueError):
            fixed_rate_policy(np.array([0.5]), approval_rate=0.0)
        with pytest.raises(ValueError):
            fixed_rate_policy(np.array([0.5, 1.2]), approval_rate=0.3)


class TestStylisedUtility:
    def test_hand_computed_case(self):
        y = np.array([0.0, 1.0, 0.0])
        codes = np.array([APPROVE, APPROVE, DECLINE])
        # funded: one good (+1) and one default (-5), decline ignored
        assert stylised_utility(y, codes, loss_gain_ratio=5.0) == pytest.approx(-4.0)

    def test_refer_and_decline_contribute_zero(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        codes = np.array([REFER, REFER, DECLINE, DECLINE])
        assert stylised_utility(y, codes, loss_gain_ratio=5.0) == 0.0

    def test_scale_in_gain(self):
        y = np.array([0.0, 0.0])
        codes = np.array([APPROVE, APPROVE])
        assert stylised_utility(y, codes, loss_gain_ratio=9.0, gain=10.0) == pytest.approx(20.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            stylised_utility(np.array([0.5]), np.array([APPROVE]), loss_gain_ratio=5.0)
        with pytest.raises(ValueError):
            stylised_utility(np.array([0.0, 1.0]), np.array([APPROVE]), loss_gain_ratio=5.0)
