"""Synthetic-only tests for the three-way decision-flip machinery (T9.3/T9.4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.decisions.flips import (
    APPROVE,
    DECLINE,
    REFER,
    distance_to_nearest_threshold,
    flip_delta_utility,
    flip_flows,
    flip_rate,
    matched_null_total_delta_utility,
    probability_changes,
    transition_matrix,
)

RNG = np.random.default_rng(99)


class TestTransitionMatrix:
    def test_counts_and_diagonal(self):
        before = np.array([APPROVE, APPROVE, REFER, DECLINE])
        after = np.array([APPROVE, REFER, REFER, APPROVE])
        m = transition_matrix(before, after)
        assert m[0, 0] == 1 and m[0, 1] == 1  # stable, approve->refer
        assert m[1, 1] == 1                   # refer stays
        assert m[2, 0] == 1                   # decline->approve
        assert m.sum() == 4

    def test_flip_rate_matches_off_diagonal(self):
        before = np.array([0, 0, 1, 2, 2])
        after = np.array([0, 1, 1, 0, 0])
        assert flip_rate(before, after) == pytest.approx(3 / 5)

    def test_identical_codes_never_flip(self):
        c = np.array([0, 1, 2, 0, 1])
        assert flip_rate(c, c) == 0.0
        m = transition_matrix(c, c)
        assert (m == np.diag(np.diag(m))).all()

    def test_flow_asymmetry(self):
        before = np.array([0, 0, 2, 1])
        after = np.array([1, 2, 0, 1])
        flows = flip_flows(before, after)
        assert flows["n_flips"] == 3
        assert flows["toward_decline"] == 2   # 0->1, 0->2
        assert flows["toward_approve"] == 1   # 2->0
        assert flows["asymmetric_flips"] == 1

    def test_validation(self):
        with pytest.raises(ValueError):
            transition_matrix(np.array([3, 0]), np.array([0, 0]))
        with pytest.raises(ValueError):
            transition_matrix(np.array([0, 1]), np.array([0]))


class TestProbabilityChanges:
    def test_flipped_pairs_only(self):
        before = np.array([0.2, 0.8])
        after = np.array([0.3, 0.4])
        codes_before = np.array([APPROVE, DECLINE])
        codes_after = np.array([APPROVE, REFER])  # only the second flips
        d = probability_changes(before, after, codes_before, codes_after)
        assert d.tolist() == pytest.approx([0.4 - 0.8])

    def test_validation(self):
        with pytest.raises(ValueError):
            probability_changes(
                np.array([0.2, 1.2]),
                np.array([0.2, 0.3]),
                np.array([0, 0]),
                np.array([0, 0]),
            )


class TestDistanceToNearestThreshold:
    def test_band_specific_distances(self):
        t_low, t_high = 0.15, 0.25
        p = np.array([0.10, 0.20, 0.30])
        c = np.array([APPROVE, REFER, DECLINE])
        d = distance_to_nearest_threshold(p, c, t_low=t_low, t_high=t_high)
        assert d.tolist() == pytest.approx([0.05, 0.05, 0.05])

    def test_validation(self):
        with pytest.raises(ValueError):
            distance_to_nearest_threshold(np.array([0.5]), np.array([0]), t_low=0.3, t_high=0.2)


class TestFlipDeltaUtility:
    def test_hand_computed(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        before = np.array([APPROVE, DECLINE, REFER, APPROVE])
        after = np.array([APPROVE, APPROVE, APPROVE, REFER])
        res = flip_delta_utility(y, before, after, loss_gain_ratio=5.0)
        # loan 0: +1 -> +1 (0); loan 1: 0 -> -5 (-5); loan 2: 0 -> +1 (+1);
        # loan 3: -5 -> 0 (+5); total +1
        assert res["total"] == pytest.approx(1.0)
        assert res["per_loan"].tolist() == pytest.approx([0.0, -5.0, 1.0, 5.0])
        assert res["flipping_loans"] == 3
        assert res["util_flipping_loans"] == 3

    def test_no_change_no_delta(self):
        c = np.array([APPROVE, DECLINE, REFER])
        y = np.array([0.0, 0.0, 0.0])
        res = flip_delta_utility(y, c, c, loss_gain_ratio=5.0)
        assert res["total"] == 0.0 and res["per_loan"].sum() == 0.0


class TestMatchedNull:
    def _fixture(self, n: int = 400):
        p = np.clip(RNG.normal(0.5, 0.12, size=n), 0.0, 1.0)
        y = (RNG.uniform(size=n) < p).astype(float)
        return p, y

    def test_null_distributes_finite_totals(self):
        p, y = self._fixture()
        before = np.where(p < 0.45, APPROVE, DECLINE)
        null = matched_null_total_delta_utility(
            y, p, before, t_low=0.45, t_high=0.55, draws=50, seed=11, loss_gain_ratio=5.0
        )
        assert len(null) == 50
        assert np.isfinite(null).all()

    def test_noop_after_arm_exceeds_the_broken_pairing_null(self):
        p, y = self._fixture()
        before = np.where(p < 0.45, APPROVE, DECLINE)
        obs = flip_delta_utility(y, before, before, loss_gain_ratio=5.0)["total"]
        assert obs == pytest.approx(0.0)
        null = matched_null_total_delta_utility(
            y, p, before, t_low=0.45, t_high=0.55, draws=100, seed=7, loss_gain_ratio=5.0
        )
        # the null decouples scores from outcomes, so it loses information;
        # an unchanged (informative) arm must sit at or above every draw
        assert obs >= null.max()

    def test_permuted_after_arm_rests_inside_the_null(self):
        p, y = self._fixture(600)
        before = np.where(p < 0.45, APPROVE, DECLINE)
        rng = np.random.default_rng(101)
        codes_after = np.where(rng.permutation(p) < 0.45, APPROVE, DECLINE)
        obs = flip_delta_utility(y, before, codes_after, loss_gain_ratio=5.0)["total"]
        null = matched_null_total_delta_utility(
            y, p, before, t_low=0.45, t_high=0.55, draws=200, seed=13, loss_gain_ratio=5.0
        )
        # a literally broken-pairing arm is just another null draw
        assert null.min() <= obs <= null.max()

    def test_seed_reproduces_null_exactly(self):
        p, y = self._fixture(300)
        before = np.where(p < 0.42, APPROVE, DECLINE)
        kwargs = {"t_low": 0.42, "t_high": 0.58, "draws": 40, "seed": 5, "loss_gain_ratio": 9.0}
        n1 = matched_null_total_delta_utility(y, p, before, **kwargs)
        n2 = matched_null_total_delta_utility(y, p, before, **kwargs)
        assert np.array_equal(n1, n2)

    def test_null_totals_of_plausible_magnitude(self):
        p, y = self._fixture(600)
        before = np.where(p < 0.45, APPROVE, DECLINE)
        null = matched_null_total_delta_utility(
            y, p, before, t_low=0.45, t_high=0.55, draws=100, seed=3, loss_gain_ratio=5.0
        )
        bound = 5.0 * y.sum() + len(y)
        assert null.max() < bound
        assert null.min() > -bound

    def test_validation(self):
        p, y = self._fixture(50)
        before = np.where(p < 0.45, APPROVE, DECLINE)
        with pytest.raises(ValueError):
            matched_null_total_delta_utility(y, p, before, t_low=0.5, t_high=0.2, draws=10, seed=1, loss_gain_ratio=5.0)
        with pytest.raises(ValueError):
            matched_null_total_delta_utility(y, p, before, t_low=0.4, t_high=0.6, draws=1, seed=1, loss_gain_ratio=5.0)
