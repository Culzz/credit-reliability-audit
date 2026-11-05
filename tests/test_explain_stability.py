"""Synthetic-only tests for the explanation-stability metrics (T8.3/T8.7)."""

from __future__ import annotations

import numpy as np
import pytest

from src.explain.stability import (
    attribution_distribution_distance,
    cosine_importance,
    global_importance,
    jaccard_at_k,
    rank_agreement,
    rank_vector,
)

RNG = np.random.default_rng(12345)


def _attributions(
    n: int = 5_000,
    f: int = 8,
    loud_features: tuple[int, ...] = (0, 1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sparse-ish attribution matrix: only `loud_features` carry signal."""
    rng = rng or RNG
    V = rng.normal(0.0, 0.001, size=(n, f))
    for j in loud_features:
        V[:, j] += rng.normal(0.2, 0.05, size=n)
    return V


class TestGlobalImportance:
    def test_mean_absolute_per_feature(self):
        V = np.array([[1.0, -2.0], [-1.0, 2.0]])
        assert global_importance(V).tolist() == pytest.approx([1.0, 2.0])

    def test_loud_features_dominate(self):
        V = _attributions()
        imp = global_importance(V)
        assert np.argsort(imp)[-2:].tolist() == [0, 1]


class TestRankVector:
    def test_ranks_descending(self):
        V = _attributions(2_000, 5, loud_features=(2,))
        r = rank_vector(V)
        assert r.min() == 1.0 and r.max() == 5.0
        assert r[2] == 1.0  # the one loud feature is most important
        assert sorted(r) == [1, 2, 3, 4, 5]


class TestJaccardAtK:
    def test_same_distribution_identical_topk(self):
        a = _attributions(loud_features=(0, 1))
        b = _attributions(loud_features=(0, 1))
        assert jaccard_at_k(a, b, k=2) == pytest.approx(1.0)

    def test_disjoint_signal_features_low_at_significant_k(self):
        a = _attributions(loud_features=(6, 7))
        b = _attributions(loud_features=(0, 1))
        # disjoint pure-signal features differ in their topks... except the
        # two loud features still dominate other noise columns; with k=2 the
        # top sets must mismatch by construction (different loud columns)
        assert jaccard_at_k(a, b, k=1) == pytest.approx(0.0)
        assert jaccard_at_k(a, b, k=2) <= 0.5

    def test_all_zero_importances_agree(self):
        a = np.zeros((100, 4))
        assert jaccard_at_k(a, a, k=2) == 1.0

    def test_k_flags(self):
        with pytest.raises(ValueError):
            jaccard_at_k(np.zeros((3, 2)), np.zeros((3, 2)), k=0)


class TestRankAgreement:
    def test_identical_distributions_full_agreement(self):
        a = _attributions()
        agg = rank_agreement(a, a)
        assert agg["spearman"] == pytest.approx(1.0)
        assert agg["kendall"] == pytest.approx(1.0)

    def test_shifted_signal_ranking_agrees_but_vector_shifts(self):
        local = np.random.default_rng(7)
        a = _attributions(rng=local, loud_features=(0, 1))
        b = _attributions(rng=local, loud_features=(0, 1, 2, 3))
        agg = rank_agreement(a, b)
        # 2 loud vs 4 loud features: the loud columns keep their top ranks
        # while the quiet columns reshuffle, so Spearman stays moderate-to-
        # high; the point is that the rank view alone under-detects the
        # change, which the distribution metric does catch (next assert)
        assert agg["spearman"] > 0.5
        assert attribution_distribution_distance(a, b) > 0.03

    def test_no_ties_degenerate(self):
        with pytest.raises(ValueError):
            rank_agreement(np.zeros((10, 1)), np.zeros((10, 1)))


class TestCosineImportance:
    def test_identical_vectors(self):
        a = _attributions()
        assert cosine_importance(a, a) == pytest.approx(1.0)

    def test_orthogonal_importances(self):
        n, f = 4_000, 4
        a = np.zeros((n, f))
        b = np.zeros((n, f))
        a[:, 0] += 0.2
        b[:, 1] += 0.2
        assert cosine_importance(a, b) == pytest.approx(0.0)

    def test_all_zero_pair(self):
        assert cosine_importance(np.zeros((5, 3)), np.zeros((5, 3))) == 1.0

    def test_mismatched_feature_counts(self):
        with pytest.raises(ValueError):
            cosine_importance(np.zeros((5, 3)), np.zeros((5, 4)))


class TestAttributionDistributionDistance:
    def test_identical_distributions_zero(self):
        a = _attributions()
        assert attribution_distribution_distance(a, a) == pytest.approx(0.0)

    def test_shifted_mean_detected(self):
        n = 4_000
        a = _attributions(n=n)
        b = _attributions(n=n) + np.c_[np.full(n, 0.5), np.zeros((n, 7))]
        d = attribution_distribution_distance(a, b)
        # only one of eight columns shifted by 0.5: the mean distance is
        # ~0.5/8 (plus cross-sample noise from the independent draws)
        assert d == pytest.approx(0.5 / 8, abs=0.02)
        # a clean re-draw without the shift sits an order lower
        d0 = attribution_distribution_distance(a, _attributions(n=n))
        assert d > 5 * d0

    def test_validation(self):
        with pytest.raises(ValueError):
            attribution_distribution_distance(np.zeros((3, 2)), np.zeros((3, 3)))
        with pytest.raises(ValueError):
            attribution_distribution_distance(np.zeros((3, 2)), np.zeros((3, 2)) + np.nan)
