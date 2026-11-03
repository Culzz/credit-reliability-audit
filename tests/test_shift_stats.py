"""Synthetic-input tests for the descriptive shift statistics (T4.2 helpers).

Phase 4 statistics never touch real data in tests — frames below are built
from fixed random draws with a pinned seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.shift import (
    categorical_js_divergence,
    ks_statistic,
    psi_numeric,
    quantile_bin_edges,
    smd,
    wilson_interval,
)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(3116)


def _series(rng, n, loc, scale):
    return pd.Series(rng.normal(loc, scale, size=n))


class TestPsi:
    def test_identical_draws_give_small_psi(self, rng):
        ref = _series(rng, 20_000, 0.0, 1.0)
        cur = _series(rng, 20_000, 0.0, 1.0)
        assert psi_numeric(ref, cur) < 0.02

    def test_strong_shift_gives_large_psi(self, rng):
        ref = _series(rng, 20_000, 0.0, 1.0)
        cur = _series(rng, 20_000, 2.5, 1.0)
        assert psi_numeric(ref, cur) > 1.0

    def test_shifted_data_positive_in_both_directions(self, rng):
        ref = _series(rng, 10_000, 0.0, 1.0)
        cur = _series(rng, 10_000, 1.0, 1.0)
        # PSI is not order-symmetric because bins derive from the reference;
        # a real shift must show up strongly either way.
        assert psi_numeric(ref, cur) > 0.3
        assert psi_numeric(cur, ref) > 0.3

    def test_missing_values_are_dropped_not_binned(self, rng):
        ref = _series(rng, 5_000, 0.0, 1.0)
        cur = _series(rng, 5_000, 0.0, 1.0)
        cur_with_na = cur.copy()
        cur_with_na.iloc[:1000] = np.nan
        assert psi_numeric(ref, cur_with_na) == pytest.approx(psi_numeric(ref, cur), abs=0.01)

    def test_constant_reference_does_not_crash_and_detects_shift(self, rng):
        ref = pd.Series([1.0] * 1_000)
        cur = pd.Series([5.0] * 1_000)
        value = psi_numeric(ref, cur)
        assert value > 3.0  # everything sits in the guard bin

    def test_empty_current_raises(self, rng):
        ref = _series(rng, 100, 0.0, 1.0)
        cur = pd.Series([np.nan] * 50)
        with pytest.raises(ValueError):
            psi_numeric(ref, cur)


class TestEdges:
    def test_edges_have_infinite_guards(self, rng):
        edges = quantile_bin_edges(_series(rng, 1_000, 0.0, 1.0), n_bins=10)
        assert edges[0] == -np.inf and edges[-1] == np.inf
        assert len(np.unique(edges)) == len(edges)
        assert np.all(np.diff(edges) >= 0)


class TestKs:
    def test_same_distribution_small_d(self, rng):
        ref = _series(rng, 20_000, 0.0, 1.0)
        cur = _series(rng, 20_000, 0.0, 1.0)
        assert ks_statistic(ref, cur) < 0.035

    def test_shifted_distribution_big_d(self, rng):
        ref = _series(rng, 10_000, 0.0, 1.0)
        cur = _series(rng, 10_000, 2.0, 1.0)
        assert ks_statistic(ref, cur) > 0.5


class TestSmd:
    def test_zero_when_identical_proportions(self, rng):
        ref = _series(rng, 5_000, 0.0, 1.0)
        cur = _series(rng, 5_000, 0.0, 1.0)
        assert abs(smd(ref, cur)) < 0.15

    def test_positive_for_shifted_current(self, rng):
        ref = _series(rng, 5_000, 0.0, 1.0)
        cur = _series(rng, 5_000, 1.0, 1.0)
        assert smd(ref, cur) > 0.5

    def test_equal_means_zero_variance_is_zero(self):
        assert smd(pd.Series([1.0] * 10), pd.Series([1.0] * 10)) == 0.0


class TestCategoricalJs:
    def test_identical_distributions_zero(self):
        ref = pd.Series(["a", "b", "c", "a", "b", "c"])
        assert categorical_js_divergence(ref, ref.copy()) == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_categories_maximal(self):
        ref = pd.Series(["a", "a", "a", "a", "a", "a"])
        cur = pd.Series(["b", "b", "b", "b", "b", "b"])
        # jensenshannon returns the JS distance (sqrt of divergence),
        # so disjoint supports hit 1.0 in base-2 units up to smoothing.
        assert categorical_js_divergence(ref, cur) == pytest.approx(1.0, abs=1e-3)

    def test_na_forms_own_category(self):
        ref = pd.Series(["a", "a", "a", "a", "a", "a"])
        cur = pd.Series([np.nan] * 6)
        value = categorical_js_divergence(ref, cur)
        assert value > 0.9


class TestWilson:
    def test_well_known_row(self):
        # p = 0.1, n = 1000, z = 1.96: analytic Wilson centres at 0.10192,
        # half-width 0.0186; bounds asserted from the analytic value.
        lo, hi = wilson_interval(100, 1000)
        assert 0.080 < lo < 0.090 and 0.112 < hi < 0.128

    def test_binomial_proportion_centered_near_p(self):
        lo, hi = wilson_interval(300, 1200)
        p = 300 / 1200
        assert 0.0 < lo < p < hi < 0.4
        assert min(abs(p - lo), abs(hi - p)) < 0.05

    def test_empty_total_nan(self):
        lo, hi = wilson_interval(0, 0)
        assert np.isnan(lo) and np.isnan(hi)
