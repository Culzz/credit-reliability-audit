"""Reproducibility checks (PLAN.md T6.7).

Re-runs one small configuration through the full harness path
(preprocessor fit -> train -> predict -> evaluate with bootstrap) and
requires identical outputs. The fresh-clone reproduction sweep is a
manual gate activity and stays out of CI; what CI can hold is: same
code + same seed + same inputs => same numbers, end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.harness import evaluate, make_estimator, make_preprocessor

RNG = np.random.default_rng(20261106)


def _plumbing_like_frame(n: int = 1_200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = RNG.normal(size=(n, 6))
    truth = 1.0 / (1.0 + np.exp(-(1.4 * x[:, 0] - 1.1 * x[:, 2] + 0.9 * x[:, 3])))
    y = (RNG.uniform(size=n) < truth).astype(float)
    x[RNG.uniform(size=(n, 6)) < 0.02] = np.nan
    dev = np.arange(0, n // 2)
    val = np.arange(n // 2, n)
    return x, y, (dev, val)


SMALL_CONFIG = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 25, "min_samples_leaf": 20},
    "xgboost": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
    "lightgbm": {
        "n_estimators": 40,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "subsample_freq": 1,
        "min_child_samples": 20,
    },
}

BOOTSTRAPS = {"logistic_regression": 50, "random_forest": 50, "xgboost": 30, "lightgbm": 30}


@pytest.mark.parametrize("family", list(SMALL_CONFIG))
def test_full_path_reproduces_exactly(family):
    x, y, (dev, val) = _plumbing_like_frame()

    def run_once():
        pre = make_preprocessor(scale=True).fit(x[dev])
        est = make_estimator(family, SMALL_CONFIG[family], seed=20180101)
        est.fit(pre.transform(x[dev]), y[dev])
        p_val = est.predict_proba(pre.transform(x[val]))[:, 1]
        return evaluate(
            y[val], p_val, n_bootstrap=BOOTSTRAPS[family], seed=20180101
        )

    r1 = run_once()
    r2 = run_once()

    def same_numbers(a, b):
        assert set(a) == set(b)
        for k in a:
            assert a[k] == pytest.approx(b[k], abs=1e-9)

    # parallel tree predictions are not bit-stable across runs (openMP
    # summation order), so identical-within-1e-9 is the T6.7 tolerance line
    same_numbers(r1.point, r2.point)
    same_numbers(r1.ci_low, r2.ci_low)
    same_numbers(r1.ci_high, r2.ci_high)


def test_preprocessor_independently_reproducible():
    x, _y, (dev, val) = _plumbing_like_frame()
    pre1 = make_preprocessor(scale=True).fit(x[dev])
    pre2 = make_preprocessor(scale=True).fit(x[dev])
    assert np.array_equal(pre1.transform(x[dev]), pre2.transform(x[dev]))
    assert np.array_equal(pre1.transform(x[val]), pre2.transform(x[val]))


def test_bootstrap_design_ignores_global_rng_state():
    # the bootstrap design depends only on its own seed: two evaluate calls
    # with identical inputs return identical CIs even if the GLOBAL generator
    # has been consumed differently between the calls
    n = 600
    y = (RNG.uniform(size=n) < 0.35).astype(float)
    p = np.linspace(0.05, 0.95, n)

    _ = RNG.normal(size=(25, 4))  # consume the global stream between calls
    e1 = evaluate(y, p, n_bootstrap=30, seed=17)
    _ = RNG.normal(size=(90, 11))  # consume a different amount
    e2 = evaluate(y, p, n_bootstrap=30, seed=17)
    assert e1.point == e2.point
    assert e1.ci_low == e2.ci_low
    assert e1.ci_high == e2.ci_high
