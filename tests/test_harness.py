"""Synthetic-only tests for the baseline training harness (no data files)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src.models.harness import (
    FAMILIES,
    METRICS,
    evaluate,
    make_estimator,
    make_preprocessor,
)

RNG = np.random.default_rng(202)


# --------------------------------------------------------------------------
# estimator factory
# --------------------------------------------------------------------------
class TestMakeEstimator:
    def test_known_families(self):
        assert FAMILIES == (
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        )
        assert make_estimator("logistic_regression", {"C": 0.5, "max_iter": 100}, 1) is not None
        assert isinstance(make_estimator("logistic_regression", {}, 1), LogisticRegression)
        assert isinstance(make_estimator("random_forest", {"n_estimators": 5}, 1), RandomForestClassifier)
        # boosted families import inside the factory; both keys construct
        assert make_estimator("xgboost", {"n_estimators": 5}, 1) is not None
        assert make_estimator("lightgbm", {"n_estimators": 5}, 1) is not None

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown model family"):
            make_estimator("gradboost", {}, 1)


# --------------------------------------------------------------------------
# preprocessing is fitted on the training window only
# --------------------------------------------------------------------------
class TestPreprocessor:
    def test_scaling_uses_train_statistics_only(self):
        x_train = RNG.normal(0.0, 1.0, size=(500, 1))
        x_val = RNG.normal(5.0, 1.0, size=(200, 1))
        pre = make_preprocessor(scale=True).fit(x_train)
        scaled = pre.transform(x_val)
        # val rows scaled with train mean/std stay far from standard normal
        assert abs(float(np.mean(scaled)) - 5.0) < 0.5

    def test_median_imputation(self):
        x_train = np.array([[1.0], [2.0], [np.nan], [4.0]])
        x_val = np.array([[np.nan]])
        pre = make_preprocessor(scale=False).fit(x_train)
        assert pre.transform(x_val)[0, 0] == pytest.approx(2.0)  # train medians


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
class TestEvaluate:
    def test_perfect_scores(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        p = np.array([0.01, 0.02, 0.97, 0.98])
        res = evaluate(y, p)
        assert res.point["roc_auc"] == 1.0
        assert res.n_bootstrap == 0
        assert res.point["accuracy"] == 1.0
        assert res.point["brier"] == pytest.approx(np.mean((p - y) ** 2))
        assert res.point["logloss"] == pytest.approx(log_loss(y, p, labels=[0, 1]))

    def test_constant_scores_give_half_auc(self):
        y = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        p = np.full(5, 0.5)
        res = evaluate(y, p)
        assert res.point["roc_auc"] == pytest.approx(0.5)

    def test_fixed_threshold_confusion_terms(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([0.6, 0.7, 0.2, 0.1])
        res = evaluate(y, p, threshold=0.5)
        # hard calls: [1,1,0,0] -> tp=1, fp=1, fn=1, tn=1
        assert res.point["precision"] == pytest.approx(0.5)
        assert res.point["recall"] == pytest.approx(0.5)
        assert res.point["f1"] == pytest.approx(0.5)
        assert res.point["balanced_accuracy"] == pytest.approx(0.5)
        # threshold sensitivity: 0.15 flips the second call
        res2 = evaluate(y, p, threshold=0.15)
        assert res2.point["recall"] == pytest.approx(1.0)

    def test_metrics_whitelist_order(self):
        y = np.array([0.0, 1.0])
        p = np.array([0.3, 0.7])
        res = evaluate(y, p)
        assert set(res.point) == set(METRICS)

    def test_bootstrap_cis_contain_point_estimate(self):
        n = 400
        y = (RNG.uniform(size=n) < 0.3).astype(float)
        p = np.clip(y * 0.3 + RNG.uniform(0, 0.5, size=n), 0.0, 1.0)
        res = evaluate(y, p, n_bootstrap=200, seed=7)
        assert res.n_bootstrap == 200
        for m in ("roc_auc", "brier", "accuracy"):
            assert res.ci_low[m] <= res.point[m] <= res.ci_high[m]

    def test_same_seed_reproduces_cis_exactly(self):
        n = 300
        y = (RNG.uniform(size=n) < 0.4).astype(float)
        p = RNG.uniform(size=n)
        r1 = evaluate(y, p, n_bootstrap=100, seed=42)
        r2 = evaluate(y, p, n_bootstrap=100, seed=42)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_input_validation(self):
        with pytest.raises(ValueError):
            evaluate(np.array([0.5]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            evaluate(np.array([0.5, 1.2]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            evaluate(np.array([0.5, 0.5]), np.array([0.5, 1.3]))
        with pytest.raises(ValueError):
            evaluate(np.array([0.0, 1.0]), np.array([0.5, 0.5]), threshold=1.0)
        with pytest.raises(ValueError):  # bootstrap without a seed is a bug
            evaluate(np.array([0.0, 1.0, 1.0]), np.array([0.4, 0.5, 0.6]), n_bootstrap=10)
        with pytest.raises(ValueError):
            evaluate(np.array([0.0, 1.0, 1.0]), np.array([0.4, 0.5, 0.6]), n_bootstrap=1, seed=1)


# --------------------------------------------------------------------------
# end-to-end fit / predict determinism on synthetic tabular data
# --------------------------------------------------------------------------
def _synthetic_frame(n: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = RNG.normal(size=(n, 3))
    truth = 1.0 / (1.0 + np.exp(-(2.0 * x[:, 0] - 1.5 * x[:, 1])))
    y = (RNG.uniform(size=n) < truth).astype(float)
    x[RNG.uniform(size=(n, 3)) < 0.02] = np.nan  # sparse missingness
    return x, y, truth


class TestFitPredict:
    def test_logistic_recovers_signal(self):
        x, y, _truth = _synthetic_frame()
        pre = make_preprocessor(scale=True).fit(x)
        est = make_estimator("logistic_regression", {"C": 10.0, "max_iter": 2000}, seed=0).fit(pre.transform(x), y)
        p = est.predict_proba(pre.transform(x))[:, 1]
        assert roc_auc_score(y, p) > 0.85

    def test_same_seed_same_predictions(self):
        x, y, _truth = _synthetic_frame(500)
        for family, params in [("logistic_regression", {"C": 1.0}), ("random_forest", {"n_estimators": 10})]:
            out = []
            for _ in range(2):
                pre = make_preprocessor(scale=True).fit(x)
                est = make_estimator(family, params, seed=123).fit(pre.transform(x), y)
                out.append(est.predict_proba(pre.transform(x))[:, 1])
            assert np.array_equal(out[0], out[1]), family
