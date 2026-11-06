"""Synthetic-only tests for the SHAP wrapper (PLAN.md T8.1)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.explain.shap_wrap import explain_matrix, make_explainer, matrix_close

RNG = np.random.default_rng(60606)


def _additive_data(n: int = 2_000, seed: np.random.Generator | None = None):
    rng = seed or RNG
    x = rng.normal(size=(n, 4))
    # additive truth: y depends linearly on x0 and x3 only
    score = 1.2 * x[:, 0] - 0.8 * x[:, 3]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-score))).astype(int)
    return x, y


BG = 300


class TestLinearExact:
    def test_linear_model_log_odds_additivity(self):
        import shap

        x, y = _additive_data()
        model = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
        exp = make_explainer(model, x[:BG], family="logistic_regression")
        vals = explain_matrix(exp, x[:200])
        assert vals.shape == (200, 4)
        # log-odds additivity: margin = sum(shap) + expected_value
        margins = model.decision_function(x[:200])
        base = float(exp.expected_value)
        assert np.allclose(vals.sum(axis=1) + base, margins, atol=1e-6)
        del shap

    def test_loud_columns_dominate(self):
        x, y = _additive_data()
        model = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
        exp = make_explainer(model, x[:BG], family="logistic_regression")
        vals = explain_matrix(exp, x[:500])
        imp = np.abs(vals).mean(axis=0)
        assert sorted(np.argsort(imp)[-2:].tolist()) == [0, 3]


class TestTreeExplainer:
    def test_shapes_and_determinism(self):
        x, y = _additive_data(1_500)
        model = RandomForestClassifier(n_estimators=20, min_samples_leaf=20, random_state=1).fit(x, y)
        exp = make_explainer(model, x[:BG], family="random_forest")
        v1 = explain_matrix(exp, x[:100])
        v2 = explain_matrix(exp, x[:100])
        assert v1.shape == (100, 4)
        assert matrix_close(v1, v2)

    def test_informative_column_separates(self):
        x, y = _additive_data(2_000)
        model = RandomForestClassifier(n_estimators=40, min_samples_leaf=10, random_state=2).fit(x, y)
        exp = make_explainer(model, x[:BG], family="random_forest")
        vals = explain_matrix(exp, x[:300])
        imp = np.abs(vals).mean(axis=0)
        # columns 1 and 2 are inert; their mean |shap| must be far smaller
        assert imp[0] > 2 * imp[1]
        assert imp[3] > 2 * imp[2]

    def test_path_dependent_variant_differs_and_still_finite(self):
        x, y = _additive_data(2_000)
        model = RandomForestClassifier(n_estimators=25, min_samples_leaf=10, random_state=3).fit(x, y)
        exp_pd = make_explainer(model, x[:BG], family="random_forest", path_dependent=True)
        vals = explain_matrix(exp_pd, x[:200])
        assert vals.shape == (200, 4)
        assert np.all(np.isfinite(vals))

    def test_regressor_also_sanctioned(self):
        x, _y = _additive_data(1_200)
        model = RandomForestRegressor(n_estimators=15, min_samples_leaf=30, random_state=4).fit(x, x[:, 0])
        exp = make_explainer(model, x[:BG], family="random_forest")
        vals = explain_matrix(exp, x[:100])
        assert vals.shape == (100, 4)

    def test_unknown_family_and_validation(self):
        x, y = _additive_data(400)
        model = LinearRegression().fit(x, x[:, 0])
        with pytest.raises(ValueError, match="unknown family"):
            make_explainer(model, x, family="gradboost")
        with pytest.raises(ValueError, match="fully degenerate"):
            make_explainer(model, np.zeros((5, 4)), family="logistic_regression")
        # a SINGLE constant column among varied rows is attribute-zero, not
        # forbidden (plumbing windows regularly contain never-firing
        # one-hot columns)
        Xc = np.column_stack([x[:300], np.zeros(300)])
        lin_model = LogisticRegression(C=1.0, max_iter=2000).fit(Xc, y[:300])
        exp_c = make_explainer(lin_model, Xc, family="logistic_regression")
        vals_c = explain_matrix(exp_c, Xc[:100])
        assert np.all(vals_c[:, -1] == 0.0)
        tree = RandomForestClassifier(n_estimators=5, min_samples_leaf=10, random_state=5).fit(x, y)
        good = make_explainer(tree, x[:50], family="random_forest")
        with pytest.raises(ValueError):
            explain_matrix(good, np.array([[0.0, 1.0], [np.inf, 2.0]]))
