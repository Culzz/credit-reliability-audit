"""Tests for the real-battery execution helpers (DEV-1 layout)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from scripts import run_deep_battery as battery


class _Ctx:
    def __init__(self, groups):
        self.groups = groups
        self.group_names = sorted(groups)


def test_requested_families_keeps_fast_first_order():
    got = battery._requested_families("random_forest,logistic_regression")
    assert got == ("logistic_regression", "random_forest")


def test_requested_families_rejects_unknown():
    with pytest.raises(SystemExit, match="unknown families"):
        battery._requested_families("catboost")


def test_rf_importance_shape_and_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    est = RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=0
    ).fit(X, y)
    ctx = _Ctx({"signal": [0, 1], "noise_a": [2, 3], "noise_b": [4, 5]})
    V, G = battery._rf_importance(est, X[:100], y[:100], ctx, 123)
    assert V.shape == (1, 6)
    assert G.shape == (3,)
    assert np.all(G >= 0.0)
    signal = ctx.group_names.index("signal")
    assert G[signal] > G[ctx.group_names.index("noise_a")]
    assert G[signal] > G[ctx.group_names.index("noise_b")]


def test_merge_family_tables_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(battery, "TABLES", tmp_path)
    for stem in battery.OUTPUT_STEMS:
        for family in ("logistic_regression", "xgboost"):
            pd.DataFrame([{"family": family, "value": 1.0}]).to_csv(
                battery._family_path(stem, "real", family), index=False
            )
    battery.merge_family_tables(("logistic_regression", "xgboost"), "real")
    for stem in battery.OUTPUT_STEMS:
        merged = pd.read_csv(battery._merged_path(stem, "real"))
        assert list(merged["family"]) == ["logistic_regression", "xgboost"]


def test_merge_family_tables_refuses_missing_family(tmp_path, monkeypatch):
    monkeypatch.setattr(battery, "TABLES", tmp_path)
    pd.DataFrame([{"family": "logistic_regression"}]).to_csv(
        battery._family_path(battery.OUTPUT_STEMS[0], "real",
                             "logistic_regression"), index=False
    )
    with pytest.raises(FileNotFoundError, match="cannot merge"):
        battery.merge_family_tables(
            ("logistic_regression", "xgboost"), "real"
        )
