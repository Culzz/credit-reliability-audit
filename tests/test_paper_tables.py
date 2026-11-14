"""Tests for the paper-table builder (deterministic table aggregation)."""

from __future__ import annotations

import pandas as pd

from scripts import run_paper_tables as tables


def test_table2_orders_metrics_and_windows():
    agg = pd.DataFrame({
        "family": ["b", "b", "a", "a"],
        "kind": ["seed", "boot", "seed", "seed"],
        "window": ["2014Q1", "2014Q1", "val", "2014Q1"],
        "metric": ["citl", "citl", "slope", "citl"],
        "value": [0.2, 9.0, 1.1, 0.4],
    })
    out = tables.table2(agg)
    assert list(out.columns) == ["family", "window", "citl", "slope"]
    assert list(out["family"]) == ["a", "a", "b"]
    assert list(out["window"].astype(str)) == ["val", "2014Q1", "2014Q1"]
    assert out.loc[out.family == "b", "citl"].iloc[0] == 0.2
    assert out.loc[(out.family == "a") & (out.window.astype(str) == "2014Q1"),
                   "citl"].iloc[0] == 0.4


def test_table3_keeps_confirmatory_order_and_columns():
    stats = pd.DataFrame({
        "hypothesis": ["RQ4.2", "H1a", "H1c-lag", "H2c"],
        "scope": ["x"] * 4, "statistic": ["s"] * 4,
        "value": [1.0] * 4, "ci_low": [0.0] * 4, "ci_high": [2.0] * 4,
        "p_raw": [0.01] * 4, "p_holm": [0.02] * 4,
        "verdict": ["not_rejected"] * 4, "note": [""] * 4,
    })
    out = tables.table3(stats)
    assert list(out["hypothesis"]) == ["H1a", "H2c", "RQ4.2"]
    assert "p_holm" in out.columns and "note" in out.columns


def test_table_s2_lag_mean_and_count():
    arms = pd.DataFrame({
        "family": ["x", "x", "x"],
        "kind": ["seed", "seed", "boot"],
        "arm": ["platt", "platt", "platt"],
        "window": ["2014Q1", "2014Q1", "2014Q1"],
        "metric": ["recovery"] * 3,
        "value": [0.4, 0.6, 0.99],
    })
    out = tables.table_s2(arms)
    assert len(out) == 1
    assert out.loc[0, "label_age_quarters"] == 3
    assert out.loc[0, "mean_recovery"] == 0.5
    assert out.loc[0, "n_refits"] == 2
