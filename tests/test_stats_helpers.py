"""Tests for the interpretable companions to the H1c recovery ratio (DEV-3)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_real_statistics as rs


def test_h1c_interpretable_reports_median_share_and_brier_change():
    rows = []
    for cfg, arm, window, before, after, rec in [
        (0, "platt", "2014Q1", 0.10, 0.11, -5.0),
        (1, "platt", "2014Q1", 0.10, 0.07, 0.4),
        (0, "isotonic", "2014Q2", 0.10, 0.10, 0.0),
        (1, "isotonic", "2014Q2", 0.10, 0.13, -2.0),
    ]:
        for metric, value in (("brier_before", before), ("brier_after", after),
                              ("recovery", rec)):
            rows.append({"family": "xgboost", "config_index": cfg,
                         "kind": "seed", "refit_id": 0, "arm": arm,
                         "window": window, "metric": metric,
                         "value": value})
    out = rs.h1c_interpretable(pd.DataFrame(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row.family == "xgboost"
    assert row.n_cells == 4
    assert row.median_recovery == -1.0
    assert row.negative_share == 0.5
    assert row.mean_dbrier == pytest.approx(0.0025)
