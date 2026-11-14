"""Tests for the post-freeze verdict renderer."""

from __future__ import annotations

import pandas as pd

from scripts import report_verdicts as verdicts


def test_render_verdicts_includes_confirmatory_rows_and_floors():
    stats = pd.DataFrame({
        "hypothesis": ["H1a", "H1c-lag", "RQ4.1"],
        "statistic": ["mean delta", "rho", "delta rho"],
        "value": [0.0312, -0.4, 0.1],
        "ci_low": [0.01, float("nan"), -0.02],
        "ci_high": [0.05, float("nan"), 0.2],
        "p_raw": [0.001, float("nan"), 0.2],
        "p_holm": [0.004, float("nan"), 0.4],
        "verdict": ["reject_H0", "exploratory", "not_rejected"],
        "note": ["anchor", "within H1c", "region negligible"],
    })
    floors = pd.DataFrame({
        "family": ["xgboost", "xgboost"],
        "dimension": ["stability", "flip"],
        "n_pairs": [50, 50],
    })
    text = verdicts.render_verdicts(stats, floors)
    assert "| H1a | mean delta | 0.0312 | [0.01, 0.05] | 0.001 | 0.004 | reject_H0 |" in text
    assert "| H1c-lag | rho | -0.4 | — | — |" in text
    assert "| RQ4.1 | delta rho | 0.1 | [-0.02, 0.2] | 0.2 | 0.4 | not_rejected |" in text
    assert "not run" in text  # missing confirmatory rows are declared, not hidden
    assert "| xgboost | flip | 50 |" in text
