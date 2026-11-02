"""Synthetic-only tests for sample preparation (T3.2/T3.3). No real data ever."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.prepare import (
    build_target,
    filter_term_months,
    normalise_loan_status,
    parse_issue_date,
    vintage_quarter,
)


@pytest.fixture()
def statuses() -> list[str]:
    return [
        "Fully Paid",
        "Charged Off",
        "Current",
        "In Grace Period",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    ]


def test_normalise_merge_collapses_legacy_prefix(statuses):
    s = pd.Series(statuses)
    out = normalise_loan_status(s, legacy="merge")
    assert out.tolist()[7:] == ["Fully Paid", "Charged Off"]  # legacy rows land on their base statuses
    assert out.str.startswith("Does not meet").sum() == 0
    assert out.nunique() == 7


def test_normalise_exclude_flags_legacy(statuses):
    out = normalise_loan_status(pd.Series(statuses), legacy="exclude")
    assert (out == "LEGACY_EXCLUDED").sum() == 2
    assert out.iloc[0] == "Fully Paid"


def test_parse_issue_date_and_vintages():
    s = pd.Series(["Dec-2015", " Jan-2008 ", "Apr-2008"])
    parsed = parse_issue_date(s)
    assert parsed.dt.strftime("%Y-%m").tolist() == ["2015-12", "2008-01", "2008-04"]
    assert vintage_quarter(s).astype(str).tolist() == ["2015Q4", "2008Q1", "2008Q2"]


def test_parse_issue_date_rejects_garbage():
    with pytest.raises(ValueError, match="parse failures"):
        parse_issue_date(pd.Series(["Dec-2015", "NOT A DATE"]))


def test_filter_term_months_handles_raw_strings():
    df = pd.DataFrame({"term": [" 36 months", "60 months", " 36 ", " 36 months"]})
    assert len(filter_term_months(df, months=36)) == 2
    assert filter_term_months(df["term"].to_frame(), months=60).shape[0] == 1


def test_build_target_counts_default_and_unresolved(statuses):
    df = pd.DataFrame({"loan_status": statuses})
    y = build_target(df, legacy="merge")
    # resolved: Fully Paid(0), Charged Off(1), Default(1), legacy-Fully(0), legacy-Charged(1) -> 5
    assert y.notna().sum() == 5
    assert (y == 0).sum() == 2
    assert (y == 1).sum() == 3
    assert y.dtype.name == "Int64"


def test_build_target_unknown_status_raises():
    with pytest.raises(ValueError, match="not covered"):
        build_target(pd.DataFrame({"loan_status": ["Fully Paid", "Mystery Status"]}))


def test_build_target_accepts_config_override(statuses):
    df = pd.DataFrame({"loan_status": statuses})
    cfg = {
        "default_statuses": ["Charged Off"],
        "paid_statuses": ["Fully Paid"],
        "excluded_unresolved": ["Current", "Default", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"],
    }
    y = build_target(df, legacy="exclude", target_cfg=cfg)
    assert (y == 1).sum() == 1  # legacy rows are excluded, plain Charged Off remains
    assert y.isna().sum() == 7  # Current/Grace/Late16/Late31/Default unresolved + 2 legacy excluded
