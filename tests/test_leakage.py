"""Leakage and consistency guards (PLAN.md T3.8). Synthetic data only."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import build_feature_frame, keep_features
from tests.test_features import base_frame

FORBIDDEN_POST_ORIG = [
    "total_pymnt", "total_pymnt_inv", "last_pymnt_amnt", "last_pymnt_d",
    "next_pymnt_d", "last_fico_range_low", "last_fico_range_high", "recoveries",
    "out_prncp", "loan_status",
]


def test_no_forbidden_column_in_features():
    df = base_frame()
    for c in FORBIDDEN_POST_ORIG:
        if c in df.columns:
            continue
        df[c] = 1.0 if c not in ("last_pymnt_d", "next_pymnt_d") else "Dec-2016"
    with pytest.raises(ValueError, match="KEEP audit decision"):
        build_feature_frame(df)


def test_no_target_and_no_issue_date_out():
    out = build_feature_frame(base_frame())
    assert "loan_status" not in out.columns
    assert "y" not in out.columns
    assert "issue_d" not in out.columns


def test_every_feature_has_audit_decision():
    feats = keep_features()
    forbidden = {"total_pymnt", "member_id", "url", "zip_code", "policy_code", "issue_d"}
    assert not (set(feats) & forbidden)


def test_no_datetime_columns_in_feature_frame():
    out = build_feature_frame(base_frame())
    dtypes = out.dtypes
    assert not any(pd.api.types.is_datetime64_any_dtype(t) for t in dtypes)


def test_schema_identical_across_cohorts():
    a = base_frame()
    b = base_frame()
    b["issue_d"] = ["Jun-2012", "Jun-2012"]
    out_a = build_feature_frame(a)
    out_b = build_feature_frame(b)
    assert list(out_a.columns) == list(out_b.columns)


def test_credit_age_features_are_numeric():
    out = build_feature_frame(base_frame())
    assert pd.api.types.is_numeric_dtype(out["credit_age_months"])
    assert set(out["cr_line_after_issue"].dropna().unique()).issubset({0, 1})


def test_issue_date_absent_from_audit_whitelist():
    assert "issue_d" not in keep_features()
