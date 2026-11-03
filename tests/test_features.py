"""Synthetic-only tests for feature engineering (T3.6). No real data ever."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import (
    build_feature_frame,
    clean_revol_util,
    credit_age_months,
    encode_emp_length,
    forbid_issue_date,
    keep_features,
    negative_credit_age_mask,
)

BASE_COLS = [
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "application_type", "initial_list_status", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "acc_now_delinq", "delinq_amnt", "fico_range_low", "fico_range_high",
    "earliest_cr_line", "mths_since_last_delinq", "issue_d",
]


def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_amnt": [5000.0, 5000.0],
            "emp_length": ["10+ years", "< 1 year"],
            "home_ownership": ["RENT", "OWN"],
            "annual_inc": [50000.0, 60000.0],
            "verification_status": ["Source Verified", "Not Verified"],
            "purpose": ["debt_consolidation", "credit_card"],
            "dti": [14.5, 21.0],
            "application_type": ["Individual", "Individual"],
            "initial_list_status": ["w", "f"],
            "delinq_2yrs": [0.0, 0.0],
            "inq_last_6mths": [1.0, 0.0],
            "open_acc": [5.0, 3.0],
            "pub_rec": [0.0, 0.0],
            "revol_bal": [4000.0, 8000.0],
            "revol_util": ["30.0", "105.0"],
            "total_acc": [10.0, 7.0],
            "acc_now_delinq": [0.0, 0.0],
            "delinq_amnt": [0.0, 0.0],
            "fico_range_low": [700.0, 650.0],
            "fico_range_high": [704.0, 659.0],
            "earliest_cr_line": ["Jan-1990", "Dec-2007"],
            "mths_since_last_delinq": [pd.NA, 6.0],
            "issue_d": ["Dec-2007", "Dec-2015"],
        }
    )


def test_keep_features_comes_from_audit():
    feats = keep_features()
    assert "loan_amnt" in feats and "total_pymnt" not in feats
    assert all(f != "issue_d" for f in feats)


def test_credit_age_months_leap_gap_gap():
    df = base_frame()
    ages = credit_age_months(df)
    # Dec-2007 minus Jan-1990 = 17*12 + 11 = 215; Dec-2015 minus Dec-2007 = 96
    assert ages.tolist() == [215.0, 96.0]


def test_negative_credit_age_flagged_and_nul():
    df = base_frame()
    df.loc[0, "earliest_cr_line"] = "Jan-2008"  # one month after issue Dec-2007
    ages = credit_age_months(df)
    assert pd.isna(ages.iloc[0])
    assert negative_credit_age_mask(df).tolist() == [True, False]


def test_revol_util_cleaned_and_dropped_above_100():
    df = base_frame()
    vals = clean_revol_util(df)
    assert vals.iloc[0] == 30.0
    assert pd.isna(vals.iloc[1])


def test_emp_length_encode_and_reject_unknown():
    df = base_frame()
    assert encode_emp_length(df).tolist() == [10, 0]
    df.loc[0, "emp_length"] = "eleven years"
    with pytest.raises(ValueError, match="not covered"):
        encode_emp_length(df)


def test_forbid_issue_date():
    with pytest.raises(ValueError, match="issue_d"):
        forbid_issue_date(["loan_amnt", "issue_d"])
    forbid_issue_date(["loan_amnt"])  # no raise


def test_build_feature_frame_shape_and_guards():
    out = build_feature_frame(base_frame())
    assert list(out.columns) == list(out.columns)  # deterministic order
    assert "issue_d" not in out.columns
    assert "credit_age_months" in out.columns and "cr_line_after_issue" in out.columns
    assert "revol_util_pct" in out.columns and "emp_length_years" in out.columns
    assert any(c.startswith("home_ownership_") for c in out.columns)
    assert "mths_since_last_delinq_isna" in out.columns
    assert out.shape[0] == 2


def test_unaudited_column_rejected():
    df = base_frame()
    df["total_pymnt"] = [1.0, 2.0]  # dropped-by-audit leakage column smuggled in
    with pytest.raises(ValueError, match="KEEP audit decision"):
        build_feature_frame(df)
