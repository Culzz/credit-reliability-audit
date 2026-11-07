"""Tests for the PROPOSED feature-group resolver (synthetic columns only)."""

from __future__ import annotations

import pytest

from src.features.groups import grouped_partition, load_groups, resolve_column_groups


def test_config_partition_of_sources():
    doc = load_groups()
    sources = [f for members in doc["feature_groups"].values() for f in members]
    assert len(sources) == len(set(sources))
    # the 24 audited KEEP features must all be grouped (config-side check)
    from src.features.build import keep_features

    assert sorted(sources) == sorted(keep_features())


def test_engineered_and_onehot_resolution():
    columns = [
        "loan_amnt",
        "annual_inc",
        "dti",
        "emp_length_years",
        "credit_age_months",
        "cr_line_after_issue",
        "total_acc",
        "open_acc",
        "revol_util_pct",
        "revol_bal",
        "fico_range_low",
        "fico_range_high",
        "acc_now_delinq",
        "mths_since_last_record",
        "mths_since_last_record_isna",
        "purpose_debt_consolidation",
        "purpos"
        + "e_small_business",
        "home_ownership_RENT",
        "home_ownership_na",
        "verification_status_verified",
        "initial_list_status_w",
        "application_type_JOINT",
        "inq_last_6mths",
    ]
    out = resolve_column_groups(columns)
    assert out["emp_length_years"] == "borrower_capacity"
    assert out["credit_age_months"] == "credit_history_tenure"
    assert out["cr_line_after_issue"] == "credit_history_tenure"
    assert out["revol_util_pct"] == "utilization"
    assert out["mths_since_last_record_isna"] == "delinquency_public_record"
    assert out["purpose_debt_consolidation"] == "application_context"
    assert out["home_ownership_RENT"] == "application_context"
    assert out["home_ownership_na"] == "application_context"
    assert out["application_type_JOINT"] == "application_context"
    assert out["inq_last_6mths"] == "application_context"


def test_unassigned_column_is_an_error():
    with pytest.raises(ValueError, match="full partition"):
        resolve_column_groups(["loan_amnt", "tot_col_amt_not_audited"])
    with pytest.raises(ValueError, match="full partition"):
        resolve_column_groups(["random_new_engineered_name"])


def test_grouped_partition_covers_all_columns():
    columns = [
        "loan_amnt", "annual_inc", "emp_length_years", "total_acc",
        "open_acc", "credit_age_months", "pub_rec", "delinq_2yrs",
        "delinq_amnt", "acc_now_delinq", "mths_since_last_delinq",
        "mths_since_last_delinq_isna", "mths_since_last_major_derog",
        "mths_since_last_major_derog_isna", "mths_since_last_record",
        "mths_since_last_record_isna", "fico_range_low", "fico_range_high",
        "revol_bal", "revol_util_pct", "purpose_medical",
        "verification_status_na", "home_ownership_OWN",
        "initial_list_status_f", "application_type_INDIVIDUAL",
        "inq_last_6mths", "dti",
    ]
    partition = grouped_partition(columns)
    flat = sorted(c for members in partition.values() for c in members)
    assert flat == sorted(columns)
    assert set(partition) == {
        "borrower_capacity",
        "credit_history_tenure",
        "delinquency_public_record",
        "credit_score",
        "utilization",
        "application_context",
    }
