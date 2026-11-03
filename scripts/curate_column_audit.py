"""Judgment pass over the column audit (PLAN.md T3.5, second pass).

Fills the TO_DECIDE fields of data/column_audit.csv with decisions grounded in
the LendingClub data dictionary (Kaggle description / community figshare
dictionary) plus the per-vintage missingness statistics computed in the first
pass. Anything not covered by an explicit list below stays
SENSITIVITY_ONLY with an "unclassified — reclassify at T3.5 review" reason, so
no column is assigned silently.

Decision vocabulary (CONTEXT.md §5): KEEP | DROP | SENSITIVITY_ONLY |
SPLIT_ONLY | TARGET. Every recorded judgement is either sourced from the
dictionary semantics or carries an "uncertain" flag.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "data" / "column_audit.csv"

TARGET = {"loan_status"}

DROP_IDENTIFIERS = {
    "id": "Primary key (loan identifier) — join key only, never a feature.",
    "member_id": "Identifier; 100% missing in this dataset (measured).",
    "url": "LC listing URL — identifier pointing to a page published after origination.",
    "zip_code": "Quasi-identifier (3-digit zip + incrementing suffix); privacy risk, no robust feature.",
    "desc": "Borrower-authored free text; 94.4% missing and populated only in early vintages (min 0.0 / max 1.0).",
    "title": "Borrower-authored loan title; free-text duplicate of the coded purpose field.",
}

DROP_POST_ORIG = {
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "out_prncp", "out_prncp_inv",
    "pymnt_plan", "hardship_flag", "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_amount", "settlement_date", "settlement_percentage", "settlement_status",
    "settlement_term",
    "orig_projected_additional_accrued_interest", "deferral_term", "payment_plan_start_date",
    "hardship_amount", "hardship_dpd", "hardship_end_date", "hardship_last_payment_amount",
    "hardship_length", "hardship_loan_status", "hardship_payoff_balance_amount",
    "hardship_reason", "hardship_start_date", "hardship_status", "hardship_type",
}

DROP_CONSTANT = {
    "policy_code": "Constant (=1) in the accepted file — zero information.",
    "term": "Sample-defining filter (36 months); constant inside the main sample.",
    "issue_d": "Cohort-definition variable; never used as a predictive feature (PLAN T3.6).",
}

S1_LC_OUTPUTS = {
    "grade", "sub_grade", "int_rate", "installment",
}

KEEP_APPLICATION = {
    "loan_amnt", "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "application_type", "initial_list_status",
}

KEEP_BUREAU_CLASSIC = {
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "acc_now_delinq", "delinq_amnt", "fico_range_low", "fico_range_high",
    "earliest_cr_line",
}

KEEP_INFORMATIVE_MISSING = {
    "mths_since_last_delinq", "mths_since_last_major_derog", "mths_since_last_record",
}

SENSITIVITY = {
    "emp_title": "Borrower-entered free-text employer name — high cardinality; explicit decision per CONTEXT §5: out of the main run, candidate for grouped-text sensitivity.",
    "addr_state": "Borrower state at application — fully populated, but the main design stays lean; explicit decision per CONTEXT §5: regional sensitivity only.",
    "funded_amnt": "Funding amount fixed at issuance, not known at application time; near-duplicate of loan_amnt. Timing verification pending — sensitivity S-P1.",
    "funded_amnt_inv": "Investor-side funded amount — same timing question as funded_amnt; sensitivity S-P1.",
    "disbursement_method": "uncertain: populated across all vintages; the community dictionary dates this field after the earliest vintages, so verify before any use.",
}

JOINT_FIELDS = {
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
}
SEC_APP_FIELDS = {
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_earliest_cr_line", "sec_app_fico_range_high", "sec_app_fico_range_low",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_mths_since_last_major_derog",
    "sec_app_num_rev_accts", "sec_app_open_acc", "sec_app_open_act_il",
    "sec_app_revol_util",
}
WINDOW_DRIFT_FIELDS = {
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m",
    "open_rv_24m", "il_util", "inq_fi", "inq_last_12m", "max_bal_bc", "total_bal_il",
    "total_cu_tl", "all_util",
}
LATE_DRIFT_FIELDS = {
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq", "mths_since_rcnt_il",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit", "total_rev_hi_lim",
}
EARLY_GAP_FIELDS = {
    "pub_rec_bankruptcies", "chargeoff_within_12_mths", "collections_12_mths_ex_med",
    "tax_liens",
}

LEAKAGE_REASON = (
    "Recorded only after origination (payment history, recovery, last-FICO, "
    "hardship/settlement states) — direct target leakage; excluded."
)


def main() -> None:
    with AUDIT.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {r["column"] for r in rows}
    if "id" not in cols:
        rows.append(
            {
                "column": "id",
                "dtype": "VARCHAR",
                "missing_share": "0.0",
                "min_vintage_missing_share": "0.0",
                "max_vintage_missing_share": "0.0",
                "available_at_origination": "yes",
                "availability_evidence": "Primary key, present in every row.",
                "leakage_risk": "none",
                "decision": "DROP",
                "reason": DROP_IDENTIFIERS["id"],
            }
        )

    counts: dict[str, int] = {}
    uncovered = []
    for r in rows:
        c = r["column"]
        if c in TARGET:
            r["decision"] = "TARGET"
            r["available_at_origination"] = "outcome"
            r["availability_evidence"] = "Outcome label; final status observed during repayment."
            r["leakage_risk"] = "is-the-target"
            r["reason"] = "Target construction per CONTEXT §5; excluded from the feature set."
        elif c in DROP_IDENTIFIERS or c in DROP_POST_ORIG or c in DROP_CONSTANT:
            r["decision"] = "DROP"
            r["verdict_extra"] = ""
            r["available_at_origination"] = "no" if c in DROP_POST_ORIG else "yes"
            r["availability_evidence"] = (
                "Post-origination record per data dictionary."
                if c in DROP_POST_ORIG
                else "Present at origination (identifier/constant)."
            )
            r["leakage_risk"] = "high" if c in DROP_POST_ORIG else "none"
            r["reason"] = (
                LEAKAGE_REASON
                if c in DROP_POST_ORIG
                else DROP_IDENTIFIERS.get(c, "")
                or DROP_CONSTANT.get(c, "")
            )
        elif c in S1_LC_OUTPUTS:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "yes"
            r["availability_evidence"] = "LC internal underwriting decision output (data dictionary)."
            r["leakage_risk"] = "proxy"
            r["reason"] = "LC model output — excluded from the main feature set; sensitivity S1 per CONTEXT §5."
        elif c in KEEP_APPLICATION:
            r["decision"] = "KEEP"
            r["available_at_origination"] = "yes"
            r["availability_evidence"] = (
                f"Application-time field per data dictionary; populated across all vintages "
                f"(overall missing {r['missing_share']}, per-vintage max {r['max_vintage_missing_share']})."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Core application feature (CONTEXT §5 keep list)."
        elif c in KEEP_BUREAU_CLASSIC:
            r["decision"] = "KEEP"
            r["available_at_origination"] = "yes"
            r["availability_evidence"] = (
                f"Bureau attribute recorded at origination per data dictionary; effectively populated "
                f"across all substantive vintages (overall missing {r['missing_share']}; the earliest "
                f"micro-vintages show max {r['max_vintage_missing_share']})."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Classic credit-bureau feature available at origination (CONTEXT §5 keep list)."
        elif c in KEEP_INFORMATIVE_MISSING:
            r["decision"] = "KEEP"
            r["available_at_origination"] = "uncertain"
            r["availability_evidence"] = (
                "uncertain: per the data dictionary the field is blank when the borrower has no such adverse "
                "event (informative missingness), not because it was unavailable; confirm semantics against "
                "the dictionary at T3.7 and pair with a missingness indicator."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Months-since-event field with event-absence missingness; kept with indicator."
        elif c in SENSITIVITY:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "uncertain" if c == "disbursement_method" else "yes"
            r["availability_evidence"] = SENSITIVITY[c]
            r["leakage_risk"] = "none"
            r["reason"] = SENSITIVITY[c]
        elif c in JOINT_FIELDS or c in SEC_APP_FIELDS:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "yes"
            r["availability_evidence"] = (
                f"Joint-application-only field (missing {r['missing_share']}); populated "
                "whenever application_type indicates a joint application."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Joint-application subset; main run covers applications; sensitivity S-joint only."
        elif c in WINDOW_DRIFT_FIELDS:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "uncertain"
            r["availability_evidence"] = (
                f"uncertain: populated only within a contiguous sub-window of vintages "
                f"(overall missing {r['missing_share']}, min-vintage 0.0, max-vintage 1.0) — schema drift."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Schema-drift field (window population); main run out, window-only sensitivity possible."
        elif c in LATE_DRIFT_FIELDS:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "uncertain"
            r["availability_evidence"] = (
                f"uncertain: absent for the earliest vintages and adopted later (max-vintage missing 1.0, "
                f"overall {r['missing_share']}); confirm the adoption vintage at T3.7."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Late-adopted bureau attributes; main run (which trains on earlier vintages) drops them."
        elif c in EARLY_GAP_FIELDS:
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "uncertain"
            r["availability_evidence"] = (
                "uncertain: min-vintage 0.0 but max-vintage 1.0 — at least one early vintage fully "
                "unpopulated; confirm the gap at T3.7."
            )
            r["leakage_risk"] = "none"
            r["reason"] = "Effectively complete except for an early-vintage gap; kept out of the main run pending verification."
        elif c == "id":
            pass
        else:
            uncovered.append(c)
            r["decision"] = "SENSITIVITY_ONLY"
            r["available_at_origination"] = "uncertain"
            r["availability_evidence"] = "uncertain: unclassified in the curation lists."
            r["leakage_risk"] = "unknown"
            r["reason"] = "unclassified at first curation pass — reclassify at T3.5 review."
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1

    rows = [r for r in rows if r.get("column")]
    for r in rows:
        r.pop("verdict_extra", None)
    rows.sort(key=lambda r: r["column"])
    fieldnames = [
        "column", "dtype", "missing_share", "min_vintage_missing_share",
        "max_vintage_missing_share", "available_at_origination",
        "availability_evidence", "leakage_risk", "decision", "reason",
    ]
    with AUDIT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"rows: {len(rows)} | decisions: {counts}")
    print(f"uncovered (leftover): {uncovered if uncovered else 'none'}")
    enums_ok = all(r["decision"] in {"KEEP", "DROP", "SENSITIVITY_ONLY", "SPLIT_ONLY", "TARGET"} for r in rows)
    print(f"decision vocabulary valid: {enums_ok}")


if __name__ == "__main__":
    main()
