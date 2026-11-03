"""Feature engineering (PLAN.md T3.6).

Deterministic, state-free transformations from audited base columns into the
modelling feature frame. Guardrails:

- Base columns must carry a KEEP decision in data/column_audit.csv — anything
  else raises, so no unaudited column can slip into a feature matrix (T3.8).
- `issue_d` is cohort metadata and is never a feature.
- No fitting happens here: encodings are complete category maps, so nothing
  can leak through a fitted statistic. Any train-fitted scaler/imputer is
  added later, in the Phase 4 pipeline, where it is confined to training
  splits by the loader.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "data" / "column_audit.csv"

# Columns built here from audited inputs (and their audit-derived inputs).
CREDIT_AGE_SOURCE = "earliest_cr_line"

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

CATEGORICAL = ["home_ownership", "verification_status", "purpose", "application_type", "initial_list_status"]


def keep_features(audit_path: Path | None = None) -> list[str]:
    """Base columns with a KEEP audit decision (the feature whitelist)."""
    import csv

    path = audit_path or AUDIT
    with path.open(encoding="utf-8", newline="") as f:
        return sorted(r["column"] for r in csv.DictReader(f) if r["decision"] == "KEEP")


def assert_audited(columns: list[str], audit_path: Path | None = None) -> None:
    """Raise unless every requested base column carries a KEEP decision."""
    allowed = set(keep_features(audit_path)) | {"issue_d", CREDIT_AGE_SOURCE}
    unapproved = sorted(set(columns) - allowed)
    if unapproved:
        raise ValueError(f"columns without a KEEP audit decision: {unapproved!r}")


def credit_age_months(df: pd.DataFrame) -> pd.Series:
    """Months between the earliest credit line and the issue date.

    Negative values mean the credit-line field postdates issuance (recorded
    anomalies): they are returned as NA, and `cr_line_after_issue` flags them.
    """
    cr = pd.to_datetime(df["earliest_cr_line"].astype("string").str.strip(), format="%b-%Y", errors="coerce")
    issue = pd.to_datetime(df["issue_d"].astype("string").str.strip(), format="%b-%Y", errors="coerce")
    months = (issue.dt.year - cr.dt.year) * 12 + (issue.dt.month - cr.dt.month)
    return pd.Series(pd.NA, index=df.index, dtype="Float64").mask(months.notna() & (months >= 0), months.astype("Float64"))


def negative_credit_age_mask(df: pd.DataFrame) -> pd.Series:
    """True where earliest_cr_line parses later than issue_d (T3.6 check)."""
    cr = pd.to_datetime(df["earliest_cr_line"].astype("string").str.strip(), format="%b-%Y", errors="coerce")
    issue = pd.to_datetime(df["issue_d"].astype("string").str.strip(), format="%b-%Y", errors="coerce")
    return (cr > issue).fillna(False)


def clean_revol_util(df: pd.DataFrame) -> pd.Series:
    """Utilisation as a numeric 0-100 percent percentage (self-reported cap 100?)."""
    v = df["revol_util"]
    if v.dtype.kind not in "fiu":
        v = v.astype("string").str.strip().str.rstrip("%")
    out = pd.to_numeric(v, errors="coerce").astype("Float64")
    return out.mask((out < 0) | (out > 100))


def encode_emp_length(df: pd.DataFrame) -> pd.Series:
    """Employment length as the ordinal number of years."""
    s = df["emp_length"].astype("string").str.strip()
    out = s.map(EMP_LENGTH_MAP).astype("Int64")
    if s.notna().any() and out.isna().loc[s.notna()].any():
        bad = sorted(s.loc[s.notna() & out.isna()].unique())
        raise ValueError(f"emp_length strings not covered by the encoding table: {bad!r}")
    return out


def forbid_issue_date(columns: list[str]) -> None:
    """`issue_d` is cohort metadata, never a feature (PLAN T3.6)."""
    if any(re.sub(r"_(?:feat|feature|input)$", "", c) == "issue_d" or c == "issue_d" for c in columns):
        raise ValueError("issue_d must not be used as a predictive feature")


def build_feature_frame(df: pd.DataFrame, audit_path: Path | None = None) -> pd.DataFrame:
    """Assemble the modelling frame from audited KEEP columns.

    Deterministic transforms only:
    - numeric passthrough for audited numeric columns,
    - ordinal encode emp_length, clean revol_util,
    - credit age in months from earliest_cr_line vs issue_d (+ anomaly flag),
    - missingness indicators for the informative-missingness ages columns,
    - one-hot encoding for the configured categoricals (fixed by construction,
      so no train-time fitting is needed at this stage).
    """
    assert_audited(list(df.columns), audit_path)
    forbid_issue_date(df.columns.drop("issue_d", errors="ignore"))

    out = pd.DataFrame(index=df.index)
    for c in keep_features(audit_path):
        if c == "issue_d" or c not in df.columns:
            continue
        if c in CATEGORICAL:
            dummies = pd.get_dummies(df[c].astype("string"), prefix=c, dummy_na=True, dtype="uint8")
            out = pd.concat([out, dummies], axis=1)
            continue
        if c == "emp_length":
            out["emp_length_years"] = encode_emp_length(df)
            continue
        if c == "revol_util":
            out["revol_util_pct"] = clean_revol_util(df)
            continue
        if c == CREDIT_AGE_SOURCE:
            out["credit_age_months"] = credit_age_months(df)
            out["cr_line_after_issue"] = negative_credit_age_mask(df).astype("uint8")
            continue
        s = df[c]
        if s.dtype.kind not in "fiu":
            s = pd.to_numeric(s, errors="coerce")
        out[c] = s.astype("Float64")
    assert "earliest_cr_line" not in out.columns  # raw string never passes through

    # Missingness indicators for the months-since-event ages fields (kept with
    # event-absence semantics, but the raw blankness stays visible as a flag).
    for c in [x for x in keep_features(audit_path) if x.startswith("mths_since_")]:
        if c in df.columns:
            out[f"{c}_isna"] = df[c].isna().astype("uint8")
            if c not in out.columns:
                vals = df[c]
                if vals.dtype.kind not in "fiu":
                    vals = pd.to_numeric(vals, errors="coerce")
                out[c] = vals.astype("Float64")

    forbid_issue_date(list(out.columns))
    return out
