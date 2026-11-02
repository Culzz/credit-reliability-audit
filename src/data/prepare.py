"""Sample preparation: vintage parsing, 36-month filter, target construction.

PLAN.md T3.2 (filter + date parsing) and T3.3 (target). Pure row/label logic —
no feature engineering, no Parquet IO. The legacy-status merge implements the
proposed D-3 treatment and is PI-confirmed before the frozen protocol adopts
it (configs/data.yaml `legacy_policy`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PREFIX = re.compile(r"^Does not meet the credit policy\.\s*Status:\s*(.+)$", re.IGNORECASE)

LegacyPolicy = Literal["merge", "exclude", "keep"]


def load_target_config(path: Path | None = None) -> dict:
    """Target status lists from configs/data.yaml (single source of truth)."""
    cfg_path = path or REPO_ROOT / "configs" / "data.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["data"]["target"]


def normalise_loan_status(s: pd.Series, legacy: LegacyPolicy = "merge") -> pd.Series:
    """Canonicalise statuses; optionally merge the legacy prefix variants.

    legacy="merge": "Does not meet the credit policy. Status:X" -> "X".
    legacy="exclude": those rows become "LEGACY_EXCLUDED" (droppable upstream).
    legacy="keep": leave the strings untouched (inventory mode).
    """
    stripped = s.astype("string").str.strip()
    if legacy == "keep":
        return stripped
    m = stripped.str.extract(LEGACY_PREFIX, expand=False)
    merged = stripped.mask(m.notna(), m)
    if legacy == "exclude":
        legacy_mask = stripped.str.match(r"^Does not meet the credit policy", na=False)
        merged = merged.mask(legacy_mask, "LEGACY_EXCLUDED")
        return merged
    return merged


def parse_issue_date(s: pd.Series) -> pd.Series:
    """Parse `issue_d` strings like 'Dec-2015' into month-start Timestamps.

    Raises if anything fails to parse — silent NaT would corrupt vintages.
    """
    parsed = pd.to_datetime(s.astype("string").str.strip(), format="%b-%Y", errors="coerce")
    if parsed.isna().any():
        bad = s[parsed.isna()].unique()[:5]
        raise ValueError(f"issue_d parse failures, e.g. {bad!r}")
    return parsed


def vintage_quarter(s: pd.Series) -> pd.Series:
    """Quarterly Period from parsed issue dates (freq='Q', calendar)."""
    return parse_issue_date(s).dt.to_period("Q")


def filter_term_months(df: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    """Keep rows whose `term` string parses to `months` (raw files store ' 36 months')."""
    terms = df["term"].astype("string").str.strip().str.extract(r"^(\d+)\s*months?$", expand=False)
    if terms.isna().all():
        raise ValueError("term column does not match '<n> months' pattern")
    keep = terms.astype("Int64") == months
    return df.loc[keep.fillna(False), :].copy()


def build_target(
    df: pd.DataFrame,
    legacy: LegacyPolicy = "merge",
    *,
    target_cfg: dict | None = None,
) -> pd.Series:
    """Binary target: 1 = default statuses, 0 = fully paid, NaN = unresolved.

    Unresolved statuses and (with legacy="exclude") legacy rows become NaN so
    callers filter them out explicitly. Unknown status strings raise — never
    invent a silently empty class. Config values stay PROPOSED until Gate 4/5.
    """
    cfg = target_cfg or load_target_config()
    status = normalise_loan_status(df["loan_status"], legacy=legacy)
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out[status.isin(cfg["default_statuses"])] = 1
    out[status.isin(cfg["paid_statuses"])] = 0
    unresolved = status.isin(cfg["excluded_unresolved"]) | status.eq("LEGACY_EXCLUDED")
    out[unresolved] = pd.NA
    unknown = out.isna() & ~(unresolved | status.isna())
    if unknown.any():
        raise ValueError(f"loan_status strings not covered by the config: {sorted(status[unknown].unique())!r}")
    return out
