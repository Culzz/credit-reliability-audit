"""Single point of data access (PLAN.md T0.5).

Phase 0 provides the role gating and vintage-window logic; the actual Parquet
IO lands in T2.3 (`src/data/load_raw.py`) and will call these functions. Until
then, no module may read `data/` except through this package — checked by the
tests in `tests/` and by review before any gate.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml

from .guard import ProtocolLockError, protocol_is_frozen, require_unlocked

__all__ = [
    "ProtocolLockError",
    "filter_by_vintage_window",
    "load_descriptive",
    "load_split",
    "protocol_is_frozen",
    "require_unlocked",
    "vintage_window",
]

DEFAULT_SPLITS = Path(__file__).resolve().parents[2] / "configs" / "splits.yaml"

TARGET_LIKE_COLUMNS = {"y", "target", "loan_status", "default", "p", "predicted_probability"}


def _parse_quarter(s: str) -> pd.Period:
    """Parse a vintage like '2012Q1' into a quarterly Period."""
    return pd.Period(str(s).strip().upper(), freq="Q")


def vintage_window(
    role: str,
    *,
    splits_path: Path | None = None,
    flag: Path | None = None,
) -> dict[str, str]:
    """Resolve the vintage window for a role from configs/splits.yaml.

    Enforces the test-cohort lock. Returns the window dict so callers filter
    data; it never reads data itself (IO enters in T2.3 behind this gate).
    """
    require_unlocked(role, flag=flag)
    path = DEFAULT_SPLITS if splits_path is None else Path(splits_path)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)["splits"]

    window = cfg.get(role, {})
    start, end = window.get("start"), window.get("end")
    if start is None or end is None:
        if role == "test":
            raise ValueError(
                "Test window is not set yet: it is fixed at Gate 4 from the "
                "cohort feasibility table (descriptive evidence only), never "
                "before, and loading test data additionally requires "
                "protocol/PROTOCOL_FROZEN (PLAN.md T0.5)."
            )
        raise ValueError(f"Split window for role={role!r} is incomplete in {path}.")
    return {"start": str(start), "end": str(end)}


# Backwards-friendly alias per the PLAN.md §4.1 name.
def load_split(role: str, *, splits_path: Path | None = None, flag: Path | None = None) -> dict[str, str]:
    """Alias of `vintage_window` (role-gated split window resolution)."""
    return vintage_window(role, splits_path=splits_path, flag=flag)


def filter_by_vintage_window(
    df: pd.DataFrame,
    *,
    vintage_col: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Rows whose vintage lies in [start, end] inclusive (e.g. 2012Q1..2013Q2)."""
    if vintage_col not in df.columns:
        raise KeyError(f"vintage column {vintage_col!r} missing from frame")
    lo, hi = _parse_quarter(start), _parse_quarter(end)
    if lo > hi:
        raise ValueError(f"window start {lo} is after window end {hi}")
    vint = df[vintage_col].map(_parse_quarter)
    mask = (vint >= lo) & (vint <= hi)
    return df.loc[mask, :]


def load_descriptive(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    vintage_col: str | None = None,
) -> pd.DataFrame:
    """Descriptive-access surface for pre-freeze shift statistics (Phase 4).

    Returns features/counts only: drops anything target-like or
    prediction-like. This exists so Phase 4 shift code can see later vintages
    *without* ever touching labels or model outputs, per CONTEXT.md §4 rule 3.
    """
    cols = df.columns.tolist()
    if feature_columns is None:
        keep = [c for c in cols if str(c).lower() not in TARGET_LIKE_COLUMNS]
    else:
        keep = list(feature_columns)
        leaked = [c for c in keep if str(c).lower() in TARGET_LIKE_COLUMNS]
        if leaked:
            raise ProtocolLockError(
                f"load_descriptive(): refused target-like columns {leaked}; "
                "descriptive role exposes features and counts only."
            )
    out = df.loc[:, keep]
    if vintage_col is not None and vintage_col in out.columns:
        out = filter_by_vintage_window(
            out, vintage_col=vintage_col, start="2007Q1", end="2099Q4"
        )
    return out
