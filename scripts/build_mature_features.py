"""Mature 36-month feature cache for the Phase 4 descriptive pipelines.

Selects the audited KEEP columns + metadata (id, vintage, y resolved status)
for every 36-month loan from the accepted-raw Parquet through DuckDB, applies
the deterministic T3.6 feature transform once, and caches the result as
data/interim/mature_36m_features.parquet (gitignored).

Guards:
- Features enter through src.features.build_feature_frame only (audit-checked).
- Labels are carried in a separate `y` column and are NEVER passed as a
  feature source or into the transform; missingness-based audit flags come
  from audited base columns only.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

from src.data.prepare import build_target
from src.features.build import build_feature_frame, keep_features

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET = REPO_ROOT / "data" / "interim" / "accepted_raw.parquet"
OUT = REPO_ROOT / "data" / "interim" / "mature_36m_features.parquet"


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["data"]
    term = int(cfg["term_months"])

    con = duckdb.connect()
    cols = ", ".join(f'"{c}"' for c in ["id", "loan_status", "term", "issue_d"] + keep_features())
    df = con.execute(
        f"SELECT {cols} FROM read_parquet('{PARQUET.as_posix()}') WHERE TRIM(term) = '{term} months'"
    ).df()
    con.close()
    print(f"36-month rows pulled: {len(df):,}")

    df["issue_date"] = df["issue_d"].astype("string").str.strip()
    df["vintage"] = pd.PeriodIndex(df["issue_date"], freq="Q").astype(str)
    df["y"] = build_target(df, legacy="merge")

    features = build_feature_frame(df[keep_features() + ["issue_d"]])
    # float32 keeps the cache at a manageable size; no statistic here depends
    # on 64-bit precision (counts and labels kept separately).
    features = features.astype("float32")
    out = pd.concat(
        [df[["id", "vintage", "y"]].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"engineered features: {features.shape[0]:,} rows x {features.shape[1]} columns")
    print(f"written: {OUT}")


if __name__ == "__main__":
    import pandas as pd

    main()
