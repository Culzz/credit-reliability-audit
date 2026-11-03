"""Plumbing sample for pre-Gate-5 development (PLAN.md T3.9).

A fixed-seed 3% sample stratified by vintage quarter of all 36-month loans
issued in the configured development years (2012-2013, PROPOSED). Every
pipeline development run before Gate 5 uses only this sample; the mature
cohort full sample stays untouched until the protocol freezes.

Outputs:
  data/interim/plumbing_sample.parquet  (gitignored) sampled rows with target
  reports/tables/plumbing_sample.csv    (committed) per-vintage counts
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

from src.data.prepare import build_target
from src.data.prepare import parse_issue_date as _parse_issue_date
from src.features.build import keep_features

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["data"]
    plumb = cfg["plumbing_sample"]
    years = [int(y) for y in plumb["vintage_years"]]
    share = float(plumb["share"])
    seed = int(plumb["seed"])
    term = int(cfg["term_months"])

    parquet = (REPO_ROOT / "data" / "interim" / "accepted_raw.parquet").as_posix()
    con = duckdb.connect()
    cols_sql = ", ".join(f'"{c}"' for c in ["id", "loan_status", "term", "issue_d"] + keep_features())
    df = con.execute(
        f"""
        SELECT {cols_sql}
        FROM read_parquet('{parquet}')
        WHERE TRIM(term) = '{term} months'
        """
    ).df()
    con.close()

    df["issue_date"] = _parse_issue_date(df["issue_d"])
    df["vintage"] = df["issue_date"].dt.to_period("Q").astype(str)
    df = df[df["vintage"].str[:4].isin(map(str, years))].reset_index(drop=True)
    df["y"] = build_target(df, legacy="merge")

    sampled = (
        df.groupby("vintage", group_keys=False)
        .apply(lambda g: g.sample(frac=share, random_state=seed), include_groups=False)
        .reset_index()
    )
    sampled = df.loc[sampled["index"]].copy() if "index" in sampled.columns else df.loc[sampled.index]

    agg = df.groupby("vintage").agg(
        n_total=("id", "size"),
        n_resolved=("y", lambda s: int(s.notna().sum())),
    )
    agg["n_unresolved"] = agg["n_total"] - agg["n_resolved"]
    agg["n_sampled"] = sampled.groupby("vintage").size()
    counts = agg.reset_index()[["vintage", "n_total", "n_resolved", "n_unresolved", "n_sampled"]]

    out_parquet = REPO_ROOT / "data" / "interim" / "plumbing_sample.parquet"
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(out_parquet, index=False)

    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "plumbing_sample.csv"
    counts.to_csv(path, index=False)

    print(f"development vintages: {years} | share {share:.0%} | seed {seed}")
    print(f"rows in universe: {len(df):,} | sampled: {len(sampled):,}")
    print(counts.to_string(index=False))
    print(f"resolved in sample: {int(sampled['y'].notna().sum()):,}")
    print(f"written: {out_parquet} and {path}")

    # Smoke check: the sample must support full pipeline development (T3.9).
    # Metadata columns (id, y, vintage, ...) are deliberately not passed into
    # the transform — the audit guard should never see them as features.
    from src.features.build import build_feature_frame

    feats_frame = build_feature_frame(sampled[keep_features() + ["issue_d"]])
    print(
        f"smoke transform: {feats_frame.shape[0]:,} rows x "
        f"{feats_frame.shape[1]} engineered features, all audited sources"
    )


if __name__ == "__main__":
    main()
