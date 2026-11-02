"""Vintage maturity audit (PLAN.md T3.4).

Filters to 36-month loans (T3.2), constructs the target (T3.3), and computes
the per-vintage maturity table: totals, resolved outcome counts, unresolved
share U_q, default rate, and the mature flag (issue + 36 months <= snapshot,
and U_q below the PROPOSED threshold in configs/data.yaml).

Outputs:
  data/interim/loans_36m.parquet      one row per 36-month loan with y + vintage
  reports/tables/vintage_maturity.csv the per-vintage feasibility evidence
The run is logged to the data-feasibility MLflow experiment.
"""

from __future__ import annotations

import duckdb
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.data.prepare import build_target, filter_term_months, parse_issue_date

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["data"]
    parquet = REPO_ROOT / "data" / "interim" / "accepted_raw.parquet"
    con = duckdb.connect()
    df = con.execute(
        f"SELECT id, loan_status, term, issue_d FROM read_parquet('{parquet.as_posix()}')"
    ).df()
    con.close()

    n_all = len(df)
    df = filter_term_months(df, months=int(cfg["term_months"]))
    df["issue_date"] = parse_issue_date(df["issue_d"])
    df["vintage"] = df["issue_date"].dt.to_period("Q").astype(str)
    df["y"] = build_target(df["loan_status"].to_frame(), legacy="merge")
    legacy_merged = df["loan_status"].str.startswith("Does not meet the credit policy", na=False)

    status = df["loan_status"].str.strip()
    unresolved = status.isin(cfg["target"]["excluded_unresolved"])

    (
        df.assign(y=df["y"].astype("Int64"), legacy_merged=legacy_merged)
        .loc[:, ["id", "issue_date", "vintage", "loan_status", "y", "legacy_merged"]]
        .to_parquet(REPO_ROOT / "data" / "interim" / "loans_36m.parquet", index=False)
    )

    g = df.assign(n=1, unresolved=unresolved, legacy=legacy_merged)
    maturity = (
        g.groupby("vintage")
        .agg(
            n_total=("n", "sum"),
            n_fully_paid=("y", lambda s: int((s == 0).sum())),
            n_default=("y", lambda s: int((s == 1).sum())),
            n_unresolved=("unresolved", "sum"),
            n_legacy_merged=("legacy", "sum"),
        )
        .reset_index()
    )
    maturity["n_resolved"] = maturity["n_fully_paid"] + maturity["n_default"]
    maturity["unresolved_share"] = (maturity["n_unresolved"] / maturity["n_total"]).round(6)
    maturity["default_rate_resolved"] = (
        maturity["n_default"] / maturity["n_resolved"].replace(0, pd.NA)
    ).astype("Float64").round(6)

    snapshot = pd.Timestamp(cfg["snapshot_date"]) + pd.offsets.MonthEnd(0)
    latest_issue = df.groupby("vintage")["issue_date"].max()
    mature_window = latest_issue + pd.DateOffset(months=int(cfg["term_months"])) <= snapshot
    maturity["mature_by_window"] = maturity["vintage"].isin(mature_window[mature_window].index)
    maturity["mature"] = maturity["mature_by_window"] & (
        maturity["unresolved_share"] < float(cfg["max_unresolved_share"])
    )
    maturity["mature"] = maturity["mature"].astype(bool)
    maturity = maturity.sort_values("vintage").reset_index(drop=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "vintage_maturity.csv"
    maturity.to_csv(out, index=False)

    print(f"36-month loans: {len(df):,} of {n_all:,}")
    print(f"resolved: {int(maturity['n_resolved'].sum()):,} | unresolved: {int(maturity['n_unresolved'].sum()):,}")
    print(maturity.to_string(index=False))
    print(f"\nwritten: {out} and data/interim/loans_36m.parquet")

    client = MlflowClient(tracking_uri=f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(client.tracking_uri)
    mlflow.set_experiment("data-feasibility")
    with mlflow.start_run(run_name="vintage-maturity-36m"):
        mlflow.log_metrics(
            {
                "n_loans_36m": len(df),
                "n_resolved": int(maturity["n_resolved"].sum()),
                "n_unresolved": int(maturity["n_unresolved"].sum()),
                "n_mature_vintages": int(maturity["mature"].sum()),
            }
        )
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
