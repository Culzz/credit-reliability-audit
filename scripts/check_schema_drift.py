"""Schema-drift check over the sample (PLAN.md T3.7).

For every KEEP feature (data/column_audit.csv) and every quarterly vintage in
the PROPOSED maturity window, computes the populated share on 36-month loans
and applies configs/data.yaml `min_populated_share` (=0.90). Any feature with
even one failing vintage is dropped from the main feature set — the drop list
and reasons are written to reports/tables/schema_drift.csv.

The rule is deliberately mechanical: fields whose blankness is event-absence
(months-since-event) fail it wherever they are sparse; the alternative
(event-absence recoding) is a PI decision, not silently applied here.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml

from src.features.build import keep_features

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))["data"]
    thr = float(cfg["min_populated_share"])
    window = [pd.Period(v, freq="Q") for v in cfg["maturity_window"]]
    term = int(cfg["term_months"])

    parquet = (REPO_ROOT / "data" / "interim" / "accepted_raw.parquet").as_posix()
    feats = keep_features()
    con = duckdb.connect()
    cols_sql = ", ".join(f'"{c}"' for c in feats + ["issue_d", "term"])
    df = con.execute(
        f"""
        SELECT {cols_sql}
        FROM read_parquet('{parquet}')
        WHERE TRIM(term) = '{term} months'
        """
    ).df()
    con.close()

    df["vintage"] = pd.to_datetime(df["issue_d"].str.strip(), format="%b-%Y").dt.to_period("Q").astype(str)
    in_window = df["vintage"].between(str(window[0]), str(window[1]))
    df = df.loc[in_window, feats + ["vintage"]].reset_index(drop=True)

    shares = df.groupby("vintage")[feats].agg(lambda s: s.notna().mean())
    rows = []
    for f in feats:
        s = shares[f]
        failing = s[s < thr]
        rows.append(
            {
                "feature": f,
                "min_populated_share": round(float(s.min()), 6),
                "max_populated_share": round(float(s.max()), 6),
                "n_failing_vintages": len(failing),
                "failing_vintages": ";".join(str(v) for v in failing.index),
                "decision": "DROP" if len(failing) else "KEEP",
                "reason": (
                    f"populated below {thr} in {len(failing)} vintage(s) of the maturity window"
                    if len(failing)
                    else f"populated >= {thr} in every vintage of the window"
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "schema_drift.csv"
    out.to_csv(path, index=False)

    dropped = out.query("decision == 'DROP'")["feature"].tolist()
    print(f"features checked: {len(feats)} | dropped by rule: {len(dropped)}")
    print(out.to_string(index=False))

    # T3.6 anomaly check on real data: negative credit ages (cr line after issue).
    con = duckdb.connect()
    neg = con.execute(
        f"""
        SELECT count(*) FILTER (WHERE strptime(earliest_cr_line, '%b-%Y') > strptime(issue_d, '%b-%Y')) AS neg,
               count(*) AS total_redo
        FROM read_parquet('{parquet}')
        WHERE TRIM(term) = '{term} months'
        """
    ).fetchone()
    con.close()
    print(f"negative credit ages (earliest_cr_line > issue_d): {neg[0]} of {neg[1]}")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
