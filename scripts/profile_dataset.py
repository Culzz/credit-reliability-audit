"""Profile the converted dataset (PLAN.md T2.4).

Writes small, committed summary tables under reports/tables/:
  data_profile.csv          dataset-level metrics
  column_missingness.csv    missing share per column (descending)
  loan_status_counts.csv    raw loan_status distribution
  issue_terms.csv           term representation
  numeric_ranges.csv        min/max/count for key numeric columns

Everything runs through DuckDB SQL against the Parquet file, so no step
materialises the full table in memory.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

# repo root for a file directly inside scripts/ : scripts -> repo
REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
NUMERIC_CHECKS = ["loan_amnt", "funded_amnt", "annual_inc", "dti", "revol_bal", "revol_util", "delinq_2yrs"]


def main() -> None:
    parquet = REPO_ROOT / "data" / "interim" / "accepted_raw.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} missing; run `python -m src.data.load_raw` first.")
    t = f"read_parquet('{parquet.as_posix()}')"
    con = duckdb.connect()

    cols = _columns(con, t)
    n_rows, dupe_ids, n_issue_months, earliest, latest = con.execute(
        f"""
        SELECT count(*) AS n_rows,
               (SELECT count(*) FROM (SELECT id FROM {t} GROUP BY id HAVING count(*) > 1)) AS dup_ids,
               (SELECT count(DISTINCT strptime(issue_d, '%b-%Y')) FROM {t}) AS n_issue_months,
               min(strptime(issue_d, '%b-%Y')) AS earliest, max(strptime(issue_d, '%b-%Y')) AS latest
        FROM {t}
        """
    ).fetchone()
    status = con.execute(
        f"SELECT loan_status, count(*) AS n FROM {t} GROUP BY loan_status ORDER BY n DESC"
    ).df()
    term = con.execute(f"SELECT term, count(*) AS n FROM {t} GROUP BY term ORDER BY n DESC").df()
    miss = _missing(con, t, cols)
    ranges = _ranges(con, t, cols)
    con.close()

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "metric": ["n_rows", "n_columns", "duplicate_ids", "distinct_issue_months",
                       "earliest_issue", "latest_issue"],
            "value": [int(n_rows), len(cols), int(dupe_ids), int(n_issue_months), earliest, latest],
        }
    ).to_csv(TABLES / "data_profile.csv", index=False)
    miss.to_csv(TABLES / "column_missingness.csv", index=False)
    status.to_csv(TABLES / "loan_status_counts.csv", index=False)
    term.to_csv(TABLES / "issue_terms.csv", index=False)
    ranges.to_csv(TABLES / "numeric_ranges.csv", index=False)

    print(f"rows x columns: {int(n_rows):,} x {len(cols)} (duplicate ids: {int(dupe_ids)})")
    print(f"issue dates:    {earliest} ... {latest} ({int(n_issue_months)} distinct months)")
    print("\nloan_status counts:")
    print(status.to_string(index=False))
    print("\nterm counts:")
    print(term.to_string(index=False))
    print("\nmost-missing columns (top 10):")
    print(miss.head(10).to_string(index=False))


def _columns(con: duckdb.DuckDBPyConnection, t: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {t}").fetchall()]


def _missing(con: duckdb.DuckDBPyConnection, t: str, cols: list[str]) -> pd.DataFrame:
    parts = [
        f"SELECT '{c}' AS column_name, 1.0 * avg(CASE WHEN \"{c}\" IS NULL THEN 1.0 ELSE 0.0 END) AS missing_share FROM {t}"
        for c in cols
    ]
    df = con.execute(" UNION ALL ".join(parts)).df()
    df["missing_pct"] = (100.0 * df["missing_share"]).round(4)
    df["missing_share"] = df["missing_share"].round(6)
    return (
        df[["column_name", "missing_share", "missing_pct"]]
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )


def _ranges(con: duckdb.DuckDBPyConnection, t: str, cols: list[str]) -> pd.DataFrame:
    present = [c for c in NUMERIC_CHECKS if c in cols]
    parts = [
        f"SELECT '{c}' AS column_name, min(\"{c}\") AS min_value, max(\"{c}\") AS max_value, count(\"{c}\") AS n_nonmissing FROM {t}"
        for c in present
    ]
    return con.execute(" UNION ALL ".join(parts)).df()


if __name__ == "__main__":
    main()
