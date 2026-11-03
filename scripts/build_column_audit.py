"""Column audit statistics (PLAN.md T3.5, first pass).

For every column of the raw Parquet: dtype, overall missing share, and the
minimum/maximum missing share across quarterly vintages (schema-drift signal
for T3.7). Judgment fields — availability at origination, leakage risk, and
the keep/drop/split decision — are set to TO_DECIDE here and filled against
the data dictionary in the review pass (T3.5 second run), never guessed.

Output: data/column_audit.csv (committed; statistics only, no row counts
beyond shares).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PARQUET = REPO_ROOT / "data" / "interim" / "accepted_raw.parquet"


def main() -> None:
    parquet = PARQUET.as_posix()
    con = duckdb.connect()

    types = {
        r[0]: r[1] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet}')").fetchall()
    }
    cols = [c for c in types if c != "id"]
    out: list[dict] = []
    for c in cols:
        shares = con.execute(
            f"""
            WITH q AS (
                SELECT strptime(issue_d, '%b-%Y') AS d, "{c}" AS v
                FROM read_parquet('{parquet}')
            ), grouped AS (
                SELECT d, avg(CASE WHEN v IS NULL THEN 1.0 ELSE 0.0 END) AS avg_v FROM q GROUP BY d
            )
            SELECT (SELECT avg(CASE WHEN v IS NULL THEN 1.0 ELSE 0.0 END) FROM q) AS overall,
                   (SELECT min(avg_v) FROM grouped) AS mn,
                   (SELECT max(avg_v) FROM grouped) AS mx
            """
        ).fetchone()
        out.append(
            {
                "column": c,
                "dtype": types[c],
                "missing_share": round(float(shares[0]), 6),
                "min_vintage_missing_share": round(float(shares[1]), 6),
                "max_vintage_missing_share": round(float(shares[2]), 6),
                "available_at_origination": "TO_DECIDE",
                "availability_evidence": "TO_DECIDE",
                "leakage_risk": "TO_DECIDE",
                "decision": "TO_DECIDE",
                "reason": "TO_DECIDE",
            }
        )
    df = pd.DataFrame(out).sort_values("column").reset_index(drop=True)

    audit = REPO_ROOT / "data" / "column_audit.csv"
    audit.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(audit, index=False)
    con.close()

    print(f"columns audited: {len(df)} -> {audit.name}")
    print(df.loc[:, ["column", "missing_share", "max_vintage_missing_share"]]
          .query("missing_share > 0.99").to_string(index=False))


if __name__ == "__main__":
    main()
