"""Label-delay feasibility table, pre-freeze evidence (PLAN.md T4.5).

Implements the declared age-lag convention (CONTEXT.md Section 6):
- Candidate test cohorts S = quarterly vintages 2014Q1..2015Q4 (just after the
  PROPOSED validation window; >= 8 cohorts available, aim >= 6 test cohorts).
- For a recalibration window of width W = 12 months ending at S - a,
  a in {0, 6, 12, 18, 24, 36} months:
    - a = 0   arm "oracle"    (not deployable at decision time S)
    - 0<a<36  arm "optimistic" (labels not fully observable for a 36-month term)
    - a >= 36 arm "realistic"
  plus the static arm: the PROPOSED validation window (2013Q3-2013Q4).
- Eligible loans for a cell = resolved 36-month loans issued within the
  window months; events/nonevents counted on the resolved subset. A cell is
  FEASIBLE iff events >= min_events AND nonevents >= min_nonevents
  (configs/splits.yaml rule).

Descriptive counts only; windows are reported, never chosen, by this script
(labels are seen only as resolved-status counts of matured vintages).

Output: reports/tables/label_delay_feasibility.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
LOANS = REPO_ROOT / "data" / "interim" / "loans_36m.parquet"


def main() -> None:
    splits = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8"))["splits"]
    min_events = int(splits["min_events_per_eval_cohort"])
    min_nonevents = int(splits["min_nonevents_per_eval_cohort"])

    df = pd.read_parquet(LOANS)
    df["issue_month"] = pd.PeriodIndex(df["issue_date"].astype("string"), freq="M")
    resolved = df.loc[df["y"].notna(), ["issue_month", "y"]]
    # honour the PROPOSED validation window and the candidate test-cohort rule
    val = splits["validation"]

    test_start, test_end = "2014Q1", "2015Q4"
    quarters = pd.period_range(test_start, test_end, freq="Q")
    lags = [0, 6, 12, 18, 24, 36]

    rows = []

    def _cell(cohort: str, lag: int, arm: str, m_end: pd.Period) -> dict:
        m_start = m_end - 11
        sel = resolved[(resolved["issue_month"] >= m_start) & (resolved["issue_month"] <= m_end)]
        events = int((sel["y"] == 1).sum())
        nonevents = int((sel["y"] == 0).sum())
        feasible = "FEASIBLE" if (events >= min_events and nonevents >= min_nonevents) else "INFEASIBLE"
        return {
            "eval_cohort": cohort,
            "lag_months": lag,
            "arm": arm,
            "window_start": str(m_start),
            "window_end": str(m_end),
            "n_eligible": len(sel),
            "n_events": events,
            "n_nonevents": nonevents,
            "min_events_rule": min_events,
            "min_nonevents_rule": min_nonevents,
            "status": feasible,
        }

    for cohort in quarters:
        cohort_str = str(cohort)
        start_month = cohort.start_time.to_period("M")
        for lag in lags:
            m_end = start_month - lag
            arm = "oracle" if lag == 0 else ("optimistic" if lag < 36 else "realistic")
            rows.append(_cell(cohort_str, lag, arm, m_end))

    # static arm: validation window itself
    v_resolved = resolved[
        (resolved["issue_month"] >= pd.Period(val["start"], freq="M"))
        & (resolved["issue_month"] <= pd.Period(val["end"], freq="M"))
    ]
    rows.append(
        {
            "eval_cohort": "static",
            "lag_months": pd.NA,
            "arm": "static (validation window)",
            "window_start": str(val["start"]),
            "window_end": str(val["end"]),
            "n_eligible": len(v_resolved),
            "n_events": int((v_resolved["y"] == 1).sum()),
            "n_nonevents": int((v_resolved["y"] == 0).sum()),
            "min_events_rule": min_events,
            "min_nonevents_rule": min_nonevents,
            "status": "FEASIBLE"
            if ((v_resolved["y"] == 1).sum() >= min_events and (v_resolved["y"] == 0).sum() >= min_nonevents)
            else "INFEASIBLE",
        }
    )

    out = pd.DataFrame(rows)
    table_cols = [
        "eval_cohort","lag_months","arm","window_start","window_end",
        "n_eligible","n_events","n_nonevents","min_events_rule","min_nonevents_rule","status",
    ]
    out = out[table_cols]
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "label_delay_feasibility.csv"
    out.to_csv(path, index=False)

    n_feasible_realistic = int(((out["arm"] == "realistic") & (out["status"] == "FEASIBLE")).sum())
    n_realistic = int((out["arm"] == "realistic").sum())
    print(f"cells: {len(out)} | realistic feasible: {n_feasible_realistic}/{n_realistic} | written: {path}")
    print(out.to_string(index=False))
    assert list(out.columns[:1]) == ["eval_cohort"] and len(table_cols) == 11


if __name__ == "__main__":
    main()
