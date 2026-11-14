"""Build the manuscript's numbered tables from committed result tables.

Every cell is a deterministic aggregation of an existing CSV; nothing is
read from the raw data and no value is typed by hand. The manuscript cites
these files by name, so the tables and the prose cannot drift apart.

- Table 1  reports/tables/paper_table1_cohorts.csv   (copy of the observed
           cohort table, one row per window)
- Table 2  reports/tables/paper_table2_calibration.csv  family x window
           seed-axis means of the score/calibration metrics
- Table 3  reports/tables/paper_table3_confirmatory.csv  the eight
           confirmatory tests with Holm-adjusted p-values and verdicts
- Table S1 reports/tables/paper_tableS1_anchors.csv   deep-subset anchors
- Table S2 reports/tables/paper_tableS2_recovery.csv  family x arm x lag
           mean recovery

Outputs: reports/tables/paper_table*.csv
Use    : python -m uv run python -m scripts.run_paper_tables
"""

from __future__ import annotations

import pandas as pd

from scripts._common import TABLES

METRICS = ("roc_auc", "brier", "ece_b15q", "citl", "slope", "ae_ratio")
CONFIRMATORY = ("H1a", "H1c", "H2c", "H3a", "RQ4.1", "RQ4.2", "RQ4.3", "RQ4.4")
WINDOW_ORDER = ["train", "val"] + [
    f"{y}Q{q}" for y, q in
    [(2014, 1), (2014, 2), (2014, 3), (2014, 4),
     (2015, 1), (2015, 2), (2015, 3), (2015, 4)]
]


def table2(agg: pd.DataFrame) -> pd.DataFrame:
    seed = agg[(agg["kind"] == "seed") & (agg["metric"].isin(METRICS))]
    out = (seed.groupby(["family", "window", "metric"])["value"].mean()
           .unstack("metric").reset_index())
    cols = ["family", "window"] + [m for m in METRICS if m in out.columns]
    out = out[cols]
    out["window"] = pd.Categorical(out["window"], categories=WINDOW_ORDER,
                                   ordered=True)
    return out.sort_values(["family", "window"]).reset_index(drop=True)


def table3(stats: pd.DataFrame) -> pd.DataFrame:
    sub = stats[stats["hypothesis"].isin(CONFIRMATORY)].copy()
    sub["hypothesis"] = pd.Categorical(sub["hypothesis"], categories=CONFIRMATORY,
                                       ordered=True)
    return sub.sort_values("hypothesis")[
        ["hypothesis", "scope", "statistic", "value", "ci_low", "ci_high",
         "p_raw", "p_holm", "verdict", "note"]
    ].reset_index(drop=True)


def table_s2(arms: pd.DataFrame) -> pd.DataFrame:
    cells = arms[(arms["kind"] == "seed") & (arms["arm"] != "none")
                 & (arms["metric"] == "recovery")].copy()
    cells["lag"] = cells["window"].map({
        "2014Q1": 3, "2014Q2": 4, "2014Q3": 5, "2014Q4": 6,
        "2015Q1": 7, "2015Q2": 8, "2015Q3": 9, "2015Q4": 10,
    })
    out = (cells.groupby(["family", "arm", "lag"])["value"]
           .agg(["mean", "count"]).reset_index())
    out.columns = ["family", "arm", "label_age_quarters", "mean_recovery",
                   "n_refits"]
    return out.sort_values(["family", "arm", "label_age_quarters"]).reset_index(drop=True)


def main() -> None:
    agg = pd.read_csv(TABLES / "deep_agg_real.csv")
    stats = pd.read_csv(TABLES / "real_statistics.csv")
    arms = pd.read_csv(TABLES / "deep_arms_real.csv")
    cohorts = pd.read_csv(TABLES / "real_cohorts.csv")
    anchors = pd.read_csv(TABLES / "deep_subset.csv")

    outputs = {
        "paper_table1_cohorts.csv": cohorts,
        "paper_table2_calibration.csv": table2(agg),
        "paper_table3_confirmatory.csv": table3(stats),
        "paper_tableS1_anchors.csv": anchors,
        "paper_tableS2_recovery.csv": table_s2(arms),
    }
    for name, frame in outputs.items():
        path = TABLES / name
        frame.to_csv(path, index=False)
        print(f"wrote {path.name} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
