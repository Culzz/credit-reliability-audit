"""Cohort feasibility table, pre-Gate-4 evidence (PLAN.md T4.1).

Per mature 36-month vintage: loan volume, resolution status, event and
non-event counts, default rate, and the minimum-events rule check for using
that vintage as an evaluation cohort (>= 200 events and >= 200 non-events,
configs/splits.yaml). Descriptive evidence only — no model is fit, no window
is chosen here. The Gate 4 window choice cites this table.

Outputs:
  reports/tables/cohort_feasibility.csv
  reports/figures/cohort_volume_by_vintage.png
  reports/figures/default_rate_by_vintage.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.statistics.shift import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
FIGURES = REPO_ROOT / "reports" / "figures"
LOANS = REPO_ROOT / "data" / "interim" / "loans_36m.parquet"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8"))["splits"]
    min_events = int(splits["min_events_per_eval_cohort"])
    min_nonevents = int(splits["min_nonevents_per_eval_cohort"])

    df = pd.read_parquet(LOANS)
    resolved = df["y"].notna()
    events = (df["y"] == 1).fillna(False)
    nonevents = (df["y"] == 0).fillna(False)

    g = df.groupby("vintage", observed=True)
    table = pd.DataFrame(
        {
            "n_loans": g.size(),
            "n_resolved": resolved.groupby(df["vintage"], observed=True).sum().astype("int64"),
            "n_events": events.groupby(df["vintage"], observed=True).sum().astype("int64"),
            "n_nonevents": nonevents.groupby(df["vintage"], observed=True).sum().astype("int64"),
        }
    )
    table["n_unresolved"] = table["n_loans"] - table["n_resolved"]
    table["default_rate"] = table["n_events"] / table["n_resolved"].replace(0, pd.NA)

    ci = [wilson_interval(int(r.n_events), int(r.n_resolved)) for r in table.itertuples()]
    table["default_rate_ci_low"] = [lo for lo, _ in ci]
    table["default_rate_ci_high"] = [hi for _, hi in ci]

    table["meets_min_events"] = (table["n_events"] >= min_events).map({True: "PASS", False: "FAIL"})
    table["meets_min_nonevents"] = (table["n_nonevents"] >= min_nonevents).map({True: "PASS", False: "FAIL"})
    table["evaluable_as_cohort"] = [f"{a}-{b}" for a, b in zip(table["meets_min_events"], table["meets_min_nonevents"])]

    table = table.reset_index().rename(columns={"index": "vintage"})

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_csv = TABLES / "cohort_feasibility.csv"
    table.to_csv(out_csv, index=False)

    evaluable = table[(table["meets_min_events"] == "PASS") & (table["meets_min_nonevents"] == "PASS")]
    print(f"vintages: {len(table)} | passing both rules: {len(evaluable)}")
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(table["vintage"], table["n_loans"], marker="o", ms=3)
    ax.axvline("2012Q1", color="tab:blue", ls="--", lw=1, label="PROPOSED train start 2012Q1")
    ax.set_title("36-month loan volume by vintage quarter")
    ax.set_ylabel("loans")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "cohort_volume_by_vintage.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(table["vintage"], table["default_rate"], marker="o", ms=3, label="default rate (resolved)")
    ax.fill_between(table["vintage"], table["default_rate_ci_low"], table["default_rate_ci_high"], alpha=0.25)
    ax.set_title("Default rate by vintage quarter (resolved 36-month loans, Wilson 95% CI)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "default_rate_by_vintage.png", dpi=150)

    print(f"written: {out_csv} and two figures in {FIGURES}")


if __name__ == "__main__":
    main()
