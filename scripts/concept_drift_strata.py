"""Concept-drift diagnostics that need no final model (PLAN.md T4.4).

Default rates per feature-stratum by vintage: if the *conditional* default
rate within a fixed stratum moves across vintages, that is evidence of
concept drift beyond simple covariate shift (the stratum as such can be
explained by the feature-shift tables themselves; T4.2 covers those).

Strata are quantile bins derived from the PROPOSED training window only
(deterministic, reference-derived, no leakage): terciles of fico_range_low,
dti, and loan_amnt. Labels enter only as aggregate counts by stratum and
vintage (resolved 36-month loans); this matches the prior use in
protocol/GATE3.md Section 3. No model is fit; performance changes are never
interpreted alone as concept drift.

Inputs : data/interim/mature_36m_features.parquet
Outputs: reports/tables/concept_drift_strata.csv
         reports/figures/concept_drift_strata.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.statistics.shift import wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
FIGURES = REPO_ROOT / "reports" / "figures"
CACHE = REPO_ROOT / "data" / "interim" / "mature_36m_features.parquet"
META = ["id", "vintage", "y"]

STRATA_FEATURES = ["fico_range_low", "dti", "loan_amnt"]


def main() -> None:
    frame = pd.read_parquet(CACHE)
    frame["vintage"] = frame["vintage"].astype(str)

    splits = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8"))["splits"]
    ref_start, ref_end = splits["train"]["start"], splits["train"]["end"]
    ref = frame.loc[frame["vintage"].between(ref_start, ref_end)]

    rows = []
    vintages = sorted(frame["vintage"].unique())
    for feature in STRATA_FEATURES:
        # stratum edges come from the reference sample (trainable quantiles
        # belong to the training window only)
        edges = np.unique(
            np.sort(ref[feature].dropna().quantile([1/3, 2/3]).to_numpy(dtype=float))
        )
        for v in vintages:
            sub = frame.loc[frame["vintage"] == v, [feature, "y"]]
            stratum_codes = pd.cut(sub[feature], bins=np.concatenate(([-np.inf], edges, [np.inf])), right=False)
            for level, idx in sub.groupby(stratum_codes, observed=True).indices.items():
                ys = sub["y"].iloc[idx].dropna()
                events = int((ys == 1).sum())
                nonevents = int((ys == 0).sum())
                lo, hi = wilson_interval(events, events + nonevents)
                rows.append(
                    {
                        "vintage": v,
                        "feature": feature,
                        "stratum": str(level),
                        "n_resolved": events + nonevents,
                        "n_events": events,
                        "n_nonevents": nonevents,
                        "default_rate": events / (events + nonevents) if events + nonevents else np.nan,
                        "wilson_low": lo,
                        "wilson_high": hi,
                    }
                )

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    fig_root = TABLES / "concept_drift_strata.csv"
    table.to_csv(fig_root, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(STRATA_FEATURES), figsize=(6 * len(STRATA_FEATURES), 4.2), sharey=False)
    for ax, feature in zip(axes, STRATA_FEATURES):
        wide = table[table["feature"] == feature].pivot(index="stratum", columns="vintage", values="default_rate")
        im = ax.imshow(wide.to_numpy(dtype=float), aspect="auto", cmap="plasma")
        ax.set_xticks(range(wide.shape[1]))
        ax.set_xticklabels(wide.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(wide.shape[0]))
        ax.set_yticklabels(wide.index, fontsize=7)
        ax.set_title(f"resolved default rate: {feature} terciles", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "concept_drift_strata.png", dpi=150)

    print("strata cells:", len(table))
    print(
        "max within-stratum rate range (feature, stratum, spread):"
    )
    spread = table.groupby(["feature", "stratum"])["default_rate"].agg(lambda s: s.max() - s.min())
    print(spread.sort_values(ascending=False).head(8).to_string())
    print(f"written: {fig_root} and figures/concept_drift_strata.png")


if __name__ == "__main__":
    main()
