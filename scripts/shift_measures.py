"""Descriptive shift measures vs the PROPOSED training window (PLAN.md T4.2).

For every 36-month vintage quarter and every engineered T3.6 feature:
PSI (reference-quantile bins), KS statistic and standardised mean difference
for continuous features, Jensen-Shannon distance plus maximum proportion
change for one-hot categorical groups, and missingness-share differences.
Default-rate shift per vintage with a 95% interval is reported alongside.

Descriptive evidence only — no model is fit here, no window is chosen, and no
PSI threshold is cited as a law (PSI values inform the GATE4 memo).

Inputs : data/interim/mature_36m_features.parquet (T-cache; build first)
Outputs: reports/tables/shift_measures_numeric.csv
         reports/tables/shift_measures_categorical.csv
         reports/tables/default_rate_shift.csv
         reports/figures/psi_heatmap.png
         reports/figures/default_rate_shift.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.statistics.shift import (
    ks_statistic,
    psi_numeric,
    smd,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
FIGURES = REPO_ROOT / "reports" / "figures"
CACHE = REPO_ROOT / "data" / "interim" / "mature_36m_features.parquet"
META = ["id", "vintage", "y"]
EPS = 1e-6


def _js_from_proportions(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    from scipy.spatial.distance import jensenshannon

    q1 = np.where(p_ref <= 0, EPS, p_ref)
    q1 /= q1.sum()
    q2 = np.where(p_cur <= 0, EPS, p_cur)
    q2 /= q2.sum()
    d = jensenshannon(q1, q2, base=2.0)
    return 0.0 if np.isnan(d) else float(d)


def main() -> None:
    frame = pd.read_parquet(CACHE)
    frame["vintage"] = frame["vintage"].astype(str)
    feature_cols = [c for c in frame.columns if c not in META]
    # group one-hot columns by their feature prefix (T3.6 names them col_value)
    CATEGORICAL_PREFIXES = [
        "home_ownership_",
        "verification_status_",
        "purpose_",
        "application_type_",
        "initial_list_status_",
    ]
    numeric_cols: list[str] = []
    categories: dict[str, list[str]] = {}
    for c in feature_cols:
        matched = next((p for p in CATEGORICAL_PREFIXES if c.startswith(p)), None)
        if matched:
            categories.setdefault(matched.rstrip("_"), []).append(c)
        elif c not in ("cr_line_after_issue",) and not c.endswith("_isna"):
            numeric_cols.append(c)

    splits = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8"))["splits"]
    ref_start, ref_end = splits["train"]["start"], splits["train"]["end"]

    ref_mask = frame["vintage"].between(ref_start, ref_end)
    ref = frame.loc[ref_mask]
    assert len(ref) > 0, f"no rows in PROPOSED training window {ref_start}..{ref_end}"

    vintages = sorted(frame["vintage"].unique())
    print(f"reference window {ref_start}..{ref_end}: {len(ref):,} rows | vintages: {len(vintages)}")

    # ---- numeric features -------------------------------------------------
    num_rows = []
    ref_na = {c: float(ref[c].isna().mean()) for c in feature_cols}
    for v in vintages:
        cur = frame.loc[frame["vintage"] == v]
        n = len(cur)
        for c in numeric_cols:
            ref_all_na = ref[c].notna().sum() == 0
            cur_all_na = cur[c].notna().sum() == 0
            if ref_all_na or cur_all_na:
                # fully-missing far side (schema drift): report NA statistics,
                # the missingness-share columns still carry the drift signal
                num_rows.append(
                    {
                        "vintage": v,
                        "n": n,
                        "feature": c,
                        "psi": np.nan,
                        "ks_d": np.nan,
                        "smd": np.nan,
                        "ref_na_share": round(ref_na[c], 6),
                        "cur_na_share": round(float(cur[c].isna().mean()), 6),
                    }
                )
                continue
            psi = psi_numeric(ref[c], cur[c])
            ks = ks_statistic(ref[c], cur[c])
            s = smd(ref[c], cur[c])
            num_rows.append(
                {
                    "vintage": v,
                    "n": n,
                    "feature": c,
                    "psi": round(psi, 6),
                    "ks_d": round(ks, 6),
                    "smd": round(s, 6),
                    "ref_na_share": round(ref_na[c], 6),
                    "cur_na_share": round(float(cur[c].isna().mean()), 6),
                }
            )

    numeric_table = pd.DataFrame(num_rows)
    numeric_table["psi"] = numeric_table["psi"].astype(float)
    numeric_table["ks_d"] = numeric_table["ks_d"].astype(float)
    numeric_table["smd"] = numeric_table["smd"].astype(float)
    numeric_table["delta_na_share"] = (
        numeric_table["cur_na_share"] - numeric_table["ref_na_share"]
    ).round(6)

    # ---- categorical groups ----------------------------------------------
    cat_rows = []
    ref_prop = {
        g: ref[cols].mean(numeric_only=True).reindex(cols).fillna(0.0).to_numpy()
        for g, cols in categories.items()
    }
    for v in vintages:
        cur = frame.loc[frame["vintage"] == v]
        for g, cols in categories.items():
            p_cur = cur[cols].mean(numeric_only=True).reindex(cols).fillna(0.0).to_numpy()
            cat_rows.append(
                {
                    "vintage": v,
                    "n": len(cur),
                    "category": g,
                    "js_distance": round(_js_from_proportions(ref_prop[g], p_cur), 6),
                    "max_abs_prop_change": round(float(np.max(np.abs(p_cur - ref_prop[g]))), 6),
                }
            )
    cat_table = pd.DataFrame(cat_rows)

    # ---- default-rate shift (aggregate counts only) ------------------------
    ref_events = int((ref["y"] == 1).sum())
    ref_nonevents = int((ref["y"] == 0).sum())
    ref_rate = ref_events / (ref_events + ref_nonevents)
    dr_rows = []
    for v in vintages:
        sub = frame.loc[frame["vintage"] == v, "y"]
        events = int((sub == 1).sum(skipna=True))
        nonevents = int((sub == 0).sum(skipna=True))
        lo, hi = wilson_interval(events, events + nonevents)
        rate = events / (events + nonevents) if events + nonevents else np.nan
        se = np.sqrt(ref_rate * (1 - ref_rate) / (ref_events + ref_nonevents)
                     + rate * (1 - rate) / (events + nonevents)) if events + nonevents else np.nan
        diff = rate - ref_rate if events + nonevents else np.nan
        dr_rows.append(
            {
                "vintage": v,
                "n_events": events,
                "n_nonevents": nonevents,
                "default_rate": rate,
                "wilson_low": lo,
                "wilson_high": hi,
                "diff_vs_ref_window": diff,
                "diff_ci_low": diff - 1.96 * se,
                "diff_ci_high": diff + 1.96 * se,
            }
        )
    dr_table = pd.DataFrame(dr_rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    numeric_table.to_csv(TABLES / "shift_measures_numeric.csv", index=False)
    cat_table.to_csv(TABLES / "shift_measures_categorical.csv", index=False)
    dr_table.to_csv(TABLES / "default_rate_shift.csv", index=False)

    # ---- figures ----------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psi_wide = numeric_table.pivot(index="feature", columns="vintage", values="psi")
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(psi_wide.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(psi_wide.shape[1]))
    ax.set_xticklabels(psi_wide.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(psi_wide.shape[0]))
    ax.set_yticklabels(psi_wide.index, fontsize=5)
    ax.axvline(vintages.index(ref_end) + 0.5, color="w", ls="--", lw=1)
    fig.colorbar(im, ax=ax, shrink=0.7, label="PSI vs PROPOSED training window")
    ax.set_title("Per-feature PSI by vintage (white dashed = end of training window)")
    fig.tight_layout()
    fig.savefig(FIGURES / "psi_heatmap.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dr_table["vintage"], dr_table["default_rate"], marker="o", ms=3, label="default rate")
    ax.fill_between(dr_table["vintage"], dr_table["wilson_low"], dr_table["wilson_high"], alpha=0.25)
    ax.axvline(ref_start, color="tab:green", ls=":", lw=1)
    ax.axvline(ref_end, color="tab:green", ls=":", lw=1)
    ax.set_title("Default rate by vintage vs the training-window band (resolved loans)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "default_rate_shift.png", dpi=150)

    worst = numeric_table.groupby("vintage")["psi"].max()
    print(
        "max PSI per vintage (last 4): "
        + "; ".join(f"{v}={p:.2f}" for v, p in worst.tail(4).items())
    )
    print("top-5 drifted features overall:")
    print(
        numeric_table.groupby("feature")["psi"]
        .max()
        .sort_values(ascending=False)
        .head(5)
        .to_string()
    )
    print("categorical groups, max JS distance:")
    print(cat_table.groupby("category")["js_distance"].max().to_string())
    print("written: shift_measures_numeric.csv, shift_measures_categorical.csv, default_rate_shift.csv")


if __name__ == "__main__":
    main()
