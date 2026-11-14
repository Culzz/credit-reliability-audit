"""Paper figures from the frozen real-cohort battery (post-freeze).

Every panel is a deterministic read of a committed table; nothing is
recomputed from raw data and no value is typed by hand. Figures:

- sweep_auc_brier_real.png       validation AUC vs Brier, 30 configs/family
- citl_by_cohort_real.png        mean CITL by evaluation window (seed axis)
- ece_by_cohort_real.png         mean adaptive-bin ECE by window
- recovery_by_lag_real.png       recalibration recovery by arm and lag
- flip_rate_by_cohort_real.png   seed-pair flip rate vs the seed floor q95
- explanation_stability_real.png adjacent-cohort Jaccard vs noise floor q2.5
- rq4_delta_real.png             RQ4 delta-rho forest plot with 95% CIs

Outputs: reports/figures/*_real.png
Use    : python -m uv run python -m scripts.run_real_figures
"""

from __future__ import annotations

import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._common import REPO_ROOT, TABLES

FIGURES = REPO_ROOT / "reports" / "figures"
COHORTS = ["val"] + [f"{y}Q{q}" for y, q in
                     [(2014, 1), (2014, 2), (2014, 3), (2014, 4),
                      (2015, 1), (2015, 2), (2015, 3), (2015, 4)]]
FAMILY_COLORS = {
    "logistic_regression": "#1f77b4",
    "xgboost": "#ff7f0e",
    "lightgbm": "#2ca02c",
    "random_forest": "#d62728",
}


def _save(fig, name: str) -> pathlib.Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_sweep() -> pathlib.Path:
    sweep = pd.read_csv(TABLES / "sweep.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for family, sub in sweep.groupby("family"):
        ax.scatter(sub["auc"], sub["brier"], s=16, alpha=0.65,
                   label=family, color=FAMILY_COLORS.get(family))
    ax.set_xlabel("validation AUC")
    ax.set_ylabel("validation Brier")
    ax.set_title("Broad-sweep configs on the frozen training window")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "sweep_auc_brier_real.png")


def _mean_by_window(metric: str) -> pd.DataFrame:
    agg = pd.read_csv(TABLES / "deep_agg_real.csv")
    sub = agg[(agg["kind"] == "seed") & (agg["metric"] == metric)]
    out = (sub.groupby(["family", "window"])["value"].mean()
           .unstack("window"))
    order = [w for w in COHORTS if w in out.columns]
    return out[order]


def render_citl() -> pathlib.Path:
    table = _mean_by_window("citl")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(table.columns))
    for family, row in table.iterrows():
        ax.plot(x, row.to_numpy(), marker="o", ms=3.5, lw=1.4,
                label=family, color=FAMILY_COLORS.get(family))
    ax.axhline(0.0, color="#666666", lw=0.9, ls=":")
    ax.set_xticks(x, table.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("calibration-in-the-large (offset GLM)")
    ax.set_title("CITL by evaluation window (seed-refit mean)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "citl_by_cohort_real.png")


def render_ece() -> pathlib.Path:
    table = _mean_by_window("ece_b15q")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(table.columns))
    for family, row in table.iterrows():
        ax.plot(x, row.to_numpy(), marker="o", ms=3.5, lw=1.4,
                label=family, color=FAMILY_COLORS.get(family))
    ax.set_xticks(x, table.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ECE (adaptive equal-mass bins, B=15)")
    ax.set_title("Calibration error by evaluation window (seed-refit mean)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "ece_by_cohort_real.png")


def render_recovery() -> pathlib.Path:
    arms = pd.read_csv(TABLES / "deep_arms_real.csv")
    sub = arms[(arms["kind"] == "seed") & (arms["arm"] != "none")
               & (arms["metric"].isin(["brier_before", "brier_after"]))]
    wide = (sub.pivot_table(index=["family", "config_index", "arm", "window"],
                            columns="metric", values="value")
            .reset_index())
    wide["dbrier"] = wide["brier_after"] - wide["brier_before"]
    lags = {"2014Q1": 3, "2014Q2": 4, "2014Q3": 5, "2014Q4": 6,
            "2015Q1": 7, "2015Q2": 8, "2015Q3": 9, "2015Q4": 10}
    wide["lag"] = wide["window"].map(lags)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), sharey=True)
    for ax, arm in zip(axes, ("intercept", "platt", "isotonic")):
        for family, fam_sub in wide[wide["arm"] == arm].groupby("family"):
            mean = fam_sub.groupby("lag")["dbrier"].mean().sort_index()
            ax.plot(mean.index, mean.to_numpy(), marker="o", ms=4, lw=1.4,
                    label=family, color=FAMILY_COLORS.get(family))
        ax.axhline(0.0, color="#666666", lw=0.9, ls=":")
        ax.set_title(arm, fontsize=10)
        ax.set_xlabel("label age (quarters)")
    axes[0].set_ylabel("mean Brier change (after - before)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Deployable recalibration: Brier change vs uncalibrated, by label age",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, "recovery_by_lag_real.png")


def render_flips() -> pathlib.Path:
    flips = pd.read_csv(TABLES / "flip_pairs_real.csv")
    sub = flips[(flips["kind"] == "seed") & (flips["window"] != "train")]
    mean = (sub.groupby(["family", "window"])["flip_rate"].mean()
            .unstack("window"))
    order = [w for w in COHORTS if w in mean.columns]
    mean = mean[order]
    floors = pd.read_csv(TABLES / "real_noise_floors.csv")
    floors = floors[floors["dimension"] == "flip"].set_index("family")["q95"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(order))
    for family, row in mean.iterrows():
        ax.plot(x, row.to_numpy(), marker="o", ms=3.5, lw=1.4,
                label=family, color=FAMILY_COLORS.get(family))
        if family in floors.index:
            ax.axhline(float(floors[family]), color=FAMILY_COLORS.get(family),
                       lw=0.8, ls=":", alpha=0.7)
    ax.set_xticks(x, order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("seed-pair decision flip rate")
    ax.set_title("Decision stability at the policy cell (dotted = seed floor q95)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "flip_rate_by_cohort_real.png")


def render_explanation_stability() -> pathlib.Path:
    temp = pd.read_csv(TABLES / "temporal_pairs_real.csv")
    sub = temp[temp["metric"] == "jaccard_top10"]
    pairs = list(itertools.pairwise(COHORTS))
    pairs = [(a, b) for a, b in pairs
             if a in set(sub["window_i"]) and b in set(sub["window_j"])]
    labels = [f"{a}->{b}" for a, b in pairs]
    floors = pd.read_csv(TABLES / "real_noise_floors.csv")
    floors = floors[(floors["dimension"] == "stability")
                    & (floors["label"] == "jaccard_top10")].set_index("family")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.2
    families = sorted(sub["family"].unique())
    for pos, family in enumerate(families):
        fsub = sub[sub["family"] == family]
        vals = []
        for a, b in pairs:
            sel = fsub[(fsub.window_i == a) & (fsub.window_j == b)]["value"]
            vals.append(sel.mean() if len(sel) else np.nan)
        x = np.arange(len(pairs)) + (pos - 1.5) * width
        ax.bar(x, vals, width=width, label=family,
               color=FAMILY_COLORS.get(family), alpha=0.8)
        if family in floors.index:
            ax.axhline(float(floors.loc[family, "q2.5"]),
                       color=FAMILY_COLORS.get(family), lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(pairs)), labels, rotation=45, ha="right",
                  fontsize=8)
    ax.set_ylabel("Jaccard@10 of cohort attributions")
    ax.set_title("Temporal explanation similarity vs seed/bootstrap floor (dotted)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, "explanation_stability_real.png")


def render_rq4() -> pathlib.Path:
    rq = pd.read_csv(TABLES / "rq4_cells_real.csv")
    labels = {
        "ece": "AUC -> ECE",
        "slope_dev": "AUC -> |slope-1|",
        "expl_stability": "AUC -> explanation stability",
        "flip_rate": "AUC -> flip rate",
    }
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(rq))
    ax.errorbar(rq["delta"], y,
                xerr=[rq["delta"] - rq["ci_low"],
                      rq["ci_high"] - rq["delta"]],
                fmt="o", color="#333333", ecolor="#777777", capsize=3, ms=5)
    ax.axvline(0.0, color="#999999", lw=0.9, ls=":")
    ax.set_yticks(y, [labels.get(k, k) for k in rq["key"]], fontsize=9)
    ax.set_xlabel("delta Spearman rho vs the AUC->AUC baseline (95% CI)")
    ax.set_title("Does AUC predict the reliability outcomes?")
    fig.tight_layout()
    return _save(fig, "rq4_delta_real.png")


def main() -> None:
    renderers = [
        ("sweep.csv", render_sweep),
        ("deep_agg_real.csv", render_citl),
        ("deep_agg_real.csv", render_ece),
        ("recovery_cells_real.csv", render_recovery),
        ("flip_pairs_real.csv", render_flips),
        ("temporal_pairs_real.csv", render_explanation_stability),
        ("rq4_cells_real.csv", render_rq4),
    ]
    for required, renderer in renderers:
        if not (TABLES / required).exists():
            print(f"skip {renderer.__name__}: missing {required}")
            continue
        out = renderer()
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
