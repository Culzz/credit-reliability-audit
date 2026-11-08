"""Render the dry-run paper-figure preforms from committed tables.

Two figures, both deterministic (no timestamps, fixed axes, dpi=150):

- reliability_diagram_dryrun.png: grid families x calibration arms,
  per-arm reliability-bin segments (equal-mass bins, the same binning
  the primary ece_b15 summarises) with the diagonal as perfect
  calibration;
- sweep_auc_brier_dryrun.png: validation-AUC vs validation-Brier per
  sampled config across the broad sweep, one marker per family - the
  plumbing-scale picture of how much calibration room the search
  spaces leave (RQ4's association study needs exactly that spread).

These are plumbing-scale artifacts under the locked test window; none
of them is citable as a result.

Outputs: reports/figures/reliability_diagram_dryrun.png
         reports/figures/sweep_auc_brier_dryrun.png
Use    : python -m uv run python -m scripts.run_figures
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: no display, no GUI state
import matplotlib.pyplot as plt
import pandas as pd

from scripts._common import REPO_ROOT, TABLES

FIGURES = REPO_ROOT / "reports" / "figures"

ARM_ORDER = ["none", "intercept", "platt", "isotonic"]
ARM_COLORS = {"none": "#444444", "intercept": "#9467bd", "platt": "#1f77b4", "isotonic": "#d62728"}


def render_reliability_grid() -> pathlib.Path:
    curves = pd.read_csv(TABLES / "calibration_curves_dryrun.csv")
    families = sorted(curves["family"].unique())
    fig, axes = plt.subplots(
        len(families),
        len(ARM_ORDER),
        figsize=(3.0 * len(ARM_ORDER), 2.9 * len(families)),
        sharex=True,
        sharey=True,
    )
    for f_idx, family in enumerate(families):
        for a_idx, arm in enumerate(ARM_ORDER):
            ax = axes[f_idx][a_idx] if len(families) > 1 else axes[a_idx]
            sub = curves[(curves["family"] == family) & (curves["arm"] == arm)]
            ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", linewidth=1)
            ax.bar(
                sub["p_bar"],
                sub["y_bar"] - sub["p_bar"],
                bottom=sub["p_bar"],
                width=0.03,
                color=ARM_COLORS.get(arm, "#1f77b4"),
                alpha=0.55,
                edgecolor="none",
            )
            ax.scatter(sub["p_bar"], sub["y_bar"], s=9, color=ARM_COLORS.get(arm, "#1f77b4"))
            if f_idx == 0:
                ax.set_title(arm, fontsize=10)
            if a_idx == 0:
                ax.set_ylabel(family, fontsize=9)
            if f_idx == len(families) - 1:
                ax.set_xlabel("mean predicted", fontsize=8)
    fig.suptitle(
        "Reliability diagrams, plumbing dry-run (equal-mass bins, B=15)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "reliability_diagram_dryrun.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_sweep_scatter() -> pathlib.Path:
    sweep = pd.read_csv(TABLES / "sweep_dryrun.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for family, sub in sweep.groupby("family"):
        ax.scatter(sub["auc"], sub["brier"], s=16, alpha=0.6, label=family)
    ax.set_xlabel("validation AUC")
    ax.set_ylabel("validation Brier")
    ax.set_title("Broad-sweep configs, plumbing dry-run")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "sweep_auc_brier_dryrun.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p1 = render_reliability_grid()
    print(f"wrote {p1.relative_to(REPO_ROOT)}")
    p2 = render_sweep_scatter()
    print(f"wrote {p2.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
