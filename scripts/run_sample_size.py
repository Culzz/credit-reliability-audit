"""Dry-run the T7.5 calibration-sample-size experiment (H1d machinery).

For every family anchor: draw R subsamples of the TRAINING-window
calibration pool at each size of the pre-registered grid, fit all four
recalibration arms on the SAME draw (matched design: arms are comparable
per draw because they see identical calibration rows), apply to the
validation window, and record arm Brier/ECE per draw plus the paired
across-draw arm deltas (percentile CIs). Small-n fragility of isotonic
vs Platt is exactly what H1d targets; nothing here touches test cohorts
or results.

Outputs: reports/tables/sample_size_dryrun.csv
Use    : python -m uv run python -m scripts.run_sample_size
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.calibration.audit import ece
from src.calibration.recalibration import KINDS, fit_recalibrator
from src.models.harness import FAMILIES, make_estimator, make_preprocessor

# PROPOSED grid (protocol/HYPOTHESES.md H1d row): the 200 floor is the
# 200/200 feasibility rule's minimum; full = the plumbing training window.
SIZES = [200, 500, 1000, 2000, "full"]
REPEATS = 20
PAIRWISE_ARMS = [("isotonic", "platt"), ("isotonic", "intercept"), ("platt", "intercept")]


def main() -> None:
    dry = load_configs("dryrun")["dryrun"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    split = load_plumbing_split()
    split.assert_disjoint()

    rows = []
    for family_slot, family in enumerate(FAMILIES):
        params = dry["families"][family]
        seed0 = int(dry["seed"])
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(split.X_train)
        est = make_estimator(family, params, seed0).fit(
            pre.transform(split.X_train), split.y_train
        )
        p_train = est.predict_proba(pre.transform(split.X_train))[:, 1]
        p_val = est.predict_proba(pre.transform(split.X_val))[:, 1]
        pool_n = len(split.y_train)

        for size in SIZES:
            n = pool_n if size == "full" else min(int(size), pool_n)
            size_label = "full" if size == "full" else str(size)
            arm_scores = {arm: [] for arm in KINDS}
            for r in range(REPEATS):
                rng = np.random.default_rng([seed0, family_slot, r, n])
                idx = rng.choice(pool_n, size=n, replace=False)
                y_cal, p_cal = split.y_train[idx], p_train[idx]
                for arm in KINDS:
                    rec = fit_recalibrator(arm, y_cal, p_cal)
                    p_eval = rec.apply(p_val)
                    arm_scores[arm].append(
                        brier_score_loss(split.y_val, p_eval)
                    )
                    rows.append(
                        {
                            "family": family,
                            "size": size_label,
                            "repeat": r,
                            "arm": arm,
                            "metric": "brier_val",
                            "value": round(float(brier_score_loss(split.y_val, p_eval)), 6),
                        }
                    )
                    rows.append(
                        {
                            "family": family,
                            "size": size_label,
                            "repeat": r,
                            "arm": arm,
                            "metric": "ece_b15_val",
                            "value": round(float(ece(split.y_val, p_eval, n_bins=15)), 6),
                        }
                    )
                # matched computation: eval scores differ across arms fitted on
                # the SAME draw — decision-level pairing is the H1d statistic
            for hi_arm, lo_arm in PAIRWISE_ARMS:
                d = np.asarray(arm_scores[hi_arm]) - np.asarray(arm_scores[lo_arm])
                rows.extend(
                    {
                        "family": family,
                        "size": size_label,
                        "repeat": -1,
                        "arm": f"{hi_arm}_minus_{lo_arm}",
                        "metric": metric_name,
                        "value": round(float(val), 6),
                    }
                    for metric_name, val in (
                        ("delta_brier_mean", float(d.mean())),
                        ("delta_brier_q025", float(np.percentile(d, 2.5))),
                        ("delta_brier_q975", float(np.percentile(d, 97.5))),
                    )
                )
        print(
            f"{family:>20} | eval AUC {roc_auc_score(split.y_val, p_val):.4f} | "
            f"{len(SIZES)} sizes x {REPEATS} draws x {len(KINDS)} arms | "
            f"{time.perf_counter() - t0:.1f}s"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "sample_size_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
