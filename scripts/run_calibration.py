"""Dry-run the calibration-audit arms end to end on the plumbing sample.

Per sanctioned family anchor: fit on the training window, use the
TRAINING-window scores as the calibration set, evaluate every
recalibration arm on the validation window. Iron rule (T7.3): the arm is
fitted on eligible historical labels only, and the runner ASSERTS the
calibration cohort is disjoint from the evaluation cohort — here they are
different vintage windows by construction, checked explicitly.

Outputs: reports/tables/calibration_audit_dryrun.csv
         reports/tables/calibration_curves_dryrun.csv (reliability-bin
         segments per family x arm; same equal-mass binning as the
         primary ece_b15, fed to scripts/run_figures.py)
Use    : python -m uv run python -m scripts.run_calibration
"""

from __future__ import annotations

import time

import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.calibration.audit import (
    ae_ratio,
    brier_decomposition,
    citl_and_slope,
    decision_flips,
    ece,
    reliability_bins,
)
from src.calibration.recalibration import KINDS, fit_recalibrator
from src.models.harness import FAMILIES, make_estimator, make_preprocessor


def main() -> None:
    cfg = load_configs("dryrun")["dryrun"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    split = load_plumbing_split()
    split.assert_disjoint()
    print(f"plumbing sample: {split.shape_report} | disjoint: yes")

    rows = []
    curve_rows = []
    for family in FAMILIES:
        params = cfg["families"][family]
        seed = cfg["seed"]
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(split.X_train)
        est = make_estimator(family, params, seed).fit(pre.transform(split.X_train), split.y_train)
        p_train = est.predict_proba(pre.transform(split.X_train))[:, 1]
        p_val = est.predict_proba(pre.transform(split.X_val))[:, 1]
        seconds = time.perf_counter() - t0

        for arm in KINDS:
            rec = fit_recalibrator(arm, split.y_train, p_train)
            p_eval = rec.apply(p_val)
            dec = brier_decomposition(split.y_val, p_eval, n_bins=15)
            citl = citl_and_slope(split.y_val, p_eval)
            p_base = fit_recalibrator("none", split.y_train, p_train).apply(p_val)
            flips = decision_flips(p_base, p_eval, threshold=0.5)
            for metric, value in [
                ("brier_raw", dec.brier),
                ("brier_binned", dec.binned_brier),
                ("reliability", dec.reliability),
                ("ece_b15", ece(split.y_val, p_eval, n_bins=15, quantile_edges=True)),  # §7: adaptive equal-mass
                ("ae_ratio", ae_ratio(split.y_val, p_eval)),
                ("citl", citl.citl),
                ("slope", citl.slope),
                ("flip_rate_vs_none", flips.flip_rate),
                ("roc_auc", roc_auc_score(split.y_val, p_eval)),
            ]:
                rows.append(
                    {
                        "family": family,
                        "arm": arm,
                        "metric": metric,
                        "value": round(float(value), 6),
                        "n_calib": rec.n_calibrated,
                        "events_calib": rec.n_events,
                        "n_eval": len(split.y_val),
                        "fit_seconds": round(seconds, 2),
                    }
                )
            # reliability-diagram segments, same binning as the primary ece
            for seg in reliability_bins(split.y_val, p_eval, n_bins=15, quantile_edges=True):
                curve_rows.append({"family": family, "arm": arm, **seg})
        print(f"{family:>20} | arms x 9 metrics | {seconds:.1f}s")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "calibration_audit_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    curves = TABLES / "calibration_curves_dryrun.csv"
    pd.DataFrame(curve_rows).to_csv(curves, index=False)
    print(
        f"wrote {curves.relative_to(REPO_ROOT)} "
        f"({len(curve_rows)} bin segments)"
    )


if __name__ == "__main__":
    main()
