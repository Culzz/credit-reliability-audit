"""Dry-run the calibration-audit arms end to end on the plumbing sample.

Per sanctioned family anchor: fit on the training window, use the
TRAINING-window scores as the calibration set, evaluate every
recalibration arm on the validation window. Iron rule (T7.3): the arm is
fitted on eligible historical labels only, and the runner ASSERTS the
calibration cohort is disjoint from the evaluation cohort — here they are
different vintage windows by construction, checked explicitly.

Outputs: reports/tables/calibration_audit_dryrun.csv
Use    : python -m uv run python -m scripts.run_calibration
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.calibration.audit import brier_decomposition, citl_and_slope, decision_flips, ece
from src.calibration.recalibration import KINDS, fit_recalibrator
from src.data.loader import filter_by_vintage_window
from src.features.build import build_feature_frame, keep_features
from src.models.harness import FAMILIES, make_estimator, make_preprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
SAMPLE = REPO_ROOT / "data" / "interim" / "plumbing_sample.parquet"


def _load_yaml(rel: str) -> dict:
    with (REPO_ROOT / "configs" / rel).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    cfg = _load_yaml("dryrun.yaml")["dryrun"]
    splits = _load_yaml("splits.yaml")["splits"]
    test_cfg = splits.get("test", {})
    if test_cfg.get("start") is None or test_cfg.get("end") is None:
        print("test window unset: LOCKED; dry-run covers training vs validation only.")
    elif not (REPO_ROOT / "protocol" / "PROTOCOL_FROZEN").exists():
        raise RuntimeError("test windows set without PROTOCOL_FROZEN; refusing to proceed.")

    df = pd.read_parquet(SAMPLE)
    features = build_feature_frame(df[keep_features() + ["issue_d"]]).astype("float32")
    y = df["y"].astype(float)
    resolved = y.notna().to_numpy()

    train_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage",
        start=splits["train"]["start"], end=splits["train"]["end"],
    )
    val_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage",
        start=splits["validation"]["start"], end=splits["validation"]["end"],
    )
    # iron-rule assertion: calibration (training window) disjoint from eval
    overlap = train_df.index.intersection(val_df.index)
    if len(overlap):
        raise RuntimeError(f"calibration and evaluation cohorts overlap ({len(overlap)} loans).")

    X_train = features.loc[train_df.index].to_numpy()
    y_train = y.loc[train_df.index].to_numpy()
    X_val = features.loc[val_df.index].to_numpy()
    y_val = y.loc[val_df.index].to_numpy()
    print(
        f"plumbing sample: train(calib) {len(y_train):,} | "
        f"validation(eval) {len(y_val):,} | disjoint: yes"
    )

    rows = []
    for family in FAMILIES:
        params = cfg["families"][family]
        seed = cfg["seed"]
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(X_train)
        est = make_estimator(family, params, seed).fit(pre.transform(X_train), y_train)
        p_train = est.predict_proba(pre.transform(X_train))[:, 1]
        p_val = est.predict_proba(pre.transform(X_val))[:, 1]
        seconds = time.perf_counter() - t0

        for arm in KINDS:
            rec = fit_recalibrator(arm, y_train, p_train)
            p_eval = rec.apply(p_val)
            dec = brier_decomposition(y_val, p_eval, n_bins=15)
            citl = citl_and_slope(y_val, p_eval)
            base = fit_recalibrator("none", y_train, p_train)
            flips = decision_flips(base.apply(p_val), p_eval, threshold=0.5)
            for metric, value in [
                ("brier_raw", dec.brier),
                ("brier_binned", dec.binned_brier),
                ("reliability", dec.reliability),
                ("ece_b15", ece(y_val, p_eval, n_bins=15)),
                ("citl", citl.citl),
                ("slope", citl.slope),
                ("flip_rate_vs_none", flips.flip_rate),
                ("roc_auc", roc_auc_score(y_val, p_eval)),
            ]:
                rows.append(
                    {
                        "family": family,
                        "arm": arm,
                        "metric": metric,
                        "value": round(float(value), 6),
                        "n_calib": rec.n_calibrated,
                        "events_calib": rec.n_events,
                        "n_eval": len(y_val),
                        "fit_seconds": round(seconds, 2),
                    }
                )
        print(f"{family:>20} | arms x 8 metrics | {seconds:.1f}s")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "calibration_audit_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
