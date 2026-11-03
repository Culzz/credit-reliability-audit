"""Compute pilot on the plumbing sample (PLAN.md T4.6).

Purpose: size the Phase 5 grid honestly. One configuration per model family
(logistic regression, random forest, XGBoost, LightGBM) is fit on the
PROPOSED development split of the plumbing sample, plus one small SHAP run,
one paired AUC bootstrap, and one recalibration pair (Platt + isotonic).
Each cell logs wall time and peak working-set memory. These numbers set the
sample caps written into the protocol BEFORE the freeze — they never tune
better modelling results.

Inputs : data/interim/plumbing_sample.parquet (T3.9 sample)
Outputs: reports/tables/compute_pilot.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import build_feature_frame, keep_features

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
SAMPLE = REPO_ROOT / "data" / "interim" / "plumbing_sample.parquet"
SEED = 20180101


def _record(rows, task, detail, n, seconds, peak_mb):
    rows.append({"task": task, "detail": detail, "n_rows": n, "seconds": round(seconds, 3), "peak_mb": round(peak_mb / 1e6, 1)})


def main() -> None:
    df = pd.read_parquet(SAMPLE)
    features = build_feature_frame(df[keep_features() + ["issue_d"]]).astype("float32")
    y = df["y"].astype(float)
    resolved = y.notna().to_numpy()
    X, yv = features.to_numpy(), y.to_numpy()[resolved]

    # PROPOSED development splits mirrored on the plumbing sample by vintage
    vintage = df["vintage"].astype(str)
    dev_mask = (vintage <= "2013Q2").to_numpy() & resolved
    val_mask = (vintage >= "2013Q3").to_numpy() & resolved
    X_dev, y_dev = X[dev_mask], yv[dev_mask]
    X_val, y_val = X[val_mask], yv[val_mask]
    print(f"plumbing sample: resolved {len(yv):,} | dev {len(y_dev):,} | validation {len(y_val):,}")

    rows = []
    proc = psutil.Process()

    # ---- one configuration per family -------------------------------------
    import xgboost as xgb

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)
        ),
        "random_forest": None,
        "xgboost": None,
        "lightgbm": None,
    }
    fit_records = {}

    t0 = time.perf_counter()
    models["logistic_regression"].fit(X_dev, y_dev)
    fit_records["logistic_regression"] = (time.perf_counter() - t0, proc.memory_info().rss)
    p_lr = models["logistic_regression"].predict_proba(X_val)[:, 1]
    print(f"LR fit {fit_records['logistic_regression'][0]:.2f}s | val AUC {auc(y_val, p_lr):.4f}")

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED)
    t0 = time.perf_counter()
    rf.fit(X_dev, y_dev)
    fit_records["random_forest"] = (time.perf_counter() - t0, proc.memory_info().rss)
    p_rf = rf.predict_proba(X_val)[:, 1]
    print(f"RF fit {fit_records['random_forest'][0]:.2f}s | val AUC {auc(y_val, p_rf):.4f}")

    xg = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, n_jobs=4, random_state=SEED, eval_metric="logloss",
    )
    t0 = time.perf_counter()
    xg.fit(X_dev, y_dev)
    fit_records["xgboost"] = (time.perf_counter() - t0, proc.memory_info().rss)
    p_xgb = xg.predict_proba(X_val)[:, 1]
    print(f"XGB fit {fit_records['xgboost'][0]:.2f}s | val AUC {auc(y_val, p_xgb):.4f}")

    import lightgbm as lgb

    lg = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, subsample=0.9, colsample_bytree=0.9,
                            random_state=SEED, n_jobs=4, verbose=-1)
    t0 = time.perf_counter()
    lg.fit(X_dev, y_dev)
    fit_records["lightgbm"] = (time.perf_counter() - t0, proc.memory_info().rss)
    p_lgb = lg.predict_proba(X_val)[:, 1]
    print(f"LGBM fit {fit_records['lightgbm'][0]:.2f}s | val AUC {auc(y_val, p_lgb):.4f}")

    for fam, (sec, rss) in fit_records.items():
        _record(rows, "model_fit", fam, len(X_dev), sec, rss * 1.6)  # rough whole-process peak

    # ---- small SHAP run ----------------------------------------------------
    import shap

    rng = np.random.default_rng(SEED)
    ex_idx = rng.choice(len(X_val), min(500, len(X_val)), replace=False)
    expl = shap.TreeExplainer(xg)
    t0 = time.perf_counter()
    _ = expl.shap_values(X_val[ex_idx])
    seconds = time.perf_counter() - t0
    per_row = seconds / len(ex_idx)
    _record(rows, "shap_treeexplainer", "XGBoost, 500-row background, 500 rows explained", len(ex_idx), seconds, proc.memory_info().rss * 1.6)
    print(f"SHAP {seconds:.2f}s on 500 rows -> {per_row*10000:.1f}s per 10k rows")

    # ---- bootstrap pilot ---------------------------------------------------
    t0 = time.perf_counter()
    B = 200
    stats = np.empty(B)
    n = len(y_val)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        stats[b] = auc(y_val[idx], p_lr[idx])
    seconds = time.perf_counter() - t0
    lo, hi = np.quantile(stats, [0.025, 0.975])
    _record(rows, "bootstrap_auc", f"LR validation predictions, B={B}", B * n, seconds, proc.memory_info().rss * 1.6)
    print(f"bootstrap B=200 {seconds:.1f}s -> AUC CI [{lo:.4f},{hi:.4f}] (point {auc(y_val, p_lr):.4f})")

    # ---- recalibration pilot ----------------------------------------------
    # Platt: fit a sigmoid (intercept + slope) on the calibration logit scores
    t0 = time.perf_counter()
    cal_idx = train_test_split(np.arange(n), test_size=0.5, random_state=SEED)[0]
    logits = np.log(np.clip(p_lr, 1e-9, 1 - 1e-9) / np.clip(1 - p_lr, 1e-9, 1))[cal_idx]
    LogisticRegression().fit(
        (logits - logits.mean())[:, None] * 1.0, y_val[cal_idx]
    )
    seconds = time.perf_counter() - t0
    _record(rows, "platt_recalibration", "LR validation split, 50% calibration slice", len(cal_idx), seconds, proc.memory_info().rss * 1.6)
    print(f"platt {seconds:.3f}s")

    from sklearn.isotonic import IsotonicRegression

    t0 = time.perf_counter()
    IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_lr[cal_idx], y_val[cal_idx])
    seconds = time.perf_counter() - t0
    _record(rows, "isotonic_recalibration", "LR validation split, 50% calibration slice", len(cal_idx), seconds, proc.memory_info().rss * 1.6)
    print(f"isotonic {seconds:.3f}s")

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "compute_pilot.csv"
    table.to_csv(path, index=False)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
