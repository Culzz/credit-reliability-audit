"""Dry-run the baseline harness end to end on the plumbing sample (T5.6).

Fits one anchor configuration per sanctioned family, mirroring the T4.6
compute pilot, with the PROPOSED train/validation windows read from
configs/splits.yaml (no literals). Test cohorts stay locked: the runner
aborts with a clear message while `splits.yaml` test windows are unset.

Outputs: reports/tables/baseline_dryrun_metrics.csv
Use    : python -m uv run python -m scripts.run_baselines
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from src.data.loader import filter_by_vintage_window
from src.features.build import build_feature_frame, keep_features
from src.models.harness import FAMILIES, evaluate, make_estimator, make_preprocessor

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
        print(
            "test window unset: LOCKED until the PI ratifies Gate 4 and the "
            "protocol freezes; this dry-run covers training and validation "
            "cohorts only."
        )
    elif not (REPO_ROOT / "protocol" / "PROTOCOL_FROZEN").exists():
        raise RuntimeError(
            "test windows set but protocol/PROTOCOL_FROZEN is absent; the "
            "runner refuses to touch test cohorts pre-freeze."
        )

    df = pd.read_parquet(SAMPLE)
    features = build_feature_frame(df[keep_features() + ["issue_d"]]).astype("float32")
    y = df["y"].astype(float)
    resolved = y.notna().to_numpy()

    train_cfg = splits["train"]
    val_cfg = splits["validation"]
    train_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage", start=train_cfg["start"], end=train_cfg["end"]
    )
    val_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage", start=val_cfg["start"], end=val_cfg["end"]
    )
    X_train = features.loc[train_df.index].to_numpy()
    y_train = y.loc[train_df.index].to_numpy()
    X_val = features.loc[val_df.index].to_numpy()
    y_val = y.loc[val_df.index].to_numpy()
    print(
        f"plumbing sample: resolved {int(resolved.sum()):,} | "
        f"train {len(y_train):,} | validation {len(y_val):,}"
    )

    rows = []
    for family in FAMILIES:
        params = cfg["families"][family]
        seed = cfg["seed"]
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(X_train)
        est = make_estimator(family, params, seed).fit(pre.transform(X_train), y_train)
        p_val = est.predict_proba(pre.transform(X_val))[:, 1]
        seconds = time.perf_counter() - t0
        res = evaluate(y_val, p_val, n_bootstrap=cfg["bootstrap_B"], seed=seed)
        for m in sorted(res.point):
            rows.append(
                {
                    "family": family,
                    "metric": m,
                    "point": round(res.point[m], 6),
                    "lo": round(res.ci_low[m], 6),
                    "hi": round(res.ci_high[m], 6),
                    "n_train": len(y_train),
                    "n_val": len(y_val),
                    "seconds": round(seconds, 2),
                }
            )
        print(f"{family:>20} | AUC {res.point['roc_auc']:.4f} "
              f"[{res.ci_low['roc_auc']:.4f}, {res.ci_high['roc_auc']:.4f}] | {seconds:.1f}s")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "baseline_dryrun_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
