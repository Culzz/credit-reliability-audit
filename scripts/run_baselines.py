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

import pandas as pd

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.models.harness import FAMILIES, evaluate, make_estimator, make_preprocessor


def main() -> None:
    cfg = load_configs("dryrun")["dryrun"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    split = load_plumbing_split()
    split.assert_disjoint()
    print(f"plumbing sample: {split.shape_report}")

    rows = []
    for family in FAMILIES:
        params = cfg["families"][family]
        seed = cfg["seed"]
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(split.X_train)
        est = make_estimator(family, params, seed).fit(pre.transform(split.X_train), split.y_train)
        p_val = est.predict_proba(pre.transform(split.X_val))[:, 1]
        seconds = time.perf_counter() - t0
        res = evaluate(split.y_val, p_val, n_bootstrap=cfg["bootstrap_B"], seed=seed)
        for m in sorted(res.point):
            rows.append(
                {
                    "family": family,
                    "metric": m,
                    "point": round(res.point[m], 6),
                    "lo": round(res.ci_low[m], 6),
                    "hi": round(res.ci_high[m], 6),
                    "n_train": len(split.y_train),
                    "n_val": len(split.y_val),
                    "seconds": round(seconds, 2),
                }
            )
        print(
            f"{family:>20} | AUC {res.point['roc_auc']:.4f} "
            f"[{res.ci_low['roc_auc']:.4f}, {res.ci_high['roc_auc']:.4f}] | {seconds:.1f}s"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "baseline_dryrun_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
