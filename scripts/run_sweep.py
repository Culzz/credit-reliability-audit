"""Broad-sweep sampling dry-run (T5.5 follow-through; plumbing scale).

Draws the PROPOSED broad sweep — `broad_sweep.configs_per_family` = 30
configs x 1 seed per family from the deterministic T5.5 sampler (no data in
the draw) — fits every draw on the plumbing training window, evaluates on
the validation window ONLY (AUC, Brier, log loss), and performs the
deep-subset selection defined in models.yaml (5 configs per family at validation-AUC
quantile anchors: min, 25th, 50th, 75th, max). Validation-only model
selection is sanctioned; test cohorts stay locked.

Outputs:
  reports/tables/sweep_dryrun.csv          (per-draw AUC/Brier/logloss)
  reports/tables/deep_subset_dryrun.csv    (per-family selected config ids)
Use: python -m uv run python -m scripts.run_sweep
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.models.harness import FAMILIES, make_estimator, make_preprocessor
from src.models.spaces import load_spaces, sample_all_configs

DEEP_ANCHORS = ("min", "q25", "q50", "q75", "max")


def deep_subset_indices(sorted_auc_order: np.ndarray) -> dict[str, int]:
    """models.yaml deep_subset rule: 5 configs at validation-AUC quantile
    anchors (min, 25th, 50th, 75th, max of the sorted AUC ordering). Even-n
    median ties resolve deterministically via round-half-even (Python
    `round`), i.e. to the lower of the two central positions."""
    n = len(sorted_auc_order)
    return {
        "min": int(sorted_auc_order[0]),
        "max": int(sorted_auc_order[-1]),
        "q25": int(sorted_auc_order[round(0.25 * (n - 1))]),
        "q50": int(sorted_auc_order[round(0.50 * (n - 1))]),
        "q75": int(sorted_auc_order[round(0.75 * (n - 1))]),
    }


def main() -> None:
    dry = load_configs("dryrun")["dryrun"]
    models = load_configs("models")["models"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    n_configs = int(models["broad_sweep"]["configs_per_family"])
    spaces = load_spaces()
    for fam in FAMILIES:
        assert fam in spaces, f"missing T5.5 space for {fam}"

    split = load_plumbing_split()
    split.assert_disjoint()
    pre = make_preprocessor(scale=True).fit(split.X_train)
    X_train = pre.transform(split.X_train)
    X_val = pre.transform(split.X_val)

    base_seed = int(dry["seed"])
    rows = []
    deep_rows = []
    for fam in FAMILIES:
        params_list = sample_all_configs(fam, n_configs, base_seed)
        t0 = time.perf_counter()
        scored = []
        for i, params in enumerate(params_list):
            est = make_estimator(fam, params, base_seed).fit(X_train, split.y_train)
            p = est.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(split.y_val, p))
            brier = float(brier_score_loss(split.y_val, p))
            ll = float(log_loss(split.y_val, p, labels=[0, 1]))
            scored.append((i, auc, brier, ll))
            rows.append(
                {
                    "family": fam,
                    "config_index": i,
                    "params_json": json.dumps(params, sort_keys=True),
                    "auc": round(auc, 6),
                    "brier": round(brier, 6),
                    "log_loss": round(ll, 6),
                }
            )
        seconds = time.perf_counter() - t0

        # deep subset: 5 configs at validation-AUC quantile anchors
        # (never test data; models.yaml deep_subset rule)
        aucs = np.array([s[1] for s in scored])
        order = np.argsort(aucs)
        target = deep_subset_indices(order)
        for anchor, idx in target.items():
            deep_rows.append(
                {
                    "family": fam,
                    "anchor": anchor,
                    "config_index": idx,
                    "auc": round(aucs[idx], 6),
                }
            )
        print(
            f"{fam:>20} | {n_configs} draws | AUC "
            f"{aucs.min():.4f}..{aucs.max():.4f} (spread {aucs.max() - aucs.min():.4f}) "
            f"| {seconds:.1f}s"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    sweep_out = TABLES / "sweep_dryrun.csv"
    deep_out = TABLES / "deep_subset_dryrun.csv"
    pd.DataFrame(rows).to_csv(sweep_out, index=False)
    pd.DataFrame(deep_rows).to_csv(deep_out, index=False)
    print(f"wrote {sweep_out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    print(f"wrote {deep_out.relative_to(REPO_ROOT)} ({len(deep_rows)} rows)")


if __name__ == "__main__":
    main()
