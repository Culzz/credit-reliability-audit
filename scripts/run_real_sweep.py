"""Real-scale broad sweep on the frozen cohorts (post-freeze pipeline).

Same sampler and rules as the dry-run sweep (D-6, confirmed at freeze):
30 configs/family from the deterministic [base_seed, family_slot,
config_index] sampler, fitted every draw on the ratified train window
(2012Q1-2013Q2, 84,594 rows), scored on validation ONLY, deep-subset
rule = 5 configs/family at validation-AUC quantile anchors. Nothing in
this stage ever touches test cohorts.

Outputs: reports/tables/sweep.csv
         reports/tables/deep_subset.csv
Use    : python -m uv run python -m scripts.run_real_sweep
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts._common import REPO_ROOT, TABLES, check_test_window_guard, load_configs
from scripts._real import load_real_cohorts
from scripts.run_sweep import deep_subset_indices
from src.models.harness import FAMILIES, make_estimator, make_preprocessor
from src.models.spaces import load_spaces, sample_all_configs

BASE_SEED = 20180101  # registered sampler seed (T5.5; D-6 confirmed)


def main() -> None:
    models_path = REPO_ROOT / "configs" / "models.yaml"
    models = __import__("yaml").safe_load(models_path.read_text(encoding="utf-8"))["models"]
    n_configs = int(models["broad_sweep"]["configs_per_family"])
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))
    spaces = load_spaces()
    for fam in FAMILIES:
        assert fam in spaces, f"missing T5.5 space for {fam}"

    rc = load_real_cohorts()
    rc.assert_disjoint()
    print(f"frozen cohorts: {rc.shape_report()}")

    pre = make_preprocessor(scale=True).fit(rc.X_train)
    X_train = pre.transform(rc.X_train)
    X_val = pre.transform(rc.X_val)

    rows = []
    deep_rows = []
    for fam in FAMILIES:
        params_list = sample_all_configs(fam, n_configs, BASE_SEED)
        print(f"{fam}: sweeping {n_configs} configs on {len(rc.y_train):,} rows...")
        t0 = time.perf_counter()
        scored = []
        for i, params in enumerate(params_list):
            est = make_estimator(fam, params, BASE_SEED).fit(X_train, rc.y_train)
            p = est.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(rc.y_val, p))
            brier = float(brier_score_loss(rc.y_val, p))
            ll = float(log_loss(rc.y_val, p, labels=[0, 1]))
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
    sweep_out = TABLES / "sweep.csv"
    deep_out = TABLES / "deep_subset.csv"
    pd.DataFrame(rows).to_csv(sweep_out, index=False)
    pd.DataFrame(deep_rows).to_csv(deep_out, index=False)
    print(f"wrote {sweep_out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    print(f"wrote {deep_out.relative_to(REPO_ROOT)} ({len(deep_rows)} rows)")


if __name__ == "__main__":
    main()
