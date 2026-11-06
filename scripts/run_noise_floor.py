"""Dry-run the T8.2 Axis-C noise floor on the plumbing sample.

One anchor configuration per family (the T4.6 pilot anchors in
configs/dryrun.yaml — plumbing-only), refit under 20 seeds on the SAME
training window; for every seed pair the SHAP summaries of a fixed
explanation sample (fixed background, fixed rows, fixed feature order)
are compared with the T8.3 battery, and decision tracks at the primary
threshold are compared for the T9.2d seed-vs-seed flip floor. The
outputs ARE the noise-floor distributions the H2c/H3a thresholds read:
nothing here touches test cohorts, groups, or results.

Outputs: reports/tables/noise_floor_dryrun.csv
Use    : python -m uv run python -m scripts.run_noise_floor
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.calibration.audit import decision_flips
from src.explain.shap_wrap import explain_matrix, make_explainer
from src.explain.stability import (
    attribution_distribution_distance,
    cosine_importance,
    jaccard_at_k,
    rank_agreement,
)
from src.models.harness import FAMILIES, make_estimator, make_preprocessor

N_SEEDS = 20
N_EXPLAIN = 400      # fixed explanation rows (validation cohort head, frozen order)
N_BACKGROUND = 1000  # fixed background rows (training window head; protocol uses
                     # a 1,000-row anchor; smaller heads gave degenerate one-hot
                     # columns and the wrapper refuses those, correctly)
TOP_K = 10


def _stability_block(V1: np.ndarray, V2: np.ndarray) -> dict[str, float]:
    # the T8.3 battery: full attribution matrices in, summary scalars out
    ranks = rank_agreement(V1, V2)
    return {
        "jaccard_top10": jaccard_at_k(V1, V2, TOP_K),
        "spearman": ranks["spearman"],
        "kendall": ranks["kendall"],
        "cosine": cosine_importance(V1, V2),
        "wasserstein_mean": attribution_distribution_distance(V1, V2),
    }


def main() -> None:
    dry = load_configs("dryrun")["dryrun"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    split = load_plumbing_split()
    split.assert_disjoint()

    pre = make_preprocessor(scale=True).fit(split.X_train)
    X_train_t = pre.transform(split.X_train)
    X_val_t = pre.transform(split.X_val)
    # explanations live in the transformed feature space (what the model
    # consumes): imputed, standardised columns, fixed order
    X_explain = X_val_t[:N_EXPLAIN]
    y_explain = split.y_val[:N_EXPLAIN]
    X_background = X_train_t[:N_BACKGROUND]

    base_seed = int(dry["seed"])
    rows = []
    for family in FAMILIES:
        params = dry["families"][family]
        t0 = time.perf_counter()
        Vs: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for k in range(N_SEEDS):
            seed = base_seed + k
            est = make_estimator(family, params, seed).fit(X_train_t, split.y_train)
            ps.append(est.predict_proba(X_explain)[:, 1])
            expl = make_explainer(est, X_background, family=family)
            Vs.append(explain_matrix(expl, X_explain))
        fit_seconds = time.perf_counter() - t0

        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                for metric, value in _stability_block(Vs[i], Vs[j]).items():
                    rows.append(
                        {
                            "family": family,
                            "seed_i": base_seed + i,
                            "seed_j": base_seed + j,
                            "metric": metric,
                            "value": round(float(value), 6),
                        }
                    )
                flips = decision_flips(ps[i], ps[j], threshold=0.5)
                rows.append(
                    {
                        "family": family,
                        "seed_i": base_seed + i,
                        "seed_j": base_seed + j,
                        "metric": "decision_flip_rate",
                        "value": round(float(flips.flip_rate), 6),
                    }
                )
                rows.append(
                    {
                        "family": family,
                        "seed_i": base_seed + i,
                        "seed_j": base_seed + j,
                        "metric": "auc_gap",
                        "value": round(
                            float(
                                roc_auc_score(y_explain, ps[i])
                                - roc_auc_score(y_explain, ps[j])
                            ),
                            6,
                        ),
                    }
                )
        print(
            f"{family:>20} | {N_SEEDS} seeds -> {N_SEEDS * (N_SEEDS - 1) // 2} pairs "
            f"| eval AUCs {roc_auc_score(y_explain, ps[0]):.4f}.."
            f"{roc_auc_score(y_explain, ps[-1]):.4f} | {fit_seconds:.1f}s"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "noise_floor_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
