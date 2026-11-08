"""Dry-run the T8.2 Axis-C noise floor on the plumbing sample.

Two refit axes feed the reference distributions (CONTEXT 7: "the
distribution of the same metric over all pairs of seed-retrained models
AND over bootstrap-refit models on identical data"):

- seed axis: one anchor configuration per family (T4.6 pilot anchors in
  configs/dryrun.yaml) refit under 20 seeds on the SAME training window;
- bootstrap axis: the same anchor refit on B seeded bootstrap resamples
  of the training rows (estimator seed FIXED, so variation here is the
  resample, not the seed).

For every pair the SHAP summaries of a fixed explanation sample (fixed
background, fixed rows, fixed feature order) are compared with the T8.3
battery plus the T5.4 grouped partition metrics, and decision tracks at
the primary threshold for the T9.2d flip floor. Nothing here touches
test cohorts, groups results, or results.

Outputs: reports/tables/noise_floor_dryrun.csv (with a `kind` column)
Use    : python -m uv run python -m scripts.run_noise_floor
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.calibration.audit import decision_flips
from src.explain.groups import group_attributions
from src.explain.shap_wrap import explain_matrix, make_explainer
from src.explain.stability import (
    attribution_distribution_distance,
    cosine_importance,
    jaccard_at_k,
    rank_agreement,
)
from src.features.groups import resolve_column_groups
from src.models.harness import FAMILIES, make_estimator, make_preprocessor

N_SEEDS = 20
N_BOOTS = 30        # plumbing-scale bootstrap axis (T8.2 spec: 30 resamples)
N_EXPLAIN = 400      # fixed explanation rows (validation cohort head, frozen order)
N_BACKGROUND = 1000  # fixed background rows (training window head; protocol uses
                     # a 1,000-row anchor; smaller heads gave degenerate one-hot
                     # columns and the wrapper refuses those, correctly)
TOP_K = 10


def _group_partition(columns: list[str]) -> dict[str, list[int]]:
    """T5.4 resolver -> {group: design-column indices}, in design order."""
    mapping = resolve_column_groups(columns)
    groups: dict[str, list[int]] = {}
    for idx, col in enumerate(columns):
        groups.setdefault(mapping[col], []).append(idx)
    return groups


def build_feature_frame_columns() -> list[str]:
    """The design-matrix column order, reconstructed deterministically from
    the same assets the split loader used (full sample frame -> KEEP features
    + issue_d; astype does not reorder). One-hot levels must come from the
    FULL frame, so a head-only probe would under-count columns."""
    import pandas as pd  # bundled - used only for the column list

    from src.features.build import build_feature_frame, keep_features

    df = pd.read_parquet(REPO_ROOT / "data" / "interim" / "plumbing_sample.parquet")
    feats = build_feature_frame(df[keep_features() + ["issue_d"]])
    return list(feats.columns)


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


def _grouped_block(G1: np.ndarray, G2: np.ndarray) -> dict[str, float]:
    denom = np.linalg.norm(G1) * np.linalg.norm(G2)
    sp = float(spearmanr(G1, G2).statistic)
    return {
        "grouped_cosine": float(np.dot(G1, G2) / denom) if denom > 0 else 0.0,
        "grouped_spearman": float(sp) if np.isfinite(sp) else 0.0,
        "grouped_max_gap": float(np.abs(G1 - G2).max()),
    }


def _pair_rows(
    family: str,
    kind: str,
    a: int,
    b: int,
    Vi: np.ndarray,
    Vj: np.ndarray,
    pi: np.ndarray,
    pj: np.ndarray,
    Gi: np.ndarray,
    Gj: np.ndarray,
    y_eval: np.ndarray,
    base_row: dict,
) -> list[dict]:
    rows = []
    common = {"family": family, "kind": kind, "a": a, "b": b}
    for metric, value in {**_stability_block(Vi, Vj), **_grouped_block(Gi, Gj)}.items():
        rows.append({**base_row, **common, "metric": metric, "value": round(float(value), 6)})
    flips = decision_flips(pi, pj, threshold=0.5)
    rows.append({
        **base_row,
        **common,
        "metric": "decision_flip_rate",
        "value": round(float(flips.flip_rate), 6),
    })
    rows.append({
        **base_row,
        **common,
        "metric": "auc_gap",
        "value": round(float(roc_auc_score(y_eval, pi) - roc_auc_score(y_eval, pj)), 6),
    })
    return rows


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

    # T5.4 partition of the design columns (fixed order -> fixed group order)
    design_columns = build_feature_frame_columns()
    groups = _group_partition(design_columns)
    group_names = sorted(groups)

    rows = []
    for family_slot, family in enumerate(FAMILIES):
        params = dry["families"][family]
        t0 = time.perf_counter()

        # --- seed axis ---
        Vs: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for k in range(N_SEEDS):
            est = make_estimator(family, params, base_seed + k).fit(
                X_train_t, split.y_train
            )
            ps.append(est.predict_proba(X_explain)[:, 1])
            expl = make_explainer(est, X_background, family=family)
            Vs.append(explain_matrix(expl, X_explain))
        Gs = [
            np.array([group_attributions(V, groups)[g] for g in group_names])
            for V in Vs
        ]
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                rows.extend(
                    _pair_rows(
                        family,
                        "seed",
                        base_seed + i,
                        base_seed + j,
                        Vs[i],
                        Vs[j],
                        ps[i],
                        ps[j],
                        Gs[i],
                        Gs[j],
                        y_explain,
                        {},
                    )
                )
        seed_seconds = time.perf_counter() - t0

        # --- bootstrap axis (same estimator seed; resampled ROWS only) ---
        t1 = time.perf_counter()
        Vs_b: list[np.ndarray] = []
        ps_b: list[np.ndarray] = []
        n_train = len(split.y_train)
        for b in range(N_BOOTS):
            rng = np.random.default_rng([base_seed, family_slot, 1_000_000 + b])
            idx = rng.integers(0, n_train, size=n_train)
            est = make_estimator(family, params, base_seed).fit(
                X_train_t[idx], split.y_train[idx]
            )
            ps_b.append(est.predict_proba(X_explain)[:, 1])
            expl = make_explainer(est, X_background, family=family)
            Vs_b.append(explain_matrix(expl, X_explain))
        Gs_b = [
            np.array([group_attributions(V, groups)[g] for g in group_names])
            for V in Vs_b
        ]
        for i in range(N_BOOTS):
            for j in range(i + 1, N_BOOTS):
                rows.extend(
                    _pair_rows(
                        family,
                        "bootstrap_refit",
                        i,
                        j,
                        Vs_b[i],
                        Vs_b[j],
                        ps_b[i],
                        ps_b[j],
                        Gs_b[i],
                        Gs_b[j],
                        y_explain,
                        {},
                    )
                )
        boot_seconds = time.perf_counter() - t1

        print(
            f"{family:>20} | seed {N_SEEDS}->{N_SEEDS * (N_SEEDS - 1) // 2} pairs "
            f"({seed_seconds:.1f}s) | boot {N_BOOTS}->{N_BOOTS * (N_BOOTS - 1) // 2} "
            f"pairs ({boot_seconds:.1f}s)"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "noise_floor_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
