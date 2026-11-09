"""Real deep battery on the frozen cohorts (post-freeze pipeline).

Per family: the 5 deep-subset configs (the real-sweep validation-AUC
quantile anchors, D-6) x 20 seed refits x 30 bootstrap-refit draws
(T4.6 caps, D-8). For every fitted model the battery records:

- per-window score aggregates (training scores = calibration set,
  validation, each ratified test cohort): AUC, Brier, adaptive-bin B=15
  ECE, CITL + slope (CONTEXT 7 two-fit convention), A/E ratio, mean
  prediction;
- decision bands at the configured policy cell (primary loss/gain
  ratio and refer half-width from decisions.yaml) with pairwise flip
  tracks between every pair of refits (T9.2d real-scale flip floor);
- SHAP attributions on a capped 10,000-row explanation sample (T4.6:
  validation head, frozen order; background = training head 1,000)
  with the pairwise stability battery across refit pairs (T8.3 +
  T5.4 grouped metrics) and the frozen-model cross-cohort battery
  (the seed-0 model explained on the head of each eval window).

Bootstrap refits resample the training ROWS only (estimator seed
pinned), so seed-vs-seed variation and resample-vs-resample variation
stay separable: kind = "seed" | "boot" | "cohort" (frozen-model).

Families run in fast-first order; logistic_regression first, random
forest last (the heaviest).

Outputs: reports/tables/deep_agg_real.csv
         reports/tables/flip_pairs_real.csv
         reports/tables/stability_pairs_real.csv
         reports/tables/temporal_pairs_real.csv
Use    : python -m uv run python -m scripts.run_deep_battery [--pilot]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts._common import REPO_ROOT, TABLES, load_configs
from scripts._real import load_real_cohorts
from scripts.run_noise_floor import _grouped_block, _stability_block
from src.calibration.audit import ae_ratio, citl_and_slope, ece
from src.decisions.flips import flip_flows
from src.decisions.policy import breakeven_threshold, threshold_policy
from src.explain.groups import group_attributions
from src.explain.shap_wrap import explain_matrix, make_explainer
from src.models.harness import make_estimator, make_preprocessor
from src.models.spaces import sample_all_configs

BASE_SEED = 20180101
N_BACKGROUND = 1000
N_EXPLAIN_CAP = 10000  # T4.6: explanation rows per configuration
N_SEEDS = 20
N_BOOTS = 30
TOP_K = 10
FAMILY_ORDER_FAST_FIRST = (
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "random_forest",
)


@dataclass
class Ctx:
    windows: dict  # window name -> transformed X
    y_eval: dict  # window name -> y
    groups: dict  # group name -> design-column indices (55-col design)
    group_names: list
    t_low: float
    t_high: float
    window_order: list


def build_context(pilot: bool) -> Ctx:
    from src.features.groups import resolve_column_groups

    rc = load_real_cohorts()
    rc.assert_disjoint()
    print(f"frozen cohorts: {rc.shape_report()}")
    pre = make_preprocessor(scale=True).fit(rc.X_train)
    windows = {
        "train": pre.transform(rc.X_train),
        "val": pre.transform(rc.X_val),
    }
    y_eval = {"train": rc.y_train, "val": rc.y_val}
    for q in sorted(rc.test):
        windows[q] = pre.transform(rc.test[q]["X"])
        y_eval[q] = rc.test[q]["y"]

    mapping = resolve_column_groups(list(rc.feature_columns))
    groups: dict = {}
    for idx, col in enumerate(list(rc.feature_columns)):
        groups.setdefault(mapping[col], []).append(idx)

    dec = load_configs("decisions")["decisions"]
    primary_ratio = float(dec["breakeven"]["primary"])
    half = float(dec["refer_band_half_width"])
    p_star = breakeven_threshold(primary_ratio)
    return Ctx(
        windows=windows,
        y_eval=y_eval,
        groups=groups,
        group_names=sorted(groups),
        t_low=p_star - half,
        t_high=p_star + half,
        window_order=["train"] + [w for w in sorted(windows) if w != "train"],
    )


def _window_aggregate_rows(
    est, ctx: Ctx, family: str, config_index: int, kind: str, refit_id: int,
) -> list:
    """Per-window score aggregates for one fitted model."""
    rows = []
    for wname, Xw in ctx.windows.items():
        p = est.predict_proba(Xw)[:, 1]
        y = ctx.y_eval[wname]
        citl = citl_and_slope(y, p)
        rows += [
            _agg(family, config_index, kind, refit_id, wname, "roc_auc", roc_auc_score(y, p)),
            _agg(family, config_index, kind, refit_id, wname, "brier", brier_score_loss(y, p)),
            _agg(family, config_index, kind, refit_id, wname, "log_loss",
                 log_loss(y, p, labels=[0, 1])),
            _agg(family, config_index, kind, refit_id, wname, "ece_b15q",
                 ece(y, p, n_bins=15, quantile_edges=True)),
            _agg(family, config_index, kind, refit_id, wname, "citl", citl.citl),
            _agg(family, config_index, kind, refit_id, wname, "slope", citl.slope),
            _agg(family, config_index, kind, refit_id, wname, "ae_ratio", ae_ratio(y, p)),
            _agg(family, config_index, kind, refit_id, wname, "mean_p", p.mean()),
        ]
    return rows


def _agg(family, config_index, kind, refit_id, window, metric, value) -> dict:
    return {
        "family": family,
        "config_index": config_index,
        "kind": kind,
        "refit_id": refit_id,
        "window": window,
        "metric": metric,
        "value": round(float(value), 6),
    }


def _codes(p: np.ndarray, ctx: Ctx) -> np.ndarray:
    return threshold_policy(p, t_low=ctx.t_low, t_high=ctx.t_high)


def _pair_flip_rows(family, config_index, kind, i, j, codes_i, codes_j,
                    ctx: Ctx) -> list:
    rows = []
    for wname in ctx.window_order:
        y = ctx.y_eval[wname]
        flows = flip_flows(codes_i[wname], codes_j[wname])
        rows.append(
            {
                "family": family,
                "config_index": config_index,
                "kind": kind,
                "i": i,
                "j": j,
                "window": wname,
                "n": len(y),
                "n_flips": flows["n_flips"],
                "flip_rate": round(flows["n_flips"] / len(y), 6),
                "toward_decline": flows["toward_decline"],
            }
        )
    return rows


def _stability_rows(family, config_index, kind, i, j, Vi, GVi, Vj, GVj) -> list:
    rows = []
    for metric, value in {**_stability_block(Vi, Vj),
                          **_grouped_block(GVi, GVj)}.items():
        rows.append(
            {
                "family": family,
                "config_index": config_index,
                "kind": kind,
                "i": i,
                "j": j,
                "metric": metric,
                "value": round(float(value), 6),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="tiny run: logistic regression only, 2 seeds, 3 boots")
    parser.add_argument("--deep-table", type=str, default="deep_subset.csv",
                        help="deep-subset table under reports/tables (pilot may "
                        "point at a synthetic stand-in)")
    args = parser.parse_args()

    ctx = build_context(pilot=args.pilot)
    families = FAMILY_ORDER_FAST_FIRST
    n_seeds, n_boots = (N_SEEDS, N_BOOTS)
    if args.pilot:
        families = ("logistic_regression",)
        n_seeds, n_boots = 2, 3

    models_path = REPO_ROOT / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    assert int(models["deep_subset"]["configs_per_family"]) == 5

    deep = pd.read_csv(TABLES / args.deep_table)
    spaces = {fam: sample_all_configs(fam, 30, BASE_SEED) for fam in FAMILY_ORDER_FAST_FIRST}

    background = ctx.windows["train"][:N_BACKGROUND]
    val_head = ctx.windows["val"][:N_EXPLAIN_CAP]
    cohort_heads = {q: X[:N_EXPLAIN_CAP] for q, X in ctx.windows.items()
                    if q not in ("train",)}
    agg_rows, flip_rows, stab_rows, temp_rows = [], [], [], []
    for slot, family in enumerate(families):
        params_list = spaces[family]
        sub = deep[deep["family"] == family]
        for _, rec in sub.iterrows():
            cfg_idx = int(rec["config_index"])
            params = params_list[cfg_idx]
            t0 = time.perf_counter()

            # ---- seed axis: 20 fits, cached predictions, pairwise in RAM
            Ps: dict[int, dict] = {}
            Vs, GVs = {}, {}
            for k in range(n_seeds):
                est = make_estimator(family, params, BASE_SEED + k).fit(
                    ctx.windows["train"], ctx.y_eval["train"]
                )
                Ps[k] = {w: est.predict_proba(ctx.windows[w])[:, 1]
                         for w in ctx.windows}
                agg_rows += _window_aggregate_rows(est, ctx, family, cfg_idx, "seed", k)
                expl = make_explainer(est, background, family=family)
                Vs[k] = explain_matrix(expl, val_head)
                GVs[k] = np.array([group_attributions(Vs[k], ctx.groups)[g]
                                   for g in ctx.group_names])
            CK = {k: {w: _codes(Ps[k][w], ctx) for w in ctx.windows} for k in Ps}
            for i in range(n_seeds):
                for j in range(i + 1, n_seeds):
                    flip_rows += _pair_flip_rows(family, cfg_idx, "seed", i, j,
                                                 CK[i], CK[j], ctx)
                    stab_rows += _stability_rows(family, cfg_idx, "seed", i, j,
                                                 Vs[i], GVs[i], Vs[j], GVs[j])
            del Ps, CK

            # ---- bootstrap axis: resampled ROWS only, estimator seed pinned
            n_train = len(ctx.y_eval["train"])
            P_b: dict[int, dict] = {}
            V_b, GV_b = {}, {}
            for b in range(n_boots):
                rng = np.random.default_rng([BASE_SEED, slot, 1_000_000 + b])
                idx = rng.integers(0, n_train, size=n_train)
                est = make_estimator(family, params, BASE_SEED).fit(
                    ctx.windows["train"][idx], ctx.y_eval["train"][idx]
                )
                P_b[b] = {w: est.predict_proba(ctx.windows[w])[:, 1] for w in ctx.windows}
                agg_rows += _window_aggregate_rows(est, ctx, family, cfg_idx, "boot", b)
                expl = make_explainer(est, background, family=family)
                V_b[b] = explain_matrix(expl, val_head)
                GV_b[b] = np.array([group_attributions(V_b[b], ctx.groups)[g]
                                    for g in ctx.group_names])
            CK_b = {b: {w: _codes(P_b[b][w], ctx) for w in ctx.windows} for b in P_b}
            for i in range(n_boots):
                for j in range(i + 1, n_boots):
                    flip_rows += _pair_flip_rows(family, cfg_idx, "boot", i, j,
                                                 CK_b[i], CK_b[j], ctx)
                    stab_rows += _stability_rows(family, cfg_idx, "boot", i, j,
                                                 V_b[i], GV_b[i], V_b[j], GV_b[j])
            del P_b, CK_b

            # ---- frozen model (seed 0): cross-cohort explanation battery
            est0 = make_estimator(family, params, BASE_SEED).fit(
                ctx.windows["train"], ctx.y_eval["train"]
            )
            Vw, GVw = {}, {}
            for wname, head in cohort_heads.items():
                expl = make_explainer(est0, background, family=family)
                Vw[wname] = explain_matrix(expl, head)
                GVw[wname] = np.array([group_attributions(Vw[wname], ctx.groups)[g]
                                       for g in ctx.group_names])
            wnames = ["val"] + sorted(w for w in Vw if w != "val")
            for a in range(len(wnames)):
                for b in range(a + 1, len(wnames)):
                    wi, wj = wnames[a], wnames[b]
                    for metric, value in {**_stability_block(Vw[wi], Vw[wj]),
                                          **_grouped_block(GVw[wi], GVw[wj])}.items():
                        temp_rows.append({
                            "family": family,
                            "config_index": cfg_idx,
                            "window_i": wi,
                            "window_j": wj,
                            "metric": metric,
                            "value": round(float(value), 6),
                        })
            seconds = time.perf_counter() - t0
            print(f"{family:>20} cfg{cfg_idx:<2} | seeds {n_seeds} + boots "
                  f"{n_boots} | {seconds:.1f}s", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("deep_agg_real.csv", agg_rows),
        ("flip_pairs_real.csv", flip_rows),
        ("stability_pairs_real.csv", stab_rows),
        ("temporal_pairs_real.csv", temp_rows),
    ):
        out = TABLES / name
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
