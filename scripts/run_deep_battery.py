"""Real deep battery on the frozen cohorts (post-freeze pipeline).

DEV-1 (2026-11-12, PI-approved before any result was seen): the
registered 20-seed x 30-bootstrap design with 10,000-row interventional
TreeSHAP is arithmetically infeasible on the project machine (measured:
random-forest fit 207 s, interventional SHAP 1.96 s/row = 5.5 h per
10,000-row RF model; 255 RF models alone). The lean ratified design is
5 seed refits + 5 bootstrap refits per config (5 configs/family), SHAP
capped at 1,000 explanation rows for LR/XGB/LGBM, and per-feature
permutation importance (R=3, 1,000 rows, the T8.5 sanctioned machinery)
for random forest instead of SHAP. See protocol/DEVIATIONS.md DEV-1.

Per family: the 5 deep-subset configs (the real-sweep validation-AUC
quantile anchors, D-6) x 5 seed refits x 5 bootstrap-refit draws.
For every fitted model the battery records:

- per-window score aggregates (training scores = calibration set,
  validation, each ratified test cohort): AUC, Brier, adaptive-bin B=15
  ECE, CITL + slope (CONTEXT 7 two-fit convention), A/E ratio, mean
  prediction;
- decision bands at the configured policy cell (primary loss/gain
  ratio and refer half-width from decisions.yaml) with pairwise flip
  tracks between every pair of refits (T9.2d real-scale flip floor);
- attributions on a capped explanation sample (validation head, frozen
  order; background = training head 1,000 for the SHAP families) with
  the pairwise stability battery across refit pairs (T8.3 + T5.4
  grouped metrics) and the frozen-model cross-cohort battery (the
  seed-0 model explained on the head of each eval window).

Bootstrap refits resample the training ROWS only (estimator seed
pinned), so seed-vs-seed variation and resample-vs-resample variation
stay separable: kind = "seed" | "boot" | "cohort" (frozen-model).

Families run in fast-first order; logistic_regression first, random
forest last (the heaviest).

Outputs: reports/tables/deep_agg_real.csv
         reports/tables/flip_pairs_real.csv
         reports/tables/stability_pairs_real.csv
         reports/tables/temporal_pairs_real.csv
         reports/tables/deep_arms_real.csv
         reports/tables/group_pairs_real.csv
         (written per family as *_real__<family>.csv after every config,
         then concatenated by --merge-only; the merged schema is the one
         scripts.run_real_statistics consumes)
Use    : python -m uv run python -m scripts.run_deep_battery [--pilot]
         python -m uv run python -m scripts.run_deep_battery --families random_forest
         python -m uv run python -m scripts.run_deep_battery --merge-only
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
N_EXPLAIN_CAP = 1000   # DEV-1: 10,000 -> 1,000 explained rows (SHAP families)
N_SEEDS = 5            # DEV-1: frozen design had 20 seed refits
N_BOOTS = 5            # DEV-1: frozen design had 30 bootstrap refits
RF_PERM_ROWS = 1000    # DEV-1: RF explanation sample (permutation importance)
RF_PERM_REPEATS = 3    # DEV-1: seeded joint permutations per feature
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
    preds: dict, ctx: Ctx, family: str, config_index: int, kind: str,
    refit_id: int,
) -> list:
    """Per-window score aggregates for one fitted model (predictions cached
    once per model and reused here and by the recalibration arms)."""
    rows = []
    for wname in ctx.windows:
        p = preds[wname]
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


# recalibration arms: fitted on the model's TRAINING-window scores
# (iron rule: historical eligible labels only), evaluated per window
# from the model's cached predictions; the per-window "oracle" refits
# the same arm ON the eval window itself and gives H1c's partial-
# recovery denominator
def _arm_point_rows(preds: dict, ctx: Ctx, family: str,
                    config_index: int, kind: str, refit_id: int) -> list:
    from src.calibration.recalibration import KINDS, fit_recalibrator

    def row(arm, window, metric, value) -> dict:
        return {
            "family": family,
            "config_index": config_index,
            "kind": kind,
            "refit_id": refit_id,
            "arm": arm,
            "window": window,
            "metric": metric,
            "value": round(float(value), 6),
        }

    rows = []
    for arm in KINDS:
        rec = fit_recalibrator(arm, ctx.y_eval["train"], preds["train"])
        for wname in ctx.window_order:
            if wname == "train":
                continue
            y = ctx.y_eval[wname]
            p_before = preds[wname]
            p_after = rec.apply(p_before)
            b_before = brier_score_loss(y, p_before)
            oracle = fit_recalibrator(arm, y, p_before)
            b_oracle = float(brier_score_loss(y, oracle.apply(p_before)))
            b_after = float(brier_score_loss(y, p_after))
            citl = citl_and_slope(y, p_after)
            denom = b_before - b_oracle
            recovery = (
                1.0 - (b_after - b_oracle) / denom
                if abs(denom) > 1e-12
                else float("nan")  # undefined when the window needs no repair
            )
            flows = flip_flows(_codes(p_before, ctx), _codes(p_after, ctx))
            for metric, value in (
                ("brier_before", b_before),
                ("brier_after", b_after),
                ("brier_oracle", b_oracle),
                ("recovery", recovery),
                ("citl_after", citl.citl),
                ("slope_after", citl.slope),
                ("ece_after", ece(y, p_after, n_bins=15, quantile_edges=True)),
                ("flip_rate_vs_none", flows["n_flips"] / len(y)),
            ):
                rows.append(row(arm, wname, metric, value))
    return rows



def _codes(p: np.ndarray, ctx: Ctx) -> np.ndarray:
    return threshold_policy(p, t_low=ctx.t_low, t_high=ctx.t_high)


def _rf_importance(est, X, y, ctx: Ctx, seed: int):
    """DEV-1 explanation surrogate for random forest.

    Per-feature permutation importance (T8.5 machinery, loss = Brier
    score, R seeded joint permutations of a single-feature block) on the
    capped explanation sample, aggregated to the T5.4 groups. Returns a
    1-row attribution matrix (the stability battery consumes global
    importance vectors) and the 6 group importances. Negative draws are
    clipped to zero: a permutation that improves the loss carries no
    reliance. Brier rather than log loss because deep 400-tree forests
    emit near-0/1 probabilities whose log loss explodes on a 1,000-row
    sample; Brier is bounded and still a proper scoring rule.
    """
    from src.explain.groups import grouped_permutation_importance

    Xa = np.asarray(X, dtype=float)
    singleton = {f"f{j}": [j] for j in range(Xa.shape[1])}
    raw = grouped_permutation_importance(
        lambda Z: est.predict_proba(Z)[:, 1],
        y,
        Xa,
        singleton,
        metric=lambda yy, pp: float(np.mean((pp - yy) ** 2)),
        metric_is_loss=True,
        repeats=RF_PERM_REPEATS,
        seed=seed,
    )
    vec = np.array([max(0.0, raw[f"f{j}"]["mean"])
                    for j in range(Xa.shape[1])], dtype=float)
    V = vec.reshape(1, -1)
    GV = np.array([group_attributions(V, ctx.groups)[g]
                   for g in ctx.group_names])
    return V, GV


def _explanation_matrices(est, family: str, X, y, ctx: Ctx, seed: int):
    if family == "random_forest":
        return _rf_importance(est, X, y, ctx, seed)
    expl = make_explainer(est, background=ctx.windows["train"][:N_BACKGROUND],
                          family=family)
    V = explain_matrix(expl, X)
    GV = np.array([group_attributions(V, ctx.groups)[g]
                   for g in ctx.group_names])
    return V, GV


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


def _stability_rows(family, config_index, kind, i, j, Vi, GVi, Vj, GVj,
                    include_wasserstein: bool = True) -> list:
    block = _stability_block(Vi, Vj)
    if not include_wasserstein:
        # DEV-1: RF explanations are global permutation-importance
        # vectors (one row), so a per-feature attribution distribution
        # does not exist; the metric is omitted for that family.
        block.pop("wasserstein_mean", None)
    rows = []
    for metric, value in {**block, **_grouped_block(GVi, GVj)}.items():
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


def _group_pair_rows(family, config_index, kind, i, j,
                     Vi, Vj, ctx: Ctx) -> list:
    """Per-group restricted stability battery (H2c's noise-floor and
    temporal-similarity inputs): cosine and Spearman of the attribution
    profiles computed on each group's columns only."""
    from scipy.stats import spearmanr

    from src.explain.stability import cosine_importance

    rows = []
    for group in ctx.group_names:
        cols = ctx.groups[group]
        Vi_g, Vj_g = Vi[:, cols], Vj[:, cols]
        gi = np.abs(Vi_g).mean(axis=0)
        gj = np.abs(Vj_g).mean(axis=0)
        sp = float(spearmanr(gi, gj).statistic)
        rows.append(
            {
                "family": family,
                "config_index": config_index,
                "kind": kind,
                "i": i,
                "j": j,
                "group": group,
                "group_cosine": cosine_importance(Vi_g, Vj_g),
                "group_spearman": round(sp, 6) if np.isfinite(sp) else 0.0,
                "group_mean_gap": round(float(np.abs(gi - gj).mean()), 6),
            }
        )
    return rows


OUTPUT_STEMS = (
    "deep_agg", "flip_pairs", "stability_pairs", "temporal_pairs",
    "deep_arms", "group_pairs",
)


def _family_path(stem: str, suffix: str, family: str):
    return TABLES / f"{stem}_{suffix}__{family}.csv"


def _merged_path(stem: str, suffix: str):
    return TABLES / f"{stem}_{suffix}.csv"


def _write_family_tables(rows: dict, suffix: str, family: str) -> None:
    """Write one family's partial tables (crash-safe: called after every
    config, so a crash loses at most the config in flight)."""
    TABLES.mkdir(parents=True, exist_ok=True)
    for stem in OUTPUT_STEMS:
        pd.DataFrame(rows[stem]).to_csv(
            _family_path(stem, suffix, family), index=False
        )


def merge_family_tables(families: tuple, suffix: str) -> None:
    """Concatenate the per-family tables into the merged *_<suffix>.csv
    tables read by scripts.run_real_statistics (identical schema)."""
    for stem in OUTPUT_STEMS:
        parts = []
        for family in families:
            path = _family_path(stem, suffix, family)
            if not path.exists():
                raise FileNotFoundError(
                    f"cannot merge: missing {path.name}; finish that family first"
                )
            parts.append(pd.read_csv(path))
        columns = list(parts[0].columns)
        for family, part in zip(families, parts):
            if list(part.columns) != columns:
                raise ValueError(
                    f"schema mismatch in {stem} for {family}: "
                    f"{list(part.columns)} != {columns}"
                )
        merged = pd.concat(parts, ignore_index=True)
        out = _merged_path(stem, suffix)
        merged.to_csv(out, index=False)
        print(f"merged {out.name} ({len(merged)} rows from "
              f"{len(families)} families)", flush=True)


def _requested_families(spec: str | None) -> tuple:
    if not spec:
        return FAMILY_ORDER_FAST_FIRST
    wanted = [f.strip() for f in spec.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in FAMILY_ORDER_FAST_FIRST]
    if unknown:
        raise SystemExit(f"unknown families {unknown}; "
                         f"expected a subset of {list(FAMILY_ORDER_FAST_FIRST)}")
    # fast-first order regardless of how the caller listed them
    return tuple(f for f in FAMILY_ORDER_FAST_FIRST if f in wanted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true",
                        help="tiny run: logistic regression only, 2 seeds, 3 boots")
    parser.add_argument("--deep-table", type=str, default="deep_subset.csv",
                        help="deep-subset table under reports/tables (pilot may "
                        "point at a synthetic stand-in)")
    parser.add_argument("--families", type=str, default=None,
                        help="comma-separated subset of families; default all "
                        "in fast-first order")
    parser.add_argument("--suffix", type=str, default="real",
                        help="table suffix (real = frozen pipeline)")
    parser.add_argument("--merge-only", action="store_true",
                        help="only concatenate per-family tables; no fitting")
    args = parser.parse_args()

    families = _requested_families(args.families)
    suffix = args.suffix
    if args.pilot:
        if args.families:
            families = _requested_families(args.families)
        else:
            families = ("logistic_regression",)
        suffix = "pilot"

    if args.merge_only:
        merge_family_tables(families, suffix)
        return

    ctx = build_context(pilot=args.pilot)
    n_seeds, n_boots = (N_SEEDS, N_BOOTS)
    if args.pilot:
        n_seeds, n_boots = 2, 3

    models_path = REPO_ROOT / "configs" / "models.yaml"
    models = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    assert int(models["deep_subset"]["configs_per_family"]) == 5

    deep = pd.read_csv(TABLES / args.deep_table)
    spaces = {fam: sample_all_configs(fam, 30, BASE_SEED) for fam in FAMILY_ORDER_FAST_FIRST}

    val_head = ctx.windows["val"][:N_EXPLAIN_CAP]
    val_head_y = ctx.y_eval["val"][:N_EXPLAIN_CAP]
    cohort_heads = {q: X[:N_EXPLAIN_CAP] for q, X in ctx.windows.items()
                    if q not in ("train",)}
    cohort_head_y = {q: y[:N_EXPLAIN_CAP] for q, y in ctx.y_eval.items()
                     if q not in ("train",)}
    for family in families:
        slot = FAMILY_ORDER_FAST_FIRST.index(family)  # global slot: RNG key
        rows = {stem: [] for stem in OUTPUT_STEMS}
        params_list = spaces[family]
        sub = deep[deep["family"] == family]
        for _, rec in sub.iterrows():
            cfg_idx = int(rec["config_index"])
            params = params_list[cfg_idx]
            t0 = time.perf_counter()

            # ---- seed axis: 5 fits, cached predictions, pairwise in RAM
            Ps: dict[int, dict] = {}
            Vs, GVs = {}, {}
            est0 = None
            for k in range(n_seeds):
                est = make_estimator(family, params, BASE_SEED + k).fit(
                    ctx.windows["train"], ctx.y_eval["train"]
                )
                if k == 0:
                    est0 = est
                Ps[k] = {w: est.predict_proba(ctx.windows[w])[:, 1]
                         for w in ctx.windows}
                rows["deep_agg"] += _window_aggregate_rows(
                    Ps[k], ctx, family, cfg_idx, "seed", k)
                rows["deep_arms"] += _arm_point_rows(
                    Ps[k], ctx, family, cfg_idx, "seed", k)
                Vs[k], GVs[k] = _explanation_matrices(
                    est, family, val_head, val_head_y, ctx,
                    BASE_SEED + 11_000_000 + k)
            CK = {k: {w: _codes(Ps[k][w], ctx) for w in ctx.windows} for k in Ps}
            for i in range(n_seeds):
                for j in range(i + 1, n_seeds):
                    rows["flip_pairs"] += _pair_flip_rows(
                        family, cfg_idx, "seed", i, j, CK[i], CK[j], ctx)
                    rows["stability_pairs"] += _stability_rows(
                        family, cfg_idx, "seed", i, j,
                        Vs[i], GVs[i], Vs[j], GVs[j],
                        include_wasserstein=family != "random_forest")
                    rows["group_pairs"] += _group_pair_rows(
                        family, cfg_idx, "seed", i, j, Vs[i], Vs[j], ctx)
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
                rows["deep_agg"] += _window_aggregate_rows(
                    P_b[b], ctx, family, cfg_idx, "boot", b)
                rows["deep_arms"] += _arm_point_rows(
                    P_b[b], ctx, family, cfg_idx, "boot", b)
                V_b[b], GV_b[b] = _explanation_matrices(
                    est, family, val_head, val_head_y, ctx,
                    BASE_SEED + 12_000_000 + b)
            CK_b = {b: {w: _codes(P_b[b][w], ctx) for w in ctx.windows} for b in P_b}
            for i in range(n_boots):
                for j in range(i + 1, n_boots):
                    rows["flip_pairs"] += _pair_flip_rows(
                        family, cfg_idx, "boot", i, j, CK_b[i], CK_b[j], ctx)
                    rows["stability_pairs"] += _stability_rows(
                        family, cfg_idx, "boot", i, j,
                        V_b[i], GV_b[i], V_b[j], GV_b[j],
                        include_wasserstein=family != "random_forest")
                    rows["group_pairs"] += _group_pair_rows(
                        family, cfg_idx, "boot", i, j, V_b[i], V_b[j], ctx)
            del P_b, CK_b

            # ---- frozen model (seed 0): cross-cohort explanation battery
            # (est0 is the seed-0 fit already trained above: identical
            # params and seed, so no refit is needed)
            Vw, GVw = {}, {}
            for w_idx, (wname, head) in enumerate(cohort_heads.items()):
                Vw[wname], GVw[wname] = _explanation_matrices(
                    est0, family, head, cohort_head_y[wname], ctx,
                    BASE_SEED + 21_000_000 + w_idx)
            wnames = ["val"] + sorted(w for w in Vw if w != "val")
            for a in range(len(wnames)):
                for b in range(a + 1, len(wnames)):
                    wi, wj = wnames[a], wnames[b]
                    stab = {**_stability_block(Vw[wi], Vw[wj]),
                            **_grouped_block(GVw[wi], GVw[wj])}
                    if family == "random_forest":
                        stab.pop("wasserstein_mean", None)
                    for metric, value in stab.items():
                        rows["temporal_pairs"].append({
                            "family": family,
                            "config_index": cfg_idx,
                            "window_i": wi,
                            "window_j": wj,
                            "metric": metric,
                            "value": round(float(value), 6),
                        })
                    rows["group_pairs"] += _group_pair_rows(
                        family, cfg_idx, "cohort", wi, wj, Vw[wi], Vw[wj], ctx
                    )
            seconds = time.perf_counter() - t0
            _write_family_tables(rows, suffix, family)
            print(f"{family:>20} cfg{cfg_idx:<2} | seeds {n_seeds} + boots "
                  f"{n_boots} | {seconds:.1f}s | partial tables written",
                  flush=True)

    merge_family_tables(families, suffix)


if __name__ == "__main__":
    main()
