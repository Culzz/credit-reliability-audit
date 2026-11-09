"""Confirmatory statistics on the frozen real-cohort battery (Phase 7).

Implements protocol/HYPOTHESES.md literally on the real battery tables
(*_real.csv under reports/tables):

- H1a  CITL deterioration, shifted cohorts vs validation (delta > 0),
       cluster bootstrap over configurations, B = 2000.
- H1b  (exploratory) slope change vs validation, own-scale magnitudes.
- H1c  recalibration recovery (partial recovery iff CI excludes 1),
       per arm x lag cells; lag-dependence via rank ordering.
- H2c  per-group temporal similarity vs the seed/bootstrap noise floor
       at its 2.5th percentile; share of (family, group) cells exceeding.
- H3a  recalibration flips vs the seed-vs-seed floor (T9.2d) 95th pct.
- RQ4.1-4.4  within-pair Spearman over the configurations (averaged
       over the seven adjacent test-cohort pairs) of AUC_t vs the
       reliability outcome at t+1, each delta-tested against the paired
       AUC-vs-AUC baseline on a shared cluster bootstrap design
       (cluster = configuration, per statistics.yaml rq4).
- Holm step-down applied ONCE over the 8 confirmatory p-values.

Unit of analysis: configuration (family x config_index); seed-axis rows
are averaged within cohort first. Every quantity is computed from the
battery tables; bootstrap seeds are fixed explicit literals.

Outputs: reports/tables/real_noise_floors.csv
         reports/tables/recovery_cells_real.csv
         reports/tables/rq4_cells_real.csv
         reports/tables/real_statistics.csv
Use    : python -m uv run python -m scripts.run_real_statistics
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts._common import TABLES
from src.statistics.inference import holm_pvalues

B_HEADLINE = 2000        # statistics.yaml bootstrap.B_headline
STAT_SEED = 20261109     # fixed bootstrap-design seed (explicit literal)
CUTOFF_COHORTS = tuple(  # ratified test cohorts, frozen order
    f"{y}Q{q}" for y, q in [
        (2014, 1), (2014, 2), (2014, 3), (2014, 4),
        (2015, 1), (2015, 2), (2015, 3), (2015, 4),
    ]
)
LAGS = {  # quarters from the end of the calibration window (2013Q2)
    "2014Q1": 3, "2014Q2": 4, "2014Q3": 5, "2014Q4": 6,
    "2015Q1": 7, "2015Q2": 8, "2015Q3": 9, "2015Q4": 10,
}
ADJACENT_PAIRS = tuple(itertools.pairwise(CUTOFF_COHORTS))
EQUIVALENCE = (-0.3, 0.3)  # rq4.equivalence_region_spearman
PRACTICAL_ANCHOR = 0.02    # H1a anchor: refer-band half-width
RQ_KEYS = ("ece", "slope_dev", "expl_instability", "flip_rate")
RQ_IDS = {
    "ece": "RQ4.1", "slope_dev": "RQ4.2",
    "expl_instability": "RQ4.3", "flip_rate": "RQ4.4",
}
CONFIRMATORY = ("H1a", "H1c", "H2c", "H3a", "RQ4.1", "RQ4.2",
                "RQ4.3", "RQ4.4")


def _q(x, p: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), p))


def _boot_p(delta_hat: float, deltas: np.ndarray) -> float:
    """Two-sided bootstrap p against 0."""
    if not np.isfinite(delta_hat) or delta_hat == 0.0:
        return 1.0
    p_le = float((deltas <= 0).mean())
    p_ge = float((deltas >= 0).mean())
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _boot_p_vs_reference(delta_hat: float, deltas: np.ndarray,
                         reference: float) -> float:
    """Two-sided bootstrap p against a fixed reference value."""
    return _boot_p(delta_hat - reference, deltas - reference)


def _cluster_idxs(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def _region_ok(name_lo_hi, lo: float, hi: float) -> str:
    low, high = name_lo_hi
    if lo > low and hi < high:
        return "negligible"
    if (lo < low or hi > high) and (lo > high or hi < low):
        return "meaningful"
    return "inconclusive"


def load_tables() -> dict:
    names = (
        "deep_agg_real", "deep_arms_real", "flip_pairs_real",
        "stability_pairs_real", "temporal_pairs_real", "group_pairs_real",
    )
    out = {}
    for name in names:
        path = TABLES / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing battery table {path.name}: run scripts.run_deep_battery"
            )
        out[name] = pd.read_csv(path)
    return out


# ---------------------------------------------------------------- floors
def noise_floors(stab: pd.DataFrame, groups: pd.DataFrame,
                 flips: pd.DataFrame) -> pd.DataFrame:
    """T8.2 Axis C floors (seed + bootstrap axes pooled) and the T9.2d
    seed-vs-seed flip floor, per family."""
    rows = []
    for fam, sub in stab.groupby("family"):
        pooled = pd.concat([sub[sub.kind == "seed"], sub[sub.kind == "boot"]])
        for metric, msub in pooled.groupby("metric"):
            vals = msub["value"].to_numpy()
            rows.append({
                "family": fam, "dimension": "stability", "label": metric,
                "q2.5": _q(vals, 2.5), "q50": _q(vals, 50.0),
                "q95": _q(vals, 95.0), "n_pairs": len(vals),
            })
    for fam, sub in groups.groupby("family"):
        pooled = pd.concat([sub[sub.kind == "seed"], sub[sub.kind == "boot"]])
        for grp, gsub in pooled.groupby("group"):
            for col in ("group_cosine", "group_spearman", "group_mean_gap"):
                vals = gsub[col].to_numpy()
                rows.append({
                    "family": fam, "dimension": "group",
                    "label": f"{grp}:{col}",
                    "q2.5": _q(vals, 2.5), "q50": _q(vals, 50.0),
                    "q95": _q(vals, 97.5), "n_pairs": len(vals),
                })
    seed_flips = flips[(flips.kind == "seed") & (flips.window != "train")]
    for fam, sub in seed_flips.groupby("family"):
        vals = sub["flip_rate"].to_numpy()
        rows.append({
            "family": fam, "dimension": "flip", "label": "seed_floor",
            "q2.5": _q(vals, 2.5), "q50": _q(vals, 50.0),
            "q95": _q(vals, 95.0), "n_pairs": len(vals),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------- per-config units
def window_units(agg: pd.DataFrame) -> pd.DataFrame:
    """Seed-axis means per (family, config_index, window); metric columns."""
    seed = agg[agg.kind == "seed"]
    units = seed.groupby(["family", "config_index", "window", "metric"])[
        "value"].mean()
    return units.unstack("metric").reset_index().rename_axis(columns=None)


def h1c_cells(arms: pd.DataFrame) -> pd.DataFrame:
    """Per (family, config, arm, cohort) recovery means over seed refits."""
    sub = arms[(arms.kind == "seed") & (arms.arm != "none")
               & (arms.window.isin(CUTOFF_COHORTS))
               & (arms.metric == "recovery")]
    cells = sub.groupby(["family", "config_index", "arm", "window"])["value"].mean()
    cells = cells.reset_index().rename(columns={"value": "recovery",
                                                "window": "cohort"})
    cells["lag"] = cells["cohort"].map(LAGS)
    return cells


def flip_units(flips: pd.DataFrame) -> pd.DataFrame:
    """Seed-pair mean flip rate per (family, config, eval window)."""
    sub = flips[(flips.kind == "seed") & (flips.window != "train")]
    grouped = sub.groupby(["family", "config_index", "window"])["flip_rate"].mean()
    return grouped.reset_index().rename(columns={"flip_rate": "flip_rate_seed"})


def h1a_statistic(units: pd.DataFrame, fams: list):
    """H1a: mean over configurations of the per-config mean, over the
    eight ratified cohorts, of |CITL_cohort| - |CITL_val|."""
    fam_cfg_pairs: list[tuple] = []
    per_cfg: list[float] = []
    autofam = units.groupby("family")["config_index"].unique()
    for fam in fams:
        for cfg in sorted(autofam[fam]):
            citl = (units[(units.family == fam)
                          & (units.config_index == cfg)]
                    .set_index("window"))["citl"]
            per_cfg.append(float(np.mean(
                [abs(citl[c]) - abs(citl["val"]) for c in CUTOFF_COHORTS])))
            fam_cfg_pairs.append(fam)
    per_cfg = np.asarray(per_cfg, dtype=float)
    rng = np.random.default_rng(STAT_SEED)
    deltas = np.array([per_cfg[_cluster_idxs(len(per_cfg), rng)].mean()
                       for _ in range(B_HEADLINE)])
    desc = {
        fam: float(np.mean(per_cfg[[i for i, f in enumerate(fam_cfg_pairs)
                                    if f == fam]]))
        for fam in fams
    }
    return per_cfg, deltas, desc


def h1b_slopes(units: pd.DataFrame, fams: list) -> dict:
    """H1b (exploratory): mean slope delta vs validation per family."""
    out = {}
    autofam = units.groupby("family")["config_index"].unique()
    for fam in fams:
        vals = []
        for cfg in sorted(autofam[fam]):
            slope = (units[(units.family == fam)
                           & (units.config_index == cfg)]
                     .set_index("window"))["slope"]
            vals.append(float(np.mean([slope[c] - slope["val"]
                                       for c in CUTOFF_COHORTS])))
        out[fam] = float(np.mean(vals))
    return out


def h1c_statistic(cells: pd.DataFrame):
    """H1c: pooled recovery with cluster bootstrap over configurations;
    lag-dependence Spearman over arm x lag cell means."""
    clusters = cells.groupby(["family", "config_index"])["recovery"].mean()
    vals = clusters.to_numpy()
    rng = np.random.default_rng(STAT_SEED)
    deltas = np.array([vals[_cluster_idxs(len(vals), rng)].mean()
                       for _ in range(B_HEADLINE)])
    lag_mean = cells.groupby("lag")["recovery"].mean()
    order = spearmanr(lag_mean.index.to_numpy(), lag_mean.to_numpy()).statistic
    return vals, deltas, lag_mean, float(order)


def h2c_statistic(group_pairs: pd.DataFrame, floors_g: pd.DataFrame,
                  fams: list):
    """H2c: share of (family, group) cells whose temporal median
    group_cosine sits below the floor's 2.5th percentile."""
    temp = group_pairs[(group_pairs.kind == "cohort")
                       & group_pairs.i.isin(CUTOFF_COHORTS)
                       & group_pairs.j.isin(CUTOFF_COHORTS)]
    flags = []
    for fam in fams:
        f_fam = floors_g[floors_g.family == fam]
        t_fam = temp[temp.family == fam]
        for grp in sorted(t_fam["group"].unique()):
            f_g = f_fam[f_fam.label == f"{grp}:group_cosine"]
            if f_g.empty:
                continue
            floor_lo = float(f_g["q2.5"].iloc[0])
            vals = t_fam[t_fam.group == grp]["group_cosine"].to_numpy()
            flags.append(float(np.median(vals)) < floor_lo)
    flags = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(STAT_SEED)
    shares = np.array([flags[_cluster_idxs(len(flags), rng)].mean()
                       for _ in range(B_HEADLINE)])
    share = float(flags.mean())
    return share, shares


def h3a_statistic(arms: pd.DataFrame, floors_flip: pd.DataFrame,
                  fams: list):
    """H3a: mean recalibration flip rate (non-none arms, eval windows,
    seed axis) minus the family seed-floor 95th percentile."""
    sub = arms[(arms.kind == "seed") & (arms.arm != "none")
               & (arms.window != "train")
               & (arms.metric == "flip_rate_vs_none")]
    obs = sub.groupby(["family", "config_index"])["value"].mean().reset_index()
    fam_floor = {r["family"]: float(r["q95"])
                 for _, r in floors_flip.iterrows()}
    fam_stat = {}
    for fam in fams:
        if fam not in fam_floor:
            continue
        fam_obs = obs[obs.family == fam]["value"].to_numpy()
        fam_stat[fam] = float(fam_obs.mean() - fam_floor[fam])
    pooled = float(np.mean(list(fam_stat.values())))
    ids = obs[["family", "config_index"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(STAT_SEED)
    deltas = np.empty(B_HEADLINE)
    for b in range(B_HEADLINE):
        picks = ids.iloc[_cluster_idxs(len(ids), rng)]
        sample = obs.merge(picks, on=["family", "config_index"], how="inner")
        fam_obs = sample.groupby("family")["value"].mean()
        deltas[b] = float(np.mean([
            fam_obs[f] - fam_floor[f] for f in fam_obs.index if f in fam_floor
        ]))
    return pooled, deltas, fam_stat, obs


# ============================ RQ4.1-4.4 machinery ==========================
def _pairwise_auc_stab(units: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    """Frozen-model jaccard_top10 of adjacent cohort pairs, per cluster."""
    t = temporal[(temporal.window_i.isin(CUTOFF_COHORTS))
                 & (temporal.window_j.isin(CUTOFF_COHORTS))]
    pairs = {p[0]: p[1] for p in ADJACENT_PAIRS}
    t = t[t.window_j == t.window_i.map(pairs)]
    t = t[t.metric == "jaccard_top10"]
    piv = t.pivot_table(index=["family", "config_index"], columns="window_i",
                        values="value")
    return piv


def rq4_statistics(units: pd.DataFrame, fu: pd.DataFrame,
                   temporal: pd.DataFrame) -> pd.DataFrame:
    """Pooled RQ4.x deltas with one shared cluster bootstrap design.

    Per adjacent cohort pair (t, t+1): Spearman over the configuration
    points for AUC_t vs each outcome at t+1, and vs AUC_t+1 (baseline).
    The pooled estimate averages the per-pair rhos; the delta to the
    baseline is recomputed on every resample of the SAME cluster design
    (clusters resampled with replacement, scipy Spearman handles ties)."""
    cluster_keys = (units[["family", "config_index"]]
                    .drop_duplicates().reset_index(drop=True))
    idx = pd.MultiIndex.from_frame(cluster_keys)

    pivot = lambda col: (units.pivot_table(index=["family", "config_index"],
                                           columns="window", values=col)
                         .reindex(idx))
    AUC, ECE, SLO = pivot("roc_auc"), pivot("ece_b15q"), pivot("slope")
    FLIP = (fu.pivot_table(index=["family", "config_index"], columns="window",
                           values="flip_rate_seed").reindex(idx))
    STAB = _pairwise_auc_stab(units, temporal).reindex(idx)

    def rho(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return np.nan
        v = spearmanr(a[mask], b[mask]).statistic
        return float(v) if np.isfinite(v) else np.nan

    rng = np.random.default_rng(STAT_SEED)
    designs = [_cluster_idxs(len(cluster_keys), rng) for _ in range(B_HEADLINE)]

    boot_rho = {k: np.empty((B_HEADLINE, len(ADJACENT_PAIRS)))
                for k in (*RQ_KEYS, "auc")}
    point_rho = {k: np.empty(len(ADJACENT_PAIRS)) for k in (*RQ_KEYS, "auc")}

    for p_idx, (t, t1) in enumerate(ADJACENT_PAIRS):
        a_t, a_t1 = AUC[t].to_numpy(), AUC[t1].to_numpy()
        metrics = {
            "ece": ECE[t1].to_numpy(),
            "slope_dev": np.abs(1.0 - SLO[t1].to_numpy()),
            "expl_instability": STAB[t].to_numpy(),
            "flip_rate": FLIP[t1].to_numpy(),
            "auc": a_t1,
        }
        point_rho["auc"][p_idx] = rho(a_t, a_t1)
        for key, m in metrics.items():
            point_rho[key][p_idx] = rho(a_t, m)
        for b, sel in enumerate(designs):
            ra_t, ra_t1 = a_t[sel], a_t1[sel]
            boot_rho["auc"][b, p_idx] = rho(ra_t, ra_t1)
            for key in RQ_KEYS:
                boot_rho[key][b, p_idx] = rho(ra_t, metrics[key][sel])

    rows = []
    n_clusters = len(cluster_keys)
    def pooled(arr2d):
        vals = arr2d[np.isfinite(arr2d)]
        return float(vals.mean()) if len(vals) else np.nan

    for key in RQ_KEYS:
        boot_point = np.array([pooled(boot_rho[key][b])
                               for b in range(B_HEADLINE)])
        delta_point = (pooled(point_rho[key])
                       - pooled(point_rho["auc"]))
        rho_base_boot = np.array([pooled(boot_rho["auc"][b])
                                  for b in range(B_HEADLINE)])
        delta_boot = boot_point - rho_base_boot
        p = _boot_p(delta_point, delta_boot)
        n_pairs = int(np.isfinite(point_rho[key]).sum())
        rows.append({
            "key": key, "n_clusters": n_clusters, "n_pairs": n_pairs,
            "rho_outcome": round(pooled(point_rho[key]), 6),
            "rho_base": round(pooled(point_rho["auc"]), 6),
            "delta": round(delta_point, 6),
            "rho_ci_low": round(_q(boot_point, 2.5), 6),
            "rho_ci_high": round(_q(boot_point, 97.5), 6),
            "ci_low": round(_q(delta_boot, 2.5), 6),
            "ci_high": round(_q(delta_boot, 97.5), 6),
            "p_raw": round(p, 6),
            "region": _region_ok(EQUIVALENCE, _q(boot_point, 2.5),
                                 _q(boot_point, 97.5)),
        })
    return pd.DataFrame(rows)


# ================================ main =====================================
def main() -> None:
    t0 = time.perf_counter()
    tables = load_tables()
    units = window_units(tables["deep_agg_real"])
    fams = sorted(units["family"].unique())
    floors = noise_floors(tables["stability_pairs_real"],
                          tables["group_pairs_real"],
                          tables["flip_pairs_real"])
    floors.to_csv(TABLES / "real_noise_floors.csv", index=False)

    stats_rows: list[dict] = []
    descriptives: list[dict] = []

    def add_row(hyp, scope, statistic, value, lo, hi, p_raw, note=""):
        stats_rows.append({
            "hypothesis": hyp, "scope": scope, "statistic": statistic,
            "value": round(float(value), 6) if np.isfinite(value) else np.nan,
            "ci_low": round(float(lo), 6) if np.isfinite(lo) else np.nan,
            "ci_high": round(float(hi), 6) if np.isfinite(hi) else np.nan,
            "p_raw": round(float(p_raw), 6) if np.isfinite(p_raw) else np.nan,
            "note": note,
        })

    # ---- H1a
    per_cfg, deltas, h1a_desc = h1a_statistic(units, fams)
    add_row("H1a", f"{len(per_cfg)} configurations",
            "mean(|CITL_cohort| - |CITL_val|)", float(per_cfg.mean()),
            _q(deltas, 2.5), _q(deltas, 97.5), _boot_p(float(per_cfg.mean()), deltas),
            f"practical anchor |dCITL| >= {PRACTICAL_ANCHOR}")
    for fam, val in h1a_desc.items():
        descriptives.append({"hypothesis": "H1a", "family": fam,
                             "value": round(val, 6)})

    # ---- H1b (exploratory, own-scale descriptives)
    for fam, val in h1b_slopes(units, fams).items():
        descriptives.append({"hypothesis": "H1b", "family": fam,
                             "value": round(val, 6)})

    # ---- H1c
    cells = h1c_cells(tables["deep_arms_real"])
    cells.to_csv(TABLES / "recovery_cells_real.csv", index=False)
    vals, deltas_c, _lag_mean, lag_order = h1c_statistic(cells)
    rec_hat = float(vals.mean())
    add_row("H1c", f"{len(vals)} configurations x "
            f"{cells.arm.nunique()} arms x {len(CUTOFF_COHORTS)} cohorts",
            "mean recovery", rec_hat, _q(deltas_c, 2.5), _q(deltas_c, 97.5),
            _boot_p_vs_reference(rec_hat, deltas_c, 1.0),
            "partial recovery iff CI excludes 1")
    add_row("H1c-lag", "arm x lag cell means", "Spearman(lag, recovery)",
            lag_order, np.nan, np.nan, np.nan,
            "exploratory within H1c")

    # ---- H2c
    share, shares = h2c_statistic(tables["group_pairs_real"],
                                  floors[floors.dimension == "group"], fams)
    add_row("H2c", f"{int(shares.size and len(shares))} family-group cells",
            "share below floor 2.5th pct (group_cosine)", share,
            _q(shares, 2.5), _q(shares, 97.5), _boot_p(share, shares),
            "temporal median < floor q2.5 per cell")

    # ---- H3a
    pooled, deltas_f, fam_stat, obs = h3a_statistic(
        tables["deep_arms_real"], floors[floors.dimension == "flip"], fams)
    add_row("H3a", f"{len(obs)} configuration x family rows",
            "mean recalibration flip rate - seed floor q95", pooled,
            _q(deltas_f, 2.5), _q(deltas_f, 97.5), _boot_p(pooled, deltas_f),
            "floor per family (T9.2d)")
    for fam, val in fam_stat.items():
        descriptives.append({"hypothesis": "H3a", "family": fam,
                             "value": round(val, 6)})

    # ---- RQ4.x
    rq = rq4_statistics(units, flip_units(tables["flip_pairs_real"]),
                        tables["temporal_pairs_real"])
    rq.to_csv(TABLES / "rq4_cells_real.csv", index=False)
    for _, row in rq.iterrows():
        add_row(RQ_IDS[row.key],
                f"{row.n_clusters} clusters, {row.n_pairs} pairs",
                f"delta rho(AUC_t, {row.key}_t+1) vs rho(AUC_t, AUC_t+1)",
                row.delta, row.ci_low, row.ci_high, row.p_raw,
                f"rho outcome {row.rho_outcome} (CI "
                f"{row.rho_ci_low}..{row.rho_ci_high}, region {row.region}); "
                f"baseline {row.rho_base}")

    # ---- Holm once, over the 8 confirmatory p-values
    conf = [r for r in stats_rows if r["hypothesis"] in CONFIRMATORY]
    adj = holm_pvalues([r["p_raw"] for r in conf])
    for r, a in zip(conf, adj):
        r["p_holm"] = round(float(a), 6)
        r["verdict"] = "reject_H0" if a < 0.05 else "not_rejected"
    for r in stats_rows:
        r.setdefault("p_holm", np.nan)
        r.setdefault("verdict", "exploratory")

    out = pd.DataFrame(stats_rows)
    out.to_csv(TABLES / "real_statistics.csv", index=False)
    pd.DataFrame(descriptives).to_csv(TABLES / "real_descriptives.csv",
                                      index=False)
    print(out[["hypothesis", "statistic", "value", "ci_low", "ci_high",
               "p_raw", "p_holm", "verdict"]].to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

