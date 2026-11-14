"""Post-hoc analyses (labeled, not confirmatory) added after the frozen
pipeline ran, in response to external review. They reuse committed
per-configuration tables only; no raw data is required.

Outputs `reports/tables/post_hoc_analyses.csv` with columns
    analysis, label, statistic, value, ci_low, ci_high, n, note

Analyses:
- rq4_between    Spearman rho(AUC_t, outcome_{t+1}) across the 20 deep
                 configurations, pooled over the 7 adjacent cohort pairs
                 ("does a higher-AUC configuration score better on the
                 reliability outcome?"). Cluster bootstrap over configs.
- rq4_within     pooled Pearson corr of (dAUC, dOutcome) over config x pair,
                 both deltas demeaned ("does a configuration whose AUC
                 changes see its reliability change?"). Cluster bootstrap.
- rq4_leadlag    OLS coefficient of AUC_t in outcome_{t+1} ~ outcome_t + AUC_t
                 ("does AUC at t predict the outcome at t+1 beyond the
                 outcome at t?"). Cluster bootstrap over configs.
- h1c_abs_brier  mean Brier change (after - before) per family, Brier units;
                 per label age without CI (few clusters per cell).
- h1c_abs_citl   mean CITL change (recalibrated - uncalibrated) per family.
- h1c_orc_gain   distribution of the oracle gain (brier_before -
                 brier_oracle) that denominates the frozen H1c ratio.
- shift_dose     Spearman, across the 8 test cohorts, of the adversarial
                 validation AUC vs the mean |CITL| / |ECE| change.
- hy_granularity H1a-style CITL change at half-year cohort granularity.

Bootstrap: cluster (config_index) percentile, B = 1000,
STAT_SEED = 20261109. Every row is labeled post-hoc in the manuscript
appendix; none of these estimates enters a confirmatory test.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TABLES = pathlib.Path(__file__).resolve().parents[1] / "reports" / "tables"
STAT_SEED = 20261109
B = 1000
ADJACENT = [(f"{y}Q{q}", f"{y}Q{q+1}" if q < 4 else f"{y+1}Q1")
            for y, qs in [(2014, (1, 2, 3, 4)), (2015, (1, 2, 3))]
            for q in qs]  # 7 pairs 2014Q1 -> ... -> 2015Q4
ROWS: list[dict] = []


def add(analysis, label, statistic, value, lo=np.nan, hi=np.nan,
        n=np.nan, note=""):
    ROWS.append({"analysis": analysis, "label": label, "statistic": statistic,
                 "value": round(float(value), 6) if np.isfinite(value) else np.nan,
                 "ci_low": round(float(lo), 6) if np.isfinite(lo) else np.nan,
                 "ci_high": round(float(hi), 6) if np.isfinite(hi) else np.nan,
                 "n": n, "note": note})


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def boot_ci(stat_fn, rng, n_clusters: int):
    stats = np.asarray([stat_fn(rng.integers(0, n_clusters, n_clusters))
                        for _ in range(B)], dtype=float)
    return float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))


# ---------------------------------------------------------------- matrices
# The unit of analysis is (family, config_index) pairs: config_index values
# repeat across families, so keys must include the family.
agg = load("deep_agg_real")
agg = agg[agg["kind"] == "seed"]
units = sorted({(fam, int(cfg)) for fam, cfg in
                agg.groupby(["family", "config_index"]).groups})
vals = {}
for (fam, cfg, w, m), v in agg.set_index(
        ["family", "config_index", "window", "metric"]).value.items():
    vals[(fam, int(cfg), w, m)] = float(v)
n_cfg = len(units)  # 20


def cell(fam, cfg, w, m):
    return vals.get((fam, cfg, w, m), np.nan)


def slope_dev_fn(fam, cfg, w):
    return abs(cell(fam, cfg, w, "slope") - 1.0)


fp = load("flip_pairs_real")
fp = fp[(fp["kind"] == "seed")
        & fp.window.isin([w for p in ADJACENT for w in p])]
flip_mat = fp.groupby(["family", "config_index", "window"])["flip_rate"].mean()
flip_lookup = {(f, int(c), w): float(v)
               for (f, c, w), v in flip_mat.items()}


def flip_rate_fn(fam, cfg, w):
    return flip_lookup.get((fam, cfg, w), np.nan)


OUTCOMES = {
    "ece": lambda fam, cfg, w: cell(fam, cfg, w, "ece_b15q"),
    "slope_dev": slope_dev_fn,
    "flip_rate": flip_rate_fn,
}

rng = np.random.default_rng(STAT_SEED)


def between_rho(idx, outcome_fn):
    """Mean over pairs of Spearman rho across the resampled units.
    `idx` holds resampled cluster indices into 0..n_units-1."""
    rhos = []
    for t, tp in ADJACENT:
        sel = [units[j] for j in idx]
        xs = np.array([cell(f, c, t, "roc_auc") for f, c in sel])
        ys = np.array([outcome_fn(f, c, tp) for f, c in sel])
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 3 or np.std(xs[ok]) == 0 or np.std(ys[ok]) == 0:
            rhos.append(np.nan)
            continue
        rhos.append(spearmanr(xs[ok], ys[ok]).statistic)
    return float(np.nanmean(rhos))


def within_corr(idx, outcome_fn):
    """Pooled Pearson corr of (dAUC, dOutcome), both demeaned, over the
    resampled units x 7 pairs."""
    dx, dy = [], []
    for j in idx:
        fam, cfg = units[j]
        for t, tp in ADJACENT:
            da = cell(fam, cfg, tp, "roc_auc") - cell(fam, cfg, t, "roc_auc")
            dm = outcome_fn(fam, cfg, tp) - outcome_fn(fam, cfg, t)
            if np.isfinite(da) and np.isfinite(dm):
                dx.append(da)
                dy.append(dm)
    dx, dy = np.asarray(dx), np.asarray(dy)
    if len(dx) < 20:
        return np.nan
    dx, dy = dx - dx.mean(), dy - dy.mean()
    if np.std(dx) == 0 or np.std(dy) == 0:
        return np.nan
    return float(np.corrcoef(dx, dy)[0, 1])


def leadlag_coef(idx, outcome_fn):
    """OLS outcome_{t+1} ~ outcome_t + AUC_t (both demeaned); returns the
    AUC_t coefficient, pooled over resampled units x 7 pairs."""
    X0, X1, y = [], [], []
    for j in idx:
        fam, cfg = units[j]
        for t, tp in ADJACENT:
            m_t = outcome_fn(fam, cfg, t)
            m_tp = outcome_fn(fam, cfg, tp)
            a_t = cell(fam, cfg, t, "roc_auc")
            if all(np.isfinite(v) for v in (m_t, m_tp, a_t)):
                X0.append(m_t)
                X1.append(a_t)
                y.append(m_tp)
    if len(y) < 20:
        return np.nan
    X0, X1, y = map(np.asarray, (X0, X1, y))
    Xd = np.column_stack([np.ones(len(y)), X0 - X0.mean(), X1 - X1.mean()])
    yd = y - y.mean()
    coef, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    return float(coef[2])


for name, fn in OUTCOMES.items():
    ids = np.arange(n_cfg)
    point = between_rho(ids, fn)
    lo, hi = boot_ci(lambda i, fn=fn: between_rho(i, fn), rng, n_cfg)
    add("rq4_between", name,
        "mean Spearman rho(AUC_t, outcome_t+1) across configs",
        point, lo, hi, n_cfg, "post-hoc; resampling unit = configuration")

    pw = within_corr(ids, fn)
    lo_w, hi_w = boot_ci(lambda i, fn=fn: within_corr(i, fn), rng, n_cfg)
    add("rq4_within", name,
        "pooled demeaned Pearson corr(dAUC, dOutcome), config x pair",
        pw, lo_w, hi_w, n_cfg, "post-hoc")

    pl = leadlag_coef(ids, fn)
    lo_l, hi_l = boot_ci(lambda i, fn=fn: leadlag_coef(i, fn), rng, n_cfg)
    add("rq4_leadlag", name,
        "OLS coefficient of AUC_t in outcome_t+1 ~ outcome_t + AUC_t",
        pl, lo_l, hi_l, n_cfg, "post-hoc")

# ------------------------------------------------- H1c absolute scale
arms_long = load("deep_arms_real")
arms = (arms_long[(arms_long["kind"] == "seed") & (arms_long.arm != "none")]
        .pivot_table(index=["family", "config_index", "kind", "refit_id",
                            "arm", "window"],
                     columns="metric", values="value")
        .reset_index())
arms["dbrier"] = arms.brier_after - arms.brier_before
arms["oracle_gain"] = arms.brier_before - arms.brier_oracle

for fam, sub in arms.groupby("family"):
    cfgs = np.array(sorted(sub.config_index.unique()))
    point = float(sub[sub.config_index.isin(cfgs)].dbrier.mean())
    lo, hi = boot_ci(
        lambda i, sub=sub, cfgs=cfgs:
            float(sub[sub.config_index.isin(cfgs[i])].dbrier.mean()),
        rng, len(cfgs))
    add("h1c_abs_brier", fam, "mean Brier change (after - before)",
        point, lo, hi, len(cfgs), "post-hoc; Brier units; arms pooled")

LAGS = [w for w in sorted(arms.window.unique())
        if w.startswith("20")]  # the 8 test cohorts only
for fam, sub in arms[arms.window.isin(LAGS)].groupby("family"):
    for w, s in sub.groupby("window"):
        add("h1c_abs_brier_lag", f"{fam}@{w}",
            "mean Brier change (after - before)", s.dbrier.mean(),
            n=int(s.config_index.nunique()),
            note="post-hoc; per label age; no CI (few clusters per cell)")

for fam, sub in arms.groupby("family"):
    cfgs = np.array(sorted(sub.config_index.unique()))
    add("h1c_orc_gain", fam, "median oracle gain (Brier units)",
        float(sub.oracle_gain.median()), n=len(cfgs),
        note="post-hoc; denominator of the frozen recovery ratio")
add("h1c_orc_gain", "share_gain_below_1e3",
    "share of cells with oracle gain < 1e-3",
    float((arms.oracle_gain < 1e-3).mean()),
    n=int(arms.oracle_gain.notna().sum()),
    note="post-hoc; ill-conditioning of the frozen ratio")

# CITL change: recalibrated vs uncalibrated on the same refits
agg_all = load("deep_agg_real")
citl_b = (agg_all[(agg_all.metric == "citl") & agg_all.window.isin(LAGS)]
          .set_index(["family", "config_index", "kind", "refit_id",
                      "window"]).value)
a_idx = pd.MultiIndex.from_arrays(
    [arms.family, arms.config_index, arms["kind"], arms.refit_id, arms.window])
dcitl = (arms.set_index(["family", "config_index", "kind", "refit_id",
                         "window"]).citl_after
         - citl_b.reindex(a_idx)).dropna()
for fam, s in dcitl.groupby(level=0):
    add("h1c_abs_citl", fam,
        "mean CITL change (after - uncalibrated)", float(s.mean()),
        n=int(s.index.get_level_values("config_index").nunique()),
        note="post-hoc; CITL units")

# ------------------------------------------------- shift dose-response
adv_raw = load("adversarial_validation")
adv = (adv_raw[adv_raw.eval_cohort.isin(LAGS)]
       .set_index("eval_cohort").auc)
per_cohort = (agg[agg.metric == "citl"]
              .pivot_table(index="window", columns="family",
                           values="value", aggfunc="mean"))
val_abs = (agg[(agg.metric == "citl") & (agg.window == "val")]
           .groupby("family").value.mean())
delta = {w: float((per_cohort.loc[w] - val_abs).abs().mean())
         for w in LAGS if w in per_cohort.index}
xs = np.array([adv[w] for w in delta])
ys = np.array(list(delta.values()))
add("shift_dose", "citl_change",
    "Spearman(shift AUC, mean |CITL change|)", float(spearmanr(xs, ys).statistic),
    n=len(delta), note="post-hoc; 8 cohorts, no CI")

ece_cohort = (agg[agg.metric == "ece_b15q"]
              .pivot_table(index="window", columns="family",
                           values="value", aggfunc="mean"))
ece_val = (agg[(agg.metric == "ece_b15q") & (agg.window == "val")]
           .groupby("family").value.mean())
delta_e = {w: float((ece_cohort.loc[w] - ece_val).abs().mean())
           for w in LAGS if w in ece_cohort.index}
xs_e = np.array([adv[w] for w in delta_e])
ys_e = np.array(list(delta_e.values()))
add("shift_dose", "ece_change",
    "Spearman(shift AUC, mean |ECE change|)", float(spearmanr(xs_e, ys_e).statistic),
    n=len(delta_e), note="post-hoc; 8 cohorts, no CI")

# ------------------------------------------------- half-year granularity
hy = agg[agg.metric == "citl"].copy()
hy["half_year"] = hy.window.map({
    "2014Q1": "2014H1", "2014Q2": "2014H1", "2014Q3": "2014H2",
    "2014Q4": "2014H2", "2015Q1": "2015H1", "2015Q2": "2015H1",
    "2015Q3": "2015H2", "2015Q4": "2015H2"})
hy_cfg = hy.groupby(["config_index", "half_year"]).value.mean()
val_cfg = agg[(agg.metric == "citl") & (agg.window == "val")] \
    .groupby("config_index").value.mean()
deltas = (hy_cfg.unstack().sub(val_cfg, axis=0).abs().mean(axis=0))
add("hy_granularity", "citl_change_mean",
    "mean(|CITL_half_year| - |CITL_val|), H1a statistic at half-year",
    float(deltas.mean()), n=4,
    note="post-hoc; sensitivity to cohort granularity; "
         "quarterly value 0.045384")

# ------------------------------------------------- rho noise floor
# Split-half noise of the between-config rho: per unit, average each
# metric over seed refits {0,1} and {2,3,4} separately, compute the
# between-config rho for each half, and report the half-to-half
# difference over the 7 pairs. If this spread is large, low rho with
# AUC may partly be measurement noise rather than absence of
# association. Restricted to deep_agg outcomes (ece, slope_dev); the
# flip-rate table has no refit axis.
refits = {}
for (fam, cfg, refit, w, m), v in agg.set_index(
        ["family", "config_index", "refit_id", "window",
         "metric"]).value.items():
    refits[(fam, int(cfg), int(refit), w, m)] = float(v)
HALF_A, HALF_B = (0, 1), (2, 3, 4)


def half_mean(fam, cfg, w, m, half):
    vs = [refits[(fam, cfg, r, w, m)] for r in half
          if (fam, cfg, r, w, m) in refits]
    return float(np.mean(vs)) if vs else np.nan


def half_rho(outcome_name: str, half) -> float:
    rhos = []
    for t, tp in ADJACENT:
        xs = np.array([half_mean(f, c, t, "roc_auc", half)
                       for f, c in units])
        if outcome_name == "ece":
            ys = np.array([half_mean(f, c, tp, "ece_b15q", half)
                           for f, c in units])
        else:
            ys = np.array([abs(half_mean(f, c, tp, "slope", half) - 1)
                           for f, c in units])
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 3:
            rhos.append(spearmanr(xs[ok], ys[ok]).statistic)
    return float(np.nanmean(rhos))


for name in ("ece", "slope_dev"):
    add("rho_noise", name,
        "half-split difference of between-config rho (A-half minus B-half)",
        float(half_rho(name, HALF_A) - half_rho(name, HALF_B)),
        n=n_cfg,
        note="post-hoc; seeds {0,1} vs {2,3,4}; mean over 7 pairs; "
             "flip rate excluded (no refit axis)")

# ---------------------------------------------------------------- write
out = pd.DataFrame(ROWS)
out.to_csv(TABLES / "post_hoc_analyses.csv", index=False)
print(out.to_string(index=False))
