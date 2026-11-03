"""Descriptive shift statistics for Phase 4 (PLAN.md T4.2, T4.4).

Pure, state-free functions: they see two samples and return a number. No
labels, no dates, no model outputs ever enter this module — these are the
descriptive-only diagnostics that Gate 4 window choices may cite.

Conventions:
- Missing values are dropped before any numeric measure (PSI, KS, SMD);
  missingness itself is audited separately by the missingness flags.
- PSI uses quantile bins derived from the reference sample; bin edges are
  infinite at both ends so shifting distributions cannot fall out of range.
- Zero-count categories inside PSI get an additive smoothing constant
  (EPSILON proportion share), which is the standard practical treatment;
  results are descriptive only and thresholds are never cited as laws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "categorical_js_divergence",
    "ks_statistic",
    "psi_numeric",
    "quantile_bin_edges",
    "smd",
    "wilson_interval",
]

EPSILON = 1e-6  # proportion smoothing for empty bins in PSI
DEFAULT_BINS = 10


def quantile_bin_edges(reference: pd.Series, *, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Inner quantile cut points of the reference sample, with infinite guards.

    Returns edges ordered ascending, starting at -inf and ending at +inf;
    duplicate quantiles collapse so constant reference columns yield one
    interior bin.
    """
    ref = pd.Series(reference).dropna().astype(float)
    if ref.empty:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    inner = np.unique(np.sort(np.asarray(ref.quantile(qs), dtype=float)))
    inner = inner[np.isfinite(inner)]
    return np.concatenate(([-np.inf], inner, [np.inf]))


def _bin_proportions(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    vals = pd.Series(values).dropna().astype(float)
    if vals.empty:
        raise ValueError("no observed values to bin")
    idx = np.digitize(vals.to_numpy(), edges[1:-1], right=True)
    counts = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def psi_numeric(
    reference: pd.Series,
    current: pd.Series,
    *,
    n_bins: int = DEFAULT_BINS,
    epsilon: float = EPSILON,
) -> float:
    """Population stability index across reference-derived quantile bins.

    Both sides are compared as proportion vectors over the same bins; empty
    bins receive `epsilon` normalised within their own vector, then
    PSI = sum (p_ref - p_cur) * ln(p_ref / p_cur).
    """
    edges = quantile_bin_edges(reference, n_bins=n_bins)
    p_ref = _bin_proportions(reference, edges)
    p_cur = _bin_proportions(current, edges)

    def _smooth(p: np.ndarray) -> np.ndarray:
        q = np.where(p <= 0.0, epsilon, p)
        return q / q.sum()

    a, b = _smooth(p_ref), _smooth(p_cur)
    return float(np.sum((a - b) * np.log(a / b)))


def ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Two-sample Kolmogorov–Smirnov D statistic (NA-dropped, unweighted)."""
    from scipy.stats import ks_2samp

    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)
    if ref.empty or cur.empty:
        raise ValueError("KS needs non-empty samples on both sides")
    return float(ks_2samp(ref.to_numpy(), cur.to_numpy()).statistic)


def smd(reference: pd.Series, current: pd.Series) -> float:
    """Standardised mean difference (current − reference) over pooled SD.

    Pooled SD uses unweighted variances of the two samples; if the pooled sd
    is zero (two identical constants) the statistic is 0.0.
    """
    r = pd.Series(reference).dropna().astype(float)
    c = pd.Series(current).dropna().astype(float)
    if r.empty or c.empty:
        raise ValueError("SMD needs non-empty samples on both sides")
    s1, s2 = r.var(ddof=1), c.var(ddof=1)
    pooled = np.sqrt((s1 + s2) / 2.0)
    if pooled == 0.0:
        return 0.0
    return float((c.mean() - r.mean()) / pooled)


def categorical_js_divergence(
    reference: pd.Series,
    current: pd.Series,
    *,
    epsilon: float = EPSILON,
    base: float = 2.0,
) -> float:
    """Jensen–Shannon divergence (in `base` units, symmetric, <= log2 base) of
    two categorical distributions, NAs forming their own category."""
    from scipy.spatial.distance import jensenshannon

    r = pd.Series(reference).astype("object")
    c = pd.Series(current).astype("object")
    r = r.fillna("__NA__").replace(pd.NA, "__NA__")
    c = c.fillna("__NA__").replace(pd.NA, "__NA__")
    levels = sorted(map(str, set(r.unique()) | set(c.unique())))
    pr = r.astype(str).value_counts(normalize=True)
    pc = c.astype(str).value_counts(normalize=True)

    def _vec(p: pd.Series) -> np.ndarray:
        v = np.array([p.get(level, 0.0) for level in levels], dtype=float)
        v = np.where(v <= 0.0, epsilon, v)
        return v / v.sum()

    d = jensenshannon(_vec(pr), _vec(pc), base=base)
    return 0.0 if np.isnan(d) else float(d)


def wilson_interval(events: int, total: int, *, alpha: float = 0.05):
    """Wilson score interval for a proportion, with the usual z_* correction."""
    if total <= 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / 2.0)
    p = events / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (float((centre - half) / denom), float((centre + half) / denom))
