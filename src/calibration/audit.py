"""Calibration and decision-audit metrics (PLAN.md Phases 5 and 7).

Pure functions over prediction vectors and resolved labels only. Nothing
here reads data, fits scorecards, or touches calendars — callers own the
cohorts; this module owns the yardsticks.

Conventions:
- All inputs are 1-D arrays of equal length; labels are 0/1 floats.
- The Brier decomposition is the classic Murphy (1973) partition on discrete
  forecast bins: BS = uncertainty - resolution + reliability, where
  "reliability" is the within-bin frequency-matching error and "resolution"
  rewards bin-level separation. Bin edges are caller-supplied per protocol.
- ECE is reported on the same bins with quantile option, and its caveat is
  returned alongside (non-proper, binning-sensitive).
- CITL/slope come from a logistic calibration refit of the target on the
  logit of the forecast: slope 1 and intercept 0 describe a perfectly
  calibrated score; small calibration samples make the slope estimate
  unstable (H1d is expected to test this).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BrierDecomposition:
    brier: float         # raw (continuous) Brier score of the submitted scores
    binned_brier: float  # Brier score with scores replaced by their bin means
    uncertainty: float
    resolution: float
    reliability: float
    n_bins_used: int


def _bin_masks(
    probabilities: np.ndarray,
    edges: np.ndarray,
    *,
    quantile_edges: bool,
) -> list[np.ndarray]:
    if quantile_edges:
        qs = np.linspace(0.0, 1.0, len(edges) + 2)[1:-1] if len(edges) else np.array([])
        edges = np.unique(np.sort(np.quantile(probabilities, qs)))
    inner = edges[(edges > 0.0) & (edges < 1.0)]
    bounds = np.concatenate(([0.0], inner, [1.0]))
    # left-closed [lo, hi) except the last bin, which includes 1.0
    masks = []
    for i in range(len(bounds) - 1):
        lower_ok = probabilities >= bounds[i]
        upper_ok = probabilities < bounds[i + 1] if i + 1 < len(bounds) - 1 else probabilities <= bounds[i + 1]
        masks.append(lower_ok & upper_ok)
    return masks


def brier_decomposition(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    edges: np.ndarray | None = None,
    quantile_edges: bool = False,
    n_bins: int = 10,
    min_bin_count: int = 1,
) -> BrierDecomposition:
    """Murphy-style decomposition of the (binary) Brier score.

    `edges` overrides automatic equal-width binning (use the protocol's bin
    definition). Values already in [0,1]; targets 0/1.

    Reporting convention: the partition identity
    BS = uncertainty - resolution + reliability is exact for the *binned*
    forecast (scores replaced by their bin means); a raw continuous score
    additionally carries the within-bin spread of the scores, so both the
    raw and the binned Brier are returned and the identity is checked
    against the binned one. The gap between them is the within-bin score
    spread and shrinks as bins refine.
    """
    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("targets and probabilities must be equal-length 1-D arrays")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be binary")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be >= 1")

    if edges is None:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    uncertainty = base * (1.0 - base)

    masks = _bin_masks(p, edges, quantile_edges=quantile_edges)
    reliability = 0.0
    resolution = 0.0
    binned = 0.0
    used = 0
    bin_mean = np.zeros_like(p)
    for m in masks:
        nb = int(m.sum())
        if nb < min_bin_count:
            continue
        used += 1
        share = nb / len(y)
        p_bar = float(p[m].mean())
        y_bar = float(y[m].mean())
        bin_mean[m] = p_bar
        reliability += share * (p_bar - y_bar) ** 2
        resolution += share * (y_bar - base) ** 2
        binned += share * (y_bar * (1.0 - y_bar) + (p_bar - y_bar) ** 2)
    return BrierDecomposition(
        brier=brier,
        binned_brier=binned,
        uncertainty=uncertainty,
        resolution=resolution,
        reliability=reliability,
        n_bins_used=used,
    )


@dataclass(frozen=True)
class CitlResult:
    citl: float          # required intercept on the logit scale (want 0)
    slope: float         # required slope on the logit scale (want 1)
    n: int
    n_events: int


def citl_and_slope(targets: np.ndarray, probabilities: np.ndarray) -> CitlResult:
    """Logistic calibration-in-the-large and slope via refit on the logit.

    Fits targets ~ a + b * logit(p); returns (a, b) as the calibration
    intercept (CITL) and slope deviation signature per the statistical plan.
    Uses an offset-free logistic fit with a single logit-space covariate.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(targets, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    if len(set(np.unique(y))) != 2:
        raise ValueError("CITL needs both outcome classes present")
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    # C=np.inf keeps the refit unregularised (sklearn 1.8 deprecates
    # penalty=None, whose fallback silently penalises the slope).
    fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000).fit(logit, y)
    return CitlResult(
        citl=float(fit.intercept_[0]),
        slope=float(fit.coef_[0][0]),
        n=len(y),
        n_events=int(y.sum()),
    )


@dataclass(frozen=True)
class FlipReport:
    threshold: float
    n: int
    n_flips: int
    flip_rate: float
    flips_to_decline: int
    flips_to_approve: int


def decision_flips(
    probabilities_before: np.ndarray,
    probabilities_after: np.ndarray,
    threshold: float = 0.5,
) -> FlipReport:
    """Count loan-level decision disagreements between two score vectors."""
    a = np.asarray(probabilities_before, dtype=float)
    b = np.asarray(probabilities_after, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("score vectors must be equal-length 1-D arrays")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must lie in (0,1)")
    before = a >= threshold
    after = b >= threshold
    flips = before != after
    return FlipReport(
        threshold=threshold,
        n=len(a),
        n_flips=int(flips.sum()),
        flip_rate=float(flips.mean()),
        flips_to_decline=int((before & ~after).sum()),
        flips_to_approve=int((~before & after).sum()),
    )


def mean_within_cohort_spearman(per_cohort_spearman: list[float]) -> float:
    """Average of per-cohort within-cohort Spearman rho values (RQ4 primary)."""
    vals = np.asarray([v for v in per_cohort_spearman if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())
