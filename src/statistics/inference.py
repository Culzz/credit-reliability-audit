"""Confirmatory inference helpers: Holm adjustment and paired deltas.

- Holm step-down: the multiplicity rule in configs/statistics.yaml
  (confirmatory family listed there). Returns adjusted p-values in the
  original input order, enforcing monotonicity in the sorted order so no
  adjusted value is smaller than its predecessor.

- Paired bootstrap delta CIs: two scoring systems are compared ON THE
  SAME resample design (the paired-where-comparing rule), per replicate
  the same indices resample both score vectors and one shared delta is
  recorded. Percentile CI as in the main evaluation path.

Pure functions; no data access, no literals from configs.
"""

from __future__ import annotations

import numpy as np


def holm_pvalues(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, order-preserving.

    `p_values` must be within [0,1] and finite. With m tests, adj_i is the
    running maximum of (m - rank + 1) * p_(rank) over the sorted order,
    then rescaled to the original ordering and clipped at 1.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie within [0,1] and be finite")

    m = len(p)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank in range(m):
        running = max(running, (m - rank) * sorted_p[rank])
        adjusted_sorted[rank] = running
    out = np.empty_like(adjusted_sorted)
    out[order] = adjusted_sorted
    return np.clip(out, 0.0, 1.0)


def paired_bootstrap_delta(
    targets,
    scores_a,
    scores_b,
    *,
    metric,
    metric_is_loss: bool,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Percentile CI of metric(scores_a) - metric(scores_b) on shared resamples.

    `metric(y, p) -> float`; `metric_is_loss` flips the delta sign so that
    POSITIVE always means A is worse than B (a loss gap measured as
    loss(a) - loss(b), or a gain gap as gain(b) - gain(a)).
    Returns point delta, low and high percentiles, and n_bootstrap.
    """
    y = np.asarray(targets, dtype=float)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1 or len(y) == 0:
        raise ValueError("inputs must be non-empty 1-D arrays")
    if not (len(y) == len(a) == len(b)):
        raise ValueError("targets and both score vectors must align")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be binary")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2")

    rng = np.random.default_rng(seed)
    n = len(y)
    point_pa, point_pb = a.copy(), b.copy()
    delta_point = _signed_delta(y, point_pa, point_pb, metric, metric_is_loss)
    deltas = np.empty(n_bootstrap, dtype=float)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        deltas[k] = _signed_delta(y[idx], a[idx], b[idx], metric, metric_is_loss)
    return {
        "delta": float(delta_point),
        "lo": float(np.percentile(deltas, 2.5)),
        "hi": float(np.percentile(deltas, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def _signed_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, metric, metric_is_loss: bool) -> float:
    va = float(metric(y, a))
    vb = float(metric(y, b))
    raw = va - vb
    return raw if metric_is_loss else -raw
