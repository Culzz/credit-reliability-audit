"""Explanation-stability metrics (PLAN.md T8.3/T8.7, RQ3/RQ4).

Operates on attribution MATRICES with a FIXED feature order (the caller
owns the feature order and the convention; nothing here re-sorts
silently). Every comparator takes two attribution matrices for the same
feature set and returns a scalar joint-stability figure:

- global importance: mean |shap| per feature,
- rank agreement: Spearman / Kendall on the importance ranks,
- top-k set agreement: Jaccard@k on the top-k important features,
- vector agreement: cosine on the ranked importance vectors,
- distribution distance: mean per-feature Wasserstein-1 distance between
  the attribution distributions (population shift has a signature here
  that rank agreement alone cannot see).

All functions are pure: no model, no data file, no background sampling
(sampling lives in the SHAP run scripts). Validation is symmetric, so
reversed arguments only swap the two populations being compared.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr, wasserstein_distance


def _attr_matrix(V: np.ndarray, name: str = "attributions") -> np.ndarray:
    v = np.asarray(V, dtype=float)
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D (loans x features) array")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite")
    return v


def global_importance(V: np.ndarray) -> np.ndarray:
    """Mean |shap| per feature, over the explanation sample (rows)."""
    return np.abs(_attr_matrix(V)).mean(axis=0)


def rank_vector(V: np.ndarray, *, descending: bool = True) -> np.ndarray:
    """Feature importance ranks (1 = most important)."""
    imp = global_importance(V)
    ranks = np.empty(len(imp), dtype=float)
    order = np.argsort(-imp) if descending else np.argsort(imp)
    ranks[order] = np.arange(1, len(imp) + 1)
    return ranks


def jaccard_at_k(V1: np.ndarray, V2: np.ndarray, k: int) -> float:
    """|topk(V1) and topk(V2)| / |topk(V1) or topk(V2)|."""
    a = _attr_matrix(V1)
    b = _attr_matrix(V2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("attribution matrices share the fixed feature order")
    if not (1 <= k <= a.shape[1]):
        raise ValueError(f"k must lie in [1, n_features] (got {k})")
    tol = 1e-12
    if np.all(np.abs(global_importance(a)) < tol) and np.all(np.abs(global_importance(b)) < tol):
        return 1.0  # both all-zero importances: identical trivially
    top_a = set(np.argsort(-global_importance(a))[:k].tolist())
    top_b = set(np.argsort(-global_importance(b))[:k].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def rank_agreement(V1: np.ndarray, V2: np.ndarray) -> dict[str, float]:
    """Spearman and Kendall rho between the two importance rank vectors."""
    a = _attr_matrix(V1)
    b = _attr_matrix(V2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("attribution matrices share the fixed feature order")
    if a.shape[1] < 2:
        raise ValueError("rank agreement needs at least two features")
    r1, r2 = rank_vector(a), rank_vector(b)
    sp = float(spearmanr(r1, r2).statistic)
    kt = float(kendalltau(r1, r2).statistic)
    return {"spearman": 0.0 if not np.isfinite(sp) else sp,
            "kendall": 0.0 if not np.isfinite(kt) else kt}


def cosine_importance(V1: np.ndarray, V2: np.ndarray) -> float:
    """Cosine similarity of the two mean-|shap| importance vectors."""
    a = _attr_matrix(V1)
    b = _attr_matrix(V2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("attribution matrices share the fixed feature order")
    x = global_importance(a)
    z = global_importance(b)
    nx, nz = np.linalg.norm(x), np.linalg.norm(z)
    if nx == 0.0 and nz == 0.0:
        return 1.0
    if nx == 0.0 or nz == 0.0:
        return 0.0
    return float(np.dot(x, z) / (nx * nz))


def attribution_distribution_distance(V1: np.ndarray, V2: np.ndarray) -> float:
    """Mean per-feature Wasserstein-1 distance between attribution columns.

    Rows of the matrices are the attribution values for one feature across
    the explanation sample; sensitive to population shape, not only to the
    mean importance ordering.
    """
    a = _attr_matrix(V1)
    b = _attr_matrix(V2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("attribution matrices share the fixed feature order")
    dists = [
        wasserstein_distance(a[:, j].ravel(), b[:, j].ravel())
        for j in range(a.shape[1])
    ]
    return float(np.mean(dists))
