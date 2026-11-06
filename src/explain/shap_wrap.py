"""SHAP infrastructure wrapper (PLAN.md T8.1).

Thin, convention-pinning wrapper over `shap`:

- boosted/ensemble families (xgboost, lightgbm, random_forest):
  TreeExplainer with `feature_perturbation="interventional"` against a
  caller-fixed background sample (the protocol's 1,000-row anchor) and
  `model_output="raw"` (the margin, i.e. log-odds for these link
  functions). The path-dependent variant is only ever a robustness
  comparison, built with an explicit flag.
- logistic regression: LinearExplainer (exact for additive linear models)
  on the same background, also raw output.

The feature order is whatever the caller fixed when building X — this
wrapper never re-orders, drops, or renames features. Sizing decisions
(how many explanation rows per cohort, background size, version caps)
are call-site arguments; nothing about them is hard-coded here.

Determinism: given identical trees, backgrounds and explanation rows,
the returned matrix is a pure function — the same matrix each time. The
wrapper returns a plain ndarray (n_explain x n_features) to keep the
downstream stability metrics decoupled from shap's container types.
"""

from __future__ import annotations

import numpy as np


def _finite_2d(M, name: str) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite 2-D array")
    return arr


def make_explainer(model, background, *, family: str, path_dependent: bool = False):
    """Build the sanctioned explainer for one frozen model + background.

    Background discipline: a FULLY degenerate background (at least one
    column with variance, but rows otherwise identical — concretely: every
    column constant) hangs LinearExplainer indefinitely, so it is rejected
    here. A SINGLE constant column among otherwise varied rows is fine:
    both LinearExplainer and interventional TreeSHAP attribute exactly 0
    to it (verified in the synthetic tests), so rare-category one-hot
    columns that never fire in a plumbing-scale window do not block work.
    """
    import shap

    bg = _finite_2d(background, "background")
    if np.all(bg.std(axis=0) <= 0.0):
        raise ValueError("background rows must vary (fully degenerate background)")
    if path_dependent:
        if family == "logistic_regression":
            raise ValueError("path-dependent variant applies to tree families only")
        return shap.TreeExplainer(model, data=bg, feature_perturbation="tree_path_dependent")
    if family == "logistic_regression":
        return shap.LinearExplainer(model, bg)
    if family in ("xgboost", "lightgbm", "random_forest"):
        return shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
    raise ValueError(f"unknown family {family!r}; sanctioned: logistic/xgboost/lightgbm/random_forest")


def explain_matrix(explainer, X_explain) -> np.ndarray:
    """(n_explain x n_features) raw-scale attribution matrix."""
    Xe = _finite_2d(X_explain, "X_explain")
    kwargs: dict = {}
    if type(explainer).__name__ == "TreeExplainer":
        # LightGBM's float32 leaf accumulation breaks shap's strict sum
        # additivity check at the 1e-3 level (observed on real shapes);
        # the downstream stability battery consumes differences between
        # matrices computed identically, so the strict check is off here
        # and correctness rests on the fixed background + fixed order.
        kwargs["check_additivity"] = False
    out = explainer(Xe, **kwargs)
    values = getattr(out, "values", None)
    if values is None:  # shap API fallback for older versions
        values = explainer.shap_values(Xe)
    arr = np.asarray(values, dtype=float)
    if isinstance(values, list):  # multi-output (RF) collapses to class 1
        arr = np.asarray(values[-1] if len(values) == 2 else values, dtype=float)
    if arr.ndim == 3:  # newer shap returns (n, features, outputs)
        arr = arr[:, :, -1]
    if arr.shape != Xe.shape:
        raise ValueError(f"attribution shape {arr.shape} does not match features {Xe.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("explanations contain non-finite values")
    return arr


def matrix_close(A: np.ndarray, B: np.ndarray, *, atol: float = 1e-9, rtol: float = 1e-7) -> bool:
    """Allclose with default tolerances exposed for the reproducibility checks."""
    a, b = _finite_2d(A, "A"), _finite_2d(B, "B")
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))
