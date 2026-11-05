"""Baseline training harness (PLAN.md T6.1/T6.2) and metric evaluation.

Responsibilities kept narrow and pure:
- preprocessing fitted on the training window only (the caller decides what
  the training window is; nothing here reads data),
- one estimator constructor per sanctioned family with hyperparameters and
  seeds supplied by the caller (configs, never literals in model code),
- fixed metric definitions (ROC-AUC, PR-AUC, Brier, log loss, accuracy,
  precision, recall, F1, balanced accuracy) with percentile bootstrap
  confidence intervals on a single shared resample design (seeded).

Class weighting: none (main-run rule, configs/models.yaml); resampling:
none. Everything stochastic enters through `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")

METRICS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "logloss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "balanced_accuracy",
)


def make_preprocessor(*, scale: bool = True) -> Pipeline:
    """Median imputation (+ standardisation) fitted on the training window.

    `scale` matters for logistic regression and margin methods; trees are
    invariant, but one shared preprocessing keeps run-to-run comparisons
    simple and costs nothing.
    """
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_estimator(family: str, params: dict, seed: int):
    """Construct one unfitted estimator for a sanctioned model family.

    `params` carries only model hyperparameters; the seed is injected here so
    configs never embed randomness. Unknown hyperparameter keys surface as
    the estimator's own constructor errors (no silent dropping).
    """
    if family == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if family == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if family == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
            **params,
        )
    if family == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"unknown model family {family!r}; expected one of {FAMILIES}")


def _validate(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probabilities must be equal-length 1-D arrays")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y_true must be binary")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    return y, p


def _point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    hard = (p >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, hard)),
        "precision": float(precision_score(y, hard, zero_division=0)),
        "recall": float(recall_score(y, hard, zero_division=0)),
        "f1": float(f1_score(y, hard, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)),
    }


@dataclass(frozen=True)
class EvalResult:
    point: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    n_bootstrap: int


def evaluate(
    y_true,
    probabilities,
    *,
    threshold: float = 0.5,
    n_bootstrap: int | None = None,
    seed: int | None = None,
) -> EvalResult:
    """Point metrics plus percentile bootstrap CIs on shared resamples.

    One bootstrap replicate recomputes every metric on the same resampled
    indices (the paired design statistics.yaml asks for); CIs are the 2.5/97.5
    percentiles. `n_bootstrap=None` returns point estimates only.
    """
    y, p = _validate(y_true, probabilities)
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must lie in (0,1)")
    point = _point_metrics(y, p, threshold)
    if n_bootstrap is None:
        return EvalResult(point=point, ci_low={}, ci_high={}, n_bootstrap=0)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be >= 2 for interval estimates")
    if seed is None:
        raise ValueError("a seed is required for bootstrap replicate designs")

    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: np.empty(n_bootstrap, dtype=float) for m in METRICS}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        for m, v in _point_metrics(yb, pb, threshold).items():
            draws[m][b] = v
    low = {m: float(np.percentile(draws[m], 2.5)) for m in METRICS}
    high = {m: float(np.percentile(draws[m], 97.5)) for m in METRICS}
    return EvalResult(point=point, ci_low=low, ci_high=high, n_bootstrap=n_bootstrap)
