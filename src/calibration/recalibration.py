"""Recalibration arms (PLAN.md T7.3).

Four sanctioned arms over a set of scores:
- "none"       : scores passed through unchanged (the reference arm),
- "intercept"  : calibration-in-the-large — one fitted offset on the logit
                 scale (slope pinned to 1),
- "platt"      : logistic refit of the outcome on the logit of the scores
                 (fitted intercept and slope on the logit scale),
- "isotonic"   : monotone non-parametric mapping, clipped to [0,1].

Iron rule (T7.3): the calibration targets passed to `fit_recalibrator` are
eligible historical labels ONLY. The caller owns cohort eligibility; this
module enforces that the *evaluation* cohort cannot be passed implicitly —
targets are an explicit argument, and the pipeline asserts disjoint cohorts.
Sloppy use is still possible, so run_baselines-style callers must assert
the calibration cohort is disjoint from every evaluation cohort; a `tag`
argument is carried through into the result record for provenance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

KINDS = ("none", "intercept", "platt", "isotonic")

_EPS = 1e-12


def _as_arrays(targets, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=float)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("targets and scores must be equal-length 1-D arrays")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be binary")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("scores must lie in [0,1]")
    return y, p


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(pc / (1.0 - pc))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Recalibrator:
    """A fitted recalibration mapping for one arm."""

    def __init__(self, kind: str, params: dict, n_calibrated: int, n_events: int):
        if kind not in KINDS:
            raise ValueError(f"unknown recalibrator kind {kind!r}; expected one of {KINDS}")
        self.kind = kind
        self.params = params
        self.n_calibrated = n_calibrated
        self.n_events = n_events

    def apply(self, scores) -> np.ndarray:
        p = np.asarray(scores, dtype=float)
        if p.ndim != 1 or p.size == 0:
            raise ValueError("scores must be a non-empty 1-D array")
        if not np.all((p >= 0.0) & (p <= 1.0)):
            raise ValueError("scores must lie in [0,1]")
        if self.kind == "none":
            return p.copy()
        if self.kind == "intercept":
            return _expit(_logit(p) + self.params["offset"])
        if self.kind == "platt":
            return _expit(self.params["intercept"] + self.params["slope"] * _logit(p))
        # isotonic
        return np.clip(self.params["mapping"].predict(p), 0.0, 1.0)


def fit_recalibrator(kind: str, targets, scores) -> Recalibrator:
    """Fit one recalibration arm on eligible historical labels only.

    `targets`/`scores` must describe the calibration cohort, never an
    evaluation cohort (T7.3). Both outcome classes are required (the
    intercept/platt/isotonic fits are unidentified otherwise).
    """
    y, p = _as_arrays(targets, scores)
    if kind not in KINDS:
        raise ValueError(f"unknown recalibrator kind {kind!r}; expected one of {KINDS}")
    if kind == "none":
        return Recalibrator("none", {}, len(y), int(y.sum()))
    if len(np.unique(y)) < 2:
        raise ValueError(f"arm {kind!r} needs both outcome classes in the calibration set")

    if kind == "intercept":
        logits = _logit(p)

        def deviance(offset: float) -> float:
            q = _expit(logits + offset)
            return -(y * np.log(q + _EPS) + (1 - y) * np.log(1 - q + _EPS)).sum()

        res = minimize_scalar(deviance, bounds=(-10.0, 10.0), method="bounded")
        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError("intercept-only calibration fit failed to converge")
        return Recalibrator("intercept", {"offset": float(res.x)}, len(y), int(y.sum()))

    if kind == "platt":
        # C=np.inf: the calibration refit must be unregularised (sklearn 1.8
        # deprecates penalty=None; a penalised refit shrinks the slope).
        fit = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000).fit(
            _logit(p).reshape(-1, 1), y.astype(int)
        )
        return Recalibrator(
            "platt",
            {"intercept": float(fit.intercept_[0]), "slope": float(fit.coef_[0][0])},
            len(y),
            int(y.sum()),
        )

    # isotonic
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(p, y)
    return Recalibrator("isotonic", {"mapping": iso}, len(y), int(y.sum()))
