"""Threshold decision policies and stylised portfolio utilities (Phase 8).

The policy is read from configs/decisions.yaml by callers; this module
takes every quantity as an explicit argument (no literals for the
PROPOSED values live here).

Primary policy (three-band): approve when p < t_low, refer when
t_low <= p < t_high, decline otherwise. The break-even threshold follows
the loss-gain formula in decisions.yaml: p* = g / (g + L), where g is the
per-loan gain from funding a good loan and L the per-loan loss from
funding a default (L in units of g). The refer band is p* +/- h.

Second policy (fixed approval rate): the threshold is the (1 - rate)
quantile of the submitted scores, so the bottom `rate` share of scores is
approved. Ties at the threshold quantile belong to NOT-approved (strict
"p < threshold" approves) to keep the rate from overshooting.

Utility convention (stylised, funded loans only): a funded good loan
earns g (normalised to 1), a funded default costs L; indicates refer
(no funding in the stylised account) and decline contribute nothing.
These are scenario quantities, never observed LendingClub profits
(decisions.yaml A7 note).
"""

from __future__ import annotations

import numpy as np

APPROVE, REFER, DECLINE = 0, 1, 2


def breakeven_threshold(loss_gain_ratio: float) -> float:
    """p* = g / (g + L) for the stylised loss/gain pair (L in g units)."""
    if not np.isfinite(loss_gain_ratio) or loss_gain_ratio <= 0:
        raise ValueError("loss_gain_ratio must be a positive, finite number")
    L = float(loss_gain_ratio)
    return 1.0 / (1.0 + L)


def threshold_policy(
    probabilities,
    *,
    t_low: float,
    t_high: float,
) -> np.ndarray:
    """Three-band policy codes: APPROVE / REFER / DECLINE."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a non-empty 1-D array")
    if not (0.0 < t_low < t_high < 1.0):
        raise ValueError("thresholds must satisfy 0 < t_low < t_high < 1")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    codes = np.full(len(p), DECLINE, dtype=np.int8)
    codes[p < t_high] = REFER
    codes[p < t_low] = APPROVE
    return codes


def fixed_rate_policy(probabilities, *, approval_rate: float) -> tuple[np.ndarray, float]:
    """Approve the bottom `approval_rate` share of scores.

    Returns (codes, threshold) so the flip/utility records carry the
    realised threshold. Ties belong to NOT-approved (strict `p < t`).
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("probabilities must be a non-empty 1-D array")
    if not (0.0 < approval_rate < 1.0):
        raise ValueError("approval_rate must lie in (0,1)")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    threshold = float(np.quantile(p, approval_rate))
    codes = np.where(p < threshold, APPROVE, DECLINE).astype(np.int8)
    return codes, threshold


def stylised_utility(
    targets,
    codes,
    *,
    loss_gain_ratio: float,
    gain: float = 1.0,
) -> float:
    """Total stylised utility of a code vector: funded good loans earn
    g, funded defaults cost L; refer/decline contribute zero."""
    y = np.asarray(targets, dtype=float)
    c = np.asarray(codes)
    if len(y) != len(c) or len(y) == 0:
        raise ValueError("targets and codes must be equal-length, non-empty arrays")
    if np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be binary")
    funded = c == APPROVE
    funded_count = int(funded.sum())
    if funded_count == 0:
        return 0.0
    n_good = int((y[funded] == 0.0).sum())
    n_bad = funded_count - n_good
    return float(gain * n_good - gain * loss_gain_ratio * n_bad)
