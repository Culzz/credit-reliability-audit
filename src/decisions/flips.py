"""Three-way decision-flip machinery (PLAN.md T9.3/T9.4).

Works over identical loans scored twice (stale vs refreshed, uncalibrated
vs recalibrated, fixed vs rolling, seed vs seed). Everything here takes
explicit arguments; thresholds come from the configured policy, never
literals.

Metrics per comparison pair:
- 3x3 transition matrix of codes (rows = before, cols = after),
- overall flip rate (off-diagonal share),
- asymmetry between risk-band upgrades (toward decline) and downgrades
  (toward approve),
- probability changes for flipped pairs,
- distance to the nearest threshold for the pair's decisions,
- per-flip and total stylised delta-utility, with a matched-null
  distribution built by permuting the second probability vector (same
  marginal decision rates, broken loan pairing) — the null for judging
  whether observed total delta-utility differs from chance. Because the
  null breaks the score-outcome information while preserving the
  decision margins, an informative no-op arm typically sits at or above
  the entire null distribution, and a comparison pair whose observed
  total falls inside the null is statistically indistinguishable from
  chance.

Refer-band moves are reported separately from approve/decline flips by
the caller; the transition matrix carries them at full resolution.
"""

from __future__ import annotations

import numpy as np

from src.decisions.policy import APPROVE, DECLINE, REFER, threshold_policy

__all__ = [
    "APPROVE",
    "DECLINE",
    "REFER",
    "distance_to_nearest_threshold",
    "flip_delta_utility",
    "flip_flows",
    "flip_rate",
    "matched_null_total_delta_utility",
    "probability_changes",
    "transition_matrix",
]


def _codes(codes, name: str = "codes") -> np.ndarray:
    c = np.asarray(codes, dtype=np.int8)
    if c.ndim != 1 or len(c) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if np.any((c < 0) | (c > 2)):
        raise ValueError(f"{name} must contain only codes 0/1/2")
    return c


def _both_valid(codes_before, codes_after) -> tuple[np.ndarray, np.ndarray]:
    a = _codes(codes_before, "codes_before")
    b = _codes(codes_after, "codes_after")
    if a.shape != b.shape:
        raise ValueError("code vectors must have equal length")
    return a, b


def transition_matrix(codes_before, codes_after) -> np.ndarray:
    """3x3 counts: row index = before code, col index = after code."""
    a, b = _both_valid(codes_before, codes_after)
    return np.bincount(a * 3 + b, minlength=9).reshape(3, 3)


def flip_rate(codes_before, codes_after) -> float:
    """Share of identical loans whose code changes (off-diagonal share)."""
    m = transition_matrix(codes_before, codes_after)
    return float((m.sum() - np.trace(m)) / m.sum())


def flip_flows(codes_before, codes_after) -> dict[str, int]:
    """Asymmetry bookkeeping: moves toward decline vs toward approve."""
    a, b = _both_valid(codes_before, codes_after)
    # direction = sign(b - a) over flipped pairs; refer hops count as moves
    diff = b - a
    flipped = a != b
    up = int(((diff > 0) & flipped).sum())
    down = int(((diff < 0) & flipped).sum())
    return {
        "n_flips": up + down,
        "toward_decline": up,
        "toward_approve": down,
        "asymmetric_flips": abs(up - down),
    }


def probability_changes(probabilities_before, probabilities_after, codes_before, codes_after):
    """Signed probability change (after - before) for flipped pairs only."""
    pb = np.asarray(probabilities_before, dtype=float)
    pa = np.asarray(probabilities_after, dtype=float)
    a, b = _both_valid(codes_before, codes_after)
    for arr in (pb, pa):
        if arr.shape != a.shape or not np.all((arr >= 0) & (arr <= 1)):
            raise ValueError("probability vectors must match code length and lie in [0,1]")
    return pa[b != a] - pb[b != a]


def distance_to_nearest_threshold(probabilities, codes, *, t_low: float, t_high: float):
    """Distance from each submitted probability to the policy boundary it
    sits next to: |p - t_low| for approved loans, |p - t_high| for
    declined, min of the two for refer-band loans. Small distances flag
    threshold-adjacent (fragile) decisions."""
    p = np.asarray(probabilities, dtype=float)
    c = _codes(codes)
    if p.shape != c.shape or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("probabilities must match code length and lie in [0,1]")
    if not (0.0 < t_low < t_high < 1.0):
        raise ValueError("thresholds must satisfy 0 < t_low < t_high < 1")
    out = np.empty(len(p), dtype=float)
    approved = c == 0
    declined = c == 2
    refer = (~approved) & (~declined)
    out[approved] = np.abs(p[approved] - t_low)
    out[declined] = np.abs(p[declined] - t_high)
    out[refer] = np.minimum(np.abs(p[refer] - t_low), np.abs(p[refer] - t_high))
    return out


def flip_delta_utility(
    targets,
    codes_before,
    codes_after,
    *,
    loss_gain_ratio: float,
    gain: float = 1.0,
) -> dict:
    """Per-loan and total stylised delta-utility of a decision change.

    Loan-level utility: APPROVE a good loan +g, APPROVE a default -g*L,
    everything else 0 (decisions.yaml A7; refer excluded from primary
    utility). Delta = utility_after - utility_before per loan.
    """
    y = np.asarray(targets, dtype=float)
    a, b = _both_valid(codes_before, codes_after)
    if len(y) != len(a) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be binary and match code length")

    def loan_util(codes: np.ndarray) -> np.ndarray:
        u = np.zeros(len(codes), dtype=float)
        funded = codes == APPROVE
        u[funded] = np.where(y[funded] == 0.0, gain, -gain * loss_gain_ratio)
        return u

    d = loan_util(b) - loan_util(a)
    return {
        "per_loan": d,
        "total": float(d.sum()),
        "flipping_loans": int((a != b).sum()),
        "util_flipping_loans": int((d != 0.0).sum()),
    }


def matched_null_total_delta_utility(
    targets,
    probabilities_after,
    codes_before,
    *,
    t_low: float,
    t_high: float,
    draws: int,
    seed: int,
    loss_gain_ratio: float,
    gain: float = 1.0,
) -> np.ndarray:
    """Null distribution of total delta-utility under broken loan pairing.

    The second decision vector is regenerated from a random PERMUTATION of
    the after-scores: marginal decision rates are identical to the observed
    arm, but the pairing of scores to outcome rows is chance — exactly the
    "matched null" the plan asks for (1000 draws by default from config).
    Returns `draws` total delta-utilities (the caller compares the observed
    total against this distribution).
    """
    pa = np.asarray(probabilities_after, dtype=float)
    a = _codes(codes_before)
    y = np.asarray(targets, dtype=float)
    if pa.shape != a.shape or len(y) != len(a) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("vectors must be equal-length with binary targets")
    if not np.all((pa >= 0) & (pa <= 1)):
        raise ValueError("after-probabilities must lie in [0,1]")
    if draws < 2:
        raise ValueError("draws must be >= 2")
    if not (0.0 < t_low < t_high < 1.0):
        raise ValueError("thresholds must satisfy 0 < t_low < t_high < 1")

    rng = np.random.default_rng(seed)
    totals = np.empty(draws, dtype=float)
    for k in range(draws):
        codes_k = threshold_policy(rng.permutation(pa), t_low=t_low, t_high=t_high)
        totals[k] = flip_delta_utility(
            y, a, codes_k, loss_gain_ratio=loss_gain_ratio, gain=gain
        )["total"]
    return totals
