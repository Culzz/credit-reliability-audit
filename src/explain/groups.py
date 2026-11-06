"""Grouped attribution and permutation importance (PLAN.md T8.5).

Feature groups come from configs/feature_groups.yaml at freeze time
(T5.4: built from training data only, before any explanation results
exist). This module takes the group assignment as an explicit argument
and enforces the partition discipline: every feature index belongs to
exactly one named group — no overlaps, no gaps.

Two grouped importances are provided:
- group |shap|: mean over explanation rows of |sum of attributions inside
  the group| — the dependency-aware attribution summary;
- grouped permutation importance: jointly permute all rows of every
  feature in a group (R repeats, seeded), evaluate on a caller-supplied
  metric, and report (permuted - baseline) for losses / (baseline -
  permuted) for gains, normalised so positive always means "the group
  carries information the model actually uses".

Extrapolation caution for correlated groups goes in the protocol text
(Hooker, Mentch & Zhou 2021) — cited there, not re-argued here.
"""

from __future__ import annotations

import numpy as np


def _validate_group_partition(groups: dict[str, list[int]], n_features: int) -> None:
    if not groups:
        raise ValueError("group assignment must be non-empty")
    seen: list[int] = []
    for name, members in groups.items():
        if not name or not members:
            raise ValueError(f"group {name!r} must carry feature indices")
        members = list(members)
        if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= n_features for i in members):
            raise ValueError(f"group {name!r} references indices outside [0, {n_features})")
        seen.extend(members)
    unique = set(seen)
    if len(unique) != len(seen):
        raise ValueError("groups overlap: every feature must belong to exactly one group")
    missing = sorted(set(map(int, range(n_features))) - set(map(int, unique)))
    if missing:
        raise ValueError(f"groups must cover every feature exactly once; missing {missing}")


def group_attributions(V: np.ndarray, groups: dict[str, list[int]]) -> dict[str, float]:
    """Mean over explanation rows of |sum of attributions| per group."""
    v = np.asarray(V, dtype=float)
    if v.ndim != 2 or v.shape[0] == 0 or not np.all(np.isfinite(v)):
        raise ValueError("V must be a non-empty finite 2-D attribution matrix")
    _validate_group_partition(groups, v.shape[1])
    return {
        name: float(np.abs(np.asarray(v, dtype=float)[:, list(members)].sum(axis=1)).mean())
        for name, members in groups.items()
    }


def grouped_permutation_importance(
    predict,
    y_true,
    X,
    groups: dict[str, list[int]],
    *,
    metric,
    metric_is_loss: bool,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Joint row-permutation importance of whole feature groups.

    - `predict(X) -> probabilities` is the frozen pipeline (preproc+model);
    - `metric(y_true, probabilities) -> float` is the caller's score
      (e.g. log loss with metric_is_loss=True, ROC-AUC with False);
    - `repeats` seeded joint permutations per group.

    Per-group importance: signed score change from destroying the group's
    joint information, positive = informative. Sign convention: for a loss
    the raw change is permuted - baseline (permuting informative features
    raises the loss); for a gain it is baseline - permuted.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y_true, dtype=float)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D matrix")
    if ya.ndim != 1 or len(ya) != len(Xa):
        raise ValueError("y must be 1-D and aligned with X")
    if repeats < 2:
        raise ValueError("repeats must be >= 2")
    if not (0.0 < seed <= max(np.iinfo(np.int64).max, 0)):
        raise ValueError("seed must be a positive integer")
    _validate_group_partition(groups, Xa.shape[1])

    baseline = float(metric(ya, predict(Xa)))
    rng = np.random.default_rng(seed)
    n = len(Xa)

    def signed_change(joint_members: np.ndarray) -> float:
        Xp = Xa.copy()
        cols = tuple(int(j) for j in joint_members)
        idx = rng.permutation(n)
        Xp[:, cols] = Xa[idx][:, cols]  # joint row-permutation of the block
        perturbed = float(metric(ya, predict(Xp)))
        raw = perturbed - baseline
        return raw if metric_is_loss else -raw

    out: dict[str, dict[str, float]] = {}
    for name, members in groups.items():
        draws = np.array([signed_change(np.asarray(members)) for _ in range(repeats)])
        out[name] = {"mean": float(draws.mean()), "sd": float(draws.std(ddof=1))}
    return out
