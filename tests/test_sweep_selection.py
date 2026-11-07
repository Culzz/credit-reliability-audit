"""Synthetic tests for the broad-sweep runner's deep-subset selection."""

from __future__ import annotations

import numpy as np

from scripts.run_sweep import deep_subset_indices


def test_anchors_hit_min_and_max():
    order = np.array([5, 2, 8, 0, 3])  # configs ordered by ascending AUC
    out = deep_subset_indices(order)
    assert out["min"] == 5  # lowest-AUC config (index 5 in the original space)
    assert out["max"] == 3  # highest-AUC config


def test_quartile_anchors_for_30_configs():
    order = np.arange(30)
    out = deep_subset_indices(order)
    assert out["q25"] == round(0.25 * 29)   # 7
    assert out["q50"] == round(0.50 * 29)   # 15 (median, round-half-even of 14.5)
    assert out["q75"] == round(0.75 * 29)   # 22


def test_anchors_are_quantiles_of_auc_not_config_id():
    # reversed AUC order: the anchors must track the QUANTILE POSITION in
    # the ascending AUC ordering, not the config id; even-n (30 configs)
    # median ties resolve downward via round-half-even
    order = np.arange(30)[::-1]
    out = deep_subset_indices(order)
    assert out["q25"] == order[7]
    assert out["q50"] == order[14]   # order[round(14.5)] -> lower central position
    assert out["q75"] == order[22]
    assert out["min"] == order[0]
    assert out["max"] == order[29]
