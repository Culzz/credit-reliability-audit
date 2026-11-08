"""Tests for the noise-floor runner's grouped metric block."""

from __future__ import annotations

import numpy as np

from scripts.run_noise_floor import _group_partition, _grouped_block


def test_grouped_block_identical_vectors():
    g = np.array([0.5, 0.2, 0.3, 0.0, 0.1, 0.4])
    out = _grouped_block(g, g)
    assert out["grouped_cosine"] == 1.0
    assert out["grouped_spearman"] == 1.0
    assert out["grouped_max_gap"] == 0.0


def test_grouped_block_proportional_vectors():
    # scale change must not move cosine, but shows in max_gap
    g = np.array([0.5, 0.2, 0.3, 0.0, 0.1, 0.4])
    out = _grouped_block(g, 2.0 * g)
    assert out["grouped_cosine"] == 1.0
    assert out["grouped_spearman"] == 1.0
    assert out["grouped_max_gap"] == 0.5  # largest group doubles: 2*0.5 - 0.5


def test_grouped_block_rank_flip():
    a = np.array([0.10, 0.09, 0.08, 0.07, 0.06, 0.05])
    b = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    out = _grouped_block(a, b)
    assert out["grouped_spearman"] == -1.0
    # cosine stays high for magnitude-preserving reversals; the rank metric
    # is what catches ordering flips - that is WHY both are reported
    assert out["grouped_cosine"] > 0.9
    assert out["grouped_max_gap"] == 0.05


def test_group_partition_counts_and_cover():
    # the real T5.4 partition: 6 groups, cover of 52 design columns
    from scripts.run_noise_floor import build_feature_frame_columns

    columns = build_feature_frame_columns()
    groups = _group_partition(columns)
    flat = [i for members in groups.values() for i in members]
    assert len(columns) == 52
    assert sorted(flat) == list(range(52))
    assert sorted(groups) == [
        "application_context",
        "borrower_capacity",
        "credit_history_tenure",
        "credit_score",
        "delinquency_public_record",
        "utilization",
    ]
