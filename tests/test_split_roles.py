"""Tests for split-window resolution and the descriptive access surface.

Synthetic frames only (CONTEXT.md §4 rule 14). No real data exists yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import guard
from src.data.loader import (
    ProtocolLockError,
    filter_by_vintage_window,
    load_descriptive,
    load_split,
    vintage_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VINTAGES = ["2012Q1", "2012Q2", "2013Q3", "2013Q4", "2014Q1", "2015Q4"]


def synthetic_frame() -> pd.DataFrame:
    rows = []
    for i, v in enumerate(VINTAGES):
        for k in range(3):
            rows.append({"loan_id": i * 10 + k, "vintage": v, "x1": k, "y": k % 2, "p": 0.5})
    return pd.DataFrame(rows)


def tmp_splits(tmp_path: Path, with_test: bool = True) -> Path:
    cfg = {
        "splits": {
            "train": {"start": "2012Q1", "end": "2013Q2"},
            "validation": {"start": "2013Q3", "end": "2013Q4"},
            "test": {"start": "2014Q1", "end": "2015Q4"} if with_test else {"start": None, "end": None},
        }
    }
    p = tmp_path / "splits.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_window_filter_inclusive():
    out = filter_by_vintage_window(synthetic_frame(), vintage_col="vintage", start="2012Q1", end="2013Q1")
    assert sorted(out["vintage"].unique().tolist()) == ["2012Q1", "2012Q2"]


def test_window_inverted_rejected():
    with pytest.raises(ValueError, match="after window end"):
        filter_by_vintage_window(synthetic_frame(), vintage_col="vintage", start="2013Q4", end="2012Q1")


def test_window_missing_vintage_col():
    with pytest.raises(KeyError):
        filter_by_vintage_window(synthetic_frame().drop(columns=["vintage"]), vintage_col="vintage", start="2012Q1", end="2013Q1")


def test_load_split_train_validation_pre_freeze(tmp_path: Path):
    assert vintage_window("train", splits_path=tmp_splits(tmp_path)) == {"start": "2012Q1", "end": "2013Q2"}
    assert vintage_window("validation", splits_path=tmp_splits(tmp_path)) == {"start": "2013Q3", "end": "2013Q4"}
    assert load_split("train", splits_path=tmp_splits(tmp_path)) == vintage_window(
        "train", splits_path=tmp_splits(tmp_path)
    )


def test_load_split_test_resolved_after_freeze():
    """Post-freeze (Gate 5, protocol-v1, D-8): the real test window
    resolves from the shipped config plus the real frozen marker."""
    assert vintage_window("test") == {"start": "2014Q1", "end": "2015Q4"}


def test_load_split_test_needs_gate4_window(tmp_path: Path):
    """Even with the flag, an unset test window must fail loudly."""
    flag = tmp_path / "PROTOCOL_FROZEN"
    flag.write_text("PI", encoding="utf-8")
    assert guard.protocol_is_frozen(flag=flag)
    with pytest.raises(ValueError, match="Gate 4"):
        vintage_window("test", splits_path=tmp_splits(tmp_path, with_test=False), flag=flag)


def test_load_split_test_after_freeze(tmp_path: Path):
    flag = tmp_path / "PROTOCOL_FROZEN"
    flag.write_text("PI", encoding="utf-8")
    assert vintage_window("test", splits_path=tmp_splits(tmp_path), flag=flag) == {"start": "2014Q1", "end": "2015Q4"}


def test_descriptive_strips_target_like_columns():
    out = load_descriptive(synthetic_frame())
    cols = set(out.columns.str.lower())
    assert cols.isdisjoint({"y", "target", "loan_status", "default", "p", "predicted_probability"})
    assert {"loan_id", "vintage", "x1"}.issubset(out.columns)


def test_descriptive_rejects_explicit_leak_request():
    with pytest.raises(ProtocolLockError, match="refused target-like"):
        load_descriptive(synthetic_frame(), feature_columns=["x1", "y"])


def test_real_config_test_window_matches_freeze():
    """The shipped splits.yaml must carry exactly the ratified cohorts:
    eight quarterly windows 2014Q1-2015Q4 (D-8), with the real marker
    present in the repo."""
    assert guard.protocol_is_frozen()
    window = vintage_window("test")
    assert window == {"start": "2014Q1", "end": "2015Q4"}
    cfg = yaml.safe_load(
        (REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8")
    )["splits"]["test"]
    assert cfg["start"] == "2014Q1" and cfg["end"] == "2015Q4"
