"""Tests for the real-cohort loader (synthetic assets; no real parquet)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts._real as realmod


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for vintage in ["2012Q1", "2013Q3", "2014Q1", "2014Q2", "2015Q1"]:
        n = 200 if vintage.startswith("2014") or vintage == "2015Q1" else 40
        for i in range(n):
            rows.append(
                {
                    "id": f"{vintage}-{i}",
                    "vintage": vintage,
                    "y": int(rng.random() < 0.25),
                    "fico": float(rng.uniform(600, 800)),
                    "dti": float(rng.uniform(0, 40)),
                }
            )
    # one unresolved label inside a test cohort must be dropped, never imputed
    rows[80]["y"] = pd.NA  # first 2014Q1 row (40 + 40 rows precede it)
    return pd.DataFrame(rows)


def _write_splits(tmp_path: Path, min_events: int = 10, min_nonevents: int = 10) -> Path:
    doc = {
        "splits": {
            "train": {"start": "2012Q1", "end": "2012Q1"},
            "validation": {"start": "2013Q3", "end": "2013Q3"},
            "test": {"start": "2014Q1", "end": "2014Q2"},
            "min_events_per_eval_cohort": min_events,
            "min_nonevents_per_eval_cohort": min_nonevents,
        }
    }
    path = tmp_path / "splits.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_loader_constructs_and_enforces_floors(tmp_path: Path, monkeypatch):
    sample = tmp_path / "mature.parquet"
    _synthetic_frame().to_parquet(sample)
    splits = _write_splits(tmp_path)
    frozen = tmp_path / "PROTOCOL_FROZEN"
    frozen.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(realmod, "REAL_SAMPLE", sample)
    monkeypatch.setattr(realmod, "FROZEN", frozen)

    rc = realmod.load_real_cohorts(splits)
    rc.assert_disjoint()
    assert rc.train_vintages == ["2012Q1"]
    assert rc.val_vintages == ["2013Q3"]
    assert sorted(rc.test) == ["2014Q1", "2014Q2"]
    # rows with unresolved labels are dropped, never imputed: the NA row
    # sits in 2014Q1, so that cohort keeps 199 of its 200 constructed rows
    assert rc.test["2014Q1"]["n"] == 199
    assert rc.test["2014Q2"]["n"] == 200
    assert len(rc.feature_columns) == 2


def test_loader_refuses_when_frozen_marker_missing(tmp_path: Path, monkeypatch):
    sample = tmp_path / "mature.parquet"
    _synthetic_frame().to_parquet(sample)
    splits = _write_splits(tmp_path)
    monkeypatch.setattr(realmod, "REAL_SAMPLE", sample)
    monkeypatch.setattr(realmod, "FROZEN", tmp_path / "absent_marker")
    with pytest.raises(RuntimeError, match="PROTOCOL_FROZEN"):
        realmod.load_real_cohorts(splits)


def test_loader_raises_below_floor(tmp_path: Path, monkeypatch):
    sample = tmp_path / "mature.parquet"
    _synthetic_frame().to_parquet(sample)
    splits = _write_splits(tmp_path, min_events=10_000)
    frozen = tmp_path / "PROTOCOL_FROZEN"
    frozen.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(realmod, "REAL_SAMPLE", sample)
    monkeypatch.setattr(realmod, "FROZEN", frozen)
    with pytest.raises(AssertionError, match="below the frozen floor"):
        realmod.load_real_cohorts(splits)


def test_quarters_span_quarters():
    assert realmod._quarters("2014Q3", "2015Q2") == [
        "2014Q3", "2014Q4", "2015Q1", "2015Q2",
    ]
