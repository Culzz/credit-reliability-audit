"""Real-cohort loader for the frozen protocol phase (post PROTOCOL_FROZEN).

Reads the mature 36-month feature cache (data/interim/
mature_36m_features.parquet; audited KEEP features through the
deterministic T3.6 transform) and constructs the ratified cohorts from
configs/splits.yaml:

- train: 2012Q1-2013Q2 (84,594 rows, matches the plan's recorded count),
- validation: 2013Q3-2013Q4,
- test: the eight ratified quarterly cohorts 2014Q1-2015Q4, split out
  per-cohort (the unit of analysis), each required to clear the
  events/non-events floor from splits.yaml AFTER rows with unresolved
  labels are dropped (the last cohorts carry single-digit counts of
  still-open loans and are excluded, never imputed).

The loader refuses to run without protocol/PROTOCOL_FROZEN: the test
window is readable only under the frozen protocol (mirrors the dry-run
guard in scripts._common).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SAMPLE = REPO_ROOT / "data" / "interim" / "mature_36m_features.parquet"
FROZEN = REPO_ROOT / "protocol" / "PROTOCOL_FROZEN"


@dataclasses.dataclass
class RealCohorts:
    feature_columns: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_vintages: list[str]
    val_vintages: list[str]
    test: dict[str, dict]  # cohort quarter -> {"X":..., "y":..., "n":..., "n_events":..., "n_nonevents":...}

    def assert_disjoint(self) -> None:
        train, val, test = set(self.train_vintages), set(self.val_vintages), set(self.test)
        overlap = (train & val) | (train & test) | (val & test)
        if overlap:
            raise AssertionError(f"cohort vintages overlap: {sorted(overlap)}")

    def shape_report(self) -> str:
        return (
            f"train {len(self.y_train):,} | val {len(self.y_val):,} | "
            f"test cohorts {len(self.test)} "
            f"({sum(c['n'] for c in self.test.values()):,} rows)"
        )


def _quarters(start: str, end: str) -> list[str]:
    y0, q0 = int(start[:4]), int(start[-1])
    y1, q1 = int(end[:4]), int(end[-1])
    out, y, q = [], y0, q0
    while (y, q) <= (y1, q1):
        out.append(f"{y}Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def load_real_cohorts(splits_path: Path | None = None) -> RealCohorts:
    if not FROZEN.exists():
        raise RuntimeError(
            "real cohorts are readable only after the freeze: "
            "protocol/PROTOCOL_FROZEN is missing"
        )
    sp = splits_path or REPO_ROOT / "configs" / "splits.yaml"
    cfg = yaml.safe_load(sp.read_text(encoding="utf-8"))["splits"]
    min_events = int(cfg["min_events_per_eval_cohort"])
    min_nonevents = int(cfg["min_nonevents_per_eval_cohort"])

    df = pd.read_parquet(REAL_SAMPLE)
    feature_columns = [c for c in df.columns if c not in ("id", "vintage", "y")]
    resolved = df.dropna(subset=["y"])

    def window(start: str, end: str) -> pd.DataFrame:
        quarters = set(_quarters(start, end))
        return resolved[resolved.vintage.isin(quarters)]

    train = window(cfg["train"]["start"], cfg["train"]["end"])
    val = window(cfg["validation"]["start"], cfg["validation"]["end"])

    test: dict[str, dict] = {}
    for quarter in _quarters(cfg["test"]["start"], cfg["test"]["end"]):
        coh = resolved[resolved["vintage"] == quarter]
        n_events = int((coh["y"] == 1).sum())
        n_nonevents = int((coh["y"] == 0).sum())
        if n_events < min_events or n_nonevents < min_nonevents:
            raise AssertionError(
                f"cohort {quarter} below the frozen floor: "
                f"events {n_events}, nonevents {n_nonevents} "
                f"(required >= {min_events}/{min_nonevents})"
            )
        xs = coh[feature_columns].to_numpy(dtype="float32")
        test[quarter] = {
            "X": xs,
            "y": coh["y"].to_numpy(dtype="float64"),
            "n": len(coh),
            "n_events": n_events,
            "n_nonevents": n_nonevents,
        }

    return RealCohorts(
        feature_columns=feature_columns,
        X_train=train[feature_columns].to_numpy(dtype="float32"),
        y_train=train["y"].to_numpy(dtype="float64"),
        X_val=val[feature_columns].to_numpy(dtype="float32"),
        y_val=val["y"].to_numpy(dtype="float64"),
        train_vintages=_quarters(cfg["train"]["start"], cfg["train"]["end"]),
        val_vintages=_quarters(cfg["validation"]["start"], cfg["validation"]["end"]),
        test=test,
    )
