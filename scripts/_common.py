"""Shared dry-run plumbing for the run_* scripts (single source of truth).

Every run_* script loads the plumbing sample the same way and mirrors the
PROPOSED train/validation windows from configs/splits.yaml; that logic is
spelled out ONCE here to keep the three runners handling identical
cohorts, guards, and iron-rule asserts from drifting apart before the
freeze. Sample creation itself lives in scripts/build_plumbing_sample.py
(T3.9); nothing here writes samples, ever.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import filter_by_vintage_window
from src.features.build import build_feature_frame, keep_features

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
SAMPLE = REPO_ROOT / "data" / "interim" / "plumbing_sample.parquet"


def load_configs(*names: str) -> dict:
    """Read configs/<name>.yaml, unwrapping the single top-level key.

    Every config file carries exactly one top-level mapping (e.g.
    `splits:`). The returned dict maps each file NAME to that inner
    mapping, so `load_configs("splits")["splits"]["train"]` reads the
    train block directly.
    """
    out: dict = {}
    for name in names:
        with (REPO_ROOT / "configs" / f"{name}.yaml").open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or len(doc) != 1:
            raise ValueError(f"configs/{name}.yaml must hold exactly one top-level mapping")
        (key, value), = doc.items()
        if str(key) != name:
            raise ValueError(f"configs/{name}.yaml top-level key {key!r} does not match the file name")
        out[name] = value
    return out


def check_test_window_guard(splits: dict, repo_root: Path = REPO_ROOT) -> str:
    """Test-cohort guard, shared verbatim by every runner.

    Returns a short status line; raises when test windows exist but the
    freeze flag does not (the data must stay untouched pre-freeze).
    """
    test_cfg = (splits or {}).get("test", {})
    if test_cfg.get("start") is None or test_cfg.get("end") is None:
        return "test window unset: LOCKED; dry-run covers training vs validation only."
    if not (repo_root / "protocol" / "PROTOCOL_FROZEN").exists():
        raise RuntimeError(
            "test windows set without protocol/PROTOCOL_FROZEN; refusing to proceed."
        )
    return "test window resolved and protocol frozen; proceeding."


@dataclass(frozen=True)
class PlumbingSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_index: pd.Index
    val_index: pd.Index

    def assert_disjoint(self) -> None:
        overlap = self.train_index.intersection(self.val_index)
        if len(overlap):
            raise RuntimeError(
                f"calibration and evaluation cohorts overlap ({len(overlap)} loans)."
            )

    @property
    def shape_report(self) -> str:
        return f"train {len(self.y_train):,} | validation {len(self.y_val):,}"


def load_plumbing_split() -> PlumbingSplit:
    """Sample + feature matrix split by the configured vintage windows."""
    df = pd.read_parquet(SAMPLE)
    cfg = load_configs("splits")["splits"]
    features = build_feature_frame(df[keep_features() + ["issue_d"]]).astype("float32")
    y = df["y"].astype(float)
    resolved = y.notna().to_numpy()

    train_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage",
        start=cfg["train"]["start"], end=cfg["train"]["end"],
    )
    val_df = filter_by_vintage_window(
        df.loc[resolved], vintage_col="vintage",
        start=cfg["validation"]["start"], end=cfg["validation"]["end"],
    )
    return PlumbingSplit(
        X_train=features.loc[train_df.index].to_numpy(),
        y_train=y.loc[train_df.index].to_numpy(),
        X_val=features.loc[val_df.index].to_numpy(),
        y_val=y.loc[val_df.index].to_numpy(),
        train_index=train_df.index,
        val_index=val_df.index,
    )
