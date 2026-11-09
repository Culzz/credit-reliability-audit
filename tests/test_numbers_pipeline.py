"""Tests for the numbers-export / manuscript-render pipeline (CONTEXT.md 11)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from scripts.export_numbers import (
    import_baseline,
    import_calibration,
    import_curves,
    import_decisions,
    import_deep_subset,
    import_noise_floor,
    import_sample_size,
    import_sweep,
)
from scripts.render_paper import render

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestAdapters:
    def test_baseline_wide(self):
        df = pd.DataFrame(
            {
                "family": ["logistic_regression", "xgboost"],
                "metric": ["roc_auc", "roc_auc"],
                "point": [0.6209417, 0.5793123],
                "lo": [0.5, 0.4],
                "hi": [0.7, 0.6],
                "seconds": [1.5, 5.0],
                "extra_col_ignored_by_need": [0, 0],
            }
        )
        out = import_baseline(df, "baseline_dryrun_metrics.csv")
        assert out["logistic_regression.roc_auc.point"]["value"] == pytest.approx(0.620942)
        assert out["xgboost.roc_auc.hi"]["value"] == pytest.approx(0.6)
        assert out["logistic_regression.roc_auc.lo"]["table"] == "baseline_dryrun_metrics.csv"

    def test_calibration_long(self):
        df = pd.DataFrame(
            {
                "family": ["random_forest"],
                "arm": ["isotonic"],
                "metric": ["citl"],
                "value": [-1.2345678],
                "n_calib": [2538],
                "n_eval": [1779],
                "fit_seconds": [1.4],
            }
        )
        out = import_calibration(df, "calibration_audit_dryrun.csv")
        assert out["random_forest.isotonic.citl"]["value"] == pytest.approx(-1.234568)

    def test_curves(self):
        curves = pd.DataFrame(
            {
                "family": ["xgboost"],
                "arm": ["isotonic"],
                "bin": [3],
                "n": [120],
                "share": [0.06671],
                "p_bar": [0.1234567],
                "y_bar": [0.1345678],
            }
        )
        out = import_curves(curves, "calibration_curves_dryrun.csv")
        assert out["xgboost.isotonic.bin3.p_bar"]["value"] == pytest.approx(0.123457)
        assert out["xgboost.isotonic.bin3.y_bar"]["value"] == pytest.approx(0.134568)
        assert out["xgboost.isotonic.bin3.share"]["value"] == pytest.approx(0.06671)

    def test_sweep_and_deep_subset(self):
        sweep = pd.DataFrame(
            {
                "family": ["xgboost"],
                "config_index": [7],
                "params_json": ['{"learning_rate": 0.05}'],
                "auc": [0.5793123],
                "brier": [0.1234567],
                "log_loss": [0.4498765],
            }
        )
        out = import_sweep(sweep, "sweep_dryrun.csv")
        assert out["xgboost.c7.auc"]["value"] == pytest.approx(0.579312)
        assert out["xgboost.c7.log_loss"]["value"] == pytest.approx(0.449877)

        deep = pd.DataFrame(
            {
                "family": ["xgboost"],
                "anchor": ["q50"],
                "config_index": [12],
                "auc": [0.5882],
            }
        )
        out = import_deep_subset(deep, "deep_subset_dryrun.csv")
        assert out["xgboost.q50.config_index"]["value"] == 12

    def test_decisions_and_noise_and_size_long(self):
        dec = pd.DataFrame(
            {
                "family": ["lightgbm"],
                "after_arm": ["platt"],
                "ratio": [5.0],
                "half_width": [0.02],
                "metric": ["null_mean"],
                "value": [-208.171717],
            }
        )
        out = import_decisions(dec, "decision_flip_dryrun.csv")
        assert "lightgbm.platt.r5.0.h0.02.null_mean" in out

        noise = pd.DataFrame(
            {
                "family": ["xgboost"],
                "kind": ["seed"],
                "a": [1],
                "b": [2],
                "metric": ["cosine"],
                "value": [0.9967123],
            }
        )
        out = import_noise_floor(noise, "noise_floor_dryrun.csv")
        assert "xgboost.kseed.a1-b2.cosine" in out

        size = pd.DataFrame(
            {
                "family": ["xgboost"],
                "size": ["full"],
                "repeat": [3],
                "arm": ["platt_minus_intercept"],
                "metric": ["delta_brier_mean"],
                "value": [0.00123456],
            }
        )
        out = import_sample_size(size, "sample_size_dryrun.csv")
        assert "xgboost.full.draw3.platt_minus_intercept.delta_brier_mean" in out

    def test_adapter_mismatch_is_loud(self):
        df = pd.DataFrame({"wrong": ["schema"]})
        with pytest.raises(ValueError, match="adapter mismatch"):
            import_calibration(df, "x.csv")


class TestRender:
    NUMBERS: ClassVar[dict] = {
        "dryrun.a.b": {"value": 0.6209417},
        "dryrun.x.y": {"value": -208.0},
        "dryrun.int": {"value": 42.0},
    }

    def test_substitution_and_formatting(self):
        text = "AUC {{n:dryrun.a.b}} vs {{n:dryrun.x.y}} and {{n:dryrun.int}}."
        out, missing = render(text, self.NUMBERS)
        assert missing == []
        assert out == "AUC 0.620942 vs -208 and 42."

    def test_missing_key_is_reported_and_left_in_place(self):
        text = "value {{n:dryrun.absent.key}} stays literal"
        out, missing = render(text, self.NUMBERS)
        assert missing == ["dryrun.absent.key"]
        assert "{{n:dryrun.absent.key}}" in out  # never silently dropped

    def test_value_formatting_dot_g(self):
        out, missing = render("{{n:dryrun.a.b}}", self.NUMBERS)
        assert out == "0.620942"  # 6 significant digits, trailing zeros trimmed
        assert missing == []

    def test_fenced_placeholders_stay_literal(self):
        # the stub's fenced example documents the syntax; it must NOT
        # resolve into a dry-run number sitting in prose (the earlier
        # rendered stub embedded 0.620923 - that is the regression pin)
        text = (
            "prose {{n:dryrun.a.b}} continues\n\n"
            "```\n{{n:dryrun.a.b}}\n```\n\n"
            "after {{n:missing.pretty.print}} end"
        )
        out, missing = render(text, self.NUMBERS)
        assert missing == ["missing.pretty.print"]  # fence did not swallow checks
        assert "{{n:dryrun.a.b}}" in out          # fenced: still literal
        assert "0.620942" in out                  # unfenced: resolved
        assert out.count("0.620942") == 1


def test_claims_register_schema_is_fixed():
    with (REPO_ROOT / "paper" / "claims.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "claim_id",
        "sentence",
        "evidence_file",
        "number_keys",
        "status",
    ]
    assert all(len(r) == 5 for r in rows)


def test_real_export_runs_and_is_dryrun_namespaced():
    # the real export must exist, be dry-run-marked, and have keys only
    # under the dryrun namespace convention (family-level keys start with
    # a family name; the meta block carries the warning)
    import json

    data = json.loads((REPO_ROOT / "paper" / "numbers_dryrun.json").read_text(encoding="utf-8"))
    assert data["meta"]["mode"] == "dryrun"
    assert "plumbing" in data["meta"]["warning"]
    nums = data["numbers"]
    assert len(nums) > 1000
    # spot anchor: the value committed for the LR dry-run AUC
    assert nums["dryrun.logistic_regression.roc_auc.point"]["value"] == pytest.approx(0.620923)
