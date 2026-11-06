"""Export table values into paper/numbers_dryrun.json (CONTEXT.md I-11).

Every manuscript number must come from this export, never from a hand-typed
value. This dry-run variant reads ONLY the plumbing-scale dry-run tables and
writes to `paper/numbers_dryrun.json` with a loud pre-freeze marker; the real
post-freeze export (`export_numbers.py` on frozen cohorts) will use the same
adapter keys under a `main` namespace. Placeholder keys therefore cannot be
confused between plumbing artifacts and real results.

Adapters are explicit: a table whose header does not match its adapter is
refused rather than silently re-parsed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
OUT = REPO_ROOT / "paper" / "numbers_dryrun.json"

WARNING = (
    "DRY-RUN (plumbing sample, pre-freeze). NOT research results: "
    "2,538-row training / 1,779-row validation slice; not for the manuscript."
)


def _need(df: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"adapter mismatch in {table}: missing {missing}")


def _round6(value: float) -> float:
    return round(float(value), 6)


def import_baseline(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "metric", "point", "lo", "hi", "seconds"], table)
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.{row.metric}"
        for suffix, value in (("point", row.point), ("lo", row.lo), ("hi", row.hi)):
            out[f"{key}.{suffix}"] = {
                "value": _round6(value),
                "table": table,
                "columns": {"family": row.family, "metric": row.metric, "stat": suffix},
            }
    return out


def import_calibration(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(
        df,
        ["family", "arm", "metric", "value", "n_calib", "n_eval", "fit_seconds"],
        table,
    )
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.{row.arm}.{row.metric}"
        out[key] = {
            "value": _round6(row.value),
            "table": table,
            "columns": {"family": row.family, "arm": row.arm, "metric": row.metric},
        }
    return out


def import_decisions(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(
        df,
        ["family", "after_arm", "ratio", "half_width", "metric", "value"],
        table,
    )
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.{row.after_arm}.r{row.ratio}.h{row.half_width}.{row.metric}"
        out[key] = {
            "value": _round6(row.value),
            "table": table,
            "columns": {
                "family": row.family,
                "after_arm": row.after_arm,
                "ratio": row.ratio,
                "half_width": row.half_width,
                "metric": row.metric,
            },
        }
    return out


def import_noise_floor(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "seed_i", "seed_j", "metric", "value"], table)
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.s{row.seed_i}-s{row.seed_j}.{row.metric}"
        out[key] = {
            "value": _round6(row.value),
            "table": table,
            "columns": {
                "family": row.family,
                "seed_i": row.seed_i,
                "seed_j": row.seed_j,
                "metric": row.metric,
            },
        }
    return out


def import_sample_size(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "size", "repeat", "arm", "metric", "value"], table)
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.{row.size}.draw{row.repeat}.{row.arm}.{row.metric}"
        out[key] = {
            "value": _round6(row.value),
            "table": table,
            "columns": {
                "family": row.family,
                "size": row.size,
                "repeat": row.repeat,
                "arm": row.arm,
                "metric": row.metric,
            },
        }
    return out


ADAPTERS = {
    "baseline_dryrun_metrics": import_baseline,
    "calibration_audit_dryrun": import_calibration,
    "decision_flip_dryrun": import_decisions,
    "noise_floor_dryrun": import_noise_floor,
    "sample_size_dryrun": import_sample_size,
}


def main() -> None:
    numbers: dict[str, dict] = {}
    skipped: list[str] = []
    for stem, adapter in sorted(ADAPTERS.items()):
        path = TABLES / f"{stem}.csv"
        if not path.exists():
            skipped.append(stem)
            continue
        df = pd.read_csv(path)
        # every key carries the explicit dryrun namespace so placeholder
        # references can never confuse plumbing artifacts with the real,
        # post-freeze export (whose keys get the `main.` prefix)
        numbers.update(
            {f"dryrun.{key}": entry for key, entry in adapter(df, path.name).items()}
        )
    if not numbers:
        raise RuntimeError("no adapter produced any number; check reports/tables")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"meta": {"mode": "dryrun", "warning": WARNING}, "numbers": numbers},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(REPO_ROOT)} with {len(numbers)} keys")
    if skipped:
        print("skipped (missing tables):", ", ".join(skipped))


if __name__ == "__main__":
    main()
