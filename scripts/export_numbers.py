"""Export table values into the paper numbers JSON (CONTEXT.md I-11).

Every manuscript number must come from this export, never from a hand-typed
value. Two modes:

- `--source dryrun` (default) reads ONLY the plumbing-scale dry-run tables and
  writes `paper/numbers_dryrun.json` with a loud pre-freeze marker;
- `--source main` reads the frozen-cohort tables (post PROTOCOL_FROZEN) and
  writes `paper/numbers.json` under the `main.` key namespace.

Adapters are explicit: a table whose header does not match its adapter is
refused rather than silently re-parsed. Derived entries (means over configs
or lags) are computed here from observed table rows, never hand-entered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
FROZEN = REPO_ROOT / "protocol" / "PROTOCOL_FROZEN"
OUT = {
    "dryrun": REPO_ROOT / "paper" / "numbers_dryrun.json",
    "main": REPO_ROOT / "paper" / "numbers.json",
}

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


def _slug(value) -> str:
    """Placeholder-safe token: [A-Za-z0-9_.-] only."""
    out = "".join(
        ch if (ch.isalnum() or ch in "_.-") else "."
        for ch in str(value)
    )
    while ".." in out:
        out = out.replace("..", ".")
    return out.strip(".")


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
    _need(df, ["family", "kind", "a", "b", "metric", "value"], table)
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.k{row.kind}.a{row.a}-b{row.b}.{row.metric}"
        out[key] = {
            "value": _round6(row.value),
            "table": table,
            "columns": {
                "family": row.family,
                "kind": row.kind,
                "a": row.a,
                "b": row.b,
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


def import_sweep(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "config_index", "params_json", "auc", "brier", "log_loss"], table)
    out = {}
    for row in df.itertuples(index=False):
        prefix = f"{row.family}.c{row.config_index}"
        for metric, value in (("auc", row.auc), ("brier", row.brier), ("log_loss", row.log_loss)):
            out[f"{prefix}.{metric}"] = {
                "value": _round6(value),
                "table": table,
                "columns": {"family": row.family, "config_index": row.config_index, "metric": metric},
            }
    return out


def import_deep_subset(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "anchor", "config_index", "auc"], table)
    out = {}
    for row in df.itertuples(index=False):
        key = f"{row.family}.{row.anchor}.config_index"
        out[key] = {
            "value": int(row.config_index),
            "table": table,
            "columns": {"family": row.family, "anchor": row.anchor},
        }
    return out


def import_curves(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "arm", "bin", "n", "share", "p_bar", "y_bar"], table)
    out = {}
    for row in df.itertuples(index=False):
        for field in ("share", "p_bar", "y_bar"):
            key = f"{row.family}.{row.arm}.bin{row.bin}.{field}"
            out[key] = {
                "value": _round6(getattr(row, field)),
                "table": table,
                "columns": {
                    "family": row.family,
                    "arm": row.arm,
                    "bin": row.bin,
                    "field": field,
                },
            }
    return out


ADAPTERS = {
    "baseline_dryrun_metrics": import_baseline,
    "calibration_audit_dryrun": import_calibration,
    "calibration_curves_dryrun": import_curves,
    "decision_flip_dryrun": import_decisions,
    "noise_floor_dryrun": import_noise_floor,
    "sample_size_dryrun": import_sample_size,
    "sweep_dryrun": import_sweep,
    "deep_subset_dryrun": import_deep_subset,
}


# ------------------------------------------------------------ real adapters
def _num(value):
    return None if pd.isna(value) else _round6(value)


def import_real_cohorts(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["role", "window", "n", "n_events", "n_nonevents",
               "default_rate", "n_unresolved"], table)
    out = {}
    for row in df.itertuples(index=False):
        base = f"{_slug(row.role)}.{_slug(row.window)}"
        for field in ("n", "n_events", "n_nonevents", "default_rate",
                      "n_unresolved"):
            out[f"{base}.{field}"] = {
                "value": _num(getattr(row, field)),
                "table": table,
                "columns": {"role": row.role, "window": row.window,
                            "field": field},
            }
    return out


def import_real_statistics(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["hypothesis", "statistic", "value", "ci_low", "ci_high",
               "p_raw", "p_holm", "verdict"], table)
    out = {}
    for row in df.itertuples(index=False):
        hyp = _slug(row.hypothesis)
        for field in ("value", "ci_low", "ci_high", "p_raw", "p_holm"):
            out[f"{hyp}.{field}"] = {
                "value": _num(getattr(row, field)),
                "table": table,
                "columns": {"hypothesis": row.hypothesis, "field": field},
            }
        out[f"{hyp}.verdict"] = {
            "value": str(row.verdict),
            "table": table,
            "columns": {"hypothesis": row.hypothesis, "field": "verdict"},
        }
    return out


def import_real_descriptives(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["hypothesis", "family", "value"], table)
    out = {}
    for row in df.itertuples(index=False):
        out[f"{_slug(row.hypothesis)}.{_slug(row.family)}"] = {
            "value": _num(row.value),
            "table": table,
            "columns": {"hypothesis": row.hypothesis, "family": row.family},
        }
    return out


def import_real_noise_floors(df: pd.DataFrame, table: str) -> dict[str, dict]:
    renamed = df.rename(columns={"q2.5": "q2_5"})
    _need(renamed, ["family", "dimension", "label", "q2_5", "q50", "q95",
                    "n_pairs"], table)
    out = {}
    for row in renamed.itertuples(index=False):
        base = (f"{_slug(row.family)}.{_slug(row.dimension)}"
                f".{_slug(row.label)}")
        for field in ("q2_5", "q50", "q95", "n_pairs"):
            out[f"{base}.{field}"] = {
                "value": _round6(getattr(row, field)),
                "table": table,
                "columns": {"family": row.family, "dimension": row.dimension,
                            "label": row.label, "quantile": field},
            }
    return out


def import_rq4_cells(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["key", "rho_outcome", "rho_base", "delta", "ci_low",
               "ci_high", "p_raw", "region"], table)
    out = {}
    for row in df.itertuples(index=False):
        base = _slug(row.key)
        for field in ("rho_outcome", "rho_base", "delta", "ci_low", "ci_high",
                      "p_raw"):
            out[f"{base}.{field}"] = {
                "value": _num(getattr(row, field)),
                "table": table,
                "columns": {"key": row.key, "field": field},
            }
        out[f"{base}.region"] = {
            "value": str(row.region),
            "table": table,
            "columns": {"key": row.key, "field": "region"},
        }
    return out


def import_recovery_cells(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "arm", "recovery", "lag"], table)
    out = {}
    grouped = df.groupby(["family", "arm", "lag"])["recovery"].mean()
    for (family, arm, lag), value in grouped.items():
        key = f"{_slug(family)}.{_slug(arm)}.lag{int(lag)}.mean"
        out[key] = {
            "value": _round6(value),
            "table": table,
            "columns": {"family": family, "arm": arm, "lag": int(lag)},
        }
    return out


def import_battery_summary(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "config_index", "kind", "window", "metric", "value"],
          table)
    seed = df[df["kind"] == "seed"]
    grouped = seed.groupby(["family", "window", "metric"])["value"]
    out = {}
    for (family, window, metric), vals in grouped:
        base = f"{_slug(family)}.{_slug(window)}.{_slug(metric)}"
        for stat, value in (("mean", vals.mean()), ("min", vals.min()),
                            ("max", vals.max()), ("n", len(vals))):
            out[f"{base}.{stat}"] = {
                "value": _round6(value),
                "table": table,
                "columns": {"family": family, "window": window,
                            "metric": metric, "stat": stat},
            }
    return out


def import_flip_summary(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "config_index", "kind", "window", "flip_rate"], table)
    seed = df[(df["kind"] == "seed") & (df["window"] != "train")]
    grouped = seed.groupby(["family", "window"])["flip_rate"]
    out = {}
    for (family, window), vals in grouped:
        base = f"{_slug(family)}.{_slug(window)}.flip_rate"
        for stat, value in (("mean", vals.mean()), ("max", vals.max()),
                            ("n", len(vals))):
            out[f"{base}.{stat}"] = {
                "value": _round6(value),
                "table": table,
                "columns": {"family": family, "window": window,
                            "stat": stat},
            }
    return out


def import_temporal_summary(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "config_index", "window_i", "window_j", "metric",
               "value"], table)
    grouped = df.groupby(["family", "window_i", "window_j", "metric"])["value"]
    out = {}
    for (family, wi, wj, metric), vals in grouped:
        key = (f"{_slug(family)}.{_slug(wi)}.{_slug(wj)}."
               f"{_slug(metric)}.mean")
        out[key] = {
            "value": _round6(vals.mean()),
            "table": table,
            "columns": {"family": family, "window_i": wi, "window_j": wj,
                        "metric": metric},
        }
    return out


def import_group_summary(df: pd.DataFrame, table: str) -> dict[str, dict]:
    _need(df, ["family", "kind", "group", "group_cosine", "group_spearman",
               "group_mean_gap"], table)
    grouped = df.groupby(["family", "kind", "group"])
    out = {}
    for (family, kind, group), sub in grouped:
        base = f"{_slug(family)}.{_slug(kind)}.{_slug(group)}"
        for col in ("group_cosine", "group_spearman", "group_mean_gap"):
            out[f"{base}.{col}.median"] = {
                "value": _round6(sub[col].median()),
                "table": table,
                "columns": {"family": family, "kind": kind, "group": group,
                            "stat": "median", "column": col},
            }
    return out


def import_posthoc(df: pd.DataFrame, table: str) -> dict[str, dict]:
    """Generic adapter for reports/tables/post_hoc_analyses.csv (labeled
    post-hoc estimates; none enters a confirmatory test)."""
    _need(df, ["analysis", "label", "statistic", "value", "ci_low",
               "ci_high", "n", "note"], table)
    out: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        base = (f"posthoc.{_slug(row.analysis)}.{_slug(row.label)}")
        for field in ("value", "ci_low", "ci_high", "n"):
            v = getattr(row, field)
            if pd.notna(v):
                out[f"{base}.{field}"] = {
                    "value": round(float(v), 6), "table": table}
    return out


MAIN_ADAPTERS = {
    "real_cohorts": import_real_cohorts,
    "real_statistics": import_real_statistics,
    "real_descriptives": import_real_descriptives,
    "real_noise_floors": import_real_noise_floors,
    "rq4_cells_real": import_rq4_cells,
    "recovery_cells_real": import_recovery_cells,
    "deep_agg_real": import_battery_summary,
    "flip_pairs_real": import_flip_summary,
    "temporal_pairs_real": import_temporal_summary,
    "group_pairs_real": import_group_summary,
    "post_hoc_analyses": import_posthoc,
    # B = 20,000 p-floor sensitivity (Appendix C.5); namespaced so the
    # keys cannot collide with the primary export
    "real_statisticsb20k":
        lambda df, table: {f"b20k.{k}": v for k, v in
                           import_real_statistics(df, table).items()},
    "sweep": import_sweep,
    "deep_subset": import_deep_subset,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(OUT), default="dryrun")
    args = parser.parse_args()
    if args.source == "main" and not FROZEN.exists():
        raise RuntimeError("--source main requires protocol/PROTOCOL_FROZEN")
    adapters = MAIN_ADAPTERS if args.source == "main" else ADAPTERS
    prefix = f"{args.source}."

    numbers: dict[str, dict] = {}
    skipped: list[str] = []
    for stem, adapter in sorted(adapters.items()):
        path = TABLES / f"{stem}.csv"
        if not path.exists():
            skipped.append(stem)
            continue
        df = pd.read_csv(path)
        numbers.update(
            {f"{prefix}{key}": entry
             for key, entry in adapter(df, path.name).items()}
        )
    if not numbers:
        raise RuntimeError("no adapter produced any number; check reports/tables")

    out_path = OUT[args.source]
    meta = {"mode": args.source}
    if args.source == "dryrun":
        meta["warning"] = WARNING
    else:
        meta["warning"] = (
            "Frozen-cohort results (protocol-v1). Every value is read from "
            "reports/tables by scripts/export_numbers.py; none is hand-typed."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"meta": meta, "numbers": numbers}, indent=2,
                   sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {out_path.relative_to(REPO_ROOT)} with {len(numbers)} keys "
          f"({args.source})")
    if skipped:
        print("skipped (missing tables):", ", ".join(skipped))



if __name__ == "__main__":
    main()
