"""Observed cohort sizes for the frozen real design (descriptive only).

Reads the mature 36-month cache through the same frozen split config the
modelling runners use, and writes one row per window with the counts a
reader needs to judge the design: rows, resolved labels, events,
non-events, default rate, and rows dropped because the label is still
unresolved (never imputed).

The loader's frozen-window guard is the authority for the modelling
windows; this script only reports what that cache contains, so its
numbers can be cited as observed cohort facts.

Outputs: reports/tables/real_cohorts.csv
Use    : python -m uv run python -m scripts.report_real_cohorts
"""

from __future__ import annotations

import pandas as pd
import yaml

from scripts._common import REPO_ROOT, TABLES
from scripts._real import REAL_SAMPLE, _quarters


def _window_row(df: pd.DataFrame, role: str, window: str,
                quarters: list[str]) -> dict:
    sub = df[df["vintage"].isin(quarters)]
    resolved = sub.dropna(subset=["y"])
    n_events = int((resolved["y"] == 1).sum())
    n_nonevents = int((resolved["y"] == 0).sum())
    n_resolved = len(resolved)
    return {
        "role": role,
        "window": window,
        "n": n_resolved,
        "n_events": n_events,
        "n_nonevents": n_nonevents,
        "default_rate": round(n_events / n_resolved, 6) if n_resolved else 0.0,
        "n_unresolved": len(sub) - n_resolved,
    }


def main() -> None:
    splits = yaml.safe_load(
        (REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8")
    )["splits"]
    df = pd.read_parquet(REAL_SAMPLE, columns=["vintage", "y"])

    rows = [
        _window_row(df, "train", f"{splits['train']['start']}-"
                     f"{splits['train']['end']}",
                     _quarters(splits["train"]["start"],
                               splits["train"]["end"])),
        _window_row(df, "validation", f"{splits['validation']['start']}-"
                     f"{splits['validation']['end']}",
                     _quarters(splits["validation"]["start"],
                               splits["validation"]["end"])),
    ]
    for quarter in _quarters(splits["test"]["start"],
                             splits["test"]["end"]):
        rows.append(_window_row(df, "test", quarter, [quarter]))

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "real_cohorts.csv"
    out.to_csv(path, index=False)

    test = out[out.role == "test"]
    print(out.to_string(index=False))
    print(f"test cohorts {len(test)} | rows {int(test.n.sum()):,} | "
          f"events {int(test.n_events.sum()):,} | "
          f"unresolved dropped {int(out.n_unresolved.sum()):,}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
