"""Resolve design-matrix columns into PROPOSED feature groups (T5.4 draft).

Groups live in configs/feature_groups.yaml; the resolver turns them into a
partition of design-matrix columns for the grouped importance / grouped
stability machinery:

- a design column named exactly like a source feature takes that group;
- an engineered column listed in `engineered_inherit` takes its source's
  group;
- a one-hot column inheriting by prefix (`<source>_<level>`, `<source>_na`)
  takes the LONGEST matching source prefix's group;
- anything unassigned is an error — the grouping must be a full partition
  (T8.5 iron rule), never silently partial.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GROUPS_FILE = REPO_ROOT / "configs" / "feature_groups.yaml"


def load_groups() -> dict:
    with GROUPS_FILE.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if set(doc) != {"feature_groups", "engineered_inherit", "onehot_prefix_inherit"}:
        raise ValueError(
            "feature_groups.yaml must carry exactly: feature_groups, "
            "engineered_inherit, onehot_prefix_inherit"
        )
    sources = [f for members in doc["feature_groups"].values() for f in members]
    if len(sources) != len(set(sources)):
        raise ValueError("a source feature appears in more than one group")
    return doc


def resolve_column_groups(columns: list[str]) -> dict[str, str]:
    """Map every design-matrix column to its group; full partition required."""
    doc = load_groups()
    source_to_group: dict[str, str] = {}
    for group, members in doc["feature_groups"].items():
        for feature in members:
            source_to_group[feature] = group

    inherited: dict[str, str] = dict(doc["engineered_inherit"])

    out: dict[str, str] = {}
    for column in columns:
        if column in source_to_group:
            out[column] = source_to_group[column]
        elif column in inherited:
            out[column] = source_to_group[inherited[column]]
        else:
            matches = [
                (feature, group)
                for feature, group in source_to_group.items()
                if column.startswith(f"{feature}_")
            ]
            if not matches:
                raise ValueError(
                    f"design column {column!r} has no feature group "
                    "(the grouping must be a full partition)"
                )
            feature, group = max(matches, key=lambda m: len(m[0]))
            out[column] = group
    if set(out) != set(columns):
        raise RuntimeError("resolver dropped columns")  # defensive
    return out


def grouped_partition(columns: list[str]) -> dict[str, list[str]]:
    """Column partition as {group: [design columns]}, every column inside."""
    groups = resolve_column_groups(columns)
    partition: dict[str, list[str]] = {}
    for column, group in groups.items():
        partition.setdefault(group, []).append(column)
    flat = [c for members in partition.values() for c in members]
    if sorted(flat) != sorted(columns):
        raise RuntimeError("partition is not a cover of the design columns")
    return partition
