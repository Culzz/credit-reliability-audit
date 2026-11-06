"""Deterministic search-space sampler (PLAN.md T5.5).

Reads the PROPOSED spaces in configs/search_spaces.yaml and turns
(family, config index, base seed) into one concrete parameter dict —
purely from the distribution spec, never from data or results. The same
triple always yields the same dict (the pre-registration property: the
sweep that will run after the freeze is already fixed by the spaces and
the seeds, not by anything anyone looks at).

Enforcement: every stochastic row/column parameter known to weaken the
seed design (subsample, colsample_*, bagging_fraction, feature_fraction)
must be STRICTLY below 1.0 whenever it is sampled, so the deep subset's
20 seeds produce genuinely different models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACES_FILE = REPO_ROOT / "configs" / "search_spaces.yaml"

STOCHASTIC_KEYS = {
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "bagging_fraction",
    "feature_fraction",
}


def load_spaces() -> dict:
    with SPACES_FILE.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if set(doc) != {"search_spaces"}:
        raise ValueError("search_spaces.yaml must carry exactly the top key 'search_spaces'")
    return doc["search_spaces"]


def _draw(spec: dict, rng: np.random.Generator):
    dist = spec.get("dist")
    if dist == "uniform":
        value = float(rng.uniform(spec["low"], spec["high"]))
    elif dist == "log_uniform":
        low, high = float(spec["low"]), float(spec["high"])
        if not (0.0 < low < high):
            raise ValueError(f"log_uniform needs 0 < low < high, got {low}..{high}")
        value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    elif dist == "randint":
        value = int(rng.integers(spec["low"], spec["high"] + 1))
    elif dist == "log_uniform_int":
        low, high = int(spec["low"]), int(spec["high"])
        if not (0 < low <= high):
            raise ValueError(f"log_uniform_int needs 0 < low <= high, got {low}..{high}")
        value = round(float(np.exp(rng.uniform(np.log(low), np.log(high)))))
    elif dist == "choice":
        value = spec["options"][int(rng.integers(0, len(spec["options"])))]
    elif dist == "fixed":
        value = spec["value"]
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return value


def _rng_for(family: str, config_index: int, base_seed: int) -> np.random.Generator:
    # seed sequence from a list keeps draws stable across numpy versions
    family_slot = list(load_spaces()).index(family)
    return np.random.default_rng([int(base_seed), family_slot, int(config_index)])


def sample_config(family: str, config_index: int, base_seed: int) -> dict:
    spaces = load_spaces()
    if family not in spaces:
        raise ValueError(f"unknown family {family!r}; sanctioned: {sorted(spaces)}")
    spec = spaces[family]
    rng = _rng_for(family, config_index, base_seed)
    params: dict = {}
    params.update({k: v for k, v in spec.get("fixed", {}).items()})
    for name, dist_spec in spec.get("sample", {}).items():
        value = _draw(dist_spec, rng)
        if name in STOCHASTIC_KEYS and not (value is None or value < 1.0):
            raise ValueError(
                f"stochastic parameter {name} must be strictly < 1 so seeds "
                f"change predictions; drew {value}"
            )
        params[name] = value
    return params


def sample_all_configs(family: str, n_configs: int, base_seed: int) -> list[dict]:
    if n_configs < 1:
        raise ValueError("n_configs must be >= 1")
    return [sample_config(family, i, base_seed) for i in range(n_configs)]


def assert_seed_informative(family: str, base_seed: int, n_configs: int) -> None:
    """The T5.5 invariant as a runtime check: with seeds varying, tree
    boosters with stochastic params must yield differing params per seed.
    (Full model-variation is argued in the GATE4 compute note; here we pin
    the parameter-level side.)"""
    spaces = load_spaces()
    if family not in spaces:
        raise ValueError(f"unknown family {family!r}")
    sampled_keys = [k for k in spaces[family].get("sample", {}) if k in STOCHASTIC_KEYS]
    if not sampled_keys:
        return  # family has no stochastic params; seeds vary ensembles only
    first = sample_config(family, 0, base_seed)
    second = sample_config(family, 0, base_seed + 1)
    if not any(first[k] != second[k] for k in first):
        raise ValueError("seeds produced identical parameter dicts despite stochastic params")
