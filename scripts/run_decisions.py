"""Dry-run the decision layer end to end on the plumbing sample.

Wires T9.1/T9.2/T9.5 machinery: base decision codes come from the none
arm under the three-band policy read from configs/decisions.yaml
(break-even ratios and refer-band half-widths as configured), then each
recalibration arm is compared against the base on identical loans:
flip rates, directional flows, stylised delta-utility, and the
matched-null distribution of total delta-utility. The full ratio x
half-width sensitivity grid carries flip/flow/utility metrics; the null
distribution is attached to the primary policy pair (the real run
reproduces it at full scale post-freeze).

Outputs: reports/tables/decision_flip_dryrun.csv
Use    : python -m uv run python -m scripts.run_decisions
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from scripts._common import (
    REPO_ROOT,
    TABLES,
    check_test_window_guard,
    load_configs,
    load_plumbing_split,
)
from src.calibration.recalibration import KINDS, fit_recalibrator
from src.decisions.flips import (
    flip_delta_utility,
    flip_flows,
    matched_null_total_delta_utility,
)
from src.decisions.policy import breakeven_threshold, threshold_policy
from src.models.harness import FAMILIES, make_estimator, make_preprocessor


def main() -> None:
    dry = load_configs("dryrun")["dryrun"]
    decisions = load_configs("decisions")["decisions"]
    splits = load_configs("splits")["splits"]
    print(check_test_window_guard(splits))

    split = load_plumbing_split()
    split.assert_disjoint()

    primary_ratio = float(decisions["breakeven"]["primary"])
    h_primary = float(decisions["refer_band_half_width"])
    ratios = sorted({float(r) for r in decisions["breakeven"]["loss_gain_ratios"]})
    half_widths = sorted(
        {float(x) for x in decisions["sensitivity_half_widths"]} | {h_primary}
    )
    null_draws = int(decisions["null_draws"])

    rows = []
    for family in FAMILIES:
        params = dry["families"][family]
        seed = dry["seed"]
        t0 = time.perf_counter()
        pre = make_preprocessor(scale=True).fit(split.X_train)
        est = make_estimator(family, params, seed).fit(pre.transform(split.X_train), split.y_train)
        p_train = est.predict_proba(pre.transform(split.X_train))[:, 1]
        p_val = est.predict_proba(pre.transform(split.X_val))[:, 1]
        seconds = time.perf_counter() - t0

        arms = {arm: fit_recalibrator(arm, split.y_train, p_train) for arm in KINDS}
        p_base = arms["none"].apply(p_val)

        for arm, rec in arms.items():
            if arm == "none":
                continue
            p_after = rec.apply(p_val)
            for ratio in ratios:
                for h in half_widths:
                    p_star = breakeven_threshold(ratio)
                    t_low, t_high = p_star - h, p_star + h
                    codes_base = threshold_policy(p_base, t_low=t_low, t_high=t_high)
                    codes_after = threshold_policy(p_after, t_low=t_low, t_high=t_high)
                    flows = flip_flows(codes_base, codes_after)
                    du = flip_delta_utility(split.y_val, codes_base, codes_after, loss_gain_ratio=ratio)
                    for metric, value in [
                        ("t_low", t_low),
                        ("t_high", t_high),
                        ("flip_rate", flows["n_flips"] / len(split.y_val)),
                        ("toward_decline", flows["toward_decline"]),
                        ("toward_approve", flows["toward_approve"]),
                        ("total_delta_utility", du["total"]),
                    ]:
                        rows.append(
                            {
                                "family": family,
                                "after_arm": arm,
                                "ratio": ratio,
                                "half_width": h,
                                "metric": metric,
                                "value": round(float(value), 6),
                            }
                        )
                    if ratio == primary_ratio and h == h_primary:
                        null = matched_null_total_delta_utility(
                            split.y_val, p_after, codes_base,
                            t_low=t_low, t_high=t_high, draws=null_draws,
                            seed=seed, loss_gain_ratio=ratio,
                        )
                        for metric, value in [
                            ("null_mean", float(null.mean())),
                            ("null_sd", float(null.std())),
                            ("null_q025", float(np.percentile(null, 2.5))),
                            ("null_q975", float(np.percentile(null, 97.5))),
                        ]:
                            rows.append(
                                {
                                    "family": family,
                                    "after_arm": arm,
                                    "ratio": ratio,
                                    "half_width": h,
                                    "metric": metric,
                                    "value": round(value, 6),
                                }
                            )
        print(f"{family:>20} | {seconds:.1f}s fit | 3 arms x "
              f"{len(ratios) * len(half_widths)} policy cells")

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "decision_flip_dryrun.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
