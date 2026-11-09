"""Regression smoke: run_real_statistics on synthetic battery tables.

Every join, pivot, bootstrap and Holm step in the confirmatory
statistics runner must execute end-to-end; B is monkeypatched small so
the test stays sub-minute. Real verdicts come only from the real tables
(the runner refuses to run when they are missing).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import scripts.run_real_statistics as rs

FAMS = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
CFGS = [0, 7, 15, 22, 29]
METRICS = ["roc_auc", "brier", "log_loss", "ece_b15q", "citl", "slope",
           "ae_ratio", "mean_p"]
ARMS = ["intercept", "platt", "isotonic"]
ARM_METRICS = ["brier_before", "brier_after", "brier_oracle", "recovery",
               "citl_after", "slope_after", "ece_after", "flip_rate_vs_none"]
STAB_METRICS = ["grouped_cosine", "grouped_spearman", "jaccard_top10",
                "spearman", "kendall", "cosine", "wasserstein_mean"]
GROUPS = ["borrower_capacity", "credit_history_tenure",
          "delinquency_public_record", "credit_score", "utilization",
          "application_context"]


@pytest.fixture()
def synthetic_tables(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    cohs = list(rs.CUTOFF_COHORTS)

    agg = []
    for slot, fam in enumerate(FAMS):
        for cfg in CFGS:
            base_auc = 0.62 + 0.25 * rng.random() - 0.001 * slot
            for s in range(6):
                for w in ["train", "val"] + cohs:
                    t = cohs.index(w) if w in cohs else 0
                    auc = base_auc - 0.02 * t / 9 + 0.004 * rng.standard_normal()
                    ece = max(0.01, 0.10 - 0.5 * (auc - 0.55))
                    citl = -0.01 - 0.05 * t / 9 - 0.3 * (auc - 0.55)
                    slope = 1.0 - 2.2 * abs(citl) * 0.2
                    vals = {"roc_auc": auc, "brier": 0.14, "log_loss": 0.35,
                            "ece_b15q": ece, "citl": citl, "slope": slope,
                            "ae_ratio": 0.95, "mean_p": 0.17}
                    for m, v in vals.items():
                        agg.append({"family": fam, "config_index": cfg,
                                    "kind": "seed", "refit_id": s,
                                    "window": w, "metric": m,
                                    "value": round(v, 6)})
    deep_agg = pd.DataFrame(agg)

    arms = []
    for fam in FAMS:
        for cfg in CFGS:
            for arm in ARMS:
                for w in ["val"] + cohs:
                    t = cohs.index(w) if w in cohs else 0
                    for m in ARM_METRICS:
                        v = {"brier_before": 0.13, "brier_after": 0.12,
                             "brier_oracle": 0.115,
                             "recovery": 0.95 - 0.04 * t / 9,
                             "citl_after": -0.01, "slope_after": 0.98,
                             "ece_after": 0.02, "flip_rate_vs_none": 0.0025}[m]
                        arms.append({"family": fam, "config_index": cfg,
                                     "kind": "seed", "refit_id": 0,
                                     "arm": arm, "window": w, "metric": m,
                                     "value": round(v, 6)})
    deep_arms = pd.DataFrame(arms)

    flips = []
    for fam in FAMS:
        for cfg in CFGS:
            for i, j in itertools.combinations(range(4), 2):
                for w in ["val"] + cohs:
                    flips.append({"family": fam, "config_index": cfg,
                                  "kind": "seed", "i": i, "j": j,
                                  "window": w, "n": 1000, "n_flips": 0,
                                  "flip_rate": 0.0004 + 0.0008 * rng.random(),
                                  "toward_decline": 0})
    flip_pairs = pd.DataFrame(flips)

    stab = []
    for fam in FAMS:
        for cfg in CFGS:
            for kind in ("seed", "boot"):
                for i, j in itertools.combinations(range(4), 2):
                    for m in STAB_METRICS:
                        stab.append({"family": fam, "config_index": cfg,
                                     "kind": kind, "i": i, "j": j,
                                     "metric": m,
                                     "value": round(0.99, 6)})
    stability_pairs = pd.DataFrame(stab)

    gp = []
    for fam in FAMS:
        for cfg in CFGS:
            for kind in ("seed", "boot"):
                for i, j in itertools.combinations(range(4), 2):
                    for g in GROUPS:
                        gp.append({"family": fam, "config_index": cfg,
                                   "kind": kind, "i": i, "j": j, "group": g,
                                   "group_cosine": 0.97,
                                   "group_spearman": 0.95,
                                   "group_mean_gap": 0.002})
            for i, j in itertools.combinations(["val"] + cohs, 2):
                for g in GROUPS:
                    # credit_score temporal similarity deliberately low:
                    # the H2c exceed cell the share statistic must see
                    low = g == "credit_score"
                    gp.append({"family": fam, "config_index": cfg,
                               "kind": "cohort", "i": i, "j": j, "group": g,
                               "group_cosine": 0.5 if low else 0.97,
                               "group_spearman": 0.3 if low else 0.95,
                               "group_mean_gap": 0.05 if low else 0.002})
    group_pairs = pd.DataFrame(gp)

    tp = []
    for fam in FAMS:
        for cfg in CFGS:
            for i, j in itertools.combinations(cohs, 2):
                is_adjacent = rs.ADJACENT_PAIRS and (i, j) in rs.ADJACENT_PAIRS
                for m in STAB_METRICS:
                    low = is_adjacent and m == "jaccard_top10"
                    tp.append({"family": fam, "config_index": cfg,
                               "window_i": i, "window_j": j, "metric": m,
                               "value": 0.4 if low else 0.99})
    temporal_pairs = pd.DataFrame(tp)

    for name, df in [("deep_agg_real", deep_agg),
                     ("deep_arms_real", deep_arms),
                     ("flip_pairs_real", flip_pairs),
                     ("stability_pairs_real", stability_pairs),
                     ("temporal_pairs_real", temporal_pairs),
                     ("group_pairs_real", group_pairs)]:
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    monkeypatch.setattr(rs, "TABLES", tmp_path)
    monkeypatch.setattr(rs, "B_HEADLINE", 60)
    return tmp_path


def test_synthetic_battery_runs_and_holms(synthetic_tables):
    rs.main()
    out = pd.read_csv(synthetic_tables / "real_statistics.csv")
    confirmed = out[out.hypothesis.isin(rs.CONFIRMATORY)]
    assert len(confirmed) == 8
    # Holm must be populated for every confirmatory row with a verdict
    assert confirmed["p_holm"].notna().all()
    assert set(confirmed["verdict"]) <= {"reject_H0", "not_rejected"}
    # H2c: the synthetic credit_score cell sits below its floor q2.5, so
    # the share statistic must be strictly positive
    h2c = confirmed[confirmed.hypothesis == "H2c"].iloc[0]
    assert h2c["value"] > 0
    # floors table covers all families x the three dimensions
    floors = pd.read_csv(synthetic_tables / "real_noise_floors.csv")
    assert set(floors.dimension) == {"stability", "group", "flip"}
    assert floors.groupby("family").size().min() >= 7
    rq4 = pd.read_csv(synthetic_tables / "rq4_cells_real.csv")
    assert list(rq4.key) == list(rs.RQ_KEYS)
    rec = pd.read_csv(synthetic_tables / "recovery_cells_real.csv")
    assert set(rec.arm) == set(ARMS)
    assert rec.lag.is_monotonic_increasing is False  # cohort order rows
