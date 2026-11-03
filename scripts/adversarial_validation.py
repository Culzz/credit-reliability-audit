"""Adversarial validation, features only (PLAN.md T4.3).

Question: how distinguishable is each later vintage from the PROPOSED
training window by features alone (no ids, no dates, no target, no leakage)?

A one-vs-cohort logistic regression is trained on the training window
(label = 1) versus each evaluation vintage (label = 0 for that pairing),
with time-aware held-out evaluation: the model is fit only on training-window
rows plus the evaluation vintage it is paired against; imputation and scaling
are fit on that training slice only. Reported AUC values describe feature
distinguishability — larger AUC means stronger distributional shift.

Descriptive diagnostics for the GATE4 memo. No window is chosen here.

Inputs : data/interim/mature_36m_features.parquet
Outputs: reports/tables/adversarial_validation.csv
         reports/figures/adversarial_roc.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "reports" / "tables"
FIGURES = REPO_ROOT / "reports" / "figures"
CACHE = REPO_ROOT / "data" / "interim" / "mature_36m_features.parquet"
META = ["id", "vintage", "y"]


def main() -> None:
    frame = pd.read_parquet(CACHE)
    frame["vintage"] = frame["vintage"].astype(str)
    features = [c for c in frame.columns if c not in META]

    splits = yaml.safe_load((REPO_ROOT / "configs" / "splits.yaml").read_text(encoding="utf-8"))["splits"]
    ref_start, ref_end = splits["train"]["start"], splits["train"]["end"]
    ref = frame.loc[frame["vintage"].between(ref_start, ref_end)]
    assert len(ref) > 0, f"no rows in training window {ref_start}..{ref_end}"

    eval_vintages = sorted(v for v in frame["vintage"].unique() if v > ref_end)
    print(f"reference {ref_start}..{ref_end}: {len(ref):,} rows | eval vintages: {len(eval_vintages)}")

    rows = []
    curves = {}
    for v in eval_vintages:
        cur = frame.loc[frame["vintage"] == v]
        x = pd.concat([ref[features], cur[features]], axis=0, ignore_index=True)
        # pandas dtype objects can't pass the imputer; impute on float64 views
        x = x.astype("float32")
        label = np.concatenate([np.ones(len(ref)), np.zeros(len(cur))])
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        )
        model.fit(x, label)
        fpr, tpr, _ = roc_curve(label, model.predict_proba(x)[:, 1])
        score = sk_auc(fpr, tpr)
        rows.append({"eval_cohort": v, "n_ref": len(ref), "n_eval": len(cur), "auc": score})
        curves[v] = (fpr, tpr)
        print(f"  {v}: AUC {score:.4f} (n={len(cur):,})")

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "adversarial_validation.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    step = max(1, len(curves) // 10)
    for i, (v, (fpr, tpr)) in enumerate(curves.items()):
        if i % step == 0:
            ax.plot(fpr, tpr, lw=1, label=f"{v} AUC={sk_auc(fpr, tpr):.3f}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Adversarial validation: training window vs each later vintage")
    ax.legend(fontsize=7, ncol=1, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "adversarial_roc.png", dpi=150)

    print(f"written: {TABLES / 'adversarial_validation.csv'} and figures/adversarial_roc.png")


if __name__ == "__main__":
    main()
