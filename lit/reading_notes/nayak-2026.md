# Reading note — Nayak (2026), arXiv:2603.06733

**Full bibliographic record (verified on arXiv abstract page, 2 Nov 2026):**
Srikumar Nayak, *Calibrated Credit Intelligence: Shift-Robust and Fair Risk
Scoring with Bayesian Uncertainty and Gradient Boosting*, arXiv:2603.06733,
submitted 6 Mar 2026, q-fin.RM (cross-listed cs.AI, cs.LG), 7 pp, CC BY-NC-ND
4.0. NOT peer-reviewed at time of writing (venue check later).

**Read level:** FULL TEXT, 3 Nov 2026 (arXiv HTML v1). UNKNOWN sections
below resolved; remaining gaps flagged.

## Full-text findings (3 Nov 2026)
- **Dataset: Home Credit "Credit Risk Model Stability" Kaggle benchmark**
  (base table + many auxiliary tables, one row per application, WEEK_NUM +
  date_decision + binary default). **Not LendingClub.** Chronological
  week-based train/val/test split (Eq. 5); ALL preprocessing fitted
  train-only (median imputation + missingness indicators `m_ij` appended as
  inputs; frequency encoding for categoricals; train-only standardisation).
- Method (CCI): (i) variational BNN scorer (ELBO, S-sample predictive mean,
  epistemic u_epi = Var across samples + aleatoric u_ale); (ii)
  fairness-constrained GBDT objective (loss + λ·max(0, Δ_fairness − Δ_max));
  (iii) convex fusion s̃ = β·μ_gbdt + (1−β)·μ_bnn with β selected on
  validation via a drift test; (iv) **temperature scaling fitted by NLL on
  validation weeks** → calibrated ŝ.
- Metrics/baselines: AUC-ROC/AUC-PR/Recall@1%FPR; Brier + ECE +
  reliability diagram; early→late AUC-PR drop 0.017 (vs LightGBM 0.034,
  Fair-GBDT 0.030); ΔDP 0.046 / ΔEO 0.037 (vs Fair-GBDT 0.052/0.041).
  Baselines: LR, XGBoost, LightGBM, CatBoost, TabNet, BNN, Fair-GBDT.
  Results mean ± std over seeded repeats.
- Explanations: single-figure SHAP attribution on the GBDT ("audit trail"),
  **no stability or fidelity measurement of explanations**.
- No loan-level decision-flip analysis; "stable decision thresholds" is
  framed at the performance-metric level. No label-delay mechanism (future
  work lists online recalibration / periodic retraining as not yet done).
- Fairness groups used for ΔDP/ΔEO are unspecified in the extracted text
  (sensitive attribute identity uncertain).
- Limitations stated: one dataset, one label, more fairness definitions
  planned — recorded verbatim in conclusions.

## What it measures
| Axis | Covered? | Notes |
|---|---|---|
| Calibration (Brier/ECE) | Yes | temperature scaling on validation weeks (Eq. 16); reliability diagram in Fig. 3 |
| Explanation stability | No | absent from abstract |
| Fidelity vs reliance | No | absent |
| Decision flips | No | confirmed absent: "stable thresholds" is a performance-level claim, not loan-level flip analysis |
| Label-delay recalibration | No | confirmed absent: temperature scaling on validation weeks only; online recalibration is future work |
| AUC-vs-reliability relationship | No | reports trade-offs, no cross-config correlation test |

## Positioning for our Gate 1 comparison
A *method* paper (make scoring calibrated + fair under shift), not an *audit*.
Overlaps with our recalibration arm (post-hoc calibration under temporal
shift) on Home Credit rather than LendingClub; does not test whether AUC
predicts reliability, nor explanations, nor loan-level decision flips. Risk:
their "shift-aware fusion + calibration" presses on our RQ1c novelty corner —
we must (a) confirm what exactly their recalibration uses (delay semantics),
(b) keep our claim narrow (joint audit + label-delay arms + AUC-reliability).

## Limitations (full text)
Short preprint, single benchmark (Home Credit Stability), sensitive-attribute
identity unspecified in the text, no explanations audit, no decision-flip
analysis, peer-review status unknown.
