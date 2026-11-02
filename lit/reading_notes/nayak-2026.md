# Reading note — Nayak (2026), arXiv:2603.06733

**Full bibliographic record (verified on arXiv abstract page, 2 Nov 2026):**
Srikumar Nayak, *Calibrated Credit Intelligence: Shift-Robust and Fair Risk
Scoring with Bayesian Uncertainty and Gradient Boosting*, arXiv:2603.06733,
submitted 6 Mar 2026, q-fin.RM (cross-listed cs.AI, cs.LG), 7 pp, CC BY-NC-ND
4.0. NOT peer-reviewed at time of writing (venue check later).

**Read level:** abstract only. Full read (methods + limitations) REQUIRED
before Gate 1 — sections marked UNKNOWN must come from the PDF itself.

## What it does (from abstract)
- Proposes *Calibrated Credit Intelligence (CCI)*, a deployment-oriented
  framework combining:
  1. a Bayesian neural risk scorer (epistemic uncertainty, less overconfidence);
  2. a fairness-constrained gradient boosting model;
  3. a shift-aware fusion strategy + post-hoc probability calibration to
     stabilise thresholds in later periods.
- Evaluated on the Home Credit *Credit Risk Model Stability* benchmark with a
  time-consistent split.
- Headline numbers: AUC-ROC 0.912, AUC-PR 0.438, Recall@1%FPR 0.509, Brier
  0.087, ECE 0.015; smaller AUC-PR drop early→late (0.017); demographic-parity
  gap 0.046, equal-opportunity gap 0.037.

## What it measures
| Axis | Covered? | Notes |
|---|---|---|
| Calibration (Brier/ECE) | Yes | model-level, single recalibrator? UNKNOWN which method |
| Explanation stability | No | absent from abstract |
| Fidelity vs reliance | No | absent |
| Decision flips | No | "threshold stability" ≠ loan-level flip analysis; confirm in full text |
| Label-delay recalibration | Partial? | "later periods calibration" — mechanism and delay semantics UNKNOWN; confirm |
| AUC-vs-reliability relationship | No | reports trade-offs, no cross-config correlation test |

## Positioning for our Gate 1 comparison
A *method* paper (make scoring calibrated + fair under shift), not an *audit*.
Overlaps with our recalibration arm (post-hoc calibration under temporal
shift) on Home Credit rather than LendingClub; does not test whether AUC
predicts reliability, nor explanations, nor loan-level decision flips. Risk:
their "shift-aware fusion + calibration" presses on our RQ1c novelty corner —
we must (a) confirm what exactly their recalibration uses (delay semantics),
(b) keep our claim narrow (joint audit + label-delay arms + AUC-reliability).

## Limitations abstractly visible
Short preprint (7 pp), single benchmark, race-correlated fairness proxies?
(protection status features UNKNOWN), peer-review status unknown.
