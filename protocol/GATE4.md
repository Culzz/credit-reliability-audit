# GATE 4 — Shift adequacy memo (PROPOSED; PI ratifies, then the protocol freezes)

Status: PROPOSED. Written 3 Nov 2026 from the descriptive Phase 4 tables
only. No outcome-model results were consulted in choosing the windows: the
only model-adjacent diagnostic cited is the features-only adversarial
validation mandated by PLAN.md T4.3.

## 1. Evidence sources (all committed)

- `reports/tables/cohort_feasibility.csv` + `reports/figures/cohort_volume_by_vintage.png`, `default_rate_by_vintage.png` (T4.1)
- `reports/tables/shift_measures_numeric.csv`, `shift_measures_categorical.csv`, `default_rate_shift.csv` + `reports/figures/psi_heatmap.png` (T4.2)
- `reports/tables/adversarial_validation.csv` + `reports/figures/adversarial_roc.png` (T4.3)
- `reports/tables/concept_drift_strata.csv` + `reports/figures/concept_drift_strata.png` (T4.4)
- `reports/tables/label_delay_feasibility.csv` (T4.5)
- `reports/tables/compute_pilot.csv` (T4.6)
- Reference window for every shift measure: PROPOSED training window
  2012Q1–2013Q2 (84,594 loans). Statistics: `src/statistics/shift.py`, 18
  synthetic-data tests green (`tests/test_shift_stats.py`).

## 2. Cohort feasibility (T4.1)

- Fully mature 36-month vintages run 2007Q2–2015Q4.
- Every mature cohort from 2009Q4 onward passes the ≥200 events AND ≥200
  non-events rule; the micro-vintages 2008Q2–2009Q3 fail the events rule and
  are excluded from evaluation use.
- 2016+ vintages are partially unresolved (up to 85%) and are not mature; the
  evaluation side stops at 2015Q4.

**PROPOSED choice:** test cohorts = the 8 quarterly vintages 2014Q1–2015Q4
(≥6 satisfied). Each passes both min-count rules with wide margins (events
4,384–13,138; non-events 29,690–75,531).

## 3. Shift adequacy of the chosen window pair (T4.2 + T4.3)

Numeric features vs the training window, restricted to candidate test cohorts:

- Excluding the three event-absence fields (`mths_since_last_*`), the
  max per-vintage PSI is ≤ 0.17 (top mover: `pub_rec`). PSI values are cited
  descriptively; no threshold is treated as a law.
- Their (larger) PSI values, 0.63–1.25, are dominated by prevalence decay of
  the underlying record/delinquency events between train and test windows,
  consistent with the D-5(b) pending decision on those fields' semantics.
- Strongest categorical movers in the window: `initial_list_status`
  (JS 0.199 → 0.489 by 2015Q4; max proportion change 0.71 — a listing-channel
  switch), `verification_status` (JS ≤ 0.238), `purpose` (JS ≤ 0.099).
- Features-only adversarial validation AUC: 0.739 (2014Q1) → 0.882 (2015Q4);
  beyond 2016 it escalates to 0.95 as the late-adopted bureau fields
  (num_*, mo_sin_*, bc_*, …) enter. Evidence: shift inside the candidate
  window is real but moderate; post-2016 cohorts are schema-confounded and
  stay outside the main evaluation range.
- Earlier micro-vintage artifact check (D-5(a)): 2007Q2 alone shows extreme
  categorical JS (`purpose` 0.79; `verification_status` 0.60) — supporting
  dropping the 2007Q2 vintage from modelling cohorts, as proposed in D-5(a).

## 4. Concept-drift diagnostics (T4.4)

Within training-window terciles of `fico_range_low`, `dti`, `loan_amnt`,
resolved default-rate ranges across 2007Q2–2015Q4 reach 0.24–0.57 per
stratum (largest: `loan_amnt` top tercile, spread 0.563). Conditional
default rates move under fixed strata — simple covariate shift does not
explain everything; recalibration and retraining arms must both be run.

## 5. Label-delay feasibility (T4.5, blueprint-equivalent convention)

All 49 cells (8 test cohorts × lags {0,6,12,18,24,36} + static validation
arm) are FEASIBLE under the 200/200 rule: realistic-lag (a=36) recalibration
windows carry 973–3,443 events. The limitation is deployment realism (arms
are tagged oracle / optimistic / realistic in the table), not sample size.

## 6. Compute caps set from the pilot (T4.6)

Measured on the plumbing sample (3% of 2012–2013, seed 20180101):

| task | sizing finding |
|---|---|
| model fit (LR/RF/XGB/LGBM, ~2.5k dev rows) | 0.1–2.5 s per fit |
| SHAP TreeExplainer | ≈ 12.9 s per 10,000 explained rows with 500-row background → cap SHAP explanation samples at ≤ 10,000 rows per configuration in the main run |
| paired AUC bootstrap | B=200 in 1.6 s → headline B=2000 ≈ 16 s per metric per cell; keep the grid within ~2,000 fit-cells |
| Platt / isotonic recalibration | ≤ 0.4 s, negligible |

These caps enter the protocol text at freeze; they were fixed here first,
not after seeing result sensitivity.

## 7. Gate decision requested from the PI

1. Ratify test cohorts 2014Q1–2015Q4 (freeze `configs/splits.yaml` test
   window) — or amend with a descriptive-only rationale.
2. Decide D-3 (legacy-status merge), D-4 (sensitivity-parked columns),
   D-5 (2008Q1 start; months-since-event fields) before freeze.
3. Accept the T4.6 compute caps as written.
