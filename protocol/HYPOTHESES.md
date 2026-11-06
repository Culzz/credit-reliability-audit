# protocol/HYPOTHESES.md — hypothesis register

> **Status: DRAFT for T5.2 (Analyst pre-registers, PI confirms at freeze).**
> Every number below is either (a) already a configured quantity
> (`configs/*.yaml`, cited inline) or (b) a PROPOSED default stated with its
> rationale, to be confirmed or edited BEFORE the freeze — never after
> results are seen. Test statistics follow PLAN.md Phase 7/8/9 definitions
> and `src/*` implementations as committed on main.

# Decision rules common to all hypotheses

- **Alpha:** 0.05, two-sided, on headline cohort aggregates. **CIs:**
  percentile bootstrap, B = 2,000 (`configs/statistics.yaml bootstrap.B_headline`;
  B = 1,000 default per non-headline cell). Paired designs share one resample
  design between the compared arms (`src/statistics/inference.py`).
- **Multiplicity:** Holm step-down over the confirmatory family ONLY
  (8 confirmatory hypotheses below); exploratory results are reported with
  raw CIs and labelled exploratory in the paper.
- **Unit of analysis:** per CONTEXT/PLAN — configuration (family, config_id,
  seed) × evaluation cohort, aggregated to cohort level by averaging within
  cohort (`configs/statistics.yaml rq4.unit_of_analysis`).
- **Equivalence region for H4 (pre-set):** Spearman |ρ| < 0.3
  (`configs/statistics.yaml rq4.equivalence_region_spearman`). Region tested
  via the bootstrap CI of ρ against the region bounds (negligible if CI
  inside, meaningful if outside, inconclusive otherwise).

# Confirmatory family (Holm-corrected)

| ID | Hypothesis | Test statistic; direction justified? | Threshold (pre-set/PROPOSED) | Practical-effect anchor | Statistic source |
|---|---|---|---|---|---|
| H1a | Calibration-in-the-large deteriorates on shifted future cohorts relative to validation | Δ = \|CITL\| cohort − \|CITL\| validation, averaged per config over eval cohorts; direction: Δ > 0 (justified: iid-calibration-does-not-carry precedent, Ovadia 2019) | bootstrap CI of Δ excludes 0 | \|ΔCITL\| ≥ 0.02 = the refer-band half-width (`decisions.yaml`); intercept drift of this size moves operating decisions | `audit.citl_and_slope` per cohort |
| H1c | Recalibration on eligible historical labels recovers only part of the loss; recovery depends on lag | PROPOSED operationalization (CONTEXT gives the direction only): recovery = 1 − (Brier_shifted_after − Brier_oracle)/(Brier_shifted_before − Brier_oracle) per arm×lag, _oracle = same-arm recalibration fitted on the evaluation cohort itself_ | CI excluding 1 (partial recovery); lag-dependence via rank ordering of recovery across allowed lags | recovery ≥ 0.5 pre-registered as "substantial" (descriptive, not a test) | `recalibration` arms on T4.5-FEASIBLE cells |
| H2c | Temporal explanation differences exceed the seed/bootstrap noise floor for at least some feature groups | per group: stability metric vs cohort-shifted comparison measured against the seed/bootstrap noise-floor distribution (T8.2), compared at its 2.5th percentile | temporal similarity below floor-2.5% for the group counts as exceeding noise | report the share of feature groups exceeding the floor (descriptive) | `explain.stability` vs `compute-pilot`-capped T8.2 runs |
| H3a | Retraining/recalibration produce measurable decision flips | flip rate with bootstrap CI; direction: > noise floor | PROPOSED threshold: flip rate above the seed-vs-seed noise-floor 95th percentile (T9.2d ground truth), not merely > 0 | practical anchor: flips ≥ once per 1,000 scored loans at the policy cell | `decisions.flips` vs seed-floor |
| RQ4.1 | AUC↔ECE association weaker than AUC↔AUC | cohort-avg Spearman ρ(AUC_t, ECE_t+1) vs ρ(AUC_t, AUC_t+1); paired bootstrap delta ρ | delta CI excludes 0 (two-sided) | plus descriptive magnitude of ρ within the pre-set region | `inference.paired_bootstrap_delta` |
| RQ4.2 | AUC↔\|slope−1\| association limited | ρ(AUC_t, \|slope−1\|_{t+1}) vs ρ(AUC_t, AUC_t+1) | delta CI excludes 0 | same as above | same |
| RQ4.3 | AUC↔explanation instability limited | ρ(AUC_t, shap-stability_{t+1}) vs ρ(AUC_t, AUC_t+1) | delta CI excludes 0 | same as above | same |
| RQ4.4 | AUC↔flip rate limited | ρ(AUC_t, flip-rate_{t+1}) vs ρ(AUC_t, AUC_t+1) | delta CI excludes 0 | same as above | same |

# Exploratory hypotheses (raw CIs; labelled exploratory in the paper)

| ID | Hypothesis | Statistic | Note |
|---|---|---|---|
| H1b | Calibration slope changes, by a different magnitude than CITL | slope Δ vs validation, per cohort; magnitude contrast pre-reported as Δslope vs ΔCITL on their own scales (never normalised together) | exploratory: no literature direction for lending data |
| H1d | Isotonic is more vulnerable to small calibration samples than Platt or intercept-only | paired bootstrap Δ Brier on subsampled calibration sets (sizes from ACTUAL counts of the T4.5 cells); PROPOSED size grid: {200/200-rule minimal, 500, 1k, 2k, 5k, full}, completeness cut at feasibility | H1d is exploratory in this project (contextualised by TableShift/tabular precedent) |
| H2a | Fixed-model SHAP summaries vary across cohorts | Axis A stability battery (`explain.stability`) across all test cohorts, one frozen model per family | descriptive battery |
| H2b | Retrained models drift in reliance | Axis B: expanding-window model versions scored on the common evaluation population; attribution profiles compared | descriptive battery |
| H2d | SHAP vs independent reliance measures disagree for some correlated groups | Kendall τ between group |shap| and grouped-permutation importance; disagreement claimed when τ < 0.2 AND per-group gap beyond noise floor | permutation R = 20 (configured) |
| H3b | Flips concentrate near thresholds | distribution of distance-to-nearest-threshold for flipped vs non-flipped pairs; two-sample comparison (rank-based) + median ratio | closeness defined by the same policy cell as the flip |
| H3c | Flips carry heterogeneous outcome associations | default rate by flip direction vs p\*; per-flip ΔU and totals vs the matched-null distribution (1,000 draws, `decisions.yaml`) | refer moves reported separately (A7 convention) |
| H4a | AUC predicts future AUC more consistently than non-discrimination outcomes | pairwise ρ comparison battery (same as RQ4.x) + explicit rank-reversal count | confirmatory power reserved to the four RQ4.x rows to avoid double-testing |

# Noise floors (pre-registered inputs to H2c/H3a thresholds)

- T8.2 Axis C: 20 seeds × 30 bootstrap resamples on ONE fixed cohort for the
  deep subset (5 configs/family, `configs/models.yaml deep_subset`);
  noise-floor distributions stored per family and per stability metric.
- T9.2(d) seed-vs-seed flip floor: same design, seed-paired decision tracks;
  floor = distribution of flip rates across seed pairs, upper 95th pct used
  as the H3a reference.
- Compute caps from T4.6 enforce feasibility of the whole design
  (`reports/tables/compute_pilot.csv`).

# Ordering and freeze protocol

1. PI edits/tolerates every PROPOSED default above at Gate 5 (figures may be
   struck or re-anchored ONLY with a logged reason in `DECISIONS.md`).
2. On freeze the table goes verbatim into the frozen protocol; any later
   change to a threshold is a logged deviation (`protocol/DEVIATIONS.md`).
3. Confirmatory p-values arrive only from the frozen pipeline
   (`scripts/run_*` on frozen cohorts); Holm is applied once, at the end.
