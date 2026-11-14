# protocol/DEVIATIONS.md — append-only deviations after protocol-v1

Format per PLAN.md §7.5:

```
### DEV-<n> <date>
Original decision: ...  Proposed change: ...  Reason: ...
Were test results already seen: YES/NO
Consequence for interpretation: ...  Confirmatory or exploratory: ...
```

## DEV-1 — 2026-11-12 — deep-battery compute envelope

**Original decision.** The frozen deep battery (D-8, tag `protocol-v1`)
is 5 configs/family × 20 seed refits × 30 bootstrap refits, with
interventional TreeSHAP on 10,000 explanation rows per fitted model
(T4.6 cap), a 1,000-row training background, and the full 589,488-row
evaluation set per fit.

**Proposed change.**
1. Refit counts per config: 5 seed refits and 5 bootstrap refits
   (was 20 + 30). All five configs/family are retained.
2. SHAP explanation sample: 1,000 rows (was 10,000) for logistic
   regression, XGBoost and LightGBM; the 1,000-row background and the
   interventional TreeSHAP estimator are retained for those families.
3. Random-forest explanations: per-feature permutation importance from
   the sanctioned T8.5 machinery (`grouped_permutation_importance`,
   Brier score, R = 3 seeded joint permutations on 1,000 rows, negative
   draws clipped to 0), aggregated to the six T5.4 feature groups.
   The per-feature Wasserstein-1 stability metric is not defined for a
   global importance vector and is omitted for random forest only.
4. The seed-0 estimator is reused for the frozen-model cross-cohort
   battery instead of being refit (identical params and seed, so the
   model is bit-identical).
5. The battery writes crash-safe per-family tables after every config
   and concatenates them with `--merge-only`; this is an execution
   change only (identical schema and content).

**Reason.** The registered envelope is arithmetically infeasible on the
project machine (4 logical cores, 7.9 GB RAM). Measured on the real
84,594-row training window before any confirmatory result was seen:

| measurement | observed |
|---|---|
| RF cfg6 fit (deepest deep-subset anchor) | 206.9 s |
| RF predict, 59,298 validation rows | 7.1 s |
| RF interventional TreeSHAP, 1,000-row background | 1.964 s/row → 327 min per 10,000 rows |
| RF path-dependent TreeSHAP | 5.57 s/row → 928 min per 10,000 rows |
| XGB cfg15 fit / interventional SHAP | 17.0 s / 0.024 s/row |
| LGBM cfg12 fit / interventional SHAP | 11.1 s / 0.066 s/row |
| LR cfg26 fit / LinearExplainer | 0.5 s / <0.001 s/row |

The T4.6 pilot had measured XGBoost on 2,538-row plumbing models
(0.643 s for 500 rows, 500-row background) and extrapolated linearly to
10,000 rows. That extrapolation does not survive real-scale forests:
the full frozen RF battery alone (255 fits + 255 SHAP runs) is on the
order of weeks, not the ~12 h the planning note assumed.

**Were test results already seen:** NO. No confirmatory statistic,
verdict, p-value or manuscript result existed when this deviation was
written; only model-fit and explainer timings on the training /
validation windows had been observed. The frozen test cohorts had not
been scored by the battery.

**Consequence for interpretation.**
- Noise floors (T8.2 Axis C, T9.2d) now pool 10 seed pairs and 10
  bootstrap pairs per configuration, i.e. 20 pairs per configuration and
  100 pairs/family (five configurations), instead of 190 + 435 = 625
  pairs per configuration under the registered 20/30 design. The
  2.5th/95th percentile anchors are correspondingly noisier; this is
  stated in the paper's limitations and every floor row carries its
  `n_pairs`.
- H2c and RQ4.3 for random forest measure stability of a permutation-
  importance reliance profile, not SHAP. Within-family temporal-vs-
  floor comparisons remain internally consistent (both sides use the
  same attribution engine); cross-family comparisons of the explanation
  axis are descriptive only.
- All confirmatory tests remain in the registered family; Holm is still
  applied once over exactly the eight registered p-values. No threshold
  is changed.

**Confirmatory or exploratory.** Confirmatory family retained under the
reduced-precision design; the explanation-engine substitution for
random forest is explicitly labelled in the results and limitations.

## DEV-2 — 2026-11-12 — realised lag axis of the H1c recovery analysis

**Original decision.** H1c is defined per arm × lag, where the frozen
label-delay convention (`CONTEXT.md` §6) fits each recalibrator on a
12-month window ending at `S − a` for `a ∈ {0, 6, 12, 18, 24, 36}` months
relative to the evaluation cohort start `S`, with `a = 0` the oracle and
`a = 36` the first realistic lag.

**Proposed change.** No code change. The realised analysis keeps the
frozen confirmatory test (pooled recovery versus 1, CI excluding 1 means
partial recovery) but documents that the lag axis actually produced by
the battery is **label age at cohort start**, i.e. quarters from the end
of the calibration window (2013Q2) to the cohort start: 3 quarters for
2014Q1 through 10 quarters for 2015Q4. The battery fits every arm on the
model's training-window scores (2012Q1–2013Q2) and applies it unchanged
to each evaluation cohort, which is the static-arm design already used by
the pre-freeze `scripts/run_calibration.py`. The sliding 12-month window
grid is not run; its feasibility remains the pre-freeze table
`reports/tables/label_delay_feasibility.csv`.

**Reason.** The frozen arm module and the pre-freeze calibration runner
both define the deployable arm as "fitted on the training-window labels
and applied forward". Implementing the sliding grid now would require
scoring loan vintages outside the frozen model windows with models
already fitted, i.e. a new scoring pass, and would change the estimand
after the protocol was frozen. The honest options were to keep the frozen
estimand and label the lag correctly, or to replace H1c after results
were visible; the former preserves confirmatory integrity.

**Were test results already seen:** PARTIAL. Per-cohort arm descriptives
for logistic regression existed (the family that finished first), but no
pooled H1c estimate, bootstrap CI, p-value or verdict had been computed
when this deviation was written.

**Consequence for interpretation.**
- The confirmatory H1c p-value is unchanged in definition: it tests
  whether pooled historical-label recovery differs from the within-cohort
  oracle across feasible arm × cohort cells.
- The descriptive lag curve is a **label-age** curve (3–10 quarters since
  the calibration window). It is not the 0–36-month label-delay grid, and
  the manuscript, Table S2 and the recovery figure caption say so.
- The pre-freeze feasibility result — that 36-month-delayed windows would
  have enough events — remains a design finding, but the delay curve
  itself is left to future work and is listed as a limitation.
- H1c remains confirmatory; the relabelling changes no threshold, test
  statistic or multiplicity decision.

## DEV-3 — 2026-11-12 — reporting companions for the H1c recovery ratio

**Original decision.** H1c's confirmatory quantity is the pooled mean of
the per-cell recovery ratio `1 − (Brier_after − Brier_oracle) /
(Brier_before − Brier_oracle)`, with a cell treated as undefined only when
the denominator is numerically zero (`|denom| ≤ 1e-12`, implemented in
`scripts/run_deep_battery.py`).

**Proposed change.** No change to the confirmatory estimator, its CI, its
p-value, the Holm family or any threshold. Two additions to the reporting:
(1) per-family descriptive companions are exported alongside the frozen
mean — the median cell recovery, the share of cells with negative
recovery, and the mean change in Brier score (`after − before`) on the
same cells; (2) Figure 5 plots the mean Brier change by family, arm and
label age instead of the raw ratio.

**Reason.** The denominator is the within-cohort oracle's Brier gain, which
is tiny for cohorts that are already close to calibrated (median oracle
gain on the completed families is of order 1e-4 in Brier units). The mean
ratio is therefore dominated by a handful of cells with near-zero
denominators and reaches double-digit negative magnitudes that are not an
interpretable "fraction of the achievable gain". The Brier change on the
same cells is stable and answers the operational question directly: did
applying the deployable arm make the score worse?

**Were test results already seen:** YES. A preliminary pooled H1c on the
three families that had finished (logistic regression, XGBoost, random
forest) was inspected before this deviation was written, and it showed the
denominator instability. No manuscript claim, verdict file or committed
table existed at that point; the confirmatory decision rule and the frozen
ratio are unchanged, and the companions are descriptive, labelled as such,
and derived mechanically from the same committed table.

**Consequence for interpretation.**
- The H1c verdict remains the frozen test's decision against the reference
  value 1. It is reported with the mean ratio and its CI in Table 3.
- The prose does **not** interpret the mean ratio as a recovery fraction.
  It reports the median cell recovery and the mean Brier change, which are
  exported as descriptive keys (`main.H1c-median.*`, `main.H1c-brier.*`,
  `main.H1c-negshare.*`).
- The H1c conclusion is about the sign and heterogeneity of recovery:
  where the deployable arms help, by how much, and for which families they
  hurt. It is not a precise magnitude estimate.
