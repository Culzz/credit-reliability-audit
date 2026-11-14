# When a credit score stops meaning what it meant: a reliability audit of calibration, explanations and decisions across eight LendingClub cohorts

**Working paper — frozen protocol `protocol-v1` (9 November 2026).**
Every empirical number in this manuscript is a placeholder resolved from
`paper/numbers.json`, which is exported mechanically from the committed
tables in `reports/tables/`. No value is typed by hand. Claims are
registered in `paper/claims.csv` with their evidence file and number keys.

---

## Abstract

Credit scores are deployed for years after they are fitted, yet the
properties that make them useful — calibrated probabilities, explanations
that reflect what the model relies on, and stable decisions near an
operating threshold — can decay at different rates. We audit all three on
the same temporal cohorts of US consumer credit. Using 36-month LendingClub
loans, we train four model families on 2012Q1–2013Q2
(n = {{n:main.train.2012Q1-2013Q2.n}}), validate on 2013Q3–2013Q4
(n = {{n:main.validation.2013Q3-2013Q4.n}}), and evaluate on eight quarterly
cohorts from 2014Q1 to 2015Q4 ({{n:main.test.2014Q1.n}} to
{{n:main.test.2015Q4.n}} loans). Thirty configurations per family are
swept; five per family are carried into a deep battery of seed and
bootstrap refits, recalibration arms, decision-flip tracks and
attribution-stability comparisons. Calibration-in-the-large changed by
{{n:main.H1a.value}} in absolute value on average across cohorts (95% CI
{{n:main.H1a.ci_low}} to {{n:main.H1a.ci_high}}). Deployable
recalibration did not recover the within-cohort oracle gain: the frozen
recovery ratio is dominated by a near-zero denominator and is negative,
and the median cell recovery was {{n:main.H1c-median.logistic_regression}}
for logistic regression and negative for the three tree families. Only
{{n:main.H2c.value}} of family-by-group explanation cells fell below
their seed/bootstrap noise floor, which is not distinguishable from none.
Recalibration produced fewer decision flips than reseeding the same model
({{n:main.H3a.value}} relative to the seed floor, 95% CI
{{n:main.H3a.ci_low}} to {{n:main.H3a.ci_high}}). AUC in cohort `t` was
strongly associated with AUC in `t+1` but only weakly, or negatively,
with next-cohort calibration, explanation stability and flip rate; the
paired delta for ECE was {{n:main.RQ4.1.value}} (95% CI
{{n:main.RQ4.1.ci_low}} to {{n:main.RQ4.1.ci_high}}). The results argue for
monitoring reliability outcomes directly rather than treating
discrimination as a sufficient proxy, and they separate what
recalibration on historical labels can repair from what it cannot.

**Keywords:** credit risk; temporal distribution shift; calibration;
explainability; decision stability; model monitoring.

---

## 1. Introduction

A credit score is fitted on one population and used on another. The
population changes because the economy, lending policy, marketing channels
and application mix change; the scoring function does not. Much of the
applied literature measures the consequence as a loss of discrimination —
AUC or KS — and treats a small change in ranking quality as evidence that
the model is still fit for purpose. That inference is not licensed. A
model can retain its ranking of borrowers while its probabilities drift
away from observed default rates, while the features it leans on change,
and while a decision rule applied near a threshold produces a materially
different set of approvals. Each of those failures affects a different
stakeholder: the probability user, the adverse-action and model-risk
reviewer, and the borrower at the margin of a decision.

This paper is an audit rather than a method. We take one public lending
portfolio, fix a protocol before looking at outcome-model results, and ask
four questions on identical cohorts. First, how far does calibration
degrade, and how much of the loss can a recalibrator fitted only on
historically available labels recover at realistic label delays? Second,
do explanations of the model change across cohorts by more than the change
produced by reseeding or resampling the same model on the same data?
Third, does recalibration move operating decisions by more than the
seed-to-seed decision noise? Fourth, does a model's AUC tell us how any of
those reliability outcomes will behave in the next cohort?

The design has three features that distinguish it from a single-model
case study. It holds the evaluation cohorts fixed across every comparison,
so calibration, explanation and decision results describe the same loans.
It represents each model family by a swept grid of thirty configurations,
including deliberately weak and over-fitted ones, and carries five of them
— chosen at fixed validation-AUC quantiles — into the deep battery, so that
conclusions are not an artefact of one hyperparameter choice. And it compares every
temporal difference against an explicit noise floor obtained by refitting
the same configuration on the same data with different seeds and bootstrap
resamples — without that floor, "the explanation changed" is unfalsifiable.

The empirical sections report what the frozen pipeline produced,
including null and inconclusive results. Section 2 positions the questions.
Section 3 describes the data and the frozen protocol. Section 4 gives the
measurement definitions and the reduced-precision deep-battery design
(DEV-1, recorded before any confirmatory result was seen). Sections 5–7
report RQ1–RQ3, Section 8 the AUC analysis, and Sections 9–13 the
monitoring implications, discussion, limitations, conclusion and
reproducibility statement.

---

## 2. Related work

*This is a working version of the section. The project's literature gate
(`protocol/GATE1.md`) is still open: the literature matrix has not been
completed and no reference enters the manuscript until the PI has verified
authors, year, venue and pages against the primary source. The repository
therefore contains no bibliography yet, and the paragraphs below state the
comparison conceptually rather than asserting citations. A verified
bibliography will replace this note when Gate 1 closes; the empirical
sections do not depend on it.*

Four strands meet in this paper. The calibration literature has
established that probabilistic classifiers trained on independent data
lose calibration under distribution shift, that the loss is often larger
in the tails, and that post-hoc recalibration can recover part of it when
labels from the new distribution are available. That work is usually
evaluated with synthetic corruptions or with random splits, and it rarely
distinguishes label availability from label timeliness — the operational
constraint in credit, where a 36-month loan's outcome is unknown at the
moment a recalibrator would have to be fitted. Our label-delay arms make
that distinction explicit and price it as a recovery fraction.

The explainability literature has moved from "does the attribution look
plausible" to "is the attribution stable and does it agree with an
independent reliance measure". Rank correlation and top-k overlap between
explanations of two models are now standard, and the instability of
explanations across seeds is documented. What is less often reported is
how much of a cross-cohort change is simply the noise of the explanation
procedure itself. We build that floor into the design and report every
temporal comparison against it, at both the feature and the domain-group
level.

The decision literature supplies the threshold logic: under a stylised
gain-to-loss ratio, the break-even probability is fixed, and a model
change matters only if it moves loans across that break-even. Flip-rate
analysis of competing models is well established; flip-rate analysis of
the same model refitted on new data, compared with a seed-level floor, is
rarer, and it is the comparison a monitoring team actually faces.

Finally, work on the "accuracy on the line" phenomenon asks when
in-distribution performance predicts out-of-distribution performance. That
question is usually asked of discrimination. We ask it of calibration,
explanation stability and decision flips on the same cohorts, and we
pre-register an equivalence region for the association rather than
treating a non-significant correlation as evidence of independence.

The gap this study was designed against is the joint audit: calibration,
explanation and decision reliability measured on one set of temporal
cohorts of public credit data, with label-delayed recalibration and with
AUC tested against each reliability outcome. Whether that gap is genuinely
open is a question for the literature matrix, not for this paper; until
Gate 1 closes we make no priority claim.

---

## 3. Data and protocol

### 3.1 Data

We use the public LendingClub loan-level file of accepted loans. The
analysis is restricted to 36-month loans originated between 2012Q1 and
2015Q4. The outcome is a resolved binary default: `y = 1` for loans whose
status is Charged Off or Default, `y = 0` for Fully Paid. Unresolved
statuses (Current, In Grace Period, Late 16–30, Late 31–120) are excluded,
never imputed; such rows are counted per cohort in Table 1. Loans are
assigned to a cohort by issue quarter.

A vintage enters the analysis only if its 36-month term had matured by the
data snapshot, so the label is observable for every retained loan. This is
the binding constraint on the evaluation window: 2016 and later vintages
are still substantially unresolved and are excluded. Table 1 gives the
cohort sizes, event counts and default rates.

**Table 1. Frozen cohorts (resolved 36-month loans).**

| Role | Window | Loans | Defaults | Non-defaults | Default rate | Unresolved dropped |
|---|---|---|---|---|---|---|
| Train | 2012Q1–2013Q2 | {{n:main.train.2012Q1-2013Q2.n}} | {{n:main.train.2012Q1-2013Q2.n_events}} | {{n:main.train.2012Q1-2013Q2.n_nonevents}} | {{n:main.train.2012Q1-2013Q2.default_rate}} | {{n:main.train.2012Q1-2013Q2.n_unresolved}} |
| Validation | 2013Q3–2013Q4 | {{n:main.validation.2013Q3-2013Q4.n}} | {{n:main.validation.2013Q3-2013Q4.n_events}} | {{n:main.validation.2013Q3-2013Q4.n_nonevents}} | {{n:main.validation.2013Q3-2013Q4.default_rate}} | {{n:main.validation.2013Q3-2013Q4.n_unresolved}} |
| Test | 2014Q1 | {{n:main.test.2014Q1.n}} | {{n:main.test.2014Q1.n_events}} | {{n:main.test.2014Q1.n_nonevents}} | {{n:main.test.2014Q1.default_rate}} | {{n:main.test.2014Q1.n_unresolved}} |
| Test | 2014Q2 | {{n:main.test.2014Q2.n}} | {{n:main.test.2014Q2.n_events}} | {{n:main.test.2014Q2.n_nonevents}} | {{n:main.test.2014Q2.default_rate}} | {{n:main.test.2014Q2.n_unresolved}} |
| Test | 2014Q3 | {{n:main.test.2014Q3.n}} | {{n:main.test.2014Q3.n_events}} | {{n:main.test.2014Q3.n_nonevents}} | {{n:main.test.2014Q3.default_rate}} | {{n:main.test.2014Q3.n_unresolved}} |
| Test | 2014Q4 | {{n:main.test.2014Q4.n}} | {{n:main.test.2014Q4.n_events}} | {{n:main.test.2014Q4.n_nonevents}} | {{n:main.test.2014Q4.default_rate}} | {{n:main.test.2014Q4.n_unresolved}} |
| Test | 2015Q1 | {{n:main.test.2015Q1.n}} | {{n:main.test.2015Q1.n_events}} | {{n:main.test.2015Q1.n_nonevents}} | {{n:main.test.2015Q1.default_rate}} | {{n:main.test.2015Q1.n_unresolved}} |
| Test | 2015Q2 | {{n:main.test.2015Q2.n}} | {{n:main.test.2015Q2.n_events}} | {{n:main.test.2015Q2.n_nonevents}} | {{n:main.test.2015Q2.default_rate}} | {{n:main.test.2015Q2.n_unresolved}} |
| Test | 2015Q3 | {{n:main.test.2015Q3.n}} | {{n:main.test.2015Q3.n_events}} | {{n:main.test.2015Q3.n_nonevents}} | {{n:main.test.2015Q3.default_rate}} | {{n:main.test.2015Q3.n_unresolved}} |
| Test | 2015Q4 | {{n:main.test.2015Q4.n}} | {{n:main.test.2015Q4.n_events}} | {{n:main.test.2015Q4.n_nonevents}} | {{n:main.test.2015Q4.default_rate}} | {{n:main.test.2015Q4.n_unresolved}} |

### 3.2 Features and leakage control

The feature set is organised into six credit-domain groups: borrower
capacity (loan amount, income, debt-to-income ratio, employment length),
credit history and tenure (earliest credit line converted to credit age,
total and open accounts), delinquency and public records, credit score
(application-time FICO range), utilisation (revolving balance and
utilisation), and application context (purpose, verification status, home
ownership, listing channel, application type, recent inquiries). The full
partition is frozen in `configs/feature_groups.yaml` and documented in
`protocol/FEATURE_GROUPS.md`.

Every group is built only from information available at origination.
Post-origination payment, recovery, last-credit-pull and hardship fields
are excluded, as are LendingClub's own model outputs (`grade`,
`sub_grade`, `int_rate`) and `installment`, which is a deterministic
function of amount, rate and term. A column is retained only if it is
populated in at least 90% of loans in every vintage from training through
the end of the test window, which removes fields that appear only in later
years and would otherwise encode the calendar. The leakage and
schema-drift audits are reproducible from `reports/tables/` and are
covered by tests.

### 3.3 Temporal design and observed shift

The training window is 2012Q1–2013Q2, the validation window is
2013Q3–2013Q4, and the eight evaluation cohorts are 2014Q1–2015Q4. No test
row is used for fitting, early stopping, recalibrator selection or
hyperparameter choice. Model selection for the headline configuration
within a family uses validation log loss; the reliability analysis instead
uses five configurations per family spanning that family's validation-AUC
range, so conclusions do not depend on the headline choice.

Shift inside the window is real but moderate. Feature-only adversarial
validation separates training from each test cohort with increasing
success across the window, and categorical composition moves most in the
listing-channel and verification fields; numeric scorecard variables move
less. These diagnostics were computed descriptively before the protocol
was frozen and are reported in `reports/tables/adversarial_validation.csv`,
`shift_measures_numeric.csv`, `shift_measures_categorical.csv` and the
corresponding figures; they are not re-used as outcome-model evidence.

Three protocol facts matter for interpretation. First, the design observes
granted loans only: no application that LendingClub rejected has an
outcome, so all statements concern the accepted population. Second, the
file contains no reliable borrower identifier across loans, so repeated
borrowers cannot be clustered; this is treated as a limitation rather than
modelled. Third, label arrival is not observed. The protocol declares a
label-delay convention (Section 4.2); this run's deployable arms are
fitted on the training window, so the realised lag axis is label age at
cohort start rather than the 0–36-month sliding grid, and the grid is
reported only as the pre-freeze feasibility analysis (DEV-2).

### 3.4 Frozen protocol and deviations

The protocol was frozen on 9 November 2026 (tag `protocol-v1`,
`protocol/PROTOCOL_FROZEN`). Confirmatory p-values are produced only by the
frozen pipeline, and Holm adjustment is applied once over the eight
confirmatory p-values. Three post-freeze deviations are recorded, none of
which changes a confirmatory hypothesis, threshold, cohort or the
multiplicity family. DEV-1 reduces the deep battery from 20 seed and 30
bootstrap refits to 5 and 5, caps explanation samples at 1,000 rows, and
substitutes permutation importance for SHAP for the random forest; it was
written before any confirmatory result existed, with the measured timings
that motivated it. DEV-2 records that the realised lag axis of the
recovery analysis is label age at cohort start rather than the 0–36-month
sliding delay grid. DEV-3 adds stable descriptive companions to the
recovery ratio, whose denominator is too small to support a mean. The
full text of each is in `protocol/DEVIATIONS.md` and Appendix A.

---

## 4. Methods

### 4.1 Calibration

Probabilities are clipped to [1e-6, 1 − 1e-6] before any logit. For a
cohort with predictions `p` and outcomes `y`:

- **Calibration-in-the-large (CITL):** intercept of the binomial GLM
  `y ~ 1` with `logit(p)` as an offset, so the slope is pinned at one.
  Ideal value 0.
- **Calibration slope:** coefficient on `logit(p)` in the separate free
  binomial GLM `y ~ logit(p)`. Ideal value 1; deviation is `|slope − 1|`.
- **A/E ratio:** `mean(y) / mean(p)`. Ideal value 1.
- **ECE:** sort by `p`, split into 15 equal-count bins, and report
  `Σ_b (n_b/N) |mean(y_b) − mean(p_b)|`.
- **Brier score:** `mean((p − y)²)`; log loss is binary cross-entropy.

All degradation statements are relative to the validation window:
`Δmetric_q = metric_q − metric_validation`. This separates
pre-existing miscalibration from shift-induced change. The headline
confirmatory statistic H1a averages, over the twenty deep-battery
configurations, the cohort mean of `|CITL_cohort| − |CITL_validation|`
across the eight test cohorts. The practical anchor, fixed before results,
is a change of 0.02 in absolute CITL, the half-width of the decision refer
band.

### 4.2 Recalibration under simulated label delay

Three post-hoc arms are fitted on the model's training-window scores,
using only labels that would have been available: intercept-only, Platt
scaling, and isotonic regression. Each arm is applied unchanged to every
evaluation cohort. For each arm and cohort we also fit an **oracle**
recalibrator on that cohort's own labels. Recovery is

`1 − (Brier_after − Brier_oracle) / (Brier_before − Brier_oracle)`,

so 1 means the deployable recalibrator matches the within-cohort oracle
and 0 means it gives no improvement over the uncalibrated model. A cell is
undefined when the cohort needs no repair and the denominator vanishes;
such cells are excluded and counted.

Label delay is the operational constraint: a 36-month loan's outcome is
unknown when a recalibrator would have to be fitted. The protocol fixes a
convention for a 12-month recalibration window ending at `S − a`, with
age lag `a ∈ {0, 6, 12, 18, 24, 36}` months relative to the cohort start
`S`; `a = 0` is the oracle and is not deployable, intermediate lags are
optimistic, and `a = 36` is the first realistic lag. The pre-freeze
feasibility table reports that every cell passes the 200-event /
200-non-event rule.

The battery implements the deployable arm as "fitted on the training
window's labels and applied forward", which is the pre-freeze calibration
runner's design. The realised lag axis is therefore **label age at cohort
start** — three quarters for 2014Q1 through ten for 2015Q4 — not the
sliding delay grid. H1c keeps its frozen definition (pooled recovery
versus 1, with lag dependence as a rank correlation across arm-by-lag
cell means); the deviation and its interpretation are recorded as DEV-2
in `protocol/DEVIATIONS.md`, and the delay grid itself is listed as
future work and a limitation rather than being run post hoc.

### 4.3 Explanations, stability and noise floors

For logistic regression, XGBoost and LightGBM we use interventional
TreeSHAP with a 1,000-row training-window background and a 1,000-row
explanation sample drawn from the head of each evaluation window in a
fixed order. Attributions are on the log-odds scale and are aggregated to
mean absolute SHAP per feature and per feature group. For random forest,
DEV-1 substitutes per-feature permutation importance computed with the
project's grouped-permutation machinery: three seeded joint permutations
per feature, Brier loss on the same 1,000 rows, negative draws clipped to
zero, then summed within groups. The substitution is deliberate and is
labelled wherever random-forest explanation results appear; the
Wasserstein distance between per-loan attribution distributions is
undefined for a global importance vector and is omitted for that family
only.

Stability between two attribution matrices is summarised by Jaccard
overlap of the top ten features, Spearman and Kendall rank agreement of
the importance vectors, cosine similarity, and the mean per-feature
Wasserstein-1 distance. The **noise floor** for each metric is the
empirical distribution obtained from refits of the same configuration on
the same training rows: all pairs among five seed refits and all pairs
among five bootstrap refits, pooled. A temporal difference counts as
beyond noise only when the observed similarity falls below the 2.5th
percentile of that floor. H2c reports the share of family-by-group cells
whose median cross-cohort group cosine falls below the group-specific
floor.

The frozen-model cohort battery holds the seed-0 estimator fixed and
explains the head of each evaluation window with the same background; the
resulting adjacent-cohort similarities feed the explanation-instability
outcome in RQ4.3.

### 4.4 Decisions and flips

Decisions use a three-band policy: approve if `p < t_low`, refer if
`t_low ≤ p < t_high`, decline otherwise. The break-even probability is
`p* = g/(g+L)`; the primary stylised loss-to-gain ratio is `L/g = 5`, so
`p* ≈ 0.167`, and the refer band is `p* ± 0.02` (approximately 0.147 to
0.187). The utility convention is the frozen one: a flip from approve to
decline changes utility by `L·y − g·(1−y)` and the reverse by
`g·(1−y) − L·y`; moves involving the refer band are reported separately
and excluded from primary utility. These are scenario quantities on
funded loans, not observed platform profits.

The **flip rate** between two scoring functions is the share of identical
loans whose band changes. The reference floor is the distribution of flip
rates over seed pairs of the same configuration scored on the same cohort,
with its 95th percentile used as the H3a threshold. Recalibration flips
compare the uncalibrated and recalibrated band assignments of the same
model on the same cohort.

### 4.5 Battery design and DEV-1

The broad sweep fits thirty configurations per family on the training
window and scores them on validation. Five configurations per family are
then selected at the minimum, 25th, 50th, 75th and maximum validation AUC
(Table S1). Each selected configuration is refitted five times with
different estimator seeds and five times on bootstrap resamples of the
training rows with the estimator seed pinned. For every fit, predictions
are cached once and reused for the score aggregates and the recalibration
arms; decision bands are formed at the configured policy cell; and
pairwise flip and stability tracks are computed within each refit axis.
The seed-0 estimator supplies the frozen-model cohort battery.

DEV-1 reduced the registered 20 × 30 refit grid to 5 × 5 after measuring
the wall-clock cost on the real training window. It also reduced the
explanation sample from 10,000 to 1,000 rows. The consequence is that each
noise floor pools ten seed pairs plus ten bootstrap pairs per
configuration — for example
{{n:main.random_forest.stability.jaccard_top10.n_pairs}} pairs for the
random-forest Jaccard floor across five configurations — rather than 625
per configuration under the registered design, so the 2.5th and 95th
percentile anchors are noisier; every floor row carries its pair count.
All definitions, cohorts, thresholds and the confirmatory family are
untouched by the deviation.

### 4.6 Inference and multiplicity

The unit of analysis is the configuration (family × configuration index),
with seed-axis rows averaged within a cohort first. CIs are percentile
cluster bootstraps over configurations with B = 2,000 and a fixed explicit
seed. The eight confirmatory tests are H1a, H1c, H2c, H3a and RQ4.1–RQ4.4;
Holm step-down is applied once over their eight p-values at α = 0.05,
two-sided. Exploratory results are reported with raw CIs and labelled as
such. A null is reported as not rejected, not as evidence of absence.

### 4.7 Does AUC predict reliability?

For each adjacent cohort pair `(t, t+1)` we compute the Spearman
correlation across the twenty deep-battery configurations between AUC in
cohort `t` and each outcome associated with cohort `t+1`: ECE,
`|slope − 1|`, explanation stability (the Jaccard agreement of the frozen
model's top-ten features between `t` and `t+1`, so larger means more
stable), and the seed-pair flip rate. The baseline is the correlation between AUC at
`t` and AUC at `t+1`. The primary estimate averages the per-pair
correlations over the seven adjacent pairs. The delta between each
outcome correlation and the baseline is recomputed on every resample of a
single cluster bootstrap design shared by all four outcomes, so the
comparisons are paired. An equivalence region of `|ρ| < 0.3` is
pre-registered for the outcome correlations: a CI wholly inside is
described as negligible, wholly outside as meaningful, and otherwise
inconclusive.

---

## 5. Results: calibration (RQ1)

### 5.1 Level and slope by cohort

Calibration-in-the-large moved from validation to the test cohorts
(Table 2; Figure 4). Averaged over the twenty deep-battery configurations,
the mean change in absolute CITL was {{n:main.H1a.value}}
(cluster-bootstrap 95% CI {{n:main.H1a.ci_low}} to
{{n:main.H1a.ci_high}}; Holm-adjusted p = {{n:main.H1a.p_holm}}). The
null of no change in absolute CITL was rejected at the 5% level. The
family means are heterogeneous: the change is
{{n:main.H1a.lightgbm}} for LightGBM and {{n:main.H1a.xgboost}} for
XGBoost, {{n:main.H1a.logistic_regression}} for logistic regression and
{{n:main.H1a.random_forest}} for random forest.

By 2015Q4 the mean absolute CITL is
{{n:main.logistic_regression.2015Q4.citl.mean}} for logistic regression,
{{n:main.xgboost.2015Q4.citl.mean}} for XGBoost,
{{n:main.lightgbm.2015Q4.citl.mean}} for LightGBM and
{{n:main.random_forest.2015Q4.citl.mean}} for random forest, against
validation values of {{n:main.logistic_regression.val.citl.mean}},
{{n:main.xgboost.val.citl.mean}}, {{n:main.lightgbm.val.citl.mean}} and
{{n:main.random_forest.val.citl.mean}} respectively. Mean ECE in the same
cohort is {{n:main.xgboost.2015Q4.ece_b15q.mean}},
{{n:main.lightgbm.2015Q4.ece_b15q.mean}} and
{{n:main.random_forest.2015Q4.ece_b15q.mean}} for the three nonlinear
families, and {{n:main.logistic_regression.2015Q4.ece_b15q.mean}} for
logistic regression. The A/E ratio over the same cohort is
{{n:main.xgboost.2015Q4.ae_ratio.mean}},
{{n:main.lightgbm.2015Q4.ae_ratio.mean}},
{{n:main.random_forest.2015Q4.ae_ratio.mean}} and
{{n:main.logistic_regression.2015Q4.ae_ratio.mean}}.

The slope moves too, and not by the same amount as the intercept. Mean
deviation `|slope − 1|` in 2015Q4 is
{{n:main.xgboost.2015Q4.slope.mean}} for XGBoost,
{{n:main.lightgbm.2015Q4.slope.mean}} for LightGBM,
{{n:main.random_forest.2015Q4.slope.mean}} for random forest and
{{n:main.logistic_regression.2015Q4.slope.mean}} for logistic regression.
The exploratory H1b summary — mean slope change relative to validation,
averaged over cohorts, on its own scale — is
{{n:main.H1b.logistic_regression}},
{{n:main.H1b.xgboost}},
{{n:main.H1b.lightgbm}} and
{{n:main.H1b.random_forest}}. As pre-registered, these are reported
separately from CITL and are not normalised onto a common scale.

### 5.2 Recovery from historical-label recalibration

The frozen H1c statistic, the pooled mean of the recovery ratio over all
feasible arm × cohort cells, is {{n:main.H1c.value}} (95% CI
{{n:main.H1c.ci_low}} to {{n:main.H1c.ci_high}}; Holm-adjusted
p = {{n:main.H1c.p_holm}}), so the null that the deployable arms match
the within-cohort oracle was rejected; the sign of the estimate is
negative, and the stable companions below show why it is not a
partial-recovery fraction. The reference value is 1: a CI that contains 1
would mean the deployable recalibrator is statistically
indistinguishable from the within-cohort oracle.

The magnitude of that ratio needs a warning that the protocol did not
anticipate. Its denominator is the within-cohort oracle's gain in Brier
score, and for these cohorts the oracle gain is tiny relative to the Brier
score itself — because the uncalibrated scores are already close to
calibrated within a cohort. A ratio with so small a denominator
is dominated by sampling noise, and its mean is not an interpretable
fraction of the achievable gain. That is why the frozen mean is reported
as the confirmatory decision but the discussion below uses two stable
companions on the same cells (DEV-3): the median cell recovery and the
mean change in Brier score from applying the deployable arm.

By that reading the deployable arms do not recover the oracle gain for
any family. The median cell recovery is
{{n:main.H1c-median.logistic_regression}} for logistic regression,
{{n:main.H1c-median.xgboost}} for XGBoost,
{{n:main.H1c-median.lightgbm}} for LightGBM and
{{n:main.H1c-median.random_forest}} for random forest, and the share of
cells with a negative ratio is
{{n:main.H1c-negshare.logistic_regression}},
{{n:main.H1c-negshare.xgboost}},
{{n:main.H1c-negshare.lightgbm}} and
{{n:main.H1c-negshare.random_forest}} respectively. The mean Brier change
on the same cells is {{n:main.H1c-brier.logistic_regression}} for logistic
regression, {{n:main.H1c-brier.xgboost}} for XGBoost,
{{n:main.H1c-brier.lightgbm}} for LightGBM and
{{n:main.H1c-brier.random_forest}} for random forest; positive values mean
the deployable arm made the score worse than leaving it uncalibrated.

Figure 5 shows that change by family, arm and label age. The intercept-only
arm is essentially neutral everywhere. Platt scaling and isotonic
regression are close to neutral for logistic regression and clearly
harmful for the tree families, where the training-window score
distribution does not cover the shifted cohort's scores and the fitted
mapping extrapolates badly. The rank correlation between label age and
the cell-mean recovery ratio is {{n:main.H1c-lag.value}} (exploratory,
within H1c). The ordering of arms is reported per family in Table S2 rather
than pooled, because the arms have different sample-size requirements and
different failure modes.

---

## 6. Results: explanation stability (RQ2)

Adjacent-cohort agreement of the frozen model's top-ten features is
compared with the seed/bootstrap floor for the same configuration. For
XGBoost the mean adjacent Jaccard is
{{n:main.xgboost.2014Q1.2014Q2.jaccard_top10.mean}} from 2014Q1 to 2014Q2
and {{n:main.xgboost.2015Q3.2015Q4.jaccard_top10.mean}} from 2015Q3 to
2015Q4, against a floor median of
{{n:main.xgboost.stability.jaccard_top10.q50}} and a 2.5th percentile of
{{n:main.xgboost.stability.jaccard_top10.q2_5}}. LightGBM's corresponding
values are {{n:main.lightgbm.2014Q1.2014Q2.jaccard_top10.mean}},
{{n:main.lightgbm.2015Q3.2015Q4.jaccard_top10.mean}},
{{n:main.lightgbm.stability.jaccard_top10.q50}} and
{{n:main.lightgbm.stability.jaccard_top10.q2_5}}; random forest's are
{{n:main.random_forest.2014Q1.2014Q2.jaccard_top10.mean}},
{{n:main.random_forest.2015Q3.2015Q4.jaccard_top10.mean}},
{{n:main.random_forest.stability.jaccard_top10.q50}} and
{{n:main.random_forest.stability.jaccard_top10.q2_5}}. These are
permutation-importance profiles, not SHAP attributions (DEV-1), so the
random-forest comparison is internally consistent but not directly
comparable with the other families' SHAP values.

H2c asks how often the temporal group-level similarity falls below the
group-specific floor. The share of family-by-group cells below the 2.5th
percentile was {{n:main.H2c.value}} (95% CI {{n:main.H2c.ci_low}} to
{{n:main.H2c.ci_high}}; Holm-adjusted p = {{n:main.H2c.p_holm}}), so the
null of no group falling below its floor was not rejected at the 5%
level; this share is not distinguishable from none. Group-level
agreement is generally
higher than feature-level agreement, which is expected because groups
aggregate correlated columns; the group medians for XGBoost range from
{{n:main.xgboost.cohort.borrower_capacity.group_cosine.median}} for
borrower capacity to
{{n:main.xgboost.cohort.credit_score.group_cosine.median}} for credit
score, against a credit-score floor of
{{n:main.xgboost.group.credit_score.group_cosine.q2_5}}.

---

## 7. Results: decision stability (RQ3)

At the primary policy cell (`p* ≈ 0.167`, refer band ±0.02), the mean
seed-pair flip rate in 2015Q4 is
{{n:main.logistic_regression.2015Q4.flip_rate.mean}} for logistic
regression, {{n:main.xgboost.2015Q4.flip_rate.mean}} for XGBoost,
{{n:main.lightgbm.2015Q4.flip_rate.mean}} for LightGBM and
{{n:main.random_forest.2015Q4.flip_rate.mean}} for random forest; the
corresponding family seed floors (95th percentile) are
{{n:main.logistic_regression.flip.seed_floor.q95}},
{{n:main.xgboost.flip.seed_floor.q95}},
{{n:main.lightgbm.flip.seed_floor.q95}} and
{{n:main.random_forest.flip.seed_floor.q95}}. Recalibration adds a second
source of movement, and the direction is the one the hypothesis did not
expect. The frozen H3a statistic is {{n:main.H3a.value}} (95% CI
{{n:main.H3a.ci_low}} to {{n:main.H3a.ci_high}}; Holm-adjusted
p = {{n:main.H3a.p_holm}}), so the null of zero difference from the seed
floor was rejected — but the sign is negative. Recalibration moved
**fewer** loans than reseeding the same model did. The family-level
differences are {{n:main.H3a.logistic_regression}},
{{n:main.H3a.xgboost}},
{{n:main.H3a.lightgbm}} and
{{n:main.H3a.random_forest}}; the positive value for logistic regression
reflects its zero seed floor, not a large flip rate.

The size of the effect should be judged against the base rate, not against
zero: a flip rate of one in a thousand loans is the pre-registered
practical anchor, and no family exceeded it in the direction the
hypothesis expected. Figure 7 plots the cohort-level seed flip rate against
the floor; Figure 5 compares recalibration arms on the Brier scale.

---

## 8. Results: does AUC predict reliability? (RQ4)

Table 3 gives the pooled Spearman correlations and the paired delta
against the AUC-to-AUC baseline. AUC in cohort `t` correlates
{{n:main.ece.rho_base}} with AUC in `t+1`, so the baseline is close to
one; all four reliability outcomes are associated with AUC far more
weakly, and three of the four outcome correlations are negative. For ECE
the outcome correlation is
{{n:main.ece.rho_outcome}} against a baseline of {{n:main.ece.rho_base}};
the delta is {{n:main.ece.delta}} (95% CI {{n:main.ece.ci_low}} to
{{n:main.ece.ci_high}}; Holm-adjusted p = {{n:main.RQ4.1.p_holm}}), and the
correlation is {{n:main.ece.region}} by the pre-registered equivalence
rule. For `|slope − 1|` the outcome correlation is
{{n:main.slope_dev.rho_outcome}} (delta {{n:main.slope_dev.delta}}, 95% CI
{{n:main.slope_dev.ci_low}} to {{n:main.slope_dev.ci_high}};
p = {{n:main.RQ4.2.p_holm}}; {{n:main.slope_dev.region}}). For
explanation stability the outcome correlation is
{{n:main.expl_stability.rho_outcome}} (delta
{{n:main.expl_stability.delta}}, 95% CI
{{n:main.expl_stability.ci_low}} to {{n:main.expl_stability.ci_high}};
p = {{n:main.RQ4.3.p_holm}}; {{n:main.expl_stability.region}}). For the
flip rate the outcome correlation is {{n:main.flip_rate.rho_outcome}}
(delta {{n:main.flip_rate.delta}}, 95% CI {{n:main.flip_rate.ci_low}} to
{{n:main.flip_rate.ci_high}}; p = {{n:main.RQ4.4.p_holm}};
{{n:main.flip_rate.region}}). Figure 8 plots the four deltas. Only the
slope-deviation outcome is classified as meaningful by the
pre-registered region; the other three are inconclusive.

Two design facts bound the RQ4 reading. First, the logistic-regression
sweep is effectively a single model: its thirty configurations are
near-identical in validation AUC, so the five logistic-regression
clusters that enter the deep battery carry almost no cross-configuration
variation, and the twenty clusters are in practice fewer independent
points than the count suggests. Second, the deep subset is chosen at
validation-AUC quantiles and is therefore intended to span the family's
own quality range, not the pooled one;
cross-family comparisons of the delta are descriptive. The confirmatory
object is the paired delta, not the absolute correlation.

---

## 9. Monitoring recommendations

The results support six concrete practices. They are recommendations, not
validated interventions; each is tied to a result above.

1. **Monitor calibration on resolved cohorts, not only discrimination.**
   Track CITL, A/E and adaptive-bin ECE per cohort and compare them with
   the validation window. H1a tests exactly this change; the family
   cohort means in Table 2 show where it is largest.
2. **Treat AUC as a screening statistic, not a reliability certificate.**
   The RQ4 deltas compare AUC's association with calibration, explanation
   and decision outcomes against its association with future AUC. A
   monitoring dashboard that reports only the latter omits the outcomes
   that the audit finds.
3. **Fit recalibrators on the label vintage you will actually have, and
   check them against the uncalibrated score before deployment.** On
   these cohorts the tree-family arms increased Brier score; Figure 5
   shows the Brier change by family, arm and label age. The
   0–36-month delay grid is a feasibility result from the frozen design,
   not a measured curve in this run (DEV-2), so a deployable rule should
   be re-estimated on the label window a lender actually holds.
4. **Recompute the decision population, not only the average score.** At
   the operating cell, report the share of loans within the refer band and
   the flip rate between the incumbent and the candidate score. A stable
   average probability can hide a large band movement.
5. **Establish a seed/bootstrap floor before calling an explanation
   change meaningful.** The floor is cheap relative to the explanation run
   itself and converts an unfalsifiable claim into a test.
6. **Re-audit after any change of explanation engine.** The random-forest
   substitution in DEV-1 changes the object being measured; a monitoring
   rule calibrated on SHAP should not be transferred to permutation
   importance without re-basing the floor.

---

## 10. Discussion

The audit's central observation is that reliability is not one property.
On these cohorts, calibration change, explanation change and decision
change are only partly aligned, and the direction of the AUC association
differs by outcome. A team that monitors only discrimination would have
missed the calibration movement that H1a measures and the fact that
recalibration moved fewer loans than reseeding, even where the ranking of
borrowers was largely preserved. A recalibrator that changes little at
the decision boundary still needs to pass the Brier check of Section 5.2
before it is adopted.

The recovery results sharpen the operational message. Historical-label
recalibration is not a free repair, and on these cohorts it is not a
repair at all for the tree families: the deployable arm is fitted on
training-window scores whose distribution does not cover the shifted
cohort, and applying it increases Brier score relative to leaving the
model uncalibrated. For logistic regression the arms are close to
neutral. The practically relevant question for a lender is therefore not
"does recalibration work" but "which arm, fitted on which vintage, is
stable enough to apply". The arm ordering in Table S2 is the input to that
decision, and a monitoring rule should compare the recalibrated score with
the uncalibrated one on the target cohort before adopting it.

The explanation results require a careful reading. Feature-level stability
is low in absolute terms for every family, but the relevant comparison is
the noise floor: much of the apparent change is the instability of the
attribution procedure itself. The group level is more stable, which
supports reporting domain-group reliance rather than top-feature lists to
non-specialist reviewers. The random-forest substitution is a genuine
limitation of the explanation comparison, not a result about SHAP versus
permutation importance.

Two negative or inconclusive results deserve the same prominence as the
positive ones. The H1c lag association is exploratory and should not be
read as a confirmed monotone law; and the RQ4 equivalence classifications
are statements about the CI relative to a pre-set region, not proof that
AUC carries no information. Where a delta CI straddles zero, the honest
reading is that this design does not distinguish the outcome correlation
from the baseline.

What would change the conclusions? A design with more independent
configurations, especially for logistic regression, would sharpen RQ4. A
true label-arrival timestamp would replace the simulated lag convention.
Rejected-applicant outcomes would extend the audit from the accepted
population to the decision population. And a SHAP-based random-forest
battery at the original scale would remove the engine substitution. None
of these is available in the public file; all are stated as limitations.

---

## 11. Limitations

- **Accepted-loan selection.** Only granted loans have outcomes. Rejected
  applicants are absent, so the audit cannot speak to the full decision
  population and may understate shift in the declined region.
- **One platform, one period.** The data are US consumer loans from a
  single marketplace lender, 2012–2015. External validity is limited and
  no cross-platform claim is made.
- **Simulated label arrival.** True label-arrival times are not observed.
  The frozen protocol declares a 0–36-month delay grid; this run's arms
  are fitted on the training window, so the reported curve is label age
  (3–10 quarters), and the sliding delay grid is left to future work
  (DEV-2).
- **Recovery-ratio denominator.** The H1c recovery ratio divides by the
  within-cohort oracle's Brier gain, which is tiny for cohorts that are
  already close to calibrated. The frozen mean is dominated by near-zero
  denominators; the median recovery and the Brier change are reported as
  interpretable companions, and the ratio's magnitude should not be read
  as a fraction of the achievable gain (DEV-3).
- **Repeated borrowers.** No reliable cross-loan borrower identifier is
  available, so dependence between loans is not modelled.
- **SHAP assumptions.** Attributions are model-relative and assume the
  interventional background is meaningful; correlated features can make
  attributions non-unique.
- **Permutation extrapolation.** Grouped permutation importance evaluates
  the model on feature combinations that may lie off the data manifold
  when features are correlated; disagreement with SHAP is a finding, not
  an error in either method.
- **DEV-1 precision.** Five seed and five bootstrap refits per
  configuration give 100 floor pairs per family, not the 3,125 of the
  registered design. Percentile anchors are noisier; every floor row
  reports its pair count.
- **Explanation-engine substitution.** Random-forest explanations are
  permutation importances, not SHAP. Cross-family explanation comparisons
  are descriptive only.
- **Simulated thresholds and cost ratios.** The policy cell and the
  loss-to-gain ratio are stylised; utility quantities are scenario
  arithmetic, not observed profits.
- **Single explanation sample.** Explanations use 1,000 rows per model
  (DEV-1); sampling error in the attribution matrices is not propagated
  into the stability CIs.
- **LR configuration degeneracy.** The logistic-regression search space
  produced thirty near-identical fits, limiting the effective number of
  independent clusters in RQ4.
- **Mixed model not fitted.** The pre-registered secondary mixed model was
  not estimated; the primary cluster-bootstrap estimator is reported
  instead, and the omission is recorded rather than presented as a null.

---

## 12. Conclusion

On eight quarterly cohorts of 36-month LendingClub loans, calibration
deteriorated after the validation window (H1a rejected, heterogeneous by
family); group-level explanations did not move beyond the seed/bootstrap
floor (H2c not rejected); recalibration moved fewer decisions than
reseeding the same model (H3a rejected with a negative estimate); and AUC
tracked future AUC but not the reliability outcomes (all four RQ4 deltas
rejected). The confirmatory tests are reported with their Holm-adjusted
p-values and verdicts in Table 3; exploratory quantities are labelled as such. The
practical conclusion is that a reliability audit needs all three
outcomes measured on the same cohorts, an explicit noise floor for the
explanation comparisons, and a recalibration analysis that respects when
labels would actually have arrived. The frozen protocol, the deviation
record and the mechanical numbers export make every reported value
re-derivable from the repository.

---

## 13. Reproducibility statement

The analysis is reproducible from the repository
`credit-reliability-audit` at tag `protocol-v1` plus the DEV-1 commit.
The raw LendingClub file is not redistributed; `data/README.md` records
the source, version, access date and checksum. Scripts seed every
stochastic step with explicit literals: `BASE_SEED = 20180101` for the
model/battery axes and `STAT_SEED = 20261109` for the bootstrap inference
design. Committed scripts contain no `hash()` calls and no wall-clock
seeding. The pipeline is:

```
python -m uv run python -m scripts.run_sweep
python -m uv run python -m scripts.run_deep_battery
python -m uv run python -m scripts.run_real_statistics
python -m uv run python -m scripts.report_real_cohorts
python -m uv run python -m scripts.run_paper_tables
python -m uv run python -m scripts.report_verdicts
python -m uv run python -m scripts.run_real_figures
python -m uv run python -m scripts.export_numbers --source main
python -m uv run python -m scripts.render_paper --source main
```

The battery writes crash-safe per-family tables and concatenates them; the
merged schema is what `run_real_statistics` consumes. `paper/numbers.json`
is generated only from `reports/tables/`, and the renderer fails on any
placeholder whose key is missing. The full test suite covers the metric
definitions (including the offset-versus-free intercept distinction and
the 1e-6 clipping rule), the stability and permutation machinery, the
battery helpers, the export adapters and the renderer. A post-freeze
change log is maintained in `protocol/DEVIATIONS.md`.

---

## Appendix A. Post-freeze deviations

DEV-1 (2026-11-12) reduced the deep battery to five seed and five
bootstrap refits per configuration, capped SHAP explanations at 1,000
rows, substituted Brier-loss permutation importance for SHAP on random
forest, and reused the seed-0 estimator for the cohort battery. The
deviation was written before any confirmatory result was seen. Measured
timings and the full consequence statement are in
`protocol/DEVIATIONS.md`.

DEV-2 (2026-11-12) records that the recovery analysis's lag axis is
label age at cohort start (3–10 quarters since the calibration window),
not the 0–36-month sliding label-delay grid. H1c's confirmatory test is
unchanged; the grid is reported as a pre-freeze feasibility result and
left to future work. See `protocol/DEVIATIONS.md`.

DEV-3 (2026-11-12) adds stable descriptive companions to the H1c ratio —
median cell recovery, share of negative cells and mean Brier change —
because the ratio's denominator (the within-cohort oracle gain) is tiny
and its mean is not interpretable. The confirmatory estimator, CI,
p-value and Holm family are unchanged. See `protocol/DEVIATIONS.md`.

---

## Appendix B. Tables and figures

**Table 2.** Cohort-level calibration aggregates by family
(`reports/tables/paper_table2_calibration.csv`, seed-axis means of
`deep_agg_real.csv`).

**Table 3.** Confirmatory tests with cluster-bootstrap CIs and
Holm-adjusted p-values (`reports/tables/paper_table3_confirmatory.csv`,
from `real_statistics.csv`).

**Table S1.** Deep-subset anchors (`reports/tables/paper_tableS1_anchors.csv`,
from `deep_subset.csv`).

**Table S2.** Recovery by family, arm and label age
(`reports/tables/paper_tableS2_recovery.csv`, from `deep_arms_real.csv`).
Table 1 is `reports/tables/paper_table1_cohorts.csv`.

**Figures.** F1 cohort volume and default rate by vintage
(`cohort_volume_by_vintage.png`, `default_rate_by_vintage.png`); F2 shift
assessment (`psi_heatmap.png`, `default_rate_shift.png`,
`concept_drift_strata.png`, `adversarial_roc.png`); F3 performance under
shift (`sweep_auc_brier_real.png`); F4 calibration degradation by cohort
(`citl_by_cohort_real.png`, `ece_by_cohort_real.png`); F5 recalibration
arms (`recovery_by_lag_real.png`, mean Brier change — after minus before
— by family, arm and label age; negative values mean the deployable arm
improved on the uncalibrated score); F6 explanation stability
(`explanation_stability_real.png`); F7 decision flips
(`flip_rate_by_cohort_real.png`); F8 AUC versus reliability
(`rq4_delta_real.png`). No figure contains a hand-typed value; each is
rendered by a committed script from a committed table.
