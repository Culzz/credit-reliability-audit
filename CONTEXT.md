# CONTEXT.md — Credit Reliability Audit

**Read this file completely before doing anything. Then read `PLAN.md`.**
Version 1.0 · Written 2 Nov 2026 · Owner: Chima (GitHub: `Culzz`) · Repo: `credit-reliability-audit`

---

## 0. How to use this file (working conventions)

Collaborators join the project in sessions without prior memory of previous sessions. This file and `PLAN.md` are therefore the project's shared memory; re-read the relevant sections at the start of every session.

**Reading order:** this file (all of it) → `PLAN.md` (the current phase and tasks) → `notes/STATUS.md` (current state) → `notes/DECISIONS.md` (past decisions).

**Your first reply in any new session must contain:**
1. Ten bullets summarising the project in your own words (proves you understood it).
2. A list of anything in these files that is ambiguous, contradictory or missing.
3. The role you are taking (Section 9) and the task ID you will start with.

Then wait for the PI to confirm. Do not start coding or writing before that confirmation.

**If you cannot run code locally:** write the code, the tests and the exact command to run, mark the task `READY_FOR_HUMAN_RUN`, and stop. Never describe results of code you have not seen run.

---

## 1. The project in plain language

**What we are doing.** Credit-risk models are normally judged by ROC-AUC (how well they rank risky borrowers above safe ones). But lenders and regulators also need: (a) predicted probabilities that mean what they say (calibration), (b) explanations that are stable and reflect what the model actually uses, and (c) approve/decline decisions that stay defensible when the borrower population changes over time.

We audit all three on the *same* real loans, with strictly time-ordered evaluation, and ask whether AUC tells you anything about them.

**Data.** Real LendingClub accepted loans, 2007–2018Q4 (public Kaggle dataset). 36-month loans only. No synthetic data is ever used for results (synthetic data is allowed only inside unit tests).

**Type of paper.** An empirical audit. Not a new algorithm. Not a leaderboard. Negative and inconclusive results are acceptable and publishable because the protocol is frozen before final evaluation.

**Working title.** *Does Accuracy Predict Reliability? Calibration, Explanation Fidelity and Decision Stability of Credit-Risk Models Under Temporal Distribution Shift.*

**Why it matters to the PI.** The paper supports the PI's applications for fully funded postgraduate AI/ML scholarships (2027 intake) and a preprint. Quality, honesty and reproducibility matter more than volume. A reviewer should be unable to find leakage, p-hacking, fabricated references or unsupported claims.

**What "excellent" means here.** Not the most experiments. It means: a defensible gap, clean leakage-free data, a frozen protocol, uncertainty on every number, honest limitations, a reproducible repo, and a manuscript whose every number traces to a script. Originality is *earned at Gate 1* (Section 10) and never assumed.

---

## 2. People, tools and constraints

- **PI:** Chima. Strong in Python/ML. Works on **Windows (PowerShell) in VS Code**. Prefers direct, honest, detailed answers without hedging. Will run heavy jobs locally or on Colab/Kaggle.
- **Supervisor:** Dr. Obikwere Fortune. Only the PI contacts them. Collaborators never contact anyone and never fabricate supervisor feedback.
- **Stack:** Python 3.11, `uv`, pandas, NumPy, DuckDB, PyArrow/Parquet, scikit-learn, XGBoost, LightGBM, SHAP, statsmodels, SciPy, MLflow, matplotlib/seaborn, pytest, ruff, Git/GitHub, Zotero.
- **Compute:** unknown personal machine. Write memory-efficient code: Parquet + DuckDB, no repeated CSV loads, cache predictions, subsample for expensive explanation work.
- **Time:** target about 16 weeks of part-time work (about 300 hours). Cut lines in `PLAN.md` Section 2 protect the core if time runs short.
- **Privacy:** never write the PI's phone, address, ID numbers or credentials into any file. Never commit Kaggle tokens or raw data.

---

## 3. Research questions and hypotheses

**Central question.** Does predictive accuracy (ROC-AUC) adequately predict the reliability of ML credit-risk models on future borrower populations?

| ID | Question |
|---|---|
| **RQ1** | How does probability calibration degrade across future cohorts, and how much can recalibration (none / intercept-only / Platt / isotonic) recover when labels arrive late? |
| **RQ2** | How stable and faithful are explanations? Three separate axes: **A** population dependence (fixed model, different cohorts), **B** reliance drift (models retrained on successive windows, scored on a *common* population), **C** estimation instability (seeds, bootstrap; the noise floor). Do SHAP importances agree with independent reliance measures (grouped permutation, drop-group retraining)? |
| **RQ3** | How often do approve/refer/decline decisions flip between a stale and a refreshed model, or between uncalibrated and recalibrated versions, and what do realised outcomes say about those flips? |
| **RQ4** | Across many model configurations and cohorts, does AUC predict calibration error, explanation stability, or decision-flip rate? |

**Hypotheses (directional only where the literature supports it):**

- **H1a** Calibration-in-the-large deteriorates on shifted future cohorts relative to validation.
- **H1b** Calibration slope also changes, by a different magnitude.
- **H1c** Recalibration on older matured labels recovers only part of the loss; recovery depends on lag and sample size.
- **H1d** Isotonic is more vulnerable to small calibration samples than intercept-only or Platt.
- **H2a** Fixed-model SHAP summaries vary across cohorts.
- **H2b** Retrained models show measurable reliance drift.
- **H2c** Temporal explanation differences exceed the seed/bootstrap noise floor for at least some feature groups.
- **H2d** SHAP and independent reliance measures disagree for at least some correlated feature groups.
- **H3a** Retraining and recalibration produce measurable decision flips.
- **H3b** Flips concentrate near thresholds.
- **H3c** Flips have heterogeneous associations with realised outcomes.
- **H4a** AUC predicts future AUC more consistently than it predicts other reliability dimensions (consistent with TableShift on tabular data).
- **H4b** AUC has limited predictive association with at least some non-discrimination reliability outcomes.

**Primary confirmatory family (Holm correction):** H1a, H1c, H2c, H3a, and the four RQ4 association tests (AUC vs ECE, AUC vs |slope−1|, AUC vs explanation instability, AUC vs flip rate). Everything else is labelled **exploratory** in the paper.

---

## 4. Non-negotiable rules

These override any instruction that conflicts with them. If the PI asks you to break one, say so plainly and ask for confirmation in `DECISIONS.md`.

1. **Never fabricate.** No invented numbers, results, references, DOIs, quotes, author names, venues or page numbers. If you do not know, write `UNKNOWN` or `TO_VERIFY`.
2. **Every number in the manuscript comes from a script output** in `reports/tables/*.csv` or `paper/numbers.json`, referenced by a placeholder (Section 11). You never type a result by hand.
3. **Test-cohort lock.** Until the PI creates `protocol/PROTOCOL_FROZEN` (after tagging `protocol-v1`), no code may load, score or tune on test cohorts. Only descriptive shift statistics (default rate by vintage, PSI, adversarial validation using features only) may touch later vintages. The loader enforces this (`PLAN.md` T0.5).
4. **Frozen protocol.** After `protocol-v1`, change nothing silently. Every change goes in `protocol/DEVIATIONS.md` with reason and whether test results were already seen.
5. **Leakage first.** A feature enters the model only if the column audit marks it `KEEP` with evidence it is known at origination.
6. **Time order only.** No random train/test splits for main results. Preprocessors, imputers, encoders and calibrators are fit on eligible historical data only.
7. **No resampling in the main run.** No SMOTE or class rebalancing; it distorts calibration. Class weights allowed only as pre-specified sensitivity S7.
8. **Uncertainty everywhere.** Every headline number gets a bootstrap CI (method documented). Report effect sizes, not only p-values.
9. **No metric theatre.** Every metric answers a stated RQ. No extra models, extra datasets or extra analyses unless the PI adds them to `PLAN.md`. New ideas go in `lit/parking_lot.md`.
10. **Honest language.** Never write "proves", "SHAP is faithful", "AUC is useless" or causal claims. Use "associated with", "under the specified procedure", "in this sample".
11. **Scope honesty.** The sample is *accepted, funded* loans from one US peer-to-peer platform. Rejected applicants have no outcomes. Say so wherever decisions or generalisation are discussed.
12. **Do not hide failures.** If a test fails, a number looks too good, or you suspect leakage, stop and report. Suspiciously high AUC (above about 0.80 on this task) is a leakage alarm until proven otherwise.
13. **Small steps, evidence-backed.** Report commands run, file paths changed, and the first and last lines of real output. Commit messages follow `T<id>: <what> [run <run_id>]`.
14. **Synthetic data only in tests.** Use it to unit-test metric code (e.g., a perfectly calibrated simulator should give slope about 1, intercept about 0). Never in results.
15. **Human decides.** Thresholds marked PROPOSED, venue choice, novelty claims, authorship, and anything involving external contact belong to the PI.

---

## 5. Data facts and known pitfalls

**Source (PI must verify at download):** Kaggle `wordsforthewise/lending-club`, file `accepted_2007_to_2018Q4.csv.gz`. Third-party descriptions report about 2.26 million rows × 151 columns, about 1.6 GB uncompressed. Treat as reference checks, not guarantees. The PI records version, access date, SHA-256, licence text and LendingClub terms in `data/README.md`. Do not assume CC0 metadata settles redistribution; never commit data.

**Target.** `y = 1` if `loan_status` ∈ {Charged Off, Default}; `y = 0` if Fully Paid. Exclude unresolved statuses (Current, In Grace Period, Late 16–30, Late 31–120). Inspect the exact status strings in the file; handle legacy "Does not meet the credit policy" statuses explicitly (exclude with reason, or map with documentation).

**Main sample.** `term == 36 months` only (check how `term` is stored). Quarterly vintages from `issue_d`. A vintage enters only if issue date + 36 months ≤ dataset snapshot **and** its unresolved-status share is below the threshold.

**Pitfalls (each has a test or audit step in `PLAN.md`):**

- **Post-origination leakage.** Payment/recovery fields (`total_pymnt*`, `total_rec_*`, `recoveries`, `collection_recovery_fee`, `last_pymnt_*`, `out_prncp*`), last credit-pull and last-FICO fields, hardship and settlement fields, and anything outcome-derived.
- **LendingClub's own model outputs.** `grade`, `sub_grade`, `int_rate`, and `installment` (a function of amount, rate and term). Excluded from the main run; sensitivity S1 only.
- **Schema drift.** Some columns only populate from about 2015. A feature that is missing early identifies the era and will fake "shift". Rule: keep a column only if populated share ≥ threshold in every vintage of the training-plus-test span.
- **Identifiers and text.** Drop `id`, `member_id`, `url`, `desc`, `title`, `zip_code`; treat `emp_title` and `addr_state` as explicit documented decisions.
- **Footer/summary rows** may appear at file end; remove by numeric `id` check and document counts removed.
- **Right-censoring.** Loans issued near the snapshot may be mostly unresolved. Early charge-offs and fast prepayers resolve first, so including immature vintages biases outcomes.
- **No borrower identifier** is guaranteed, so borrower-clustered inference is not claimed; discuss repeated-borrower dependence as a limitation.
- **Label arrival times are not in the data.** Real operational label delays cannot be observed. We simulate delay with a declared convention (Section 6, "Label-delay convention") and say so.

**Starting `KEEP` candidates (the audit decides, not this list):** `loan_amnt`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `purpose`, `dti`, `delinq_2yrs`, `earliest_cr_line` (convert to credit age at origination), `fico_range_low/high` (application-time), `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `initial_list_status`, `application_type`, `mort_acc`, `pub_rec_bankruptcies`. Verify `funded_amnt` and `funded_amnt_inv` timing before keeping.

---

## 6. Design decisions and default parameters

Everything in this table is **PROPOSED** until the PI confirms it at Gate 4 (feasibility) and freezes it at Gate 5 (protocol). Collaborators may implement them as configuration values but must not hard-code them.

| Parameter | Default | Note |
|---|---|---|
| Train window (starting point) | 2012Q1–2013Q2 | May change at Gate 4 based on descriptive shift only |
| Validation window | 2013Q3–2013Q4 | Tuning and static recalibrator |
| Test cohorts | Quarterly, after validation, up to last mature vintage | Aim for 6 or more |
| Max unresolved share per vintage | 0.01 | Sensitivity at 0.02 (S3) |
| Min populated share per feature per vintage | 0.90 | Schema-drift rule |
| Min events / non-events per evaluation cohort | 200 / 200 | Calibration assessment |
| Min recalibration window | Platt, intercept-only: 200 events and 200 non-events; isotonic: 500 and 500 | Otherwise cell = INFEASIBLE |
| Models | Logistic regression (regularised), random forest, XGBoost, LightGBM | No others in main run |
| Configs per family (broad sweep) | 30, including deliberately weak and over-fitted | One seed each |
| Deep subset | 5 configs per family at quantiles of validation AUC (min, 25th, 50th, 75th, max) | 20 seeds each |
| Stochasticity | `subsample` and `colsample_*` < 1 in search spaces | Else seeds give identical models |
| Model selection (headline per family) | Lowest **validation log loss** | Never test data |
| Bootstrap | B = 1000 (2000 for headline CIs), percentile CIs, paired where comparing | Resample loans within cohort |
| ECE | Adaptive equal-mass bins, B = 15; sensitivity at 10, 20, 30 | Always beside slope and intercept |
| SHAP background | 1000 training-window rows, stratified by vintage, fixed seed | Same background across models in primary comparisons |
| SHAP explain sample per cohort | Broad: 3,000; deep: 20,000 | Fixed seed |
| Stability top-k | 5, 10, 15 (primary k = 10) | |
| Stability bootstrap resamples | 30 | |
| Feature groups | Domain groups + clusters with absolute Spearman ≥ 0.8 | Finalise **before** seeing explanation results |
| Decision policy | Approve if p < t_low, refer if t_low ≤ p < t_high, decline otherwise | |
| Break-even | p* = g / (g + L); loss/gain ratios L/g ∈ {3, 5, 9}, primary L/g = 5 (p* ≈ 0.167) | Stylised, not real profit |
| Refer band | t_low = p* − 0.02, t_high = p* + 0.02 | Sensitivity ±0.01, ±0.04 |
| Second policy | Fixed approval rate (rate set from validation under the fixed-threshold policy) | |
| RQ4 "negligible" region | Spearman ρ in (−0.3, +0.3) | CI inside = negligible; CI outside = meaningful; else inconclusive |
| Multiplicity | Holm over the primary confirmatory family | |

**Label-delay convention (RQ1).** For a test cohort starting at date S, a recalibration window of width W = 12 months consists of loans issued in the 12 months ending at S − a, where the *age lag* a ∈ {0, 6, 12, 18, 24, 36} months.
- a = 0 is the **oracle** reference (not deployable).
- 0 < a < 36 uses outcomes that would **not** yet be fully observable for a 36-month loan; label these "optimistic, not deployable".
- a ≥ 36 is the first **realistic** lag (full term has elapsed).
- Plus the **static** arm: the validation window.
- A cell is run only if the feasibility table (min events rule) says FEASIBLE. Because windows for a = 36 on early test cohorts fall before 2012 where volume is small, the realistic-lag analysis may be possible only for later cohorts. That is a finding to report, not a problem to hide.
- Exploratory only: approximating label-arrival timing from `last_pymnt_d` (used for timing, never as a feature). This is biased toward early events; declare it so.

**Axis B common population.** Models M1..Mk are trained on expanding windows ending at successive quarters. They are scored on a **common evaluation population** that lies after all training windows (default: the last two eligible cohorts). Otherwise population dependence and reliance drift are confounded.

---

## 7. Definitions and formulas (use exactly these)

**Probabilities.** Clip `p` to [1e-6, 1 − 1e-6] before any logit.

- **Calibration intercept (calibration-in-the-large):** fit `y ~ 1` with `logit(p)` as an *offset* in a binomial GLM; report the intercept (ideal ≈ 0).
- **Calibration slope:** fit `y ~ logit(p)` binomial GLM; report coefficient on `logit(p)` (ideal ≈ 1). Use `|slope − 1|` as deviation.
- **A/E ratio:** `mean(y) / mean(p)`.
- **ECE (adaptive):** sort by `p`, split into B equal-count bins, `ECE = Σ_b (n_b/N) · |mean(y_b) − mean(p_b)|`.
- **Brier:** `mean((p − y)²)`. **Log loss:** standard binary cross-entropy.
- **Degradation:** always **relative to validation**: `Δmetric_q = metric_q − metric_validation`. This separates pre-existing miscalibration from shift-induced degradation.

**Explanation measures** (global importance vector `I`, mean |SHAP| per feature or per feature group, on log-odds scale):
- Top-k Jaccard `|A∩B| / |A∪B|`; Spearman ρ; Kendall τ-b; cosine similarity of `I` vectors; local top-3 reason-code agreement per loan across seeds.
- **Noise floor:** the distribution of the same metric over all pairs of seed-retrained models and over bootstrap-refit models on identical data. A cohort or retraining difference counts as "beyond noise" only if the observed similarity falls below the 2.5th percentile of the noise-floor distribution.
- **Axis A:** one frozen model, explanations on each cohort with a fixed background. **Axis B:** versions M1..Mk on the common population. **Axis C:** seeds and bootstrap on one cohort.
- **Fidelity (operational):** compare group-level SHAP importance against (i) grouped permutation importance (permute the whole group jointly; drop in log loss and AUC) and (ii) drop-group retraining (retrain without the group; loss change). Report Kendall τ between rankings with CIs and list groups whose ranks differ by 2 or more. Treat disagreement as a finding. Never write "SHAP was proven faithful".

**Decision measures.** Decisions coded 0 approve, 1 refer, 2 decline. Flip rate `= mean(D_A ≠ D_B)` on identical loans. Transition matrix over the 3×3 cells.
- **Per-loan simulated gain of a flip** (g = gain per repaid loan, L = loss per defaulted loan, y observed default):
  - approve → decline: `ΔU = L·y − g·(1 − y)`
  - decline → approve: `ΔU = g·(1 − y) − L·y`
  - Moves involving *refer* are reported separately (counts, default rates); the primary utility excludes refer.
- **Null:** for each flip direction, resample the same number of non-flipped loans matched on decision under model A and on score within ±0.01, compute the same total ΔU, repeat 1000 times. Compare the observed total ΔU and the realised default rate among flipped loans with this null and with p*.
- These are stylised scenario quantities, **not** observed LendingClub profits.

**RQ4 statistics.** Unit of analysis is the configuration (family, config_id, seed). Primary estimate: within-cohort Spearman ρ between AUC and each reliability measure, averaged over cohorts; **cluster bootstrap over configurations** for the CI (a configuration's cohort rows resampled together). Secondary: mixed model `reliability ~ AUC_centred + family + cohort + (1|config)` if sample supports it. Also report: rank-reversal rate (share of config pairs where higher AUC is worse on reliability), an "all configs" vs "reasonable configs" contrast (reasonable = within a pre-stated validation-AUC distance of the family's best), and AUC-to-future-AUC ρ.

---

## 8. Repository, schemas and conventions

```
credit-reliability-audit/
  README.md  pyproject.toml  uv.lock  CITATION.cff  LICENSE
  CONTEXT.md  PLAN.md                  <- these two files, kept in repo root
  notes/         STATUS.md  DECISIONS.md  REVIEWS.md
  protocol/      PROTOCOL.md  HYPOTHESES.md  GATE1.md  GATE3.md  GATE4.md
                 DEVIATIONS.md  RESEARCH_RISKS.md  PROTOCOL_FROZEN (created by PI only)
  configs/       base.yaml data.yaml splits.yaml models.yaml calibration.yaml
                 explanations.yaml decisions.yaml statistics.yaml
  data/          README.md  column_audit.csv  raw/ interim/ processed/   (data gitignored)
  src/           data/ features/ models/ calibration/ explain/ decisions/ statistics/ visualization/
  scripts/       run_data_audit.py run_baselines.py run_rq1.py run_rq2.py run_rq3.py run_rq4.py
                 run_sensitivity.py export_numbers.py render_paper.py
  experiments/   experiment_registry.yaml
  reports/       tables/  figures/
  paper/         manuscript.md  references.bib  claims.csv  numbers.json  supplementary/
  lit/           matrix.csv  reading_notes/  parking_lot.md
  tests/
```

**Naming:** vintages `YYYYQn` (e.g., `2014Q3`); run IDs `<family>-<config_id>-s<seed>-<train_end>`; one config hash per run (SHA-256 of canonical YAML).

**`predictions.parquet`** (one row per loan per run per recalibration arm):
`loan_id:int64, run_id:str, family:str, config_id:str, seed:int, train_start:str, train_end:str, eval_cohort:str, y:int8, p:float32, recal_arm:str(none|intercept|platt|isotonic), recal_window:str, lag_months:int|null`

**`metrics_long.parquet`** (every metric, tidy):
`run_id, family, config_id, seed, train_window, eval_cohort, recal_arm, lag_months, metric, value, ci_low, ci_high, n, n_events, boot_B, created_utc, git_sha`

All figures and tables are produced by scripts from these two files. No notebook state is allowed to be the only source of a result. Log every run in MLflow (config hash, data hash, git SHA, seed, metrics, artefact paths); never log raw applicant rows.

---

## 9. Roles, handoffs and review

| Role | Does | Notes |
|---|---|---|
| **Builder** | Data pipeline, features, models, metrics code, explanation code, tests, run scripts, figure scripts | The collaborator stronger at code |
| **Analyst-Writer** | Literature matrix and notes, protocol text, statistical analysis plan and code review, manuscript drafting, claim register | The collaborator stronger at long-form writing and statistics |
| **PI (human)** | Downloads data, runs heavy jobs, verifies every citation, confirms or changes PROPOSED parameters, freezes protocol, contacts supervisor, submits | Chima |

Swap roles if one collaborator proves weaker. The PI assigns roles in `notes/STATUS.md`.

**Mandatory cross-review.** Builder reviews the Analyst's statistical code and formulas for implementation errors and leakage. Analyst reviews the Builder's metric definitions against Section 7 and checks every manuscript number against `numbers.json`. Reviews are logged in `notes/REVIEWS.md` using the template below. A task is done only after review.

**Templates**

`notes/STATUS.md`
```
## Current phase / gate:
## Active role assignments:
## Last completed task (ID, date, commit):
## Next task (ID):
## Blockers (needs PI):
## Open questions:
```

`notes/DECISIONS.md` (append only)
```
### D-<n> <date> <short title>
Context: ...  Options: ...  Decision (by PI, or collaborator-proposed and PI-confirmed): ...  Consequence: ...
```

`notes/REVIEWS.md`
```
### R-<task id> reviewer: <model>  date:
Checked: <files, formulas, tests>  Findings: <list>  Verdict: PASS | PASS_WITH_FIXES | FAIL
```

**Per-task report format (every collaborator uses this):**
```
TASK <id> — status: DONE | READY_FOR_HUMAN_RUN | BLOCKED
Files created/changed: ...
Commands to run: ...
Evidence (verbatim output, first and last lines): ...
Tests: <names and result>
Assumptions I made: ...
Risks / things I am unsure about: ...
Next task proposed: ...
```

If collaborators disagree, state both positions with evidence in `DECISIONS.md` and let the PI decide.

---

## 10. Literature status and positioning

**Rule: nothing enters `references.bib` until the PI has opened the source (publisher, proceedings, arXiv or DOI page) and confirmed authors, year, venue and pages.** Citations reproduced from memory are unreliable and often plausible-looking but wrong. Mark every unverified item `TO_VERIFY`.

**Closest competitor papers — must be read in full before Gate 1** (both exist as arXiv preprints; neither is established as peer-reviewed):
- John (2025), *Fair and Explainable Credit-Scoring under Concept Drift: Adaptive Explanation Frameworks for Evolving Populations*, arXiv:2511.03807. Proposes adaptive SHAP variants benchmarked against static SHAP with rank/Jaccard stability. A *fix*, not an audit of fidelity, calibration or decision flips (based on abstract level reading; confirm author name order on the PDF).
- Nayak (2026), *Calibrated Credit Intelligence: Shift-Robust and Fair Risk Scoring with Bayesian Uncertainty and Gradient Boosting*, arXiv:2603.06733. A *method paper* on calibration and fairness under temporal shift (Home Credit data). Abstract-level reading shows no explanation or decision-flip analysis; confirm in full text.

**Anchor papers** (existence and venue to be confirmed by the PI): Ovadia et al. 2019 (NeurIPS); Marx, Calmon, Ustun 2020 (ICML); Miller et al. 2021 (ICML, "Accuracy on the Line"); Gardner, Popovic, Schmidt 2023 (NeurIPS Datasets & Benchmarks, TableShift); Krishna et al. 2024 (TMLR, disagreement problem); Guo et al. 2017 (ICML); Van Calster et al. 2019 (BMC Medicine); Lundberg & Lee 2017 (NeurIPS); Lundberg et al. 2020 (Nature Machine Intelligence); Fisher, Rudin, Dominici 2019 (JMLR); Gama et al. 2014 (ACM Computing Surveys); Lessmann et al. 2015 (EJOR); Saito & Rehmsmeier 2015 (PLOS ONE); Breiman 2001; D'Amour et al. 2022 (JMLR); Slack et al. 2020 (AIES).

**Corrections to an earlier draft's bibliography (already checked once, still re-verify):**
- "Hooker et al. 2021, ICML" is **wrong**. The paper is Hooker, Mentch & Zhou (2021), *Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance*, **Statistics and Computing 31(6): 82**, doi:10.1007/s11222-021-10057-z.
- "Hooker et al. 2019, *Investigations into feature permutation and feature importance*" could not be matched to a real paper. Treat as invalid until found.
- Kelodjou et al. 2024 (*Shaping up SHAP*, AAAI) and Subbaswamy & Saria 2020 and Sugiyama et al. 2007 were **not** independently confirmed. Look up before use.

**Regulatory context (motivation only, not technical prior art):**
- US model-risk guidance: on **17 April 2026** the Federal Reserve (**SR 26-2**), OCC (**Bulletin 2026-13**) and FDIC (**FIL-15-2026**) issued revised interagency Model Risk Management guidance that **supersedes SR 11-7 (2011) and SR 21-8**. Cite the primary Federal Reserve page (`federalreserve.gov/supervisionreg/srletters/SR2602.htm`). The letter states it is expected to be most relevant to banking organisations with over $30 billion in assets. Describe SR 11-7 only as the historical predecessor.
- EU AI Act: creditworthiness assessment of natural persons is listed as high-risk in Annex III, subject to scope and exceptions. Read the consolidated text on EUR-Lex; do not rely on summaries. Also consider ECOA/Regulation B adverse-action requirements (read the primary text).

**Novelty rule.** The candidate claim is: *no published study jointly audits calibration, explanation fidelity and decision flips on the same temporal cohorts of public credit data, with label-delay recalibration and a test of whether AUC predicts these.* This claim is **provisional**. It can enter the paper only after Gate 1 is passed from a completed literature matrix (40–60 papers, closest ones read in full). If overlap is found, narrow the claim (e.g., label-delay recalibration; outcome-justified flips; fidelity audit) rather than abandon the project. Never claim novelty on seed instability (Marx et al.) or on calibration degrading under shift in general (Ovadia et al.).

---

## 11. Writing rules for the manuscript

- **Placeholders, not typed numbers.** In `paper/manuscript.md` write `{{n:rq1.delta_ece.2015Q1.lightgbm}}`. `scripts/render_paper.py` substitutes from `paper/numbers.json`, which `scripts/export_numbers.py` builds from `reports/tables/`. If a placeholder has no value, rendering fails; do not paper over it.- **Claim register.** `paper/claims.csv` with columns `claim_id, sentence, evidence_file, number_keys, status(SUPPORTED|PARTIAL|UNSUPPORTED)`. No sentence making an empirical claim may exist without a row.
- **Structure:** Title · Abstract (about 200–250 words, written last) · Introduction · Related work (compare papers, do not list them) · Data and protocol · Methods (RQ1–RQ3) · Results (RQ1, RQ2, RQ3) · AUC analysis (RQ4) · Monitoring recommendations · Discussion · Limitations · Conclusion · Reproducibility statement.
- **Limitations must include:** accepted-loan selection and no rejected-applicant outcomes; one US platform and period; simulated (not observed) label-arrival times; possible repeated borrowers; SHAP assumptions; permutation extrapolation under correlated features; simulated thresholds and cost ratios; limited external validity.
- **Figures:** about 10–12 in the main text, each with n, CI method and cohort definition in the caption, produced by scripts from `metrics_long.parquet`. Colour-blind safe, readable at column width.
- **Tone:** precise, modest, specific. Report ranges and CIs. Distinguish confirmatory (frozen) from exploratory results in the text. Report null and inconclusive results at the same prominence as positive ones.
- **Banned phrases:** "novel framework" (unless true and earned), "state-of-the-art", "proves", "clearly shows", "revolutionary", "first-ever" (unless Gate 1 supports it), "SHAP is faithful/accurate".

---

## 12. Known risks and standard responses

| Risk | Standard response |
|---|---|
| Prior art overlaps | Narrow to what it misses; document in `GATE1.md` |
| Little measurable shift | Re-pick windows using descriptive statistics only, before freezing |
| Unresolved outcomes in late vintages | Shorten the test window; record in protocol |
| Leakage discovered late | Stop; fix column audit; rerun affected experiments; log in DEVIATIONS |
| Realistic-lag recalibration infeasible on early cohorts | Report feasibility table; run only feasible cells; state it as a finding |
| Seed variance is zero | Put `subsample`/`colsample` < 1 in search spaces; verify predictions differ across seeds |
| Underpowered AUC-vs-explanation test | Cheap SHAP (3,000 rows) for **all** configs; judge by CI against equivalence region |
| Compute limits | Parquet/DuckDB, cached predictions, fixed subsamples, Colab/Kaggle |
| H4 result is "AUC predicts everything" | Report it; show "all vs reasonable configs"; emphasise what still diverges |
| One dataset | Optional extension S10 (Home Credit) only after core is done; random-split replication is not a temporal replication |
| Scope creep or stalled progress | Apply the cut lines in `PLAN.md` Section 2 |

---

## 13. Glossary

- **Vintage / cohort:** loans originated in one calendar quarter.
- **Matured:** issue date + 36 months ≤ snapshot, with outcome resolved.
- **Calibration-in-the-large:** average predicted vs observed risk (intercept form). **Slope:** whether predictions are too extreme (<1) or too compressed (>1).
- **PSI:** population stability index, a descriptive drift measure, not a test.
- **Adversarial validation:** classifier trained to distinguish old from new rows using features only; high AUC means detectable covariate shift.
- **Noise floor:** variation in an explanation or metric caused only by seeds/bootstrap/sampling.
- **Flip:** the same loan gets a different decision under two model versions.
- **Oracle window:** a calibration window that would not be available in real time; an upper bound.
- **PROPOSED:** a parameter the PI has not yet confirmed.

---

## 14. Starter prompts for collaborator sessions

**First message:**
> I am attaching `CONTEXT.md` and `PLAN.md` for a research paper project. Read both fully. Reply with: (1) ten bullets summarising the project in your own words, (2) every ambiguity or contradiction you find, (3) which role you propose to take (Builder or Analyst-Writer) and the first task ID. Do not start any task until I confirm. Never invent numbers, citations or results; if you cannot run code, write it plus tests and mark the task READY_FOR_HUMAN_RUN.

**Per task:**
> Role: <Builder|Analyst-Writer>. Task: <ID from PLAN.md>. Current state: <paste notes/STATUS.md>. Relevant files: <paste or attach>. Follow CONTEXT.md Section 4 rules and answer in the Section 9 report format.

**Cross-review:**
> Review task <ID> completed by the other collaborator. Check against CONTEXT.md Section 7 formulas and Section 4 rules. Look specifically for leakage, test-cohort access, wrong formulas, hard-coded parameters, numbers typed by hand, and unsupported claims. Reply in the REVIEWS.md template.
