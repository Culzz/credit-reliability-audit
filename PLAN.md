# PLAN.md — Credit Reliability Audit: Execution Plan

Companion to `CONTEXT.md` (read that first; its rules and parameter table govern this plan).
Version 1.0 · 2 Nov 2026 · Task owners: **H** = PI (human), **B** = Builder role, **A** = Analyst-Writer role.

---

## 0. What this plan adds to the original blueprint

The original blueprint is kept in full. These changes fix gaps found on review. Each one is already built into the tasks below.

| # | Change | Why |
|---|---|---|
| I-1 | The two closest competitor papers (John 2025, arXiv:2511.03807; Nayak 2026, arXiv:2603.06733) are back in the Gate 1 comparison | The blueprint's table omitted the most direct prior art |
| I-2 | Regulatory text corrected: SR 26-2 (17 Apr 2026) supersedes SR 11-7; also OCC Bulletin 2026-13 and FDIC FIL-15-2026 | Verified on the Federal Reserve page; the paper must not present SR 11-7 as current |
| I-3 | Citation corrections (Hooker, Mentch & Zhou 2021 is in *Statistics and Computing*, not ICML; one "2019 Hooker" entry unmatched; several refs flagged TO_VERIFY) | Prevents wrong citations entering the bibliography |
| I-4 | **Test-cohort lock** enforced in code (`PROTOCOL_FROZEN` file) with a unit test | Makes the frozen protocol technically impossible to violate by accident |
| I-5 | **Plumbing sample** (3% of 2012–2013 rows) for debugging before the freeze | Pipeline development and testing work without touching test cohorts |
| I-6 | **Label-delay design made precise** (age-lag `a`, oracle vs optimistic vs realistic regions, feasibility rule) | The blueprint's delay design would put realistic windows before the training period on early cohorts; now handled explicitly |
| I-7 | **Loan-level flip utility** (per-flip ΔU, refer excluded from primary, matched null model) | "Share justified" was ambiguous; now precisely defined and testable |
| I-8 | **RQ4 power fix**: cheap SHAP stability for all configs; equivalence-region test; cluster bootstrap by configuration | 20 expensive configs give wide CIs; now the test is informative |
| I-9 | **Seed design resolved** (broad = 1 seed × 30 configs/family; deep = 5 configs × 20 seeds; stochastic search spaces) | Blueprint listed 5 seeds in one place and 20 in another; deterministic boosters make seeds meaningless |
| I-10 | **Degradation always relative to validation**; logistic regression as the natural calibration control | Separates baseline miscalibration from shift-induced degradation |
| I-11 | **Numbers pipeline**: manuscript placeholders, `numbers.json`, claim register | Drafted text cannot contain a hand-typed or invented number |
| I-12 | **Collaboration protocol**: STATUS, DECISIONS, REVIEWS, per-task report format, mandatory cross-review | Lets collaborators who share no session memory work without drift |
| I-13 | **Scope tiers, time budget and cut lines** | Protects the core paper if time runs short |
| I-14 | **Concrete default parameters** for every threshold the blueprint left open (marked PROPOSED) | Work can proceed; PI confirms at Gate 4/5 |
| I-15 | **Unit tests on synthetic calibrated data** for metric code | Catches implementation errors in slope/intercept/ECE before they reach results |
| I-16 | Axis B uses a **common post-training evaluation population** | Otherwise population dependence and reliance drift are confounded |

**Honest note on originality.** Nothing here guarantees a "one of a kind" paper. The design makes it rigorous, transparent and reproducible. Novelty is a claim to be earned at Gate 1 from a completed literature review, and the paper must be written so it stands on rigor even if the novelty claim has to be narrowed.

---

## 1. Scope tiers, time budget and cut lines

| Tier | Contents | Priority |
|---|---|---|
| **Core (must ship)** | Provenance, leakage audit, shift analysis, 4 model families, baseline sweeps, RQ1 incl. feasible label-delay cells, RQ4 core (AUC vs calibration, AUC vs future AUC), CIs, manuscript, repo | Never cut |
| **Flagship** | RQ2 three-axis stability + grouped permutation, RQ3 flips with outcome association and simulated utility, RQ4 vs explanation and flips, sensitivity S1–S6 | Target |
| **Extension** | Drop-group retraining at scale, conditional permutation, extra model families, S8–S10, subgroup/fairness analysis | Only if ahead of schedule |

**Estimated effort (hours) and calendar at about 20 h/week:**

| Phase | Hours | Cumulative weeks |
|---|---|---|
| 0 Infrastructure | 8 | 0.5 |
| 1 Literature and Gate 1 | 35 | 2 |
| 2 Data acquisition | 5 | 2.5 |
| 3 Sample, target, leakage audit | 25 | 3.5 |
| 4 Temporal feasibility and shift | 20 | 4.5 |
| 5 Protocol freeze | 15 | 5.5 |
| 6 Baselines and sweeps | 30 | 7 |
| 7 RQ1 calibration | 35 | 9 |
| 8 RQ2 explanations | 50 | 11.5 |
| 9 RQ3 decisions | 25 | 12.5 |
| 10 RQ4 AUC vs reliability | 20 | 13.5 |
| 11 Robustness | 20 | 14.5 |
| 12 Manuscript (drafting runs in parallel from week 4) | 50 | 16 |
| 13 Release | 10 | 16.5 |

Plan for **16–18 weeks**. The Dec 21 – Jan 3 holidays will slip things. Drafting related work (after Gate 1) and methods (after Gate 5) early keeps the end phase short.

**Cut lines (apply in this order if behind; record each in `DECISIONS.md`):**
1. Drop drop-group retraining (T8.6) and S8–S10.
2. Reduce RQ2 deep analysis to seeds + bootstrap + Axis A; report fidelity as exploratory with grouped permutation only.
3. Reduce sensitivity analyses to S1, S3, S5.
4. Reduce RQ3 to one policy (fixed threshold, L/g = 5) plus the null.
5. Never cut: leakage audit, frozen protocol, CIs, RQ1, RQ4 core, reproducibility.

**Scholarship timing.** The PI should list every deadline in `notes/STATUS.md`. Any deadline before the preprint can cite: repo link, `protocol-v1` tag, Gate 1 gap statement, shift report and the first RQ1 figures as "paper in preparation".

---

## 2. Gates

Do not start a phase until the previous gate is passed. A gate is passed only when its evidence exists in the repo and the PI signs off in `notes/STATUS.md`.

| Gate | After phase | Evidence |
|---|---|---|
| 0 | 0 | Repo pushed; env locks on a fresh clone; MLflow runs; tests run; lock guard test passes |
| 1 | 1 | Literature matrix (40–60 papers), closest papers read in full, competitor comparison, final gap statement, `GATE1.md` go/pivot |
| 2 | 2 | Dataset version, checksum, licence text, Parquet, row/column counts recorded |
| 3 | 3 | Sample definition, target, `column_audit.csv`, leakage tests passing |
| 4 | 4 | Cohort feasibility table, shift report, label-delay feasibility table, final windows chosen on descriptive evidence only |
| 5 | 5 | `PROTOCOL.md`, `HYPOTHESES.md`, all configs; supervisor feedback; tag `protocol-v1`; `PROTOCOL_FROZEN` created |
| 6 | 6 | Reproducible predictions and baseline report; reproducibility test passes |
| 7–10 | RQ1–RQ4 | Tables, figures and CIs for that RQ; cross-review PASS |
| 11 | 11 | Robustness summary table |
| 12 | 12 | Complete manuscript; claim register all SUPPORTED/PARTIAL; every number from `numbers.json` |
| 13 | 13 | Public-ready repo, fresh-clone reproduction passes, licences checked |

---

## 3. Tasks by phase

### Phase 0 — Infrastructure

- **T0.1 (H, B)** Create the repo `credit-reliability-audit` and the folder structure in `CONTEXT.md` §8 (private for now). Commit initial structure.
- **T0.2 (B)** Pin Python 3.11 with `uv`; add dependencies (pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, shap, statsmodels, duckdb, pyarrow, matplotlib, seaborn, pyyaml, tqdm, joblib, mlflow, kaggle; dev: pytest, ruff, jupyter). Commit `pyproject.toml` and `uv.lock`. *Done when* a fresh clone plus `uv sync` works.
- **T0.3 (B)** Create `configs/*.yaml` from the parameter table in `CONTEXT.md` §6 (all values PROPOSED, none hard-coded in code). `.gitignore` per blueprint (data, parquet, mlruns, tokens).
- **T0.4 (H, B)** Start MLflow and create experiments: `data-feasibility`, `baseline-models`, `temporal-evaluation`, `calibration`, `label-delay`, `explanation-stability`, `explanation-fidelity`, `decision-stability`, `auc-reliability`, `sensitivity-analysis`.
- **T0.5 (B)** **Test-cohort lock.** Implement `src/data/guard.py` and loader (spec in §4.1). *Done when* tests prove that requesting `role="test"` raises before `protocol/PROTOCOL_FROZEN` exists and that no module reads data except through the loader.
- **T0.6 (H)** Copy `CONTEXT.md` and `PLAN.md` into the repo root; create `notes/STATUS.md`, `DECISIONS.md`, `REVIEWS.md` from the templates; assign roles.
- **T0.7 (A)** Draft `protocol/RESEARCH_RISKS.md` from `CONTEXT.md` §12 with owner and trigger for each risk.
- **T0.8 (H)** Install Zotero; create collection `credit-shift-audit`. **(A)** create `lit/matrix.csv` with the 22 columns in the blueprint §1.4.

### Phase 1 — Literature and Gate 1 (target: 40–60 papers)

Analysis of papers must come from PDFs or source pages, never from memory.

- **T1.1 (A)** Write the search plan: the 11 query strings in blueprint §1.3, plus backward/forward citation steps and the databases (Google Scholar, Semantic Scholar, OpenAlex, Crossref, arXiv). **(H)** runs searches, saves results to Zotero, shares candidate titles.
- **T1.2 (H)** Verify each candidate on the publisher/arXiv/DOI page. Only verified items go in `matrix.csv` with `read_status`.
- **T1.3 (A)** Fill matrix rows from PDFs the PI supplies (never from memory). Mark unknown cells `UNKNOWN`.
- **T1.4 (H, A)** Read **in full** the two closest papers (John 2025; Nayak 2026) and the anchor papers most relevant to RQ2 and RQ4 (Krishna 2024; Gardner 2023; Miller 2021; Marx 2020; Ovadia 2019). Structured note per paper in `lit/reading_notes/`: what it does, data, shift design, metrics, what it does **not** do, limitations.
- **T1.5 (A)** Build the competitor comparison table (columns: calibration, explanation stability, fidelity, decision flips, label delay, AUC-reliability test, dataset, temporal design). Fill only from read papers.
- **T1.6 (A, H)** Write the narrative review by RQ, the unresolved methodological questions, and the final gap statement. Run the extra searches: "explanation stability temporal drift credit", "calibration under temporal shift credit scoring", "recalibration label delay", "decision flips model update lending", "LendingClub out-of-time", "AUC does not predict calibration under distribution shift tabular".
- **T1.7 (H)** Decide GO / PIVOT / STOP in `protocol/GATE1.md`.
  - GO if no study jointly covers at least two of (calibration, explanation fidelity, decision flips) under temporal shift on public credit data **and** tests AUC against them.
  - PIVOT (narrow) if one does.
  - STOP and rethink only if essentially all of RQ1–RQ4 is already covered.
- **T1.8 (A)** Regulatory background notes from **primary sources the PI supplies**: Federal Reserve SR 26-2 (page and PDF), OCC Bulletin 2026-13, FDIC FIL-15-2026, EU AI Act Annex III on EUR-Lex, Reg B adverse action. Describe SR 11-7 only as superseded predecessor.

### Phase 2 — Data acquisition

- **T2.1 (H)** Create Kaggle API token (follow Kaggle's current instructions; keep it outside the repo). Download `wordsforthewise/lending-club` into `data/raw`, unzip, locate `accepted_2007_to_2018Q4.csv.gz`.
- **T2.2 (H)** Record SHA-256 (PowerShell `Get-FileHash ... -Algorithm SHA256`), file size, access date, dataset version, licence text and LendingClub terms in `data/README.md`. If licence is unclear, note it and do not publish derived data.
- **T2.3 (B)** `src/data/load_raw.py`: DuckDB `read_csv_auto` to Parquet (ZSTD), removing footer/non-loan rows by numeric `id` check. Log rows removed. *Done when* shape and columns are printed and match expectations (about 2.26 M × 151; record any difference).
- **T2.4 (B)** `scripts/profile_dataset.py`: duplicates, missingness, dtypes, status counts, `term` representation, `issue_d` range, numeric ranges, odd categories. Output `reports/tables/data_profile.csv`.

### Phase 3 — Sample, target and leakage audit

- **T3.1 (B)** Inventory exact `loan_status` strings and counts. **(H)** decides the treatment of legacy "Does not meet the credit policy" statuses; record in `DECISIONS.md`.
- **T3.2 (B)** Filter to 36-month loans (check the stored string). Build `issue_date`, `vintage_quarter`. Validate parsing against raw values.
- **T3.3 (B)** Build target (Charged Off/Default = 1, Fully Paid = 0); exclude unresolved statuses.
- **T3.4 (B)** **Maturity table** per vintage: total loans, Fully Paid, Charged Off/Default, unresolved, unresolved share `U_q`, default rate, mature flag (issue + 36 m ≤ snapshot **and** `U_q` below threshold). Output `reports/tables/vintage_maturity.csv`.
- **T3.5 (B draft, H review)** `data/column_audit.csv` for **every** column: `column, dtype, missing_share, min_vintage_missing_share, max_vintage_missing_share, available_at_origination, availability_evidence, leakage_risk, decision (KEEP|DROP|SENSITIVITY_ONLY|SPLIT_ONLY|TARGET), reason`. Use the starting lists in `CONTEXT.md` §5 as hypotheses only. Evidence for availability must cite the data dictionary (community dictionary on figshare plus the Kaggle description); anything uncertain is flagged `uncertain`.
- **T3.6 (B)** Feature engineering: credit age = `issue_date − earliest_cr_line` (check negatives), utilisation as numeric, employment length encoding, missingness indicators, categorical encoding. All fit on training data only (pipeline object). Do **not** use `issue_d` as a predictive feature.
- **T3.7 (B)** Schema-drift check: missing share per feature per vintage; apply the `min_populated_share` rule; list dropped features and reasons.
- **T3.8 (B)** `tests/test_leakage.py` and friends: no forbidden column in the feature matrix; no target in features; every feature has an audit decision; imputers/encoders fit on training only; identical feature schema across cohorts; no future dates in training.
- **T3.9 (B)** **Plumbing sample:** a stratified-by-vintage 3% random sample of 2012–2013 rows, fixed seed, saved separately. All pipeline development before Gate 5 uses only this sample.
- **T3.10 (A)** Write `GATE3.md`: final sample definition, exclusions, feature dictionary.

### Phase 4 — Temporal feasibility and shift (no final models yet)

- **T4.1 (B)** Cohort feasibility table (vintage × counts × default rate × unresolved × mature). Plots: loan volume and default rate by quarter. *Check against the rule* of ≥ 200 events and ≥ 200 non-events per evaluation cohort and aim for ≥ 6 test cohorts.
- **T4.2 (B)** Shift measures vs the training window, per vintage: PSI per feature (state binning), KS for continuous features, standardised mean differences, categorical proportion changes, Jensen–Shannon where suitable, default-rate shift with CI. Plots: PSI heatmap, missingness heatmap, feature distributions. PSI is descriptive; do not cite thresholds as laws.
- **T4.3 (B)** Adversarial validation with features only (no ids, no dates, no leakage): train vs each later cohort; time-aware held-out evaluation; report AUC. Output ROC curves.
- **T4.4 (B)** Concept-drift diagnostics that need no final model (e.g., default rate across feature strata by vintage). Do not claim concept drift from performance changes alone.
- **T4.5 (B)** **Label-delay feasibility table** (blueprint §9.6, using the age-lag design in `CONTEXT.md` §6): for each test cohort S and lag a, eligible loans, events, non-events, FEASIBLE/INFEASIBLE under the min-events rule for each recalibrator.
- **T4.6 (B, H)** Compute pilot on the plumbing sample: one config per family, one small SHAP run, one bootstrap, one recalibration. Log runtime and peak memory in `reports/tables/compute_pilot.csv`. *Purpose*: set sample caps for SHAP and bootstrap.
- **T4.7 (A, H)** Shift adequacy memo `protocol/GATE4.md`: chosen windows, why, evidence. If shift is negligible, re-pick windows now using descriptive statistics only. **Never choose windows by looking at model results.**

### Phase 5 — Protocol freeze

- **T5.1 (A)** Write `protocol/PROTOCOL.md` with all 33 headings from blueprint §7.1, filled with confirmed values.
- **T5.2 (A, H)** `protocol/HYPOTHESES.md`: each hypothesis with test statistic, direction (only if justified), threshold, practical-effect size, confirmatory vs exploratory. Include the equivalence region for H4 and the Holm family.
- **T5.3 (B)** Finalise `configs/*.yaml` (windows, seeds, sweep ranges, explanation sizes, thresholds).
- **T5.4 (A, H)** Define **feature groups** (domain groups + clusters at |ρ| ≥ 0.8) using training data only, **before** any explanation results exist. Save as `configs/feature_groups.yaml`.
- **T5.5 (B)** Hyperparameter search spaces for all four families, with stochastic parameters (`subsample`, `colsample_*`) below 1 so seeds change predictions. Write into the protocol.
- **T5.6 (B)** Dry-run every `scripts/run_*.py` end to end on the plumbing sample. Fix bugs now; after the freeze, code fixes that change results must be logged as deviations.
- **T5.7 (H)** Send the supervisor a half-page summary plus the protocol; ask about supervision and co-authorship. Record feedback in `DECISIONS.md`.
- **T5.8 (H)** Freeze:
  ```powershell
  git add -A
  git commit -m "Freeze research protocol v1"
  git tag -a protocol-v1 -m "Frozen protocol before final model evaluation"
  git push origin main --tags
  Set-Content protocol/PROTOCOL_FROZEN (git rev-parse protocol-v1)
  git add protocol/PROTOCOL_FROZEN; git commit -m "Unlock test cohorts"; git push
  ```
  Optionally register on OSF. Collaborators never create `PROTOCOL_FROZEN`.

### Phase 6 — Baselines and configuration sweeps

- **T6.1 (B)** Preprocessing pipeline (fit on training window only) saved with its version.
- **T6.2 (B)** Training harness: logistic regression (regularised), random forest, XGBoost, LightGBM; natural class distribution; no resampling; seeds from config.
- **T6.3 (B, H runs)** Broad sweep: 30 configs/family (include weak and over-fitted), 1 seed each; tune on training/validation only; log everything. Verify prediction variability across seeds on the deep subset.
- **T6.4 (B)** Pick the **headline** configuration per family by lowest validation log loss. Pick the **deep subset**: 5 configs/family at validation-AUC quantiles.
- **T6.5 (B)** Write `predictions.parquet` (schema `CONTEXT.md` §8) for validation and every test cohort; metrics (ROC-AUC, PR-AUC, Brier, log loss, accuracy, precision, recall, F1, balanced accuracy) with bootstrap CIs into `metrics_long.parquet`.
- **T6.6 (A)** Baseline report: model comparison, ROC/PR curves, performance by cohort, degradation vs validation. Flag any AUC above about 0.80 as a leakage alarm to investigate.
- **T6.7 (B)** `tests/test_reproducibility.py`: re-run one small config, metrics match within tolerance; a fresh-clone check.

### Phase 7 — RQ1 calibration and label delay

- **T7.1 (B)** `src/calibration/metrics.py` (spec in §4.2) with unit tests on synthetic data: (i) perfectly calibrated simulator gives slope ≈ 1, intercept ≈ 0, ECE near 0; (ii) a deliberately shifted simulator gives the expected intercept sign. **(A)** reviews formulas against `CONTEXT.md` §7.
- **T7.2 (B)** Degradation: for each model and cohort compute intercept, slope, A/E, ECE (B = 15, plus 10/20/30), Brier, log loss, with bootstrap CIs; then `Δ` vs validation. Logistic regression serves as the calibration control.
- **T7.3 (B)** Recalibration arms: none, intercept-only, Platt (logistic on logit p), isotonic. Fit only on eligible historical labels; never on the evaluation cohort (assert in code).
- **T7.4 (B)** Label-delay experiment: arms × windows (static; latest with lag a ∈ {0, 6, 12, 18, 24, 36}; oracle a = 0). Run only cells marked FEASIBLE in T4.5. Label results `oracle`, `optimistic (a<36)` and `realistic (a≥36)`. Plot recovery vs lag.
- **T7.5 (B)** Calibration sample-size experiment: subsample calibration sets (sizes from actual counts) and compare Platt vs isotonic (H1d).
- **T7.6 (B)** Decompose degradation into intercept (base-rate) vs slope (ranking) components; report window sample sizes.
- **T7.7 (B, A)** Outputs: calibration curves per cohort, CITL and slope over time, ECE and Brier degradation, recovery by method, delay vs recovery, sample size vs performance; tables of metrics per model/cohort/arm; paired CIs for method comparisons. **(A)** drafts the RQ1 results paragraph only from the tables, with placeholders.
- **Gate 7 review:** cross-review PASS.

### Phase 8 — RQ2 explanation stability and fidelity

- **T8.1 (B)** SHAP infrastructure: TreeSHAP (interventional with the fixed 1,000-row background; path-dependent as robustness) for RF/XGB/LGBM, exact linear SHAP for logistic regression, log-odds scale, fixed feature order, fixed explanation samples per cohort. Cap RF depth/trees if runtime requires (document).
- **T8.2 (B)** **Axis C (noise floor):** on one fixed cohort, retrain the deep-subset configs with 20 seeds and 30 bootstrap resamples; compute all pairwise stability metrics; save the noise-floor distributions.
- **T8.3 (B)** **Axis A (population dependence):** one frozen model per family; explanations on every test cohort, same background; compute global importance, ranks, Jaccard@k, Spearman, Kendall, cosine, attribution-distribution distances.
- **T8.4 (B)** **Axis B (reliance drift):** models M1..Mk trained on expanding windows; all scored on the **common post-training evaluation population**; compare SHAP profiles and permutation importances across versions.
- **T8.5 (B)** Feature groups (from `configs/feature_groups.yaml`): group SHAP importance (mean |sum of group SHAP|), **grouped permutation importance** (joint row-permutation of the whole group, R = 20 repeats, Δ log loss and Δ AUC). Note extrapolation risk with correlated features (Hooker, Mentch & Zhou 2021) in the methods.
- **T8.6 (B, stretch)** **Drop-group retraining:** retrain without each group on the same window, evaluate on the same cohort, compare to SHAP and permutation. Run on the deep subset only. First thing cut if behind.
- **T8.7 (B)** **Broad cheap stability for all configs:** 3,000-row explanation sample per cohort, fixed background; compute stability vs the same config's Axis-C-style bootstrap refits where feasible, otherwise vs the family headline model; store per-config stability for RQ4.
- **T8.8 (A, B)** Outputs: SHAP global importance by cohort, Jaccard heatmap, Spearman matrix, attribution distributions, reliance drift plot, noise-floor vs observed difference, SHAP vs permutation vs drop-group; tables of stability, noise floor, fidelity alignment with CIs. **(A)** writes results text with the careful phrasing in `CONTEXT.md` §7 and §11.
- **Gate 8 review:** cross-review PASS; confirm the three axes are reported separately.

### Phase 9 — RQ3 decision stability

- **T9.1 (B)** Implement policies from §4.3: fixed-threshold three-way policy with p* = g/(g+L), L/g ∈ {3, 5, 9}, refer band ±0.02 (sensitivity ±0.01, ±0.04); fixed-approval-rate policy. Thresholds derived from validation and written to config.
- **T9.2 (B)** Flip comparisons on **identical loans**: (a) stale vs refreshed model; (b) uncalibrated vs recalibrated; (c) fixed model vs rolling-retrained; (d) seed vs seed (noise floor).
- **T9.3 (B)** Flip metrics: overall rate, 3×3 transition matrix, probability change, distance to nearest threshold, asymmetry of approve→decline vs decline→approve.
- **T9.4 (B)** Outcome association: default rate by flip direction vs p*; flipped vs non-flipped comparison; per-flip ΔU and total ΔU; matched-null distribution (1000 draws); bootstrap CIs. Refer moves reported separately.
- **T9.5 (B)** Sensitivity over L/g, refer band, policy type.
- **T9.6 (A, B)** Outputs: flip matrix, flip rate by cohort and probability band, probability-change distribution, default rate by flip direction, utility sensitivity. State plainly in text that outcomes exist only for accepted loans and utilities are stylised.
- **Gate 9 review:** cross-review PASS.

### Phase 10 — RQ4 AUC versus reliability

- **T10.1 (B)** Build the configuration-level table (`configuration_id, family, parameters, seed, train_window, eval_cohort, roc_auc, pr_auc, brier, log_loss, ece, slope_dev, intercept_abs, shap_stability, permutation_alignment, flip_rate`). Join from `metrics_long` and the T8.7/T9.2 outputs.
- **T10.2 (B)** Primary estimates: within-cohort Spearman ρ (AUC vs ECE, vs |slope−1|, vs explanation instability, vs flip rate), averaged across cohorts, with **cluster bootstrap over configurations** (B = 2000) CIs. Also AUC (validation) vs AUC (future cohort).
- **T10.3 (B)** Mixed model (if the sample supports it): `reliability ~ AUC_centred + family + cohort + (1 | config)`; sensitivity excluding each family.
- **T10.4 (B)** Rank-reversal rate; "all configs" vs "reasonable configs" analysis; family-specific plots.
- **T10.5 (B)** Classify each primary result as negligible / meaningful / inconclusive against the equivalence region; apply Holm over the primary family.
- **T10.6 (A)** Outputs and text: scatter plots with CIs, correlation table, regression table, multiplicity-adjusted table. Report inconclusive results as inconclusive.
- **Gate 10 review:** cross-review PASS.

### Phase 11 — Robustness and sensitivity (each a labelled analysis)

Priority order: **S1, S3, S5, S2, S4, S6, S7** (Flagship), then **S8–S10** (Extension).

- **S1** Add `grade`, `sub_grade`, `int_rate`, `installment`: effect on explanations and decisions (shows the cost of letting the platform's own score in).
- **S2** Alternative defensible temporal window (decided in advance, not for effect).
- **S3** Unresolved-share threshold 0.02 vs 0.01.
- **S4** Alternative calibration windows and regularised recalibration.
- **S5** Explanation settings: sample sizes, background choices, top-k, feature grouping thresholds, permutation variants, ECE bins.
- **S6** Decision thresholds and cost ratios.
- **S7** Class-weight arm.
- **S8** 60-month loans: **only** if the feasibility audit shows usable test cohorts after the training window (with a 2018Q4 snapshot this is likely impossible; do not force it).
- **S9** One extra model family (e.g., CatBoost) only if it answers a specific robustness question.
- **S10** External dataset (e.g., Home Credit) as a separately labelled generalisation analysis; a random-split replication is **not** temporal.
- **T11.1 (B)** Run chosen sensitivities. **(A)** builds `reports/tables/robustness_summary.csv` (primary result, alternative configuration, result with CI, direction of change, whether the conclusion changes).
- **Gate 11:** conclusions do not depend on a single arbitrary choice, or the dependency is stated.

### Phase 12 — Manuscript

Drafting starts early and runs in parallel; text depends only on gated outputs.

- **T12.1 (B)** `scripts/export_numbers.py` and `scripts/render_paper.py` (§4.5). *Done when* a missing placeholder makes rendering fail.
- **T12.2 (A)** Create `paper/claims.csv` and keep it current; no empirical sentence without a row.
- **T12.3 (A)** Related work (after Gate 1) comparing papers by RQ, using only read papers and PI-verified references.
- **T12.4 (A)** Data and methods (after Gate 5) matching `PROTOCOL.md` exactly; deviations disclosed.
- **T12.5 (A)** Results sections as each RQ gate passes (placeholders only).
- **T12.6 (A)** Introduction, discussion (empirical findings vs interpretations), monitoring recommendations (what to track beyond AUC), limitations (list in `CONTEXT.md` §11), conclusion that answers: what AUC tells us, what it does not, how calibration changed, how stable explanations were, what happened to simulated decisions, what practitioners should monitor. No claim of universal rules.
- **T12.7 (A)** Abstract last, from the finished results.
- **T12.8 (B, A)** Cross-review of the full draft: numbers vs `numbers.json`, formulas vs `CONTEXT.md` §7, language vs rules, figures vs captions.
- **T12.9 (H)** Supervisor review; incorporate feedback with records in `DECISIONS.md`.
- **T12.10 (H)** Verify every reference against its source; resolve all `TO_VERIFY`; proofread.
- **T12.11 (H)** Choose the venue from its current call-for-papers/aims page (candidate types: applied ML/decision-support journals, finance journals, trustworthy-ML or ML-in-finance workshops, TMLR). Check article type, length, data/code rules, charges. Do not pay fees for promises of fast acceptance. Format accordingly (Markdown + BibTeX via pandoc, or LaTeX template).

### Phase 13 — Release

- **T13.1 (A)** README with the 22 sections in blueprint §19.1, honest limitations, reproduction instructions.
- **T13.2 (B, H)** Fresh-clone reproduction: `uv sync` → data download instructions → `run_data_audit` → `run_baselines` → `run_full_experiments` → `export_results`. Fix undocumented steps.
- **T13.3 (H)** Release review: no credentials, no raw data, no personal information, licences checked, third-party licences listed. Add software `LICENSE`.
- **T13.4 (H)** `CITATION.cff`; tagged release archived on Zenodo for a DOI.
- **T13.5 (H)** Preprint on arXiv (check the current endorsement policy early) or OSF; keep the distinction between preprint and peer-reviewed publication clear on the CV.
- **T13.6 (H, A)** Final QA checklist (§8).

---

## 4. Implementation specifications

### 4.1 Test-cohort lock (`src/data/guard.py`)

```python
from pathlib import Path

FROZEN_FLAG = Path("protocol/PROTOCOL_FROZEN")

class ProtocolLockError(RuntimeError):
    pass

def require_unlocked(role: str) -> None:
    """Test cohorts are readable only after the PI has frozen the protocol."""
    if role == "test" and not FROZEN_FLAG.exists():
        raise ProtocolLockError(
            "Test cohorts are locked until protocol-v1 is tagged and "
            "protocol/PROTOCOL_FROZEN exists (created by the PI only)."
        )
```
`load_split(role)` reads `configs/splits.yaml`, calls `require_unlocked`, and returns only rows in that role's vintages. Descriptive shift code uses a separate function `load_descriptive()` that returns *features and counts only* (no model scoring) and is tested to expose no prediction columns.

### 4.2 Calibration and stability metrics (reference implementations)

```python
import numpy as np, statsmodels.api as sm
EPS = 1e-6
def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def calibration_intercept(y, p):            # calibration-in-the-large (slope fixed at 1)
    X = np.ones((len(y), 1))
    return float(sm.GLM(y, X, family=sm.families.Binomial(), offset=_logit(p)).fit().params[0])

def calibration_slope(y, p):
    X = sm.add_constant(_logit(p))
    return float(sm.GLM(y, X, family=sm.families.Binomial()).fit().params[1])

def ece_adaptive(y, p, n_bins=15):
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p); y, p = y[o], p[o]
    bins = np.array_split(np.arange(len(p)), n_bins)
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))

def ae_ratio(y, p): return float(np.mean(y) / np.mean(p))

def jaccard_topk(a, b, k):
    A, B = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(A & B) / len(A | B)
```
Handle separation/non-convergence in GLM fits gracefully (report NaN with a flag, never silently drop).

### 4.3 Decisions and flips

```python
import numpy as np
APPROVE, REFER, DECLINE = 0, 1, 2
def decide(p, t_low, t_high):
    p = np.asarray(p)
    return np.where(p < t_low, APPROVE, np.where(p < t_high, REFER, DECLINE))

def flip_gain(y, dA, dB, g, L):
    """Per-loan simulated gain of moving from decision A to decision B.
    approve->decline: L*y - g*(1-y);  decline->approve: g*(1-y) - L*y;  others: 0 (refer excluded)."""
    y, dA, dB = map(np.asarray, (y, dA, dB))
    gain = np.zeros(len(y), float)
    a2d = (dA == APPROVE) & (dB == DECLINE)
    d2a = (dA == DECLINE) & (dB == APPROVE)
    gain[a2d] = L * y[a2d] - g * (1 - y[a2d])
    gain[d2a] = g * (1 - y[d2a]) - L * y[d2a]
    return gain
```
Break-even `p* = g/(g+L)`; with L/g = 3, 5, 9 → p* = 0.25, 0.167, 0.10. Null model: for each flip direction, sample the same number of non-flipped loans matched on decision under A and on A-score within ±0.01, compute the same total gain, 1000 repeats.

### 4.4 Cluster bootstrap (RQ4)

```python
import numpy as np, pandas as pd
def cluster_bootstrap_ci(df, cluster_col, stat_fn, B=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    groups = {c: g for c, g in df.groupby(cluster_col)}
    ids = np.array(list(groups))
    stats = [stat_fn(pd.concat([groups[c] for c in rng.choice(ids, len(ids), replace=True)]))
             for _ in range(B)]
    return np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
```
`stat_fn` computes Spearman within each cohort and averages across cohorts.

### 4.5 Numbers pipeline

1. Scripts write tidy tables to `reports/tables/`.
2. `export_numbers.py` flattens selected cells into `paper/numbers.json` with stable keys like `rq1.delta_ece.2015Q1.lightgbm.value`, `.ci_low`, `.ci_high`.
3. The manuscript contains placeholders such as `{{n:rq1.delta_ece.2015Q1.lightgbm.value}}`.
4. `render_paper.py` substitutes and **fails** on any unresolved placeholder.
5. `claims.csv` links each claim to its evidence file and keys.

---

## 5. Experiment matrix and compute budget

| Block | Scope | Heaviness | Fallback |
|---|---|---|---|
| Broad sweep | 4 families × 30 configs × (validation + all test cohorts), 1 seed | Medium | Cache predictions; fewer configs for random forest if slow |
| Bootstrap CIs | B = 1000 per metric per cell (2000 headline) | Medium | Vectorise; reduce B for non-headline cells and say so |
| Deep subset | 5 configs/family × 20 seeds | Medium | Reduce to 10 seeds if runtime blocks; state in protocol before freeze |
| SHAP broad | all configs × cohorts × 3,000 rows | Medium | Linear exact for LR; cap RF size |
| SHAP deep | deep subset × 20,000 rows × cohorts | Heavy | Drop to 10,000 rows; run on Colab/Kaggle |
| Grouped permutation | R = 20 per group | Medium | R = 10 |
| Drop-group retraining | groups × deep subset | Heavy | Stretch; first cut |
| Flips and null | policies × comparisons × 1000 null draws | Light–medium | Vectorise |

Set caps from the T4.6 pilot and **write them into the protocol before the freeze**. Do not tune compute caps after seeing results.

---

## 6. Statistical analysis summary

- Bootstrap percentile CIs; paired resampling for model/arm comparisons (same loans for both).
- Keep uncertainty sources separate: training seed, bootstrap resample, explanation sample, cohort variation, configuration variation.
- Cohorts share training data and are not independent; say so and avoid p-value inflation language.
- Practical effect sizes are defined before test evaluation.
- Holm correction over the primary family (CONTEXT §3); everything else exploratory.
- Report negative and inconclusive results.
- Borrower-clustered inference is **not** claimed (no reliable borrower id).

---

## 7. Manuscript map

| Section | Source | Written when |
|---|---|---|
| Introduction | Gate 1 gap statement, regulatory notes | After Gate 1 (finalise last) |
| Related work | `lit/matrix.csv`, reading notes | After Gate 1 |
| Data and protocol | `PROTOCOL.md`, `column_audit.csv`, `vintage_maturity.csv`, shift report | After Gate 5 |
| Methods RQ1–RQ3 | `PROTOCOL.md`, `CONTEXT.md` §7 | After Gate 5 |
| Results RQ1 / RQ2 / RQ3 / RQ4 | `reports/tables`, `numbers.json` | After Gates 7 / 8 / 9 / 10 |
| Robustness | `robustness_summary.csv` | After Gate 11 |
| Discussion, limitations, conclusion | all above | After Gate 11 |
| Abstract | finished paper | Last |

**Figures (about 10–12, generated by scripts):** F1 framework diagram; F2 cohort characteristics; F3 shift assessment; F4 performance under shift; F5 calibration degradation; F6 label-delay recalibration; F7 population-dependent explanations; F8 stability vs noise floor; F9 SHAP vs reliance; F10 decision transitions; F11 AUC vs reliability; F12 consolidated reliability audit.

**Tables:** T1 dataset; T2 sample and cohorts; T3 feature audit summary; T4 model configurations; T5 baseline performance; T6 calibration; T7 recalibration and label delay; T8 explanation stability; T9 SHAP vs reliance alignment; T10 flips; T11 AUC–reliability associations; T12 robustness.

---

## 8. Final quality-assurance checklist

**Design:** gap supported by completed review · RQs and hypotheses defined · protocol frozen before confirmatory evaluation · deviations logged · confirmatory vs exploratory labelled.
**Data:** provenance, licence and checksum recorded · target verified · 36-month maturity rule implemented · cohorts validated · every column audited · no post-origination leakage · preprocessing fit on training only.
**Experiments:** temporal evaluation throughout · configs recorded · validation-only selection · test cohorts untouched until freeze · calibrators fit on eligible labels only · label-delay assumptions stated · three explanation axes separate · counterfactual limits acknowledged.
**Statistics:** CIs everywhere · paired comparisons · cluster bootstrap by configuration · multiplicity handled · practical effects considered · sensitivity done · negative results reported.
**Release:** all references verified by the PI · figures from scripts · tables match `numbers.json` · limitations explicit · supervisor feedback incorporated · no credentials or raw data · reproduction works from a fresh clone · licences and `CITATION.cff` present.

---

## 9. First ten actions

1. **H** Create the GitHub repo and folders (T0.1); copy `CONTEXT.md` and `PLAN.md` into it.
2. **H** Paste the starter prompt (`CONTEXT.md` §14) into collaborator sessions; confirm roles.
3. **B** Environment, configs and test-cohort lock (T0.2, T0.3, T0.5).
4. **H** Zotero and matrix setup (T0.8); list scholarship deadlines in `notes/STATUS.md`.
5. **H** Email the supervisor (half-page summary); ask about supervision/co-authorship.
6. **A, H** Start the literature search plan and read the two competitor papers in full (T1.1, T1.4).
7. **H** Kaggle token and dataset download, checksum, licence (T2.1, T2.2).
8. **B** Parquet conversion and profiling (T2.3, T2.4).
9. **B** Status inventory, 36-month filter, target, maturity table (T3.1–T3.4).
10. **Pass Gate 0, Gate 1 and Gate 2 before building any model.**
