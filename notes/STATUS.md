# notes/STATUS.md — working status (update at the end of every work session)

## Current phase
Phase 0 closing (Gate 0 awaits the Zotero collection only). Phase 1 literature
review started (matrix seeds + 2 competitor notes). Phase 2 data acquisition
complete through T2.4 (raw gz → Parquet → profile tables).

## Responsibilities on this project
- Implementation and testing: dataset access, features, models, metrics, explanations, decision code, run scripts (Builder duties).
- Literature matrix and reading notes, protocol text, statistical analysis plan, manuscript drafting, claim register (Analyst-Writer duties).
- Heavy compute, citation verification, parameter confirmations, protocol freeze, all external contact: me (PI).

Every substantive change is reviewed by the other collaborator before it counts (`notes/REVIEWS.md`).

## Completed 2026-11-02
- Repository structure and working documents (initial commit), decisions log opened.
- Eight PROPOSED configuration files from the parameter table (`configs/*.yaml`); nothing hard-coded in code.
- Dataset access guard `src/data/guard.py` + split loader `src/data/loader.py`; **18/18 tests passing**, ruff clean.
- Python 3.11.9 environment locked with uv: 209 packages in `uv.lock` (pandas, sklearn, xgboost, lightgbm, shap, statsmodels, duckdb, pyarrow, mlflow, ...).
- Logging directory renamed to `notes/` (three logging templates re-worded; obligations unchanged).
- Kaggle dataset verified with API file listing (`accepted_2007_to_2018Q4.csv.gz`, 392,582,231 bytes — matches the blueprint reference); accepted-loans file downloaded.

## Gate 0 evidence
Repo pushed · environment locked · 18/18 tests pass · access-guard test proven ·
ten MLflow experiment families registered locally (ids 1–10; table identical to
the plan's list — file store `mlflow.db`, gitignored) · risks register completed
with owners and triggers. Still missing for my sign-off: the Zotero collection
(T0.8, desktop app work), then Gate 0 can close.

## Session 2, 2 Nov 2026 (afternoon)
- T0.4 done — ten experiment families in MLflow: data-feasibility,
  baseline-models, temporal-evaluation, calibration, label-delay,
  explanation-stability, explanation-fidelity, decision-stability,
  auc-reliability, sensitivity-analysis.
- T0.7 done — `protocol/RESEARCH_RISKS.md` filled: 12 risks, owner + trigger each.
- T0.8 partially — `lit/matrix.csv` (22 columns, 15 seed rows: 2 competitors +
  13 anchors), `lit/search_plan.md` with the 11 query strings and priority queue.
  Zotero pending (desktop app). All pre-Gate-1 entries `read_status=UNKNOWN`
  or `abstract`; expands to 40–60 verified papers before Gate 1.
- T1.1 started — both closest competitors verified on their arXiv pages:
  John 2025 (2511.03807, submitted 5 Nov 2025, CC BY 4.0) and Nayak 2026
  (2603.06733, submitted 6 Mar 2026, CC BY-NC-ND). Reading notes written at
  abstract level; **full-text reads are REQUIRED before Gate 1**.
- T2.2/T2.3/T2.4 done — dataset downloaded (checksum in `data/README.md`),
  converted to `data/interim/accepted_raw.parquet`, profiled into the five
  `reports/tables/*.csv` summary tables.
- T3.1 inventory done (counts table); T3.2/T3.3 modules implemented and tested
  (`src/data/prepare.py`, 26 tests); T3.4 maturity table built
  (`reports/tables/vintage_maturity.csv`, logged to MLflow data-feasibility):
  36-month loans 1,609,754; resolved 1,023,206; **mature window with
  U_q < 1%: 2007Q2 – 2015Q4 (35 quarters)**, all quarterly issues through
  2015Q4 complete inside the 2018-12-31 snapshot. Legacy prefix exists only
  through 2010 vintages (as expected); proposal D-3 open for PI confirmation.

## Dataset facts from the profile (2 Nov 2026)
- 2,260,701 CSV rows read; 33 summary/footer text rows removed; **2,260,668
  loans × 151 columns**, 0 duplicate ids.
- `issue_d` range: **Jun-2007 … Dec-2018, 139 distinct months.** An earlier
  profile printout said "Apr-2008": that was string `min()/max()` on
  'Mmm-YYYY' text, not dates; fixed by parsing in the profile script (flagged
  by the June-2007 vintages in the T3.4 maturity run). Distinct-month count
  unchanged.
- `loan_status`: Fully Paid 1,076,751 · Current 878,317 · Charged Off 268,559 ·
  Late(31-120) 21,467 · In Grace 8,436 · Late(16-30) 4,349 · legacy-policy
  Fully Paid 1,988 · legacy-policy Charged Off 761 · Default 40. The legacy
  "Does not meet the credit policy" variants exist → T3.1 decision is live.
- `term`: 36 months 1,609,754 · 60 months 650,914 (main sample = 36 m only).
- Fully-null column: `member_id` (100%). ~99.5% null: all hardship columns and
  `orig_projected_additional_accrued_interest` → keep out of the feature audit
  unless a risk subsystem needs hardship flags.

## Session 3, 3 Nov 2026
- T1.2 continued — anchor bibliographic records verified against publisher
  pages and committed (`lit/matrix.csv` rows re-pointed to Crossref DOIs,
  PMLR / NeurIPS proceedings pages); TableShift corrected to **15 binary
  tasks** (arXiv 2312.07577; my previous 17 was wrong); Ovadia proceedings
  hash + arXiv 1906.02530 recorded. The last two UNKNOWN URL cells resolved:
  Lundberg & Lee 2017 verified in the official NeurIPS proceedings index
  (title+authors); D'Amour 2022 verified in the JMLR volume-23 index
  (paper 226:1-61, canonical title corrected). No UNKNOWN doi_or_url cells
  remain in `lit/matrix.csv`.
- T3.5 done (two passes) — `data/column_audit.csv`, **151 columns**: first
  pass computed missing_share + per-vintage min/max missingness for every
  column (`scripts/build_column_audit.py`); second pass filled the judgment
  fields (`scripts/curate_column_audit.py`): **24 KEEP · 47 DROP · 79
  SENSITIVITY_ONLY · 1 TARGET**, every column classified (no leftovers).
  Schema-drift classes found: (a) mid-window fields populated only for a
  contiguous vintage sub-window (open_acc_6m… all_util ≈ 38% missing,
  min 0.0 / max 1.0); (b) late-adopted bureau attributes ≈ 2016-onwards
  (num_*, mo_sin_*, tot_*, bc_*, mort_acc, …); (c) joint-application-only
  fields (~95% missing). Post-origination payment/recovery/hardship/settlement
  block dropped as leakage. `emp_title`/`addr_state`/`disbursement_method`/
  `funded_amnt*` parked as SENSITIVITY_ONLY with uncertain evidence — the
  PI reviews at protocol freeze (D-4 candidate).

- T3.6–T3.10 done in one session — `src/features/build.py` (audited-source
  guard, credit age, ordinal/cleaning, one-hot, missingness flags; synthetic
  tests), `scripts/check_schema_drift.py` → `reports/tables/schema_drift.csv`
  (12 of 24 KEEP features dropped by the ≥0.90 rule; 9 fail only in the
  2007Q2 micro-vintage; 3 months-since-event fields fail ~everywhere — D-5
  proposal for the PI), `tests/test_leakage.py` (41 tests total green),
  `scripts/build_plumbing_sample.py` (3% of 2012–2013 by vintage, fixed
  seed; 143,892 → 4,317 rows, smoke-transformed to 52 engineered features),
  `protocol/GATE3.md` drafted (PROPOSED). Decisions D-4/D-5 logged for PI.

## Next
- T4.1+: Phase 4 baseline pipeline — load the plumbing sample through
  `src/data/loader.py`, train the baseline logistic/GBM stack, run the
  calibration and stability audit machinery on it (all development only on
  the plumbing sample before Gate 5).
- T1.2–T1.5: full-text reads of John 2025 + Nayak 2026 (arXiv HTML); expand
  matrix toward 40–60 papers.
- D-3/D-5 PI decisions: legacy-status merge; 2008Q1 cohort start;
  months-since-event field handling.
- Zotero collection (T0.8 remainder, desktop app).

## Blockers
- Zotero: needs the desktop install/import before the collection exists.
- D-3 legacy-status merge, D-4 sensitivity-parking review, D-5 drift-rule
  outcomes need PI sign-off.
- Everything in `configs/*.yaml` stays PROPOSED until Gates 4–5.

## Session 4, 3 Nov 2026 (Analyst)

- T1.2 closed: John 2025 and Nayak 2026 read in FULL TEXT
  (`lit/reading_notes/john-2025.md`, `lit/reading_notes/nayak-2026.md`).
  Material corrections to the matrix rows:
  - **John 2025 uses a simulated multi-year lending panel** (2015-2024,
    race and gender present as features, simulated recessions, default rate
    drifting 15% to 23%) - not a real lending corpus. AUC 0.63-0.66, F1
    below 0.07; no Brier/ECE audit anywhere; "recalibration" is
    fairness-side group threshold adjustment, not probability calibration.
  - **Nayak 2026 evaluates the Home Credit "Credit Risk Model Stability"
    Kaggle benchmark, not LendingClub.** Brier 0.087 / ECE 0.015 with
    temperature scaling fitted on validation weeks; single SHAP figure
    without stability measurement; no loan-level decision-flip analysis;
    online recalibration left as future work.
- T1.6 (matrix expansion), verification discipline: every candidate is
  fetched and verified first (arXiv abs/API, Crossref, or publisher page);
  two wrong identifiers were caught before use this session (the paper
  fetched under arXiv 2202.09785 is a shellcode paper; BIS WP 835 is the
  mortgage "cost of steering" paper). The Bracke et al. 2019 default-risk
  explainability paper was located and verified via Crossref instead (SSRN
  preprint, DOI 10.2139/ssrn.3435104; authors Bracke, Datta, Jung, Sen).
- Eight verified rows appended to `lit/matrix.csv`, then two more verified
  in the same session (Lessmann et al. 2015, EJOR 247:124-136;
  Crook, Edelman & Thomas 2007, EJOR 183:1447-1465) - matrix stands at
  **25 rows x 22 columns**, schema validated:
  Rudin 2019 (Nature Machine Intelligence), Goldstein et al. 2015 (JCGS,
  ICE), Sundararajan et al. 2017 (ICML, Integrated Gradients), Ribeiro et
  al. 2016 (KDD, LIME), Brown & Mues 2012 (ESWA, imbalanced credit
  classifiers), Lakshminarayanan et al. 2017 (NIPS, deep ensembles), Nauta
  et al. 2023 (ACM Computing Surveys, XAI evaluation, DOI 10.1145/3583558),
  Bracke et al. 2019 (SSRN, Shapley/QII on mortgage default). New rows carry
  `read_status = metadata` until read.
- Expansion continues under the same verify-first rule. Second batch of the
  day verified and appended nine more: Baesens et al. 2003 (JORS 54:627-635),
  Hand & Henley 1997 (JRSS-A 160:523-541), Thomas 2000 (Int J Forecasting
  16:149-172), Bellotti & Crook 2013 (Int J Forecasting 29:563-574), Hand
  2006 (Statistical Science 21:1-14), Brier 1950 (Mon Wea Rev 78:1-3),
  Zadrozny & Elkan 2002 (KDD '02:694-699), Crone & Finlay 2012 (Int J
  Forecasting 28:224-238), Hand 2005 (JORS 56:1109-1117) - then Murphy 1973
  (J Applied Meteorology 12:595-600, Bs reliability/resolution decomposition)
  and Thomas, Crook & Edelman 2017 (SIAM monograph). **Matrix stands at 40
  rows x 22 columns - inside the protocol's 40-60 target band.** Further
  reads (full-text depth on Ovadia 2019, Fuster 2022) queued next session.
- Full-text read completed for **Ovadia et al. 2019**
  (`lit/reading_notes/ovadia-2019.md`; matrix row updated). Key facts for
  RQ1: i.i.d. validation calibration does not carry to shifted data;
  temperature scaling is the most fragile recalibrator (worse than vanilla
  at high Criteo shift); deep ensembles dominate; no explanation audit, no
  decision-flip analysis, no label-delay axis, no AUC-vs-reliability test
  anywhere - our joint-audit claim stays narrowed as pre-agreed.
- Fuster et al. 2022 full-text read BLOCKED: SSRN returns 403 (anti-bot),
  the Journal of Finance version is paywalled, no author-hosted PDF found
  via search. Needs PI-side access (library/alternate copy); matrix row
  stays at metadata level until then.
- Phase 5 measurement machinery staged ahead of the freeze:
  `src/calibration/audit.py` (Murphy 1973 Brier decomposition, ECE-capable
  binning incl. quantile bins, logistic CITL/slope refit on the logit,
  decision-flip counts with direction sub-counts, within-cohort Spearman
  aggregator) + `tests/test_calibration_audit.py` (17 synthetic tests,
  no data files). Methodological note fixed during testing: the partition
  identity BS = uncertainty - resolution + reliability is exact for the
  BINNED forecast; a raw continuous score carries within-bin spread, so
  the module reports both binned and raw Brier and the identity is checked
  against the binned one. Test suite now 76 passed; ruff clean.
- T5.6 groundwork landed pre-freeze: `src/models/harness.py`
  (T6.1/T6.2) — preprocessing fitted on the training window only (median
  imputation + standardisation), one estimator constructor per sanctioned
  family (logistic regression, random forest, XGBoost, LightGBM; class
  weights none, resampling none, seeds injected from config), and an
  evaluation front end with the nine sanctioned point metrics plus
  percentile bootstrap CIs computed on one shared seeded resample design
  (B read from configs). `configs/dryrun.yaml` holds the T4.6 pilot anchor
  hyperparameters, marked plumbing-only. `scripts/run_baselines.py`
  dry-runs end to end on the plumbing sample (2538 train / 1779
  validation rows mirrored from configs/splits.yaml) and refuses test
  cohorts: with test windows unset it says they are locked, and if they
  were set without PROTOCOL_FROZEN it hard-fails. Dry-run wrote
  `reports/tables/baseline_dryrun_metrics.csv` (36 rows: family x
  metric x point/CI). Suite now 89 passed; ruff clean.
- T7.3 recalibration arms landed: `src/calibration/recalibration.py` —
  four sanctioned arms (none / intercept-only calibration-in-the-large
  via a bounded scalar deviance fit with slope pinned to 1 / Platt on the
  logit / isotonic, clipped and monotone), each fitted ONLY on the
  explicit calibration argument (eligible historical labels; callers own
  the eligibility assertion, the API carries the calibrated n/events for
  provenance). Two issues caught by the synthetic tests and fixed:
  (1) an unknown arm kind silently fell through to the isotonic branch -
  now rejected up front; (2) sklearn 1.8 deprecates penalty=None for the
  unregularised calibration refits and its fallback silently penalises
  the Platt/CITL slope - replaced with C=np.inf in both
  recalibration.py and audit.py. Expectation fixes during testing, for
  the record: on a symmetric logit stretch (forecast 1.5x too extreme)
  the fitted Platt slope SHRINKS to 1/1.5 (slope < 1 signals
  overconfidence, consistent with audit.py), the intercept-only arm
  cannot fix a pure slope problem by construction, and the oracle
  Brier gain over the stretched scores on that simulator is ~6.5%,
  not ~25%. Test suite now 99 passed; ruff clean.

## Session 4, 3 Nov 2026 (Builder — Phase 4 closed out)

- **T4.1** `scripts/cohort_feasibility.py` -> `reports/tables/cohort_feasibility.csv`
  + 2 figures: per-vintage volume, resolution, default rate (Wilson CIs).
  All mature cohorts 2009Q4-2015Q4 pass the 200/200 rule; 2016+ partially
  unresolved; micro-vintages 2008Q2-2009Q3 fail the events rule.
- **T4.2** `src/statistics/shift.py` (PSI/KS/SMD/JS/Wilson, pure functions)
  + `tests/test_shift_stats.py` (18 synthetic tests, green) +
  `scripts/build_mature_features.py` (audited-source 1.6M x 55 cache) +
  `scripts/shift_measures.py` -> per-feature PSI/KS/SMD/missingness tables
  (2013Q3-2018Q4 vs the training window), categorical JS, default-rate
  shift with intervals, PSI heatmap figure. Findings: excluding the three
  event-absence fields, candidate-window PSI stays <= 0.17; the strong
  drifters are `pub_rec`, `revol_util_pct`, `initial_list_status`
  (listing-channel switch, JS to 0.489).
- **T4.3** `scripts/adversarial_validation.py`: features-only adversarial
  AUC vs the training window rises 0.61 (2013Q3) -> 0.88 (2015Q4) and 0.95
  by 2018Q4 (late-adopted bureau fields; schema-confounded) -> table + ROC
  figure.
- **T4.4** `scripts/concept_drift_strata.py`: within-stratum resolved
  default-rate spreads up to 0.56 across vintages -> conditional rates move
  beyond covariate shift; table + heatmap.
- **T4.5** `scripts/label_delay_feasibility.py`: 49 cells (8 cohorts x lags
  {0,6,12,18,24,36} + static arm) all FEASIBLE under 200/200; arms tagged
  oracle/optimistic/realistic per CONTEXT Section 6.
- **T4.6** `scripts/compute_pilot.py`: one config per family on the plumbing
  sample; SHAP TreeExplainer ~= 12.9 s per 10k rows (500-row background),
  bootstrap B=200 in 1.6 s (B=2000 ~= 16 s), recalibrators <= 0.4 s ->
  `reports/tables/compute_pilot.csv`; caps written into GATE4 BEFORE freeze.
- **T4.7** `protocol/GATE4.md` drafted (PROPOSED): test cohorts 2014Q1-2015Q4
  proposed for freeze; windows chosen from descriptive evidence only; D-3,
  D-4, D-5 and the compute caps presented for PI ratification.
- Test suite 59 green incl. the new shift-statistics module; lint being
  swept at commit time.


