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
- Phase 8 decision-policy machinery landed: `src/decisions/policy.py` —
  three-band threshold policy (approve if p < t_low, refer in the band,
  decline above; band edges are REFER by the policy wording), the
  break-even formula p* = g/(g+L) over the configured ratios (3/5/9),
  the fixed-approval-rate second policy (bottom-share approval, ties
  belong to NOT-approved, threshold returned for the records), and the
  stylised portfolio utility (funded good earns g, funded default costs
  L, refer/decline contribute zero; scenario quantities only, never
  observed profits). One module bug caught by tests and fixed: the
  fixed-rate threshold had its quantile side inverted (would have
  approved the RISKIEST rate-share rather than the least risky). All
  PROPOSED values passed in as arguments from configs at call sites;
  no literals here. Suite now 111 passed; ruff clean.
- T9.3/T9.4 flip machinery landed: `src/decisions/flips.py` — 3x3
  transition matrices, off-diagonal flip rate, toward-decline vs
  toward-approve flow counts with asymmetry, signed probability changes
  for flipped pairs only, threshold-adjacency distances (fragility
  flags), per-loan and total stylised delta-utility, and the
  matched-null distribution of total delta-utility (draws seeded;
  after-scores permuted so decision margins hold while loan pairing
  breaks). Design insight recorded: because the null strips the
  score-outcome information but keeps every marginal decision rate, an
  informative no-op arm sits at or above the ENTIRE null distribution
  and a comparison pair inside the null is indistinguishable from
  chance - the RQ3 interpretation sentence is pre-written from this.
  Suite now 128 passed; ruff clean.
- T8.3/T8.7 explanation-stability machinery landed:
  `src/explain/stability.py` — pure comparators over attribution
  matrices with a caller-fixed feature order: mean |shap| global
  importance, importance rank vectors, Jaccard@k on top-k sets, Spearman
  + Kendall rank agreement, cosine on importance vectors, and the mean
  per-feature Wasserstein-1 attribution-distribution distance (a
  population-shift signature ranks cannot see). Synthetic tests pin the
  under-detection property that motivates running BOTH views: a shift
  in how many features carry signal leaves Spearman at moderate-to-high
  (~0.69 in the pinned fixture) while the distribution metric separates
  the shifted pair from a clean re-draw by an order of magnitude.
  Suite now 145 passed; ruff clean.

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

## Session 5, 5-6 Nov 2026 (Builder + protocol drafting)

- ECE landed in `src/calibration/audit.py` (share-weighted mean |p_bar -
  y_bar| over the same bins as the decomposition; B configurable, plan's
  B=15 plus the 10/20/30 row set) + 5 synthetic tests. `scripts/run_calibration.py`
  dry-runs the full T7.2/T7.3 audit chain on the plumbing sample per
  family anchor: all four arms fitted on the TRAINING window only, iron
  rule asserted in code (calibration vs evaluation cohorts disjoint),
  per-arm binned/raw Brier + reliability + ECE + CITL/slope + flip rate
  vs the none arm + ROC-AUC, written to
  `reports/tables/calibration_audit_dryrun.csv` (128 rows = 4 families x
  4 arms x 8 metrics). Plumbing-scale findings consistent with RQ1
  expectations: arms fitted on the previous window leave validation CITL
  ~-0.74 essentially unchanged, Platt ~stationary, isotonic collapses
  the logit slope 0.67 -> 0.07 and flips 1.7% of decisions - the
  known fragility H1d targets. Suite now 150 passed; ruff clean.
- Decision-layer dry-run landed: `scripts/run_decisions.py` wires the
  T9.1/T9.2/T9.5 chain on the plumbing sample — three-band policy read
  from configs/decisions.yaml (p* = g/(g+L), all 9 ratio x half-width
  sensitivity cells), per-arm comparisons vs the none arm on identical
  loans (flip rates, directional flows, total delta-utility), matched
  null on the primary policy cell (1000 draws from config). Dry-run
  writes `reports/tables/decision_flip_dryrun.csv` (480 rows). Plumbing
  findings consistent with RQ3 framing: isotonic flips ~10% of
  decisions yet net delta-utility ~ +1 (flips mostly outcome-neutral
  re-assignments), every observed total sits far above its null mean
  (~ -208) - decision changes are not chance; the
  default-rate-by-flip-direction analysis (T9.4b) is what separates
  informative from cosmetic flips in the real run. Suite now 150
  passed; ruff clean.
- T6.7 reproducibility checks landed: `tests/test_reproducibility.py` —
  one small config per family re-run through the full path
  (preproc.fit -> train -> predict -> evaluate with bootstrap) with
  identical outputs required within 1e-9 (the tolerance line the plan
  asks for; parallel tree predict is openMP non-bit-stable, the synthetic
  suite pins that deliberately), plus seedless-pipeline checks:
  preprocessor re-fit determinism and bootstrap-design independence from
  the global RNG stream. Suite now 156 passed; ruff clean.
- T8.5 grouped-importance machinery landed: `src/explain/groups.py` —
  group |shap| (mean over rows of |sum of attributions| per named
  group) and grouped permutation importance (joint row-permutation of
  every feature in the block, R>1 seeded repeats, caller-supplied
  predict + metric, sign normalised so positive = informative for both
  loss and gain metrics). The group partition discipline is enforced:
  full coverage, no overlap, no silent gaps (T5.4 writes
  configs/feature_groups.yaml at freeze; this module takes the groups
  as an argument). Synthetic tests verify signal groups outweigh inert
  blocks by >50x on the attribution axis, the loss-vs-AUC sign
  conventions agree, and the same seed reproduces identical numbers.
  Suite now 163 passed; ruff clean.
- Confirmatory inference helpers landed: `src/statistics/inference.py`
  (T7.7 paired method comparisons + the statistics.yaml multiplicity
  rule) — Holm step-down adjusted p-values (order-preserving, monotone
  envelope, worked-example tests) and the paired bootstrap delta CI
  (two score systems evaluated on the SAME seeded resample design per
  replicate, sign normalised positive = A worse than B for both loss
  and gain metrics, determinism pinned). Suite now 172 passed; ruff
  clean.
- T8.1 SHAP wrapper landed: `src/explain/shap_wrap.py` — sanctioned
  explainer builders (linear exact SHAP for logistic regression;
  interventional TreeSHAP against the caller-fixed background for
  xgboost/lightgbm/random_forest; path-dependent variant only behind
  an explicit flag) on the raw/log-odds scale, fixed feature order
  (never re-sorted), plain-ndarray attribution matrices decoupled from
  shap's containers, and a degenerate-background guard. Practical
  catch recorded: shap's LinearExplainer HANGS indefinitely on a
  zero-variance background (observed as a 120s timeout), so the guard
  refuses such backgrounds up front. Synthetic tests verify the exact
  log-odds additivity of linear SHAP (margin = sum(shap) + expected),
  informative-column dominance, determinism, and shape validation.
  Suite now 179 passed; ruff clean.
- Runner consolidation: `scripts/_common.py` now owns the single
  loading path all run_* scripts share (configs reader with the
  one-top-level-key sanity check, the test-cohort guard verbatim, and
  the plumbing split with the disjointness iron rule attached
  structurally). run_baselines / run_calibration / run_decisions were
  re-cut onto it and ALL THREE re-ran end to end with numbers matching
  the pre-refactor tables exactly (LR/RF/XGB/LGBM AUCs unchanged at
  0.6209/0.6237/0.5793/0.5775), so the refactor is provably
  behaviour-neutral. Ruff `--fix` pass applied to sort the new import
  blocks. Suite state unchanged at 179 passed; ruff clean.
- T5.2 hypotheses register landed (PROPOSED, pre-freeze draft for the PI):
  `protocol/HYPOTHESES.md` rewritten from the STUB to the full register —
  every confirmatory row (H1a, H1c, H2c, H3a, RQ4.1-4.4) now carries its
  test statistic, direction (justified only where literature supports
  it), threshold, and practical-effect anchor; all thresholds are either
  pre-set config quantities (alpha 0.05, Holm over the 8 confirmatory
  tests, headline bootstrap B = 2,000, H4 equivalence region |rho| < 0.3,
  unit of analysis, matched-null draws = 1,000) or PROPOSED defaults
  stated with rationale for PI confirmation BEFORE freeze (H1a effect
  anchor |delta-CITL| >= 0.02 = the refer-band half-width; H1c recovery
  >= 0.5 as "substantial"; H3a threshold = the seed-noise-floor 95th
  percentile; H1d isotonic size grid {500, 1k, 2k, 5k, full}).
  Noise-floor inputs (T8.2: 20 seeds x 30 bootstrap resamples on one
  fixed cohort, deep subset 5 configs/family) and the freeze ordering
  (PI edits at Gate 5 only; verbatim into the frozen protocol; Holm
  applied once, at the end) are part of the register. Every config
  citation was verified against the actual file contents before writing
  (statistics.yaml / models.yaml / decisions.yaml keys checked,
  PLAN.md T8.2 wording matched verbatim).
- T5.5 search spaces landed (PROPOSED, D-6 opened for the PI):
  configs/search_spaces.yaml (four families, PROPOSED ranges with the
  T5.5 rules baked in) + src/models/spaces.py (deterministic sampler
  keyed on [base_seed, family_slot, config_index]; pure draw from the
  YAML spec, no data access) + protocol/SEARCH_SPACES.md (protocol
  page: rules 1-3, per-family range table, YAML is authoritative).
  Design rules enforced in code and tested: stochastic row/col
  parameters strictly < 1 (subsample/colsample_bytree drawn in
  [0.5, 0.95); subsample_freq fixed 1) so the deep subset's 20 seeds
  change predictions; ranges deliberately span weak to over-fitted
  corners (RQ4.x needs configs that differ in calibration behaviour);
  boosting n_estimators fixed 400 with learning-rate carrying the
  variation, checked against the T4.6 compute caps. Two harness
  collisions caught by the estimator smoke test and fixed at the YAML
  level (n_jobs on the RF fixed block, verbosity on xgboost/lightgbm
  were duplicated keys the harness already injects). 12 synthetic
  tests (determinism, the <1 invariant over 200 draws per booster,
  weak/overfit ends spanned, batch non-collision, sampled params feed
  the real estimator factory for every family). Suite now 191 passed;
  ruff clean.
- T8.2/T9.2d noise-floor runner landed: `scripts/run_noise_floor.py`
  dry-runs Axis C on the plumbing sample — one anchor config per family
  refit under 20 seeds on the same training window, fixed 400-row
  explanation sample + fixed 1,000-row background + fixed feature
  order, all 190 seed pairs per family compared on the T8.3 battery
  (Jaccard@10, Spearman, Kendall, cosine, mean Wasserstein-1) plus the
  T9.2d seed-vs-seed decision flip rate at 0.5 and the AUC gap;
  `reports/tables/noise_floor_dryrun.csv` (3,800 pair x metric rows).
  The table IS the pre-form of the H2c/H3a threshold reference: boosted
  trees show real seed floors (decision flips 1.0-3.5%, AUC gaps to
  +-3.5pp at the 2.5/97.5 percentiles), while logistic regression
  anchors at exactly zero on every metric (deterministic estimator) -
  the pipeline measures variation, it does not inject any. Two shap
  issues hit and handled in `src/explain/shap_wrap.py`:
  (1) the degenerate-background guard narrowed - a FULLY degenerate
  background (every column constant; identical rows) still hangs
  LinearExplainer and is refused, but a SINGLE constant column among
  varied rows is legal (linear exact SHAP and interventional TreeSHAP
  both attribute exactly 0 to it; pinned by a new assertion inside the
  existing test) - plumbing windows legitimately contain never-firing
  one-hot columns across the whole training window (7 of 52 columns
  here), so the old per-column refusal was a plumbing-scale hazard, not
  a real-run one;
  (2) `check_additivity=False` is passed for TreeExplainer calls only -
  LightGBM's float32 leaf accumulation breaks shap's strict sum check
  at the 1e-3 level; downstream stability metrics consume differences
  between matrices computed identically, so correctness rests on the
  fixed background and fixed order, documented in-code. Suite stays
  191 passed; ruff clean.
- T7.5 sample-size runner landed (H1d machinery): `scripts/run_sample_size.py`

  dry-runs the calibration-set-size experiment on the plumbing sample —
  20 draws per size on the PROPOSED grid (200 = the 200/200 feasibility
  floor, 500, 1k, 2k, full), all four arms fitted on the SAME draw
  (matched design: arms comparable per draw), applied to validation,
  per-draw arm Brier/ECE rows plus matched arm-delta summaries
  (isotonic-platt / isotonic-intercept / platt-intercept mean and 2.5/
  97.5 percentiles across draws; draws seeded by [family_slot, r, n] so
  repeats are deterministic). Dry-run writes
  `reports/tables/sample_size_dryrun.csv` (3,200 rows). The plumbing
  table already shows the H1d signature: isotonic's Brier disadvantage
  vs Platt shrinks monotonically with calibration size (0.0099 at n=200
  -> 0.0042 at full for the RF anchor), the small-n fragility direction
  H1d pre-registered. Suite stays 191 passed; ruff clean.
- I-11 numbers pipeline landed: `scripts/export_numbers.py` ->
  `paper/numbers_dryrun.json` (9,632 keys, every key explicitly
  dryrun-namespaced so plumbing numbers can never be confused with real
  post-freeze results; explicit per-table adapters that refuse a
  mismatched header instead of silently re-parsing) +
  `scripts/render_paper.py` (strict {{n:key}} substitution from the
  export; a placeholder without a value is a hard error listing the
  missing keys; --source main refuses to run without
  protocol/PROTOCOL_FROZEN) + paper scaffolding (manuscript.md STUB
  following the CONTEXT.md structure with no empirical sentences,
  claims.csv with the fixed 5-column register and zero rows ->
  consistent by design, README documenting the discipline). Plumbing
  stub renders end to end (1 placeholder resolved). 9 synthetic tests
  pin adapters, render substitution/formatting, missing-key liveness,
  the claims register schema, and a real-export spot anchor
  (LR dry-run AUC 0.620923). Suite now 200 passed; ruff clean.
- Cross-review R-7.2 (Builder-review of calibration statistics vs
  CONTEXT 7, logged in notes/REVIEWS.md) found one genuine DEFINITION
  ERROR and three conformance gaps, all fixed:
  (1) citl_and_slope took the intercept from a single FREE binomial fit,
  but 7 defines CITL via an OFFSET fit (slope pinned to 1) and the
  slope via a SEPARATE free fit - different parameters whenever the
  slope is not 1. fit_offset_intercept() now backs both the intercept
  recalibration arm and the audit metric (single source of truth); a
  discriminator test pins offset-intercept != free intercept under a
  skewed symmetric logit stretch (free/Platt intercept ~0 vs offset
  correction > 0), and citl == fitted arm offset is pinned to 1e-9.
  (2) clip level 1e-12 -> 1e-6 per 7's "before any logit" rule.
  (3) the A/E ratio (mean y / mean p, ideal 1) was missing entirely;
  added ae_ratio() + 4 synthetic tests + an ae_ratio metric column in
  the runner. Sampling-noise lesson recorded honestly: initial test
  tolerances rel=1e-6 at n=50k sit below the ~1.4% Bernoulli noise
  floor and were loosened to 2%.
  (4) runners' primary ECE switched to equal-mass bins
  (quantile_edges=True, B=15) per the 7 adaptive-bin definition;
  run_calibration and run_sample_size re-ran end to end, and
  numbers_dryrun.json re-exported (9,648 keys).
  Suite now 206 passed (6 new tests); ruff clean; AI sweep clean.
- [Analyst] Platt 1999 verified against a primary record and added as
  the recalibration line's 41st matrix row: Semantic Scholar record
  CorpusId 56563878 (title/author/year 1999; venue field empty in the
  API record) plus independent corroboration of the imprint - Advances
  in Large Margin Classifiers (MIT Press, Cambridge MA), pp. 61-74 -
  via the Lin & Lin 2007 Machine Learning note (DOI
  10.1007/s10994-007-5018-6) and the ACM DL record. Crossref was
  correctly skipped as unusable for pre-DOI book chapters. Row content
  marks the Platt arm lineage (parametric sigmoid fitted on a separate
  calibration set); read_status stays metadata until a full-text read.
  Matrix now 41 rows x 22 columns (still inside the 40-60 target band).
- Broad-sweep sampling dry-run landed (T5.5 follow-through):
  `scripts/run_sweep.py` draws the 30 configs/family from the
  deterministic sampler, fits every draw on the plumbing train window,
  evaluates AUC/Brier/logloss on validation only, and applies the
  models.yaml deep-subset rule (5 configs per family at validation-AUC
  quantile anchors min/q25/q50/q75/max; even-n median ties resolve
  downward via round-half-even, documented). Selection extracted as
  `deep_subset_indices()` + 3 tests pinning quantile positions vs
  config ids (the anchor must track the AUC ordering, not the config
  id). `reports/tables/sweep_dryrun.csv` (120 rows) +
  `reports/tables/deep_subset_dryrun.csv` (20 rows). Plumbing picture
  (dry-run only): tree families produce genuinely heterogeneous
  configs (AUC spreads 0.043 RF / 0.080 XGB / 0.044 LGBM; XGB log-loss
  spread 0.089), while LR regularization moves AUC by ~0.0025 across
  the whole C range; AUC-Brier correlations at this scale are strongly
  negative for trees (-0.75..-0.94), which is what the RQ4 association
  is designed to interrogate at real scale. Re-run after the
  behaviour-neutral refactor produced identical numbers. New table
  adapters added to `export_numbers.py` (10,028 keys exported). 3 new
  tests; suite now 210 passed; ruff clean; AI sweep clean.
- T5.4 feature-group draft landed (PROPOSED, D-7 opened for the PI):
  `configs/feature_groups.yaml` (6-way semantic partition of the 24
  audited KEEP features; engineered columns mapped explicitly; one-hots
  inherit by source prefix; no post-origination field anywhere) +
  `src/features/groups.py` resolver (hard error on any unassigned design
  column, so schema changes can never silently break the partition;
  longest-prefix rule for one-hot levels) + `protocol/FEATURE_GROUPS.md`
  (rationale, constraints, D-7 questions). Verified live on the real
  plumbing design matrix: all 52 columns resolve into 6 groups (30/10/4/
  4/2/2). 4 synthetic tests (config-side partition + KEEP coverage,
  engineered/one-hot resolution, unassigned-column errors,
  partition-cover property). Suite now 214 passed; ruff clean.
- Noise-floor runner extended with the grouped stability battery:
  `run_noise_floor.py` now builds the T5.4 partition over the design
  columns (deterministic full-frame column order), computes one
  importance scalar per group per seed, and reports grouped_cosine /
  grouped_spearman / grouped_max_gap on every one of the 190 pairs x 4
  families (7,600 metric rows total). Group-level reference
  distributions at plumbing scale: grouped cosine >= 0.998 for the
  boosters (group mass is seed-stable; per-feature cosine sat at
  0.994-0.997), RF grouped_spearman floor 0.886 at p2.5, absolute
  group-attribution gaps (max_gap) xgb 0.015-0.047 / lgbm 0.016-0.077
  at the 2.5/97.5 percentile band. LR anchors at exactly zero as
  before. This closes the grouped side of H2c's reference machinery
  pre-freeze. numbers_dryrun.json re-exported (12,308 keys). Suite
  stays 214 passed; ruff clean; AI sweep clean.
- Bootstrap-refit axis added to the noise-floor runner (CONTEXT 7's
  full noise-floor wording: seed-retrained pairs AND bootstrap-refit
  models on identical data): 30 seeded bootstrap resamples of the
  training rows per family, estimator seed pinned so variation is the
  resample itself, all 435 refit pairs per family compared on the same
  battery as the seed pairs; table now carries a `kind` column
  (seed / bootstrap_refit; 25,000 pair x metric rows total) and the
  export adapter follows (29,708 keys). Plumbing-scale floors:
  bootstrap resampling moves RF's per-feature attribution structure a
  lot (cosine median 0.882, p2.5 0.746) while the boosters hold
  0.92-0.96 median and grouped cosine stays >= 0.96 everywhere -
  group-level mass is more refit-stable than single columns, directly
  relevant to which H2c threshold applies. Decision-flip floors
  under refits: xgb ~4.0% / lgbm ~3.0% median, RF exactly 0 across all
  435 pairs (leaf-smoothed RF scores very rarely cross the 0.5
  threshold) - flagged for the real-cohort runner to re-check with
  real-scale training. 4 tests for the grouped metric block added
  (identical/proportional/reversed vectors + live partition cover of
  the 52 design columns; two initial test-expectation slips corrected -
  cosine stays high under magnitude-preserving reversals, which is
  WHY spearman is reported alongside). Suite now 218 passed; ruff
  clean; AI sweep clean.
- Reliability-diagram data + figure machinery landed (T7.x figure
  preforms): new `reliability_bins()` in `src/calibration/audit.py`
  (per-bin segments, quantile edges by default, same `_bin_masks`
  machinery as ece), wired into `run_calibration.py` which now also
  emits `reports/tables/calibration_curves_dryrun.csv` (205 segments,
  family x arm x bin; isotonic arms legitimately collapse to fewer
  non-empty segments because the step output repeats score values);
  new `scripts/run_figures.py` renders `reports/figures/
  reliability_diagram_dryrun.png` (4 families x 4 arms grid with the
  diagonal; the intercept arm's pure offset shift and xgboost's
  large residual under-prediction segment are directly visible) and
  `sweep_auc_brier_dryrun.png` (per-family 30-config scatter showing
  the AUC-Brier association). Conventions follow the existing figure
  scripts (matplotlib Agg, dpi=150, deterministic axes, no
  timestamps). NOTE an arm-name slip caught on first render: the
  fourth sanctioned arm is `intercept`, not `beta`; corrected against
  KINDS before committing.
- Binning off-by-two found and fixed (R-7.2 addendum): the equal-mass
  path of `_bin_masks` derived its inner cut count from the edges
  template incorrectly and delivered n_bins+2 masks, so `ece_b15` in
  the previously exported calibration/sample-size tables was computed
  over 17 bins rather than the protocol's 15. Fixed by deriving B-1
  inner cuts at k/B from the template length; pinned by a regression
  test asserting exactly-B masks (quantile and fixed-edge paths, B =
  10/15/20/30; the B=15 approximation now matches CONTEXT 7's
  adaptive-bin definition as written). `run_calibration` +
  `run_sample_size` re-ran end to end, numbers_dryrun.json
  re-exported (30,323 keys including the new curves-table adapter).
  Suite now 222 passed; ruff clean; AI sweep clean.
- Analyst-review R-11.1 (paper scaffolding vs numbers discipline)
  found and fixed a prose-boundary defect: the stub's example
  placeholder sat in an INDENTED code block, so the renderer resolved
  it and the rendered stub embedded the plumbing-scale LR AUC value
  directly in text - contrary to the README's own no-dry-run-numbers
  rule. Fix: `render_paper.py` now treats fenced code blocks as
  documentation (placeholders inside fences stay literal), the stub
  example moved into a proper fence, the README documents the rule,
  and a regression test pins resolve/missing/literal behaviour (one
  test-side slip - a placeholder missing the `n:` prefix - corrected
  before commit). Also verified: claims register schema stays fixed
  and empty; suspected heading corruption in the stub/README turned
  out to be console-only (file bytes are valid UTF-8 U+2014); GATE5
  verification-state refreshed (suite 222, both noise-floor axes,
  figure preforms, binning-fix note). Suite stays 222 passed; ruff
  clean.

## Session 2026-11-09 (post-freeze, Phase 6-7 launch)

- GATE 5 FROZEN: PI ratified in session (freeze now; D-6 and D-7 as
  drafted; T4.6 compute caps accepted) - D-8 logged; configs/splits.yaml
  test window unlocked to 2014Q1-2015Q4; PROTOCOL_FROZEN created; tag
  protocol-v1 pushed (85d81a2).
- Real-cohort layer committed (1de8663): scripts/_real.py loader refuses
  to run without PROTOCOL_FROZEN; verified live from the mature cache:
  train 2012Q1-2013Q2 = 84,594 rows (exact match to the recorded count),
  val 59,298; 8 test cohorts 445,596 rows after dropping 147 rows with
  unresolved labels; every cohort floor met (events 4,384-13,138;
  non-events 29,690-75,531 vs 200/200 minimum). 4 loader tests on
  synthetic fixtures added.
- Real battery runner completed and pilot-verified end to end on the
  frozen cohorts (LR, 5 pilot configs, 2 seeds x 3 boots: 67-131 s per
  config): per-window score aggregates (AUC/Brier/logloss/ECE-b15q/
  CITL/slope/A-E/mean-p on all 9 windows), recalibration arms fitted
  on training-window scores with per-cohort oracle recovery (arm
  machinery verified standalone: 62 rows in 4.0 s, recalibrators within
  the 0.4 s cap), pairwise flip tracks at the policy cell
  (t_low=0.1467/t_high=0.1867), seed + bootstrap noise-floor batteries,
  frozen-model cross-cohort battery, and per-group stability rows
  (added for H2c's per-group reading: cosine/Spearman/mean-gap per
  group per pair). Pilot artifact tables deleted (pre-arms code).
- The T5.4 resolver partitions the real 55-column design (group sizes
  4/4/10/2/2/33 - application_context absorbs the full-data one-hot
  levels), verified live.
- scripts/run_real_statistics.py committed (d4db109): H1a cluster
  bootstrap over configurations (B=2000, seed 20261109), H1c recovery
  cells with lag mapping (2014Q1=3 ... 2015Q4=10 quarters) + lag-order
  Spearman, H2c share-of-cells-below-floor-2.5-pct statistic, H3a
  recalibration flips vs the T9.2d seed floor q95, RQ4.1-4.4 pooled
  within-pair Spearman deltas on ONE shared cluster design, Holm once
  over the 8 confirmatory p-values. Synthetic smoke regression test
  added (B monkeypatched to 60; 16.5 s) - full batterytables consumed
  only when they exist.
- Real broad sweep live-verified: LR pass 8.9 s on 84,594 rows
  (AUC 0.6471..0.6477, spread 0.0006); RF pass in progress at
  commit time. Known background-run failure mode: first sweep launch
  died silently holding the log open (zombie python, empty log);
  relaunched with unbuffered -u and liveness confirmed from log lines.
- Suite 223 passed + 1 smoke test pending full-suite rerun; ruff
  clean; AI sweep clean at last commit.

## Session 2026-11-12 (post-freeze, deep battery execution)

- **DEV-1 logged before any result was seen** (`protocol/DEVIATIONS.md`).
  The registered 20-seed x 30-bootstrap x 10,000-row-SHAP battery is
  arithmetically infeasible on the project machine. Measured on the real
  training window: RF cfg6 fit 206.9 s; RF interventional TreeSHAP
  1.964 s/row = 327 min per 10,000-row model; RF path-dependent
  5.57 s/row; XGB 0.024 s/row; LGBM 0.066 s/row; LR <0.001 s/row.
  The T4.6 pilot's 12.9 s per 10,000 rows was an XGB-on-2.5k-rows
  extrapolation and does not hold at real-scale forest depth.
  Ratified lean design: 5 seed + 5 bootstrap refits per config (all 5
  configs kept), SHAP at 1,000 rows for LR/XGB/LGBM, RF explained by
  per-feature Brier permutation importance (R=3, 1,000 rows, T8.5
  machinery, groups summed, Wasserstein omitted for RF). The seed-0
  estimator is reused for the cohort battery; predictions are cached
  once per model (the aggregate block previously re-predicted).
- Battery runner hardened: `--families` comma subset, `--suffix`,
  `--merge-only`, crash-safe per-family tables rewritten after every
  config (`deep_agg_real__<family>.csv` etc.), and a merge step that
  refuses missing families or schema drift. LR pilot re-ran end to end
  with the new layout: 5 configs x (2 seeds + 3 boots), 51.7-78.9 s per
  config, all six merged pilot tables written; pilot artifacts deleted.
- Observed cohort facts committed as `reports/tables/real_cohorts.csv`
  (train 84,594 / val 59,298 / 8 test cohorts 445,596, 147 unresolved
  labels dropped, events 64,447 across the test cohorts). Matches the
  loader counts exactly.
- Real battery live: two background processes (RF alone; LR+XGB+LGBM)
  writing `rf_log.txt` and `boost_log.txt`, per-family tables written
  incrementally. Downstream added while it runs: `--source main` export
  adapters (statistics/descriptives/floors/RQ4/recovery/battery/
  temporal/group summaries), `scripts/report_real_cohorts.py`, and
  `scripts/run_real_figures.py` (7 deterministic real-table figures).
- New tests: 4 battery-helper tests + 3 real-adapter tests. Full-suite
  count to be re-established at the end of the run.

## Session 2026-11-14 (post-freeze results complete; release)

- **Four-family battery finished.** All 20 deep-subset configurations
  (5 per family: LR, XGBoost, LightGBM, random forest) completed 5 seed
  refits + 5 bootstrap refits with recalibration arms, flip tracks and
  stability pairs. Per-family crash-safe tables written incrementally
  (`*_real__<family>.csv`) and merged (`*_real.csv`: 16,000 agg rows,
  57,600 arm rows, 31,100 stability rows, 6,720 group rows). Job-log
  timings (rf_log.txt / boost_log.txt) deleted with the logs at release;
  the committed per-family tables are the durable evidence.
- **Confirmatory statistics** (`scripts/run_real_statistics.py`, merged
  tables, B=2000, STAT_SEED=20261109, Holm once over the 8 p-values),
  exact values from `reports/tables/real_statistics.csv`:
  H1a 0.045384 [0.016064, 0.082050] p_holm 0.002 reject_H0;
  H1c -13.660196 [-25.959563, -4.776104] p_holm 0.000 reject_H0
  (sign NEGATIVE - not "partial recovery"; see DEV-3 companions);
  H2c 0.083333 [0.000000, 0.208333] p_holm 0.261 not_rejected
  (only confirmatory null not rejected);
  H3a -0.102468 [-0.122929, -0.073043] p_holm 0.000 reject_H0
  (sign NEGATIVE: recalibration flips FEWER loans than reseeding;
  per family LR +0.01094 / XGB -0.21371 / LGBM -0.17584 / RF -0.03127);
  RQ4.1 -1.203654, RQ4.2 -1.643823, RQ4.3 -0.802744, RQ4.4 -1.123891,
  all p_holm 0.000 reject_H0; only the slope-deviation outcome
  (rho = -0.689) is classified meaningful by the pre-registered region,
  the other three inconclusive.
- Downstream regenerated deterministically: paper tables (1, 2, 3,
  S1, S2), `protocol/HYPOTHESES_VERDICTS.md`, seven real figures,
  `paper/numbers.json` (3,780 keys, `--source main`), rendered
  manuscript (all 176 placeholders resolved; renderer strict).
- **Manuscript results prose rewritten**: all five `{{n:...verdict}}`
  placeholders replaced with explicit prose; abstract, SS5.1, SS5.2,
  SS6, SS7, SS8, SS9 item 3, SS10, SS12 and Appendix B updated to the
  observed directions (H1c negative/not a fraction; H3a negative/
  fewer flips; H2c not rejected). Figure 5 description now matches the
  plotted quantity (mean Brier change by family/arm/label age).
- **Claim register filled**: `paper/claims.csv` now has 34 rows (30
  auto-extracted empirical sentences + 4 manual qualitative rows), each
  with an exact manuscript substring, evidence table(s) from
  `numbers.json` and main-prefixed number keys. Enforcing tests pass.
- DEV-1/2/3 stand as recorded in `protocol/DEVIATIONS.md`; no further
  deviations. Confirmatory definitions unchanged since the freeze.
- Final checks at release: ruff clean ("All checks passed!"); full test
  suite **243 passed** (96.6 s); both no-collaboration greps clean (only
  the known "installment" false positive); banned-phrase grep clean;
  no `verdict}}` and no `{{n:` in the rendered manuscript;
  rf_log.txt (approx 4 GB) and boost_log.txt deleted.
- Commit: `c49645d29f6183aa6fd841e3c83b549ac0ef4e2e` ("Post-freeze
  results: four-family battery, confirmatory statistics, verdicts,
  paper tables, real figures, manuscript"), pushed to origin/main;
  tag `protocol-v1` present.
