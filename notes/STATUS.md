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
