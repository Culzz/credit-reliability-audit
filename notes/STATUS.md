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
