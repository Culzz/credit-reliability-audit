# GATE3.md — draft (PROPOSED; PI reviews and confirms before any freeze)

Sample definition, exclusions, and feature dictionary for the main analysis.
Everything below is grounded in committed artifacts: `reports/tables/vintage_maturity.csv`
(T3.4), `data/column_audit.csv` (T3.5), `reports/tables/schema_drift.csv` (T3.7),
`reports/tables/plumbing_sample.csv` (T3.9). Statuses stay PROPOSED until the
protocol freezes; no `protocol/PROTOCOL_FROZEN` file exists yet.

## 1. Sample definition

- Universe: accepted loans file (2,260,668 rows × 151 columns after footer
  removal; SHA-256 recorded in `data/README.md`).
- Term filter: 36-month loans only — **1,609,754 rows**.
- Target (CONTEXT §5): `y = 1` Charged Off/Default; `y = 0` Fully Paid;
  unresolved statuses excluded. Legacy "Does not meet the credit policy"
  handling per D-3 (proposed: merge; 2,749 rows, 0.12%).
- Maturity: vintage (issue quarter) enters only if issue + 36 months ≤
  snapshot 2018-12-31 **and** unresolved share U_q < 0.01 → **35 mature
  quarters, 2007Q2–2015Q4** (PROPOSED; configs/data.yaml).

## 2. Exclusions and their measured sizes

| Exclusion | Rows removed | Where recorded |
| --- | --- | --- |
| 60-month term | 650,914 | T3.4 |
| Unresolved statuses (36m, in-window) | 586,548 of 1,609,754 across all vintages; U_q < 1% inside the window | T3.4 |
| Post-originated-train rows for frozen vintages | none (window ends 2015Q4) | T3.4 |
| 2007Q2 micro-vintage (D-5 proposal: cohort start 2008Q1) | 24 (window-internal) | T3.7 |

## 3. Target balance (36m, full panel)

Fully Paid 1,076,751 · Charged Off/Default 268,599 (incl. 40 "Default" and
legacy variants) in resolved rows; resolved default rate across the 35 mature
vintages: **min 9.09%, max 29.56%** (declining by vintage; per-vintage values
in `vintage_maturity.csv` — observed feasibility economics only, not
modeled). Post-2015 unresolved shares rise with right-censoring; that region
is excluded, not modeled.

## 4. Feature dictionary (audit-tied)

Base features = the 24 KEEP columns in `data/column_audit.csv` (every base
column references its audit row; the loader refuses unaudited columns).
Engineered set (52 columns on the plumbing sample):

- Numeric passthrough: `loan_amnt`, `annual_inc`, `dti`, `delinq_2yrs`,
  `delinq_amnt`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`,
  `total_acc`, `acc_now_delinq`, `fico_range_low`, `fico_range_high`.
- Ordinal: `emp_length` → `emp_length_years` (0–10; raw strings not accepted).
- Per cent: `revol_util` → `revol_util_pct` (0–100; out-of-range → NA).
- Credit age: `credit_age_months` = months between `earliest_cr_line` and
  `issue_d` (+ anomaly flag `cr_line_after_issue`). **Measured anomalies: 0**
  negative ages in 1,609,754 loans (T3.7 run).
- One-hot (with NA indicator): `home_ownership`, `verification_status`,
  `purpose`, `application_type`, `initial_list_status`.
- Missingness indicators + passthrough for months-since-event fields — only
  if they survive D-5 (b); the mechanical drift rule currently drops them.
- `issue_d` is cohort metadata; **never** a feature (enforced by tests).

## 5. Schema-drift rule (T3.7)

Rule: a base feature must be populated (non-null) in ≥ 0.90 of rows in every
mature vintage. Results (`schema_drift.csv`): 12 of 24 KEEP features dropped
— (a) 9 classic fields fail only in 2007Q2 (24-loan micro-vintage), (b) 3
months-since-event fields fail in ~every vintage (event-absence blankness,
not unavailability). 12 features (incl. all fixed-categoricals and
`loan_amnt`, `dti`, `annual_inc`, `emp_length`, FICO pair, `revol_bal`,
`application_type`, `initial_list_status`, `purpose`,
`verification_status`, `home_ownership`) pass in every vintage. Decisions:
D-5 (a) propose modeling cohorts start 2008Q1; (b) keep the mechanical drop
of months-since-event fields pending dictionary confirmation.

## 6. Development plumbing (T3.9) — pre-Gate-5 only

`data/interim/plumbing_sample.parquet`: fixed-seed 3% sample stratified by
vintage quarter (8 quarters of 2012–2013; 143,892-row universe → **4,317
rows**, all resolved), full feature columns carried, smoke-tested through the
feature module (52 engineered features). All pipeline development before
Gate 5 uses only this sample; seed in `configs/data.yaml`.

## 7. Leakage posture (T3.8 tests)

41 tests green (`tests/test_*`): no post-origination column in the feature
set, no target in features, no `issue_d` feature, every base feature tied to
its audit decision, no datetime columns in the feature frame, identical
schema across cohorts. Train-fitted imputation/scaling is postponed to the
Phase 4 pipeline where it is confined to training splits by the loader.

## 8. Open items before this gate text goes final

- D-3/D-5 decisions confirmed by the PI (legacy merge; 2008Q1 cohort start;
  months-since-event handling).
- `data.yaml` parameters confirmed at Gate 4, protocol frozen (Gate 5).
- Empirical read of hardship/settlement fields stays out for the main run.
