# protocol/RESEARCH_RISKS.md

Risk register for the audit. Each risk names the standard response, the owner
and the trigger that fires it. Reviewed at every gate; mitigations are logged
in `notes/DECISIONS.md` and any protocol change goes through
`protocol/DEVIATIONS.md`.

| # | Risk | Standard response | Owner | Trigger | Status |
|---|---|---|---|---|---|
| 1 | Prior art overlaps the joint audit claim | Narrow the claim to what the existing study misses; document in `protocol/GATE1.md` | Analyst-Writer + PI | literature matrix shows overlap with John 2025 / Nayak 2026 or any matrix paper | OPEN |
| 2 | Little measurable shift in the pre-selected windows | Re-pick windows using descriptive statistics only (never model results), before freezing | Builder | Gate 4 shift report finds negligible PSI/adversarial separation | OPEN |
| 3 | Late-vintage outcomes are still unresolved | Shorten the test window; record the choice in the protocol before freeze | Builder | maturity table (T3.4) shows unresolved share above threshold | OPEN |
| 4 | Leakage discovered after evaluation started | Stop; fix the column audit; rerun affected experiments; log in `DEVIATIONS.md` | Builder + PI | any leakage test failure or column-audit correction | OPEN |
| 5 | Realistic-lag recalibration infeasible on early cohorts | Report the feasibility table; run only FEASIBLE cells; state the limit as a finding | Builder | T4.5 label-delay feasibility table | OPEN |
| 6 | Seed variance is zero (deterministic boosters) | Set `subsample`/`colsample_*` < 1 in search spaces; verify predictions actually differ across seeds before relying on them | Builder | T6.3 seed-variability check | OPEN |
| 7 | AUC-vs-reliability test underpowered | Cheap SHAP (3,000 rows) for all configurations; judge by CI against the equivalence region; report inconclusive honestly | Builder + Analyst-Writer | Gate 10 CI width | OPEN |
| 8 | Compute limits on local machine | Parquet/DuckDB only, cached predictions, fixed explanation subsamples, Colab/Kaggle for heavy cells | Builder + PI | T4.6 compute pilot timings | OPEN |
| 9 | H4 result is "AUC predicts everything" | Report it; add the all-vs-reasonable-configs contrast; emphasise the dimensions that still diverge | Analyst-Writer | Gate 10 results | OPEN |
| 10 | Single-dataset external validity | Keep S10 (Home Credit) out of the core; treat any external analysis as separately labelled generalisation; never call a random split temporal | PI | after Gate 11 | OPEN |
| 11 | Scope creep or stalled progress | Apply `PLAN.md` §2 cut lines in order; record each application in `notes/DECISIONS.md` | PI + Builder | weekly status review | OPEN |
| 12 | Suspiciously high performance (AUC ≳ 0.80) | Treat as a leakage alarm until proven otherwise; investigate before continuing | Builder | first evaluation of any model | OPEN |
