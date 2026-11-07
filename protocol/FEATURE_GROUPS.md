# protocol/FEATURE_GROUPS.md — feature groups (T5.4 draft)

> **Status: PROPOSED** (logged as D-7 in `notes/DECISIONS.md`). Drafted
> before any Phase 5+ result exists. The PI confirms or edits at freeze;
> later changes are logged deviations (`protocol/DEVIATIONS.md`).

## What this is for

Grouped importance and grouped stability (PLAN T8.5; the grouped signals for
H2c and the RQ4.x reliability outcomes). Group-level attribution (mean |SHAP
sum| per group) and grouped permutation importance (joint permutation of the
whole group, R = 20 repeats, loss + AUC) both require a strict PARTITION of
the design matrix — a well-defined partition, no overlaps, no leftovers.

## Where the partition lives

- Machine-readable source of truth: `configs/feature_groups.yaml`.
- Resolver: `src/features/groups.py` (`resolve_column_groups`,
  `grouped_partition`); a design column with no group is a hard error, so
  a schema change can never silently break the partition.
- Mapping rule: exactly-named columns take their source feature's group;
  engineered columns via `engineered_inherit`; one-hot columns inherit via
  the longest matching source-name prefix (`purpose_*`, `home_ownership_na`,
  ...). All 52 plumbing design columns resolve; 4 tests pin the resolver.

## The PROPOSED partition (semantics-only, decided before results)

| Group | Source features | Reading |
|---|---|---|
| borrower_capacity | loan_amnt, annual_inc, dti, emp_length | how much burden the borrower takes on |
| credit_history_tenure | earliest_cr_line, total_acc, open_acc | how long/long the credit file is |
| delinquency_public_record | acc_now_delinq, delinq_2yrs, delinq_amnt, mths_since_last_\*, pub_rec | past adverse events (incl. event-absence indicators) |
| credit_score | fico_range_high, fico_range_low | bureau score (base of the model's signal) |
| utilization | revol_bal, revol_util_pct | revolving use |
| application_context | purpose, verification_status, home_ownership, initial_list_status, application_type, inq_last_6mths | application-time facts and channel |

Constraints already imposed and tested:

1. Exactly the 24 audited KEEP features are grouped (config-side assertion).
2. No group contains post-origination fields — the audit ruled those out.
3. Metadata (id, vintage, y, issue_*) never enters a group.
4. Solver-side caveat carried through: grouped permutation importance on
   correlated groups (score vs utilization) is read per Hooker, Mentch &
   Zhou 2021 as an extrapolation risk in the methods text.

## What the PI is asked to confirm (D-7)

(a) the six-way semantic partition above; (b) event-absence indicators
(`*_isna`) grouped WITH their adverse-event source feature (attack surface:
absence may mean no event — but the audit already decided to keep them with
event-absence semantics, so their group follows the source feature);
(c) `inq_last_6mths` inside application_context rather than delinquency.
