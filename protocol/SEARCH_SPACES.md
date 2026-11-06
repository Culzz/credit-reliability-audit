# protocol/SEARCH_SPACES.md — hyperparameter search spaces (T5.5)

> **Status: PROPOSED** (Analyst drafts, PI confirms at freeze; logged as D-6
> in `notes/DECISIONS.md`). Written before any Phase 5+ result exists:
> no range below was tuned against, or inspected after, any outcome data.
> After the freeze the ranges are part of the protocol; any change is a
> logged deviation (`protocol/DEVIATIONS.md`).

## Where the spaces live

- Machine-readable source of truth: `configs/search_spaces.yaml`.
- Deterministic sampler: `src/models/spaces.py` — `sample_config(family,
  config_index, base_seed)` and `sample_all_configs(family, n, base_seed)`;
  the seeding key `[base_seed, family_slot, config_index]` keeps every draw
  stable across numpy versions. The sampler touches no data; it is a pure
  function of the triple plus the YAML.
- Swept-set sizes: `broad_sweep` = 30 configs/family × 1 seed;
  `deep_subset` = 5 configs/family (chosen at validation-AUC quantiles on
  NEVER-test data) × 20 seeds (`configs/models.yaml`).

## The rules the spaces implement

1. **Stochastic parameters are strictly < 1** (`subsample`,
   `colsample_bytree` drawn in [0.5, 0.95); `subsample_freq = 1`) so the
   deep subset's 20 seeds produce genuinely different models — otherwise
   the seed-vs-seed noise floor (T8.2 / T9.2d) degenerates. Enforced in the
   sampler itself (`assert` in `_draw` consumers), tested across 200 draws
   per booster.
2. **Breadth spans weak to over-fitted.** The broad sweep must contain
   configurations that differ in *calibration behaviour*, not AUC clones:
   the weak end (L2 penalty 1e-3, leaves ≥ tens of rows, tight depth) and
   the over-fitted end (near-unpenalised, unbounded depth, single-row
   leaves) are both inside every family's space. This is what makes the
   AUC-vs-reliability association (RQ4.x) estimable at the configuration
   level.
3. **Compute honest.** Boosting `n_estimators` is fixed at 400 per draw with
   the learning-rate range carrying complexity variation — checked against
   the T4.6 compute pilot caps before the freeze (SHAP ≤ 10,000 rows per
   config; headline bootstrap B = 2,000).

## Per-family PROPOSED ranges (summary; YAML is authoritative)

| Family | Sampled parameters (distribution → range) | Fixed |
|---|---|---|
| logistic_regression | C: log-uniform → [1e-3, 1e3] | max_iter 5000, solver lbfgs |
| random_forest | max_depth: choice ∈ {4, 6, 8, 10, 12, ∞}; min_samples_leaf: log-uniform-int → [1, 100]; max_features: uniform → [0.3, 1.0] | n_estimators 400 |
| xgboost | learning_rate: log-uniform → [0.01, 0.3]; max_depth: int → [2, 10]; min_child_weight: log-uniform-int → [1, 50]; subsample: uniform → [0.5, 0.95); colsample_bytree: uniform → [0.5, 0.95); reg_lambda: log-uniform → [0.01, 100]; gamma: uniform → [0, 5] | n_estimators 400, objective binary:logistic, tree_method hist |
| lightgbm | learning_rate: log-uniform → [0.01, 0.3]; num_leaves: int → [8, 128]; min_child_samples: log-uniform-int → [5, 100]; subsample: uniform → [0.5, 0.95) with subsample_freq 1; colsample_bytree: uniform → [0.5, 0.95); reg_lambda: log-uniform → [0.01, 100] | n_estimators 400, objective binary |

Sanctioned families only (`configs/models.yaml`); class weights none and no
resampling in the main run (the S7 class-weight arm stays a sensitivity
analysis), so class imbalance is handled by the loss, not by the sampler.

## What stays out of scope here

- The generation of the actual 30-config draw happens at the real run (deterministic
given the base seed in the frozen config); nothing is sampled at freeze time
into documents, and no draw is ever edited by hand after results are seen.
- The deliberately weak/over-fitted corners are *not* sensitivity arms; they
are part of the main sweep by design (rule 2).
