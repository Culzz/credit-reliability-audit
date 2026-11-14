# MODEL_CARD — four model families

Summary of the model classes audited in this study. All hyperparameter
ranges are the frozen search spaces (`configs/search_spaces.yaml`,
written before any outcome-model result existed); nothing was tuned
against outcome data. Thirty configurations per family were swept on
validation; five per family — chosen at fixed validation-AUC quantiles
(min, q25, q50, q75, max) — entered the deep battery.

| Family | Estimator | Fixed settings | Sampled ranges (per configuration) |
|---|---|---|---|
| Logistic regression | `sklearn` LogisticRegression | lbfgs, max_iter 5000 | C ∈ [0.001, 1000] (log-uniform; strong-L2 weak end to near-unpenalised end) |
| Random forest | `sklearn` RandomForestClassifier | 400 trees | max_depth ∈ {4, 6, 8, 10, 12, unbounded}; min_samples_leaf ∈ [1, 100] (log-uniform int); max_features ∈ [0.3, 1.0] |
| XGBoost | `xgboost` | 400 trees, binary:logistic, hist | learning_rate ∈ [0.01, 0.3] (log); max_depth ∈ [2, 10]; min_child_weight ∈ [1, 50]; subsample, colsample_bytree ∈ [0.5, 0.95); reg_lambda ∈ [0.01, 100] (log); gamma ∈ [0, 5] |
| LightGBM | `lightgbm` | 400 trees, binary objective | learning_rate ∈ [0.01, 0.3] (log); num_leaves ∈ [8, 128]; min_child_samples ∈ [5, 100]; subsample, colsample_bytree ∈ [0.5, 0.95); reg_lambda ∈ [0.01, 100] (log) |

Design notes:

- Stochastic row/column parameters (subsample, colsample) are drawn in
  [0.5, 0.95) so that estimator seeds strictly change predictions; the
  seed/bootstrap noise floors are then informative rather than
  decorative.
- Ranges deliberately span deliberately weak (heavy regularisation,
  small leaves) to over-fitted (large depth) configurations, so the
  RQ4 association study sees configurations that differ in calibration
  behaviour, not thirty near-clones. The logistic-regression space
  produced near-identical validation fits (a recorded limitation).
- No class weighting or `scale_pos_weight` is used anywhere: the default
  rates in the cohorts are ~13% and every family is trained on the same
  unweighted objective, so family comparisons are not confounded by
  calibration-breaking class weighting.
- Explanation engines: interventional TreeSHAP (1,000-row training-window
  background, 1,000-row explanation sample) for logistic regression,
  XGBoost and LightGBM; grouped per-feature Brier permutation importance
  (R = 3, same 1,000 rows) for random forest (DEV-1 substitution).

Known limitations of these model cards: single training window
(2012Q1–2013Q2), accepted-loan population only, and the random-forest
explanation engine differs from the other families (cross-family
explanation comparisons are descriptive only).
