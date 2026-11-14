# Test suite

**243 tests, all passing** (`uv run pytest -q`, ~97 s locally).

| Area | Files |
|---|---|
| Metric definitions vs CONTEXT.md §7 (offset-GLM CITL vs free-intercept slope, 1e-6 clipping, A/E ratio, ECE bins) | `test_calibration_audit.py`, `test_recalibration.py` |
| Leakage guards (test-cohort protocol lock, split roles, descriptive access) | `test_protocol_lock.py`, `test_split_roles.py`, `test_leakage.py`, `test_real_loader.py`, `test_prepare.py` |
| Models, sweep selection, feature groups | `test_harness.py`, `test_spaces.py`, `test_sweep_selection.py`, `test_features.py`, `test_feature_groups.py` |
| Explanations (SHAP wrapper, grouped permutation, stability machinery) | `test_shap_wrap.py`, `test_explain_groups.py`, `test_explain_stability.py` |
| Decisions (policy bands, flip accounting) | `test_policy.py`, `test_dec_flips.py` |
| Statistics (cluster bootstrap, shift measures) | `test_inference.py`, `test_shift_stats.py` |
| Deep battery helpers (crash-safe per-family writes, merge guards) | `test_deep_battery_helpers.py` |
| Confirmatory statistics (Holm, floors, RQ4 pairing — smoke on synthetic data) | `test_real_statistics_smoke.py`, `test_stats_helpers.py` |
| Numbers pipeline (real-table adapters, strict renderer, claim register enforcement) | `test_numbers_pipeline.py` |
| Paper tables and verdict register | `test_paper_tables.py`, `test_report_verdicts.py` |
| Noise-floor blocks, reproducibility (explicit seeds, no `hash()`, no wall-clock seeding) | `test_noise_floor_blocks.py`, `test_reproducibility.py` |

Conventions: no real data in any test (synthetic fixtures only, CONTEXT.md §4
rule 14); the full suite must pass before any commit that touches `src/`.
CI runs `ruff check .` and `pytest -q` on every push to `main` (see
`.github/workflows/ci.yml`).
