# protocol/GATE5.md — freeze checklist (PROPOSED for the PI)

> **Status: RATIFIED 2026-11-09.** The PI ratified the freeze in session:
> cohorts 2014Q1-2015Q4 unlocked in `configs/splits.yaml`, D-6 (search
> spaces) and D-7 (feature groups) confirmed as drafted, T4.6 compute
> caps accepted, `protocol/PROTOCOL_FROZEN` created, tag `protocol-v1`.
> See D-8 in `notes/DECISIONS.md`. The checklist below is retained as
> the record of what was ratified.

## Counter-signed artifacts the freeze consumes

| Artifact | Status | Where |
|---|---|---|
| Test cohorts 2014Q1-2015Q4 (GATE4) | PASS 200/200 cells | `protocol/GATE4.md`, `configs/splits.yaml` (test: null until freeze) |
| Hypothesis register (8 confirmatory + exploratory) | PROPOSED | `protocol/HYPOTHESES.md` |
| Compute caps (SHAP <= 10k rows/config; headline B = 2000) | PROPOSED from T4.6 | `protocol/GATE4.md` |
| Hyperparameter search spaces + sampler rules | PROPOSED (D-6) | `configs/search_spaces.yaml`, `protocol/SEARCH_SPACES.md` |
| Feature-group partition (6-way) + resolver | PROPOSED (D-7) | `configs/feature_groups.yaml`, `protocol/FEATURE_GROUPS.md` |
| Broad/deep sweep sizes and selection rule | configured | `configs/models.yaml` |

## Open decision items to close at freeze

- **D-3, D-4, D-5** (from GATE4): the PI decides; see `notes/DECISIONS.md`.
- **D-6** search spaces: confirm ranges or edit before freeze.
- **D-7** feature groups: confirm partition, event-absence grouping,
  inquiry placement.
- Test window in `configs/splits.yaml`: replace `test: null` with the
  ratified 2014Q1-2015Q4 cohorts. Until then the shared loader REFUSES to
  construct test cohorts by design (`scripts/_common.py` guard).

## What the freeze changes mechanically

1. `HYPOTHESES.md` becomes verbatim-frozen protocol text (PI edits ONLY
   at this step, with strikes + reasons); Holm is applied ONCE, after
   all confirmatory tests, post-freeze.
2. Any code/data change after the freeze goes in
   `protocol/DEVIATIONS.md` with: what, why, and whether test results
   were already seen when it was made.
3. `scripts/render_paper.py --source main` unlocks (requires
   `protocol/PROTOCOL_FROZEN`); the real-cohort runners stop dry-run
   mode and the I-11 export switches to `paper/numbers.json`.
4. The dry-run tables (`reports/tables/*_dryrun.csv`) stay in the repo
   as plumbing-scale artifacts; they are never cited as results.

## Pre-freeze verification state (as of this draft)

- Suite: 222 passed, ruff clean, AI-hygiene sweep clean.
- All six Phase-5/7/8/9 runners dry-ran end to end: baselines,
  calibration audit (+ per-arm reliability-bin segments), decisions,
  noise floor (seed AND bootstrap-refit axes, grouped battery),
  sample size, broad sweep (+ deep-subset selection).
- Figure preforms render deterministically from the committed dry-run
  tables (reliability grid + sweep scatter, dpi=150).
- The equal-mass binning count regression (off-by-two) was found by the
  figure work and fixed with a count-pinning test; affected tables
  re-ran. Dry-run tables are re-derivable byte-identically at 1e-9
  from the committed code and seeds.
- Reproducibility tests pass at 1e-9 on full-path re-runs.
