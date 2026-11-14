# notes/REVIEWS.md — review log (append only)

Format:

```
### R-<task id> reviewer: <initials>  date:
Checked: <files, formulas, tests>  Findings: <list>  Verdict: PASS | PASS_WITH_FIXES | FAIL
```

Rules (CONTEXT.md §9): the builder-side review checks statistical code and
formulas for implementation errors and leakage; the analyst-side review checks
metric definitions against `CONTEXT.md` §7 and every manuscript number against
`numbers.json`. A task is done only after review, and Phase gates 7–10
additionally require a PASS verdict.

Solo-coverage note: the PI covers external contact only, so reviews below are
logged with a role label (Builder-review / Analyst-review) instead of a second
person's initials; every entry still genuinely re-derives or re-checks the
named artifacts, and a PASS_WITH_FIXES verdict is followed by the fix commit.

### R-7.2 reviewer: Builder-review (statistical code)  date: 2026-11-07
Checked: `src/calibration/audit.py` (citl_and_slope, ece, brier_decomposition,
ae_ratio), `src/calibration/recalibration.py` (intercept/platt/isotonic arms,
clipping), metric definitions against CONTEXT.md §7 line by line; runners
`run_calibration.py`, `run_sample_size.py` for metric usage.
Findings:
1. DEFINITION ERROR (fixed): `citl_and_slope` returned the intercept from ONE
   free binomial fit `y ~ a + b*logit(p)`, but §7 defines CITL via an OFFSET
   fit (`y ~ 1` with `logit(p)` as offset, slope pinned to 1) and the slope
   via a SEPARATE free fit. The offset intercept and the free-fit intercept
   are different parameters whenever the slope is not 1. Fixed:
   `fit_offset_intercept()` in recalibration.py now backs both the
   "intercept" recalibration arm and the audit metric (single source of
   truth, discriminator test pins offset-intercept != free intercept under
   slope != 1).
2. CLIP LEVEL (fixed): logits were clipped at 1e-12; §7 says clip p to
   [1e-6, 1-1e-6] before ANY logit. `_EPS` set to 1e-6 in recalibration.py;
   audit's citl_and_slope clips at 1e-6.
3. MISSING METRIC (fixed): the A/E ratio (mean(y)/mean(p), ideal 1) from §7
   was absent. Added `ae_ratio()` + tests + a runner metric column.
4. BINNING (fixed): runners' primary ECE used equal-width bins; §7's
   definition is adaptive EQUAL-MASS bins. Primary metrics now use
   `quantile_edges=True` (B=15; the 10/20/30 sensitivity rows remain).
5. Verified correct (no change): Brier decomposition identity, slope
   definition (free binomial GLM on logit(p)), decision p* = g/(g+L)
   break-even, Holm monotone envelope, paired-bootstrap shared-resample
   design, disjoint-cohort iron rules, explainer background sourced from
   the training window only.
Additional suite hardening observed during review: A/E synthetic tests
initially used rel=1e-6 tolerances at n=50k, which is BELOW the Bernoulli
sampling noise (~1.4% at that size); loosened honestly to 2%.
Verdict: PASS_WITH_FIXES -> fixes applied in the same commit set; suite 206
passed; all affected dry-run tables re-run.

### R-9.4 reviewer: Builder-review (decision layer)  date: 2026-11-08
Checked: `src/decisions/policy.py` (breakeven_threshold, threshold_policy,
fixed_rate_policy, stylised_utility), `src/decisions/flips.py`
(flip_flows, flip_delta_utility, matched_null_total_delta_utility,
distance_to_nearest_threshold), and `scripts/run_decisions.py` anchor use
against the plan's RQ3 definitions (decisions.yaml A7 stylised scenario).
Findings:
1. Verified: p* = 1/(1+L) reproduces the break-even of the stylised
   account ((1-p)g - pL > 0 iff p < g/(g+L)).
2. Verified: three-band codes are monotone with a deterministic tie rule
   (strict `<` throughout: p = t_low is REFER, p = t_high is DECLINE);
   fixed-rate threshold carries the same ties-to-not-approved rule.
3. Verified: delta-utility is per-loan u(after) - u(before) with funded
   good +g / funded default -gL, refer/decline 0; `util_flipping_loans`
   counts only utility-affecting flips.
4. Verified: the matched null permutes AFTER-scores only, so marginal
   decision rates are identical to the observed arm and only the
   score-outcome pairing is chance; resampling is under a single seeded
   Generator (deterministic).
5. Verified: runner anchors t_low/t_high = p* -/+ h from config (primary
   ratio 5 refer half-width 0.02; sensitivity ratios and half-widths all
   from decisions.yaml, none hard-coded).
No defects found; the layer's earlier dry-run numbers are reproducible
under the same seeds. Verdict: PASS.

### R-7.2 addendum (binning fix) date: 2026-11-08
While wiring the reliability-diagram export (T7.x figure preform), the
new `reliability_bins` helper surfaced an off-by-two in the shared
quantile-binning machinery (`_bin_masks`): the equal-mass path derived
n_bins+2 inner cut counts from the edges template and therefore
delivered n_bins+2 masks, so `ece_b15` in previously exported dry-run
tables (calibration_audit, sample_size) was an adaptive ECE over 17
bins, not the protocol's B=15. Fixed in `src/calibration/audit.py`
(the template length encodes B: B-1 inner cuts at k/B), pinned by a
new regression test that asserts the quantile path and the fixed-edge
path deliver exactly B masks for B = 10/15/20/30 on continuous scores.
`run_calibration` and `run_sample_size` re-ran; curves table and the
figures follow. Lesson logged: count-pinning tests belong with any
bin-count parameterised machinery.

### R-8.5 reviewer: Builder-review (grouped explanation machinery)  date: 2026-11-08
Checked: `src/explain/groups.py` (group_attributions,
grouped_permutation_importance, _validate_group_partition) and
`src/explain/stability.py` comparator usage inside the grouped noise-floor
extension of `scripts/run_noise_floor.py`.
Findings:
1. Verified: the partition validator enforces a strict cover of ALL
   feature indices with no overlap; both grouped metrics re-validate on
   every call, so a malformed partition cannot pass silently.
2. Verified: group attribution = mean over rows of |sum of group SHAP|
   (the T8.5 wording), matching the paper's group-level mass definition.
3. Verified: grouped permutation importance applies ONE row permutation
   to the WHOLE group block per repeat (`Xp[:, cols] = Xa[idx][:, cols]`),
   which preserves the within-group joint distribution and destroys only
   the cross-group association - the correct joint scheme; per-column
   independent shuffling would leak intra-group structure and
   underestimate group importance.
4. Verified: sign convention (loss: permuted - baseline; gain: baseline -
   permuted, positive = informative) is consistent with the T8.5 test
   suite's loss/gain pairing.
5. Verified: single seeded Generator consumed in group order, so the
   full grouped-importance report is reproducible run-to-run given the
   same group dict (ordered as returned by the T5.4 resolver).
6. Residual caveat (documented, not blocking): correlated groups (e.g.
   credit_score vs utilization) distort joint-permutation readouts;
   Hooker, Mentch & Zhou 2021 is cited in protocol/FEATURE_GROUPS.md and
   the methods will carry the caveat - an accepted limitation, not an
   implementation defect.
Verdict: PASS.

### R-11.1 Analyst-review (paper scaffolding vs numbers discipline)  date: 2026-11-08
Checked: `paper/manuscript.md` (stub), `paper/README.md`,
`paper/manuscript_rendered.md` (generated), `paper/claims.csv`,
`scripts/render_paper.py`, and the CONTEXT 11 discipline ("numbers are
NEVER typed by hand").
Findings:
1. DEFECT (fixed): the stub's example placeholder was an INDENTED code
   block, and the renderer resolved it, embedding the plumbing-scale
   LR AUC 0.620923 directly in the rendered stub - exactly what the
   README's own "do NOT cite dry-run numbers in prose anywhere" forbids
   at the prose boundary. Fix: the renderer now splits on fences and
   leaves placeholders inside fenced code blocks literal (fenced
   content is documentation, not claims); the stub example moved into a
   proper fence; README documents the rule; a regression test pins that
   unfenced placeholders still resolve, missing keys are still a hard
   error, and fenced keys stay untouched.
2. Verified: the placeholder-count print skips fenced segments, and
   `--source main` still refuses to run without `protocol/PROTOCOL_FROZEN`.
3. Verified: `claims.csv` has the fixed header and stays empty while the
   stub carries no empirical sentences (schema test exists).
4. Non-issue investigated: the heading em-dashes look garbled in
   console output but the file bytes are valid UTF-8 U+2014 - no file
   corruption, no change made.
Verdict: PASS_WITH_FIXES -> fixes applied in the same commit set.




### R-13.1 Builder-review (post-freeze statistics vs frozen definitions)  date: 2026-11-14
Checked: `scripts/run_real_statistics.py` against CONTEXT.md SS7 and
`configs/statistics.yaml`: H1a cluster bootstrap over the 20 deep
configurations of |CITL_cohort| - |CITL_validation| (offset-GLM CITL,
slope pinned at 1; clip 1e-6 before logit); H1c pooled recovery ratio
with the frozen estimator and lag rank correlation (DEV-2 axis:
label age 3-10 quarters); H2c share of family-by-group cells below the
2.5th percentile of the pooled seed+bootstrap group-cosine floor; H3a
recalibration flip rate minus the seed-floor q95 at the frozen policy
cell (p* approx 0.167, refer band +-0.02); RQ4.1-4.4 pooled within-pair
Spearman deltas on ONE shared cluster bootstrap design (B=2000,
STAT_SEED=20261109); Holm applied once over the 8 confirmatory
p-values. Verified: leakage guards untouched (no test rows in any fit);
explicit seeds only, no hash(), no wall-clock seeding; battery
per-family crash-safe writes and merge refuse schema drift;
`export_numbers --source main` adapters read only `reports/tables/`.
Findings:
1. Verified the H3a sign convention end to end: a NEGATIVE statistic
   means recalibration flips fewer loans than the seed floor. The
   manuscript prose was checked to state this direction (SS7, SS10,
   abstract, conclusion) - earlier draft language ("exceeds the floor")
   was corrected before the release commit.
2. Verified DEV-3 companions (median recovery, negative share, mean
   Brier change) are descriptives only and do not enter any
   confirmatory estimator; the Holm family is unchanged.
3. Non-issue investigated: LR's positive H3a excess (+0.01094) traces
   to its exactly-zero seed floor (deterministic solver), not to a
   large flip rate; documented in the manuscript prose.
Verdict: PASS.

### R-13.2 Analyst-review (manuscript numbers, claims register, release hygiene)  date: 2026-11-14
Checked: every empirical number in `paper/manuscript.md` (176
placeholders) against `paper/numbers.json` via the strict renderer
(zero missing keys, zero `{{n:` left in `manuscript_rendered.md`);
every empirical sentence against `paper/claims.csv` (34 rows; exact
substring test + key-existence test both pass; keys are main-prefixed
and trace to `reports/tables/`); formulas vs CONTEXT.md SS7; figure
captions vs the plotted tables (F5 caption corrected to "mean Brier
change by family, arm and label age" to match
`recovery_by_lag_real.png`); GATE1 neutrality (SS2 note intact, no
bibliography, no priority/"first" claim added); banned-phrase grep
clean; Table 3 vs `real_statistics.csv` re-derived independently from
the printed statistics (values match to the printed precision).
Findings:
1. Verified the two direction-sensitive readings: H1c is NOT described
   as partial recovery anywhere (denominator warning present, DEV-3
   companions carry the interpretation); H3a is described as FEWER
   flips than reseeding, never as exceeding the floor.
2. Verified `notes/STATUS.md` numbers are observed values copied from
   `real_statistics.csv`, not re-typed estimates.
3. Verified the full-suite count (243 passed) and ruff result recorded
   in `notes/STATUS.md` match the final release run.
Verdict: PASS.
