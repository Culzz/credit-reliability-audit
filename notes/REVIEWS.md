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


