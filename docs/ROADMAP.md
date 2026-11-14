# ROADMAP — remaining work (post-release, dated 2026-11-14)

The study is complete and released (`protocol-v1`, `v1.0`). The items
below come from an external review of the released repository and are
ordered roughly as the review's 5-week plan. They are grouped by what
blocks them: **[PI]** needs an account or personal action, **[RAW]**
needs the raw LendingClub file, **[CODE]** is doable from the committed
tables or code. The frozen confirmatory results are never re-edited;
everything below lands as dated, labelled addenda
(`protocol/DEVIATIONS.md`, post-hoc appendices) with the originals kept.

## Week 1 — registration, deviations, literature

1. **[PI] OSF registration (retroactive).** Upload the frozen
   `protocol/` files to osf.io/registries; state the `protocol-v1`
   commit hash and timestamp. In the paper, say plainly that
   registration happened after freezing and before confirmatory
   analysis, with the commit history as evidence — "frozen and tagged
   before confirmatory analysis; deposited on OSF on [date]" — and do
   not call it "preregistered" unless the registry timestamp precedes
   the analysis.
2. **[DONE 2026-11-14] Deviation table in the paper** (Appendix A) with
   date, reason, before/after-results timing and effect on conclusions.
3. **[PI] Literature gate (GATE1).** Verified matrix (citation, link,
   what it shows, verified date) from primary sources only: Ovadia et
   al. (uncertainty under shift), Miller et al. (accuracy vs OOD),
   Guo et al. (calibration), Slack et al. (explanation fragility),
   Rashomon-set literature, Gama et al. (concept drift), credit-specific
   drift/calibration studies, SR 11-7 and EU AI Act Annex III quoted
   from official text. Related-work draft per the four strands. If a
   paper already answers part of the question, reposition.

## Week 2 — RQ4 depth (needs deep-battery tables; some need raw data)

4. **[DONE 2026-11-14] RQ4 decomposition** (Appendix C.1):
   between-configuration, within-configuration and lead-lag analyses
   with cluster-bootstrap CIs over the 20 deep configurations.
5. **[DONE 2026-11-14] Estimand defined in Methods** (§4.7): paired
   delta of Spearman correlations, units, "bad" direction, comparison
   not absence claim.
6. **[DONE 2026-11-14] ρ measurement-noise floor** (Appendix C.1):
   split-half over seed refits; noise negligible.
7. **[DONE 2026-11-14] Equivalence-region prose**: only the
   slope-deviation outcome is classified meaningful; the others are
   inconclusive by the pre-registered |ρ| < 0.3 rule.
8. **[RAW] RQ4 on all 120 sweep configurations at cohort level** — the
   cheap per-cohort metrics (AUC, CITL, Brier) must be scored for the
   non-deep configurations; strengthens RQ4.1's power.
9. **[CODE] Hierarchical/two-level bootstrap sensitivity** (resample
   configurations and cohorts) and BCa intervals vs the current
   percentile CIs, as a labelled robustness appendix; state the cluster
   count in every table (currently in the table scopes and §4.6).

## Week 3 — recalibration mechanism, confounds, shift

10. **[DONE 2026-11-14] H1c absolute scale** (Appendix C.2): per-family
    ΔBrier with CIs, per-label-age cell means, oracle-gain distribution
    (87.5% of cells below 1e-3 Brier units), CITL change per family.
11. **[RAW] Calibrator-family comparison**: beta calibration,
    temperature scaling, calibration-set size as a factor, versus
    Platt/isotonic — turns "tree arms got worse" into a mechanism.
    Requires refits.
12. **[CODE/PI] Equal tuning budget check**: report the per-family
    split of the 120 configurations and a best-k-per-family
    sensitivity (the split is visible in `sweep.csv`).
13. **[DONE 2026-11-14] Shift dose-response** (Appendix C.3): shift
    magnitude (adversarial AUC) vs calibration loss, ρ ≈ 0.71 over the
    8 cohorts. **[RAW] Extend**: PSI/base-rate decomposition and
    covariate-vs-label-shift attribution via reweighting.

## Week 4 — sensitivity and external validity

14. **[DONE 2026-11-14] Cohort-granularity sensitivity** (Appendix C.4,
    half-year statistic).
15. **[DONE 2026-11-14] p-value floor** (B = 20,000 sensitivity,
    Appendix C.5, side files `real_statisticsb20k.csv`).
16. **[RAW] Maturity-window sensitivity**: stricter 36/60-month
    maturity rule to rule out late-cohort censoring masquerading as
    calibration drift.
17. **[RAW] Threshold sensitivity for decision stability**: flip rates
    under alternative threshold sets; justified-flip analysis (realised
    default rates of flipped loans before vs after).
18. **[PI] Second dataset**: Freddie Mac / Fannie Mae single-family
    performance data (timestamped, public) for one replication of the
    RQ4 headline; verify license and time fields before committing.
19. **[CODE] SHAP-variant sensitivity** (interventional vs
    path-dependent, background size) and alternative feature-groupings,
    as a labelled appendix.
20. **[CODE] Brier decomposition** (calibration vs refinement) and
    reliability diagrams for selected cohorts.

## Week 5 — manuscript and release

21. **[PI] Venue selection, arXiv posting, Zenodo DOI** (tag `v1.0`
    exists; create the GitHub release, then archive on Zenodo and cite
    via `CITATION.cff`).
22. **[CODE] Claim-register language-strength column** ("we find" /
    "consistent with" / "we cannot rule out") matched to evidence.
23. **[PI] CircleCI enablement**: connect the repository at
    circleci.com (free tier); the committed `.circleci/config.yml` runs
    ruff + the full suite. The GitHub Actions route is blocked by an
    account billing lock, hence CircleCI.
