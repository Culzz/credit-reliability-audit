# Working title (placeholder) — to be written post-freeze

> Status: STUB. Drafted only after the protocol freezes and the frozen-cohort
> tables exist; the numbers pipeline is already live so that every empirical
> value enters as a placeholder. This stub contains NO empirical sentences,
> hence the empty claim register (`claims.csv`) is consistent by design.

Placeholder syntax (double-brace `n:`-placeholders as documented in
paper/README.md) resolves from `numbers_dryrun.json` in dry-run mode or
`numbers.json` (--source main) after the freeze. An example key that
resolves in dry-run mode (do NOT cite dry-run numbers in prose anywhere):

    0.620923

## Planned structure (CONTEXT.md)

1. Abstract (about 200-250 words; written last)
2. Introduction
3. Related work (compare, do not list)
4. Data and protocol
5. Methods (RQ1-RQ3)
6. Results (RQ1: calibration degradation and recalibration recovery; RQ2:
   explanation stability; RQ3: decision stability)
7. AUC analysis (RQ4: does AUC predict the reliability outcomes?)
8. Monitoring recommendations
9. Discussion
10. Limitations (accepted-loan selection; one US platform and period;
    simulated label-arrival times; repeated borrowers; SHAP assumptions;
    permutation extrapolation under correlated features; simulated
    thresholds and cost ratios; limited external validity)
11. Conclusion
12. Reproducibility statement
