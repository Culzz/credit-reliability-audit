# protocol/HYPOTHESES.md — frozen hypothesis register

> **Status: STUB.** Frozen at Gate 5 by T5.2 (Analyst drafts, PI confirms).
> Each entry gets: test statistic, direction (only if literature-justified),
> threshold, practical effect size, confirmatory/exploratory label, plus the
> H4 equivalence region and the Holm family (see `configs/statistics.yaml`).

## Provisional register (from CONTEXT.md §3; NOT yet frozen)

**Confirmatory family (Holm correction):** H1a, H1c, H2c, H3a, and four RQ4
association tests (AUC vs ECE; AUC vs |slope−1|; AUC vs explanation
instability; AUC vs flip rate). Everything else is exploratory in the paper.

| Hypothesis | Direction (if justified) | Test statistic | Threshold | Effect size | Status |
|---|---|---|---|---|---|
| H1a | higher intercept deviation | calibration-in-the-large Δ vs validation | TBD before freeze | TBD | PROPOSED |
| H1b | — (exploratory) | slope Δ vs validation | TBD | TBD | PROPOSED/exploratory |
| H1c | partial recovery only | recovery fraction by arm/lag | TBD | TBD | PROPOSED |
| H1d | isotonic less stable small-n | Platt-vs-isotonic comparison | TBD | TBD | PROPOSED/exploratory |
| H2a | — (exploratory) | Axis A stability metrics | TBD | TBD | PROPOSED/exploratory |
| H2b | — (exploratory) | Axis B reliance drift | TBD | TBD | PROPOSED/exploratory |
| H2c | temporal > noise floor | similarity vs noise-floor 2.5th pct | TBD | TBD | PROPOSED |
| H2d | disagreement some groups | Kendall τ SHAP vs grouped perm/drop-group | TBD | TBD | PROPOSED/exploratory |
| H3a | flips exist | flip rate > 0 with CI | TBD | TBD | PROPOSED |
| H3b | flips near thresholds | distance-to-threshold distribution | TBD | TBD | PROPOSED/exploratory |
| H3c | heterogeneous outcome association | ΔU vs matched null | TBD | TBD | PROPOSED/exploratory |
| H4a | AUC→AUC stronger than AUC→reliability | cohort-avg Spearman + rank reversal | TBD | TBD | PROPOSED |
| H4b | limited association ≥ some outcomes | ρ with equivalence region | region (−0.3, 0.3) pre-set | TBD | PROPOSED |
