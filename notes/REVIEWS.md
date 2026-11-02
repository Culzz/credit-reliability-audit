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
