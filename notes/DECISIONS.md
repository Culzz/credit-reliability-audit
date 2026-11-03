# notes/DECISIONS.md — append-only decision log

### D-1 2026-11-02 Repository host, location and push setup
Context: work needs a private GitHub repository for protocol history and reproducibility.
Options: create the repo with the GitHub CLI vs create it through the web interface.
Decision: created via the web interface (`https://github.com/Culzz/credit-reliability-audit.git`, private) after `winget install GitHub.cli` stalled during download; plain `git` over HTTPS handles commits and pushes, and the credential manager stores authentication outside the repository.
Consequence: no CLI dependency to reproduce pushes; if repo management features are needed later, revisit installing the CLI.

### D-2 2026-11-02 Working document naming
Context: the plan's collaboration-logging directory needed a neutral name and wording that fits a professional research repository.
Options: keep the original logging-directory name vs rename to `notes/` and rewrite the three templates.
Decision: renamed to `notes/` (`STATUS.md`, `DECISIONS.md`, `REVIEWS.md`); same review obligations, same templates, no changes to the obligations themselves.
Consequence: file paths in `CONTEXT.md` §8 and `PLAN.md` updated to match; future references use `notes/`.

### D-3 2026-11-02 Legacy `loan_status` variants (T3.1) — collaborator-proposed, PI to confirm

Context: the inventory (`reports/tables/loan_status_counts.csv`) shows nine exact
strings; two are legacy policy-exception texts recorded identically by value
except for the prefix: `Does not meet the credit policy. Status:Fully Paid`
(1,988) and `Does not meet the credit policy. Status:Charged Off` (761); 2,749
of 2,260,668 loans (0.12%) in total. There is no `Issued` status in the file.

Options:
1. Merge legacy variants into their underlying status (Fully Paid / Charged
   Off) and include.
2. Drop all 2,749 legacy loans.
3. Keep as separate classes (do nothing).

Decision (proposed by the Builder, awaiting PI confirmation): Option 1 —
treat legacy variants as their underlying status for the target; the prefix
is a 2007–2009 credit-policy artefact, not an outcome difference, and 0.12%
is too small to justify losing rows this early in the panel. Sensitivity
check S-legacy proposed: exclude the 2,749 rows and re-check the RQ1
headline deltas move by less than noise.

Consequence (if confirmed): `build_target()` default `legacy="merge"`;
cohort feasibility tables unchanged; the four non-legacy unresolved
statuses (`Current`, `In Grace Period`, `Late (16-30 days)`,
`Late (31-120 days)`) remain excluded from the target, `Default` (40 loans,
legacy-era only) counts as default per `configs/data.yaml`.

### D-4 2026-11-03 T3.5 sensitivity-parked columns — PI review item
The column audit parked four fully-populated, low-risk columns out of the main feature
set pending dictionary verification: `emp_title` (free text, high cardinality),
`addr_state` (design-lean choice), `disbursement_method` (community-dictionary vintage
uncertainty), `funded_amnt`/`funded_amnt_inv` (funding fixed at issuance, near-duplicate
of `loan_amnt`). All are recorded with `uncertain` evidence in `data/column_audit.csv`.
Decision needed: confirm the parking for the frozen protocol.

### D-5 2026-11-03 Schema-drift rule outcomes (T3.7) — PI review item
Measured on 36-month loans across the PROPOSED maturity window 2007Q2–2015Q4 with the
0.90 populated-share rule (`reports/tables/schema_drift.csv`):
(a) 9 classic bureau/application fields (`acc_now_delinq`, `delinq_2yrs`, `delinq_amnt`,
    `earliest_cr_line`, `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_util`, `total_acc`)
    fail in 2007Q2 **only** (a 24-loan micro-vintage with 3 blank records; populated 87.5%).
    Proposal: modeling cohorts start 2008Q1, preserving all 24 candidate features; the
    alternative (dropping features to keep 2007Q2) is disproportionate. 2007Q3–Q4 (579
    loans) pass the rule and also pass by joining the 2008 cohorts without features.
(b) the three months-since-event fields (`mths_since_last_delinq`,
    `mths_since_last_major_derog`, `mths_since_last_record`) fail in essentially every
    vintage — their blankness is event absence, not unavailability. Dropped by the
    mechanical rule; an event-absence recoding is only legitimate after the data
    dictionary confirms the semantics (uncertain flag stays until then).
Decision needed: (a) cohort start 2008Q1, and (b) keep the mechanical drop for the
frozen protocol.

