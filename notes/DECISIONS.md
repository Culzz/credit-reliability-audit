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
is too small to justify losing rows this early in the panel. Optionally run a
sensitivity check S-legacy: exclude the 2,749 rows and re-check the RQ1
headline deltas move by less than noise.

Consequence (if confirmed): `build_target()` default `legacy="merge"`;
`splits`/cohort feasibility tables unchanged; the five non-legacy unresolved
statuses (`Current`, `In Grace Period`, `Late (16-30 days)`,
`Late (31-120 days)`) remain excluded from the target, `Default` (40 loans,
legacy-era only) counts as default per `configs/data.yaml`.
