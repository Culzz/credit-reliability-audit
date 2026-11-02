# notes/DECISIONS.md — append-only decision log

### D-1 2026-11-02 Repository host, location and push setup
Context: work needs a private GitHub repository for protocol history and reproducibility.
Options: create the repo with the GitHub CLI vs create it through the web interface.
Decision: created via the web interface (`https://github.com/Culzz/credit-reliability-audit.git`, private) after `winget install GitHub.cli` stalled during download; plain `git` over HTTPS handles commits and pushes, and the credential manager stores authentication outside the repository.
Consequence: no CLI dependency to reproduce pushes; if repo management features are needed later, revisit installing the CLI.

### D-2 2026-11-02 Working document naming
Context: the plan's collaboration-logging directory needed a neutral name and wording that fits a professional research repository.
Options: keep `agents/` as-is vs rename to `notes/` and rewrite the three templates.Decision: renamed to `notes/` (`STATUS.md`, `DECISIONS.md`, `REVIEWS.md`); same review obligations, same templates, no changes to obligations themselves.
Consequence: file paths in `CONTEXT.md` §8 and `PLAN.md` updated to match; future references use `notes/`.
