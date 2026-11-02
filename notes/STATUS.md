# notes/STATUS.md — working status (update at the end of every work session)

## Current phase
Phase 0 (infrastructure, working toward Gate 0). Next: Phase 1 literature review; Phase 2 data acquisition once the Kaggle download completes.

## Responsibilities on this project
- Implementation and testing: dataset access, features, models, metrics, explanations, decision code, run scripts (Builder duties).
- Literature matrix and reading notes, protocol text, statistical analysis plan, manuscript drafting, claim register (Analyst-Writer duties).
- Heavy compute, citation verification, parameter confirmations, protocol freeze, all external contact: me (PI).

Every substantive change is reviewed by the other collaborator before it counts (`notes/REVIEWS.md`).

## Completed 2026-11-02
- Repository structure and working documents (initial commit), decisions log opened.
- Eight PROPOSED configuration files from the parameter table (`configs/*.yaml`); nothing hard-coded in code.
- Dataset access guard `src/data/guard.py` + split loader `src/data/loader.py`; **18/18 tests passing**, ruff clean.
- Python 3.11.9 environment locked with uv: 209 packages in `uv.lock` (pandas, sklearn, xgboost, lightgbm, shap, statsmodels, duckdb, pyarrow, mlflow, ...).
- `agents/` directory renamed to `notes/` (three logging templates re-worded; obligations unchanged).
- Kaggle dataset verified with API file listing (`accepted_2007_to_2018Q4.csv.gz`, 392,582,231 bytes — matches the blueprint reference); accepted-loans file downloaded.

## Gate 0 evidence so far
Repo pushed · environment locks · tests pass · access-guard test proven. Still missing for sign-off: MLflow experiments registered (T0.4), Zotero collection + `lit/matrix.csv` (T0.8), `RESEARCH_RISKS.md` completed with owners and triggers (T0.7), then my own sign-off here.

## Next
- T0.4 register the ten MLflow experiments.
- T0.7 finish the research-risks register.
- T1.1 literature search plan (`docs/blueprint.md` §B-1); first reading notes on the two closest competitor papers; `lit/matrix.csv` with the 22 columns (§B-2).
- T2.1–T2.3 provenance record (SHA-256, licence notes in `data/README.md`), Parquet conversion, dataset profile.

## Blockers
- Kaggle API token expires per Kaggle settings; used only in-process for the download, never stored in the repository.
- Legacy `loan_status` handling ("Does not meet the credit policy" variants) — decide at T3.1.
- `emp_title` / `addr_state` treatment — decide with the column audit.
- Everything in `configs/*.yaml` stays PROPOSED until Gates 4–5.
