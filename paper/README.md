# paper/ — manuscript, numbers, claim register

Manuscript numbers are NEVER typed by hand (CONTEXT.md 11):

- `manuscript.md` — the source; empirical values appear only as
  `{{n:key}}` placeholders. Structure follows CONTEXT.md (Abstract, Intro,
  Related work, Data and protocol, Methods, Results RQ1-RQ3, AUC analysis
  RQ4, Monitoring recommendations, Discussion, Limitations, Conclusion,
  Reproducibility statement).
- `numbers_dryrun.json` — plumbing-scale dry-run export built by
  `scripts/export_numbers.py` from `reports/tables/*_dryrun.csv`; every key
  carries the loud pre-freeze warning. NOT research results.
- `numbers.json` — the real export, produced only after the freeze from the
  frozen-cohort tables. `scripts/render_paper.py --source main` refuses to
  run without `protocol/PROTOCOL_FROZEN`.
- `claims.csv` — the claim register: `claim_id, sentence, evidence_file,
  number_keys, status(SUPPORTED|PARTIAL|UNSUPPORTED)`. No sentence making
  an empirical claim may exist in the manuscript without a row here; the
  register currently holds 34 rows (30 auto-extracted empirical sentences
  plus 4 qualitative rows), and enforcing tests check that each `sentence`
  is an exact substring of `manuscript.md` and that every key exists in
  `numbers.json`.
- `manuscript_rendered.md` — renderer output (never edited directly).

Rendering is strict: a placeholder without a value fails the build.
Placeholders inside fenced code blocks are NOT rendered: a fenced
example documents the write-it-with-a-placeholder discipline instead of
resolving into a dry-run number sitting in prose.
