# Reproduction: environment, runtime, hardware

## Environment

- Python 3.11.9, locked with `uv` (`uv.lock`, 200+ packages). There is no
  floating dependency: `uv sync` reproduces the environment.
- No GPU is used anywhere in the pipeline; everything runs on CPU.
- Reference machine for all timings below: consumer laptop, 8 CPU cores,
  32 GB RAM, Windows. The pipeline is single-process per stage.

## Stage runtimes (observed on the reference machine)

| Stage | Command | Wall clock | Reads | Writes |
|---|---|---|---|---|
| Raw ingest + profile | `scripts.load_raw` → `scripts.profile_dataset` | ~10 min | raw CSV (2.26 M rows) | Parquet + profile tables |
| Adversarial validation / shift measures | `scripts.adversarial_validation`, `scripts.shift_measures` | ~5 min | Parquet | `adversarial_validation.csv`, `shift_measures_*.csv` |
| Broad sweep (30 configs × 4 families) | `python -m scripts.run_real_sweep` | ~2.5 h (RF dominant) | Parquet | `sweep.csv` |
| Deep battery (20 configs × 10 refits, cached predictions, recalibration, flips, stability) | `python -m scripts.run_deep_battery --families ...` | ~14 h total (RF ~7 h alone; LR+XGB+LGBM ~7 h) | Parquet | `*_real__<family>.csv` per family, then merged `*_real.csv` |
| Confirmatory statistics | `python -m scripts.run_real_statistics` | ~77 s (B = 2,000) | merged `*_real.csv` | `real_statistics.csv`, `real_descriptives.csv`, floors |
| Statistics sensitivity | `python -m scripts.run_real_statistics --b-headline 20000 --suffix b20k` | ~13 min | same | side files only |
| Paper tables / verdicts / figures | `run_paper_tables`, `report_verdicts`, `run_real_figures` | ~30 s | statistics + battery | `paper_table*.csv`, figures |
| Numbers export | `python -m scripts.export_numbers --source main` | ~10 s | `reports/tables/` | `paper/numbers.json` |
| Manuscript render | `python -m scripts.render_paper --source main` | ~1 s | `numbers.json` | `paper/manuscript_rendered.md` |
| Post-hoc analyses (Appendix C) | `python -m scripts.post_hoc_analyses` | ~2 min | per-config tables | `post_hoc_analyses.csv` |
| Test suite | `python -m uv run pytest -q` | ~97 s | synthetic fixtures only | — |

Timings without raw data: every stage from "Confirmatory statistics"
downward rebuilds from the committed `reports/tables/` alone.

## Determinism

- All stochastic steps use explicit literals: `BASE_SEED = 20180101`
  (model/battery axes), `STAT_SEED = 20261109` (bootstrap inference).
- No `hash()` calls, no wall-clock seeding anywhere in committed code
  (enforced by `tests/test_reproducibility.py`).
- The statistics stage is deterministic given the committed tables;
  `run_real_statistics` twice produces identical `real_statistics.csv`.
