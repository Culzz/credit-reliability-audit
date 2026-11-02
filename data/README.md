# data/README.md — dataset provenance (filled by the PI in T2.1–T2.2)

> **Status: TEMPLATE.** `data/raw` is gitignored on purpose: raw data is never
> committed. This file (committed) must let another researcher obtain the
> exact same dataset without you sending them a file.

## Source

| Field | Value |
|---|---|
| Platform | Kaggle |
| Dataset | `wordsforthewise/lending-club` |
| File | `accepted_2007_to_2018Q4.csv.gz` |
| Dataset version at download | file creation date 2019-12-17 (see Kaggle file listing) |
| Date accessed | 2026-11-02 |
| Source URL | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Dataset owner | Nathan George (`wordsforthewise`) — verified via dataset file listing; confirm description page wording |

## Integrity

| Field | Value |
|---|---|
| File size (bytes) | 392,582,231 |
| SHA-256 | `55C16F75120F897683F02E7AABCF080D0E4A20C4832FEB1D592CFA941BD62A2D` |
| Rows read from CSV | 2,260,701 |
| Non-loan footer rows removed (non-numeric `id`) | 33 |
| Parquet rows written | 2,260,668 |
| Columns | 151 |

## Licence & terms

| Field | Value |
|---|---|
| Kaggle licence metadata shown at download | `License(s): CC0-1.0` (as returned by the Kaggle API at download, 2026-11-02) |
| Underlying LendingClub terms reviewed | PENDING — inspect the dataset description page before publishing anything derived |
| Redistributable derived data? | PENDING — if unclear at release, do NOT publish derived data; publish code + summary statistics only |

Reference counts (2.26 M rows × 151 cols, ~1.68 GB uncompressed, ~393 MB
gzipped) were used as expectations to check against at the profiling step;
they are not guarantees.

## Transformations

1. Parquet conversion via DuckDB with footer-row removal by numeric-`id` check
   (T2.3) — output `data/interim/accepted_raw.parquet` (ZSTD compression),
   33 of 2,260,701 rows removed.
2. Everything downstream is recorded in `notes/DECISIONS.md` and scripts.
