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
| Dataset version at download | TO_RECORD |
| Date accessed | TO_RECORD |
| Source URL | TO_RECORD |
| Dataset owner | Nathan George (`wordsforthewise`) — VERIFY on the page |

## Integrity

| Field | Value |
|---|---|
| File size (bytes) | TO_RECORD |
| SHA-256 | TO_RECORD (`Get-FileHash ... -Algorithm SHA256`) |
| Rows after load (raw → parquet) | TO_RECORD |
| Columns | TO_RECORD (≈151 expected; record any difference) |

## Licence & terms

| Field | Value |
|---|---|
| Kaggle licence metadata shown at download | TO_RECORD (community listing may show CC0) |
| Underlying LendingClub terms reviewed | TO_RECORD |
| Redistributable derived data? | TO_DECIDE — if unclear, do NOT publish derived data |

Reference counts (2.26 M rows × 151 cols, ~1.68 GB uncompressed, ~393 MB
gzipped) are community-reported expectations to check against, not guarantees.

## Transformations

1. Parquet conversion via DuckDB with footer-row removal by numeric-`id` check
   (T2.3) — rows removed: TO_RECORD.
2. Everything downstream is recorded in `notes/DECISIONS.md` and scripts.
