"""Convert the raw LendingClub accepted-loans CSV into an analysis-ready Parquet file.

Blueprint 4.8 / PLAN.md T2.3. Reads the gzipped CSV once through DuckDB and
writes compressed Parquet; removes non-loan records (the summary/footer text
at the end of the file) by requiring a numeric `id`.

Usage:
    python -m src.data.load_raw            # uses configs/data.yaml source/paths
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROW_BAND = (2_000_000, 2_600_000)


def main(config: Path | None = None) -> None:
    cfg_path = config or REPO_ROOT / "configs" / "data.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["data"]

    source = REPO_ROOT / "data" / "raw" / Path(cfg["filename"]).name
    destination = REPO_ROOT / "data" / "interim" / "accepted_raw.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found. Download procedure: docs/blueprint.md B-9 (Kaggle); "
            "verify the file against the checksum in data/README.md."
        )

    src = source.as_posix()
    dst = destination.as_posix()
    con = duckdb.connect()
    try:
        total, bad, rows = _convert(con, src, dst)
        cols = con.execute(
            f"SELECT count(*) FROM (DESCRIBE SELECT * FROM read_parquet('{dst}'))"
        ).fetchone()[0]
    finally:
        con.close()

    print(f"total rows read:                        {total:,}")
    print(f"non-loan footer rows removed:           {bad:,}")
    print(f"parquet written:                        {rows:,} rows x {cols} columns")
    print(f"file:                                   {destination}")
    if not (EXPECTED_ROW_BAND[0] < rows < EXPECTED_ROW_BAND[1]):
        print("WARNING: row count outside the expected 2.0M-2.6M band; investigate before proceeding.")


def _convert(con: duckdb.DuckDBPyConnection, src: str, dst: str) -> tuple[int, int, int]:
    """Copy numeric-id rows to Parquet; count total rows and removed footer rows."""
    total = con.execute(
        f"SELECT count(*) FROM read_csv_auto('{src}', header = true, sample_size = -1)"
    ).fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{src}', header = true, sample_size = -1)
            WHERE TRY_CAST(id AS BIGINT) IS NOT NULL
        ) TO '{dst}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
    return int(total), int(total - rows), int(rows)


if __name__ == "__main__":
    main()
