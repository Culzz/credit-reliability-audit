"""Verify the manually downloaded LendingClub raw file against the
checksum recorded in data/README.md.

Usage:
    python -m uv run python -m scripts.verify_raw_data --path <file.csv.gz>

The raw file is never committed; this script exists so that another
researcher can prove they have the exact file the study used before
running the pipeline. Exits 0 on match, 1 on mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "55C16F75120F897683F02E7AABCF080D0E4A20C4832FEB1D592CFA941BD62A2D"
EXPECTED_BYTES = 392_582_231


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="path to accepted_2007_to_2018Q4.csv.gz")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest().upper()
    size = path.stat().st_size
    print(f"sha256: {digest}")
    print(f"bytes : {size}")
    ok = digest == EXPECTED_SHA256 and size == EXPECTED_BYTES
    print("MATCH" if ok else
          "MISMATCH — do not run the pipeline with this file; "
          "see data/README.md for the expected provenance")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
