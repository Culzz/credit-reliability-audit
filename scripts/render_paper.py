"""Render paper/manuscript.md placeholders from the dry-run numbers export.

Placeholder syntax: {{n:key}} -> paper/numbers_dryrun.json key. Rendering is
strict: any placeholder without a value is a HARD ERROR with the missing key
list (CONTEXT.md 11 - "do not paper over it"). Placeholders inside fenced
code blocks are documentation, not claims, and stay literal. The dry-run mode
renders against numbers_dryrun.json only; after the freeze a `--source main`
renders the real manuscript against the frozen export. Output:
paper/manuscript_rendered.md (never overwrites the source).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO_ROOT / "paper" / "manuscript.md"
NUMBERS = {
    "dryrun": REPO_ROOT / "paper" / "numbers_dryrun.json",
    "main": REPO_ROOT / "paper" / "numbers.json",
}
OUT = REPO_ROOT / "paper" / "manuscript_rendered.md"
PLACEHOLDER = re.compile(r"\{\{n:([a-zA-Z0-9_.\-]+)\}\}")


def render(text: str, numbers: dict) -> tuple[str, list[str]]:
    missing: list[str] = []

    def _subst(match: re.Match) -> str:
        key = match.group(1)
        entry = numbers.get(key)
        if entry is None:
            missing.append(key)
            return match.group(0)
        return format(entry["value"], ".6g").rstrip("0").rstrip(".") if isinstance(entry["value"], float) else str(entry["value"])

    # fenced code blocks are documentation, not claims: the stub's example
    # placeholder must stay a literal example instead of resolving into a
    # dry-run number sitting in prose. Splitting on ``` keeps pair parity:
    # odd segments are INSIDE a fence.
    segments = text.split("```")
    for i, chunk in enumerate(segments):
        if i % 2 == 0:
            segments[i] = PLACEHOLDER.sub(_subst, chunk)
    rendered = "```".join(segments)
    return rendered, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(NUMBERS), default="dryrun")
    args = parser.parse_args()
    if args.source == "main" and not (REPO_ROOT / "protocol" / "PROTOCOL_FROZEN").exists():
        raise RuntimeError("--source main requires protocol/PROTOCOL_FROZEN")
    text = MANUSCRIPT.read_text(encoding="utf-8")
    numbers = json.loads(NUMBERS[args.source].read_text(encoding="utf-8"))["numbers"]
    rendered, missing = render(text, numbers)
    if missing:
        raise SystemExit(
            f"{len(missing)} placeholder(s) without values: "
            + ", ".join(sorted(set(missing))[:20])
        )
    OUT.write_text(rendered, encoding="utf-8")
    # count only what the renderer owns (unfenced segments)
    segments = text.split("```")
    count = sum(len(PLACEHOLDER.findall(segments[i])) for i in range(0, len(segments), 2))
    print(f"rendered {count} placeholder(s) -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
