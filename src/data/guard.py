"""Test-cohort lock (PLAN.md T0.5, spec PLAN.md §4.1).

Test-cohort data may be loaded, scored or tuned on only after the PI has
tagged `protocol-v1` and created `protocol/PROTOCOL_FROZEN` (the PI, and only
the PI, creates the flag). The lock exists so the frozen protocol cannot be
violated by accident.

Every data read in this repository must go through `src/data/loader.py`;
nothing else is permitted to touch `data/`.
"""

from __future__ import annotations

from pathlib import Path

# Resolved from this file's location, NOT the process working directory,
# so the lock holds no matter where a script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_FLAG = REPO_ROOT / "protocol" / "PROTOCOL_FROZEN"

ALLOWED_EARLY_ROLES = ("train", "validation", "descriptive")


class ProtocolLockError(RuntimeError):
    """Raised on any attempt to access locked test cohorts pre-freeze."""


def flag_path() -> Path:
    """Current frozen-flag path (repo-root anchored). Exposed for tests."""
    return FROZEN_FLAG


def protocol_is_frozen(*, flag: Path | None = None) -> bool:
    """True iff the PI-created PROTOCOL_FROZEN marker exists."""
    return (FROZEN_FLAG if flag is None else flag).exists()


def require_unlocked(role: str, *, flag: Path | None = None) -> None:
    """Test cohorts are readable only after the protocol is frozen.

    Parameters
    ----------
    role: one of "train", "validation", "test", "descriptive".
        "descriptive" (shift statistics, features/counts only) is allowed
        pre-freeze; scoring/tuning on it must still go through
        `loader.load_descriptive()`, which never exposes labels or
        prediction columns.

    Raises
    ------
    ProtocolLockError
        If requesting role="test" before the PI creates the flag.
    """
    if role == "test" and not protocol_is_frozen(flag=flag):
        raise ProtocolLockError(
            "Test cohorts are locked until protocol-v1 is tagged and "
            "protocol/PROTOCOL_FROZEN exists (created by the PI only)."
        )
    if role not in ALLOWED_EARLY_ROLES and role != "test":
        raise ValueError(f"Unknown role: {role!r}")
