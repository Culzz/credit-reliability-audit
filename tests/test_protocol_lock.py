"""Tests that the test-cohort lock cannot be bypassed (PLAN.md T0.5).

Synthetic fixtures only; no real data is touched anywhere in tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import guard
from src.data.guard import ProtocolLockError, protocol_is_frozen, require_unlocked

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flag_is_repo_root_anchored():
    """The flag path must not depend on the process working directory."""
    assert guard.REPO_ROOT == REPO_ROOT
    assert guard.FROZEN_FLAG == REPO_ROOT / "protocol" / "PROTOCOL_FROZEN"


def test_test_role_raises_before_freeze(tmp_path: Path):
    assert not protocol_is_frozen(flag=tmp_path / "absent_flag")
    with pytest.raises(ProtocolLockError, match="locked.*protocol-v1"):
        require_unlocked("test", flag=tmp_path / "absent_flag")


@pytest.mark.parametrize("role", ["train", "validation", "descriptive"])
def test_early_roles_allowed_pre_freeze(role: str, tmp_path: Path):
    assert require_unlocked(role, flag=tmp_path / "absent_flag") is None


def test_unknown_role_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown role"):
        require_unlocked("scoring", flag=tmp_path / "absent_flag")


def test_test_role_allowed_after_flag_created(tmp_path: Path):
    flag = tmp_path / "PROTOCOL_FROZEN"
    assert not protocol_is_frozen(flag=flag)
    flag.write_text("frozen by PI", encoding="utf-8")
    assert protocol_is_frozen(flag=flag)
    assert require_unlocked("test", flag=flag) is None


def test_real_repo_frozen_after_gate5():
    """Post-freeze invariant (Gate 5, protocol-v1): the frozen marker
    exists in the real repo and the test role is unlocked in it. The
    complementary pre-freeze state stays covered above with an absent
    flag fixture."""
    assert protocol_is_frozen()
    assert require_unlocked("test") is None
