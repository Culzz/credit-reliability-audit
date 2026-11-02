"""Data access package: guard (test-cohort lock) + loader (single point of access)."""

from .guard import ProtocolLockError, protocol_is_frozen, require_unlocked

__all__ = ["ProtocolLockError", "protocol_is_frozen", "require_unlocked"]
