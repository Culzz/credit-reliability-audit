# Test suite

| File | Covers |
|---|---|
| `test_protocol_lock.py` | The test-cohort lock: requesting `role="test"` before `protocol/PROTOCOL_FROZEN` exists raises `ProtocolLockError`; train/validation/descriptive roles are unaffected; the flag path is anchored to the repository root, not the working directory; the real repository state (no flag yet, `configs/splits.yaml` test window unset) is asserted directly. |
| `test_split_roles.py` | Split-window resolution and the descriptive access surface, on synthetic frames only: inclusive vintage filters, inverted windows rejected, `load_split("test")` gated both by the flag and by a set test window, and `load_descriptive()` stripping target-like columns (and refusing them when asked explicitly). |

Run:

```powershell
uv run pytest -q
```

Conventions: no real data in any test (synthetic fixtures only, CONTEXT.md §4
rule 14); tests must pass before any commit that touches `src/`.
