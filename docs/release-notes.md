# Release Notes

## 2.0.0 - Adaptive Orchestration Expansion

### Added

- Meta-agent reflection loop with persistent post-run strategy guidance.
- Context optimization v2 with relevance-ranked sections and adaptive compression.
- Adaptive policy tuning from historical run signals (tool-step depth + evidence requirements).
- Extended memory schema for `meta_reflections` and policy-hint derivation.
- CLI `/adaptive` command for operator visibility into adaptive policy and reflection state.

### Improved

- Planner/coder context now includes meta-level strategy guidance.
- Tool-plan constraints dynamically adjust using past execution quality signals.
- Planner parsing now recovers malformed/non-canonical payloads (object/list/plaintext) into executable conservative plans.
- Recovery mode now expands numbered plaintext steps into subtasks when possible.

### CI

- Added dedicated adaptive/meta/context regression lane in CI.

### Validation

- Full suite hardening pass: **135/135 tests passing**.
- New targeted coverage:
  - `tests/test_meta_reflection.py`
  - `tests/test_context_optimization.py`
  - `tests/test_adaptive_policy.py`
  - `tests/test_reviewer_planner_handshake.py`
  - planner-hardening coverage additions in `tests/test_e2e.py`
