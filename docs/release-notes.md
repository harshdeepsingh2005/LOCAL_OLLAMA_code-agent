# Release Notes

## 2.0.0 - Adaptive Orchestration Expansion

### Added

- Meta-agent reflection loop with persistent post-run strategy guidance.
- Context optimization v2 with relevance-ranked sections and adaptive compression.
- Adaptive policy tuning from historical run signals (tool-step depth + evidence requirements).
- Extended memory schema for `meta_reflections` and policy-hint derivation.

### Improved

- Planner/coder context now includes meta-level strategy guidance.
- Tool-plan constraints dynamically adjust using past execution quality signals.

### Validation

- Full suite hardening pass: **125/125 tests passing**.
- New targeted coverage:
  - `tests/test_meta_reflection.py`
  - `tests/test_context_optimization.py`
  - `tests/test_adaptive_policy.py`
