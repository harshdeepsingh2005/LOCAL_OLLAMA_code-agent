# Testing

This guide describes how to validate correctness, resilience, and safety across the system.

## At a glance

- Prioritize deterministic tests around orchestration and safety boundaries.
- Mock external runtime dependencies unless explicitly testing integration.
- Prefer behavior assertions over implementation-coupled assertions.
- Every production bug should leave behind a regression test.

## Test layout

- `tests/test_cli.py`: CLI/session behavior and command pathways.
- `tests/test_e2e.py`: full orchestration behavior, fallback resilience, and loop outcomes.
- `tests/test_loop_controller.py`: terminal-state and iteration semantics.
- `tests/test_safety.py`: path guards, contract safety checks, and enforcement logic.

## Recommended local commands

- full suite: `pytest tests/ -v`
- safety-focused: `pytest tests/test_safety.py -v`
- e2e-focused: `pytest tests/test_e2e.py -v`
- coverage: `pytest tests/ --cov=src --cov-report=term-missing`

## Testing depth model

### Unit tests

- validate parser logic and model defaults,
- test error branches and fallback synthesis,
- avoid live external dependencies.

### Integration tests

- verify agent/core/orchestration interactions,
- assert state transitions and error propagation,
- mock unstable external surfaces.

### End-to-end tests

- exercise realistic task execution paths,
- validate malformed output recovery,
- validate pause/continue semantics and final outcomes.

## CI expectations

Pipeline generally includes:

- lint/format checks,
- type checks,
- security scan,
- test matrix (OS/Python versions),
- build/package stage after quality gates.

## Reliability rules for new tests

- isolate external runtime health checks with patching,
- use deterministic fixtures and stable test data,
- keep assertions behavior-centric,
- include failure-path assertions for safety-sensitive changes.

## Regression workflow

For every production bug:

1. add a failing test,
2. implement fix,
3. confirm regression protection,
4. extend related safety/edge-case coverage if needed.

## Flake prevention checklist

- patch environment-dependent checks,
- avoid timing-sensitive assertions,
- avoid shared mutable global state,
- avoid relying on implicit execution order,
- keep test setup local to each case.

## Coverage strategy

Prioritize coverage depth in:

- orchestrator terminal-state logic,
- contract parsing and fallback,
- safety enforcement boundaries,
- tool dispatch and error normalization.

Lower-priority breadth can be expanded after high-risk modules are consistently covered.
