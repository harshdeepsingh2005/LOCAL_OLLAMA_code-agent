# Roadmap

This roadmap organizes expected project evolution into stability, quality, capability, and platform maturity phases.

## At a glance

- Roadmap prioritizes reliability and safety before broad capability expansion.
- Each phase has concrete outcomes, not just feature wish-lists.
- Metrics are defined to keep progress measurable.
- Governance emphasizes small, test-backed, contract-aware changes.

## Phase 1 — Stability baseline

Primary goals:

- reduce CI flakiness across OS/Python matrix,
- eliminate environment-sensitive nondeterminism in tests,
- standardize fallback behavior across all agent boundaries,
- finalize consolidated documentation baseline.

Expected outcomes:

- predictable CI signal,
- lower triage noise,
- clearer release readiness criteria.

## Phase 2 — Safety hardening

Primary goals:

- harden shell/path execution surfaces,
- tighten policy enforcement semantics,
- improve incident diagnostics + rollback ergonomics,
- move advisory safety checks toward blocking with low false positives.

Expected outcomes:

- lower operational risk,
- stronger confidence in autonomous execution.

## Phase 3 — Quality and observability

Primary goals:

- reduce lint/type debt,
- expand contract validation coverage,
- improve run-state summaries and debugging fidelity,
- deepen orchestration telemetry visibility.

Expected outcomes:

- faster debugging,
- lower regression frequency,
- clearer behavioral accountability.

## Phase 4 — Capability expansion

Primary goals:

- improve semantic memory retrieval,
- enhance planner/coder tool-call strategy,
- improve reviewer issue structure and explainability,
- improve context pipeline quality for complex tasks.

Expected outcomes:

- higher task completion quality,
- lower iteration waste.

## Phase 5 — Platform maturity

Primary goals:

- plugin-style tool architecture,
- optional multi-workspace orchestration,
- policy profiles (`strict` / `balanced` / `permissive`),
- stronger formal safety and reliability guarantees.

Expected outcomes:

- broader deployment suitability,
- cleaner extensibility model.

## Milestone metrics

Track progress by:

- CI pass rate and flake rate,
- recovery success after forced failures,
- module-level test coverage trends,
- median task quality and iteration cost,
- rate of safety incidents and rollback usage.

## Execution governance

- keep milestones small and measurable,
- land contract/schema/test/doc changes together,
- prioritize safety and determinism before capability breadth.
