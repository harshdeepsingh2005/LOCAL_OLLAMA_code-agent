# Safety and Constraints

This document defines the project’s operational safety model.

## At a glance

- Safety is enforced through layered constraints, not one-off checks.
- File/path policy and contract validation are mandatory preconditions for execution.
- Runtime limits and rollback provide containment and recovery.
- Security scans complement, but do not replace, in-process safeguards.

## Safety objectives

The runtime must prevent unsafe edits and uncontrolled execution while still allowing useful autonomy.

Core protections:

- workspace-bound file operations,
- contract-validated agent input/output,
- bounded iteration/time/token execution,
- checkpoint and rollback capability,
- explicit and auditable failure semantics.

## Constraint layers

### File-system constraints

- no writes outside configured workspace roots,
- path traversal/symlink escape rejection,
- optional blocked pattern and extension policies,
- operation volume and file size limits.

### Execution constraints

- per-run token budget,
- per-completion token budget,
- loop iteration and fix-loop ceilings,
- run/command timeout limits,
- fail-fast behavior for fatal initialization states.

### Output constraints

- schema validation for all agent outputs,
- deterministic fallback on malformed model responses,
- invalid tool calls filtered or rejected,
- reviewer terminal semantics enforced by orchestrator.

## Risk classification

### High risk

- unsanitized shell execution,
- out-of-workspace writes,
- credential/secret exposure pathways,
- destructive broad edits without checkpoints.

### Medium risk

- broad rewrites across many files,
- high-churn fixer loops,
- unbounded recursive scans or transformations.

### Low risk

- read-only context gathering,
- focused deterministic edits with accompanying tests.

## Enforcement hierarchy

1. schema and contract validation,
2. file/path policy enforcement,
3. tool argument validation,
4. runtime resource bounds,
5. rollback/resume guarantees.

## Security scanning posture

CI includes quality and security-oriented checks. These should be treated as one layer of defense, not the only layer.

Critical note: command execution paths require strict least-privilege assumptions and explicit hardening.

## Operational safety checklist

- prefer minimal targeted diffs,
- run affected tests before finalizing changes,
- checkpoint before high-risk operations,
- keep safety policy files version-controlled,
- add regression tests for every safety bug.

## Incident response playbook

If unsafe behavior is detected:

1. halt active execution,
2. inspect telemetry and run artifacts,
3. rollback to last known-safe checkpoint,
4. isolate root cause,
5. add failing regression tests,
6. tighten policy/validation,
7. re-run in constrained mode.

## Safety maturity goals

- reduce advisory-only safety debt,
- tighten shell and path hardening,
- improve automatic detection of non-progress loops,
- maintain clear operator visibility on all high-risk actions.
