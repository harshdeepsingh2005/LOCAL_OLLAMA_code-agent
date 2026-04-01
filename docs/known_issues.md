# Known Issues

This file tracks current technical and operational gaps that affect reliability, safety, or developer experience.

## At a glance

- Main gaps are currently in strict quality debt, shell hardening, and test environment determinism.
- Known limitations are documented to make tradeoffs explicit and actionable.
- Mitigation priorities are ordered to reduce safety and reliability risk first.

## 1) Lint/type debt in strict mode

The repository carries notable lint and mypy debt under strict settings. Some CI checks may operate in non-blocking/advisory mode while stabilization work progresses.

Impact:

- lower static-signal quality,
- slower detection of type regressions,
- increased cleanup burden over time.

## 2) Memory retrieval depth

Persistence across sessions exists, but semantic topic-linking quality is still limited.

Impact:

- explicit facts are easier to retrieve than related themes,
- cross-topic contextual recall may be inconsistent.

## 3) Shell hardening backlog

Security scans can flag shell execution surfaces and temp-path defaults.

Impact:

- elevated risk profile for command-related pathways,
- increased scrutiny needed for tool changes.

## 4) Environment-sensitive tests

Certain integration/e2e tests can fail if runtime prerequisites are unavailable unless patched/mocked.

Impact:

- CI flakiness,
- lower confidence in failure attribution.

## 5) Quality-gate strictness inconsistency

Not all quality checks are uniformly enforced in blocking mode.

Impact:

- short-term velocity gains,
- medium-term quality debt risk.

## 6) Coverage concentration

Critical orchestration paths are reasonably covered, but some modules remain under-tested.

Impact:

- regression detection unevenness,
- higher maintenance risk in low-coverage areas.

## Mitigation priorities

1. progressively restore stricter CI enforcement,
2. harden shell/path safety surfaces,
3. improve semantic memory retrieval quality,
4. increase coverage for low-tested modules,
5. keep docs and tests synchronized with behavioral changes.
