# Architecture

The initial package implements safety and measurement foundations, not an
autonomous deployment system.

```text
Hermes sessions (not wired yet)
  → sanitized evidence / Curator signals (adapter contract)
  → generator and immutable candidates (future)
  → benchmark and evaluator foundations (current)
  → deterministic policy gate (current)
  → Git promoter and atomic deployment (future)
  → provisional monitoring / rollback (future)
```

## Role boundaries

- Generator: future session analysis and candidate creation; no production,
  evaluator-policy, verifier, or holdout access.
- Evaluator: benchmark execution and evidence; current access controller
  restricts sealed holdout reads to a process-local evaluator capability.
- Promoter: future Git/deployment authority; not implemented. Evaluation reports
  are not themselves authorization to deploy.

The process-local holdout capability prevents accidental API misuse in one
process; it is **not OS-level isolation** and does not make the repository
production-ready. Deployment must add separate Unix identities and filesystem
permissions.

## Current components

- Typed benchmark records and validation, including split/cluster leakage
  detection.
- Role-aware benchmark visibility and evaluator holdout capability.
- SQLite experiment, candidate, evaluation, task result, artifact, decision,
  and release metadata tables.
- SHA-256 content-addressed artifacts.
- Deterministic verifier protocol with command, file, regex, and JSON checks.
- Ordered hard safety/correctness/regression policy checks; no weighted score
  can override a failed hard gate.

## Planned next layers

Paired isolated baseline/candidate runners; broader verifier catalog; approved
benchmark suite manifests and sealed holdout rotation; candidate branch/worktree
management; signed promotion manifests; atomic release switching and rollback;
session hooks, dreaming, optional Qwen judge, and post-promotion monitoring.
