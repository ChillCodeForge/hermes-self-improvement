# Architecture and implementation status

This repository implements safety and evaluation components for a future
Hermes improvement service. It is separate from the Hermes Agent runtime and
does not yet collect live sessions or deploy changes automatically.

```text
manual JSON export, Hermes BatchRunner trajectory import, or optional Hermes on_session_finalize plugin
  → redact and queue
  → LIGHT/REM/DEEP evidence analysis and benchmark proposals
  → immutable candidate record referencing a Git commit/diff (created through
    the API or by `candidate create` from operator-supplied content and either
    direct metadata or an explicitly selected Dreaming proposal)
  → paired Docker command evaluation + deterministic policy
  → signed evaluator report
  → skills-only Git promotion + atomic release pointer
  → monitoring API (observations supplied by an external collector)
  → stable or rollback
```

## Implemented components

- Session-export redaction and durable SQLite queue; an opt-in Hermes
  finalization plugin backed by the observed SessionDB message-loader API;
  benchmark eligibility and exact task fingerprint proposals. Caller-supplied
  read-only Curator usage signals can be injected into LIGHT Dreaming; no live
  Hermes Curator API adapter is wired.
- Benchmark schema checks, split/cluster leakage checks including a
  conservative lexical near-duplicate guard, and separate-root loading for
  sealed holdout. Semantic task clustering is not implemented.
- Experiment, candidate, run, evaluation, artifact, decision, and release
  persistence; SHA-256 content-addressed artifacts.
- Deterministic verifier interfaces and hard-gate promotion policy.
- Paired runner orchestration, confidence intervals, Ed25519 report signatures,
  and a Docker runner for declared command-based benchmarks.
- Loopback-first Qwen-compatible API adapter, blind swapped pairwise judgment,
  and optional model usage/latency metrics.
- Version-gated strategy interface and iterative optimization API; they do
  not include a verified external GEPA/DSPy integration or automatic skill
  generation. The candidate CLI only commits content supplied by an operator.
- Git promoter restricted to skill paths, with evidence and diff checks,
  release manifests, atomic symlink switching, and rollback.
- Post-promotion monitoring decision API plus bounded JSONL CLI ingestion for
  trusted collector observations; no live Hermes telemetry collector is wired.

## Trust boundaries

Each role runs as its own Unix user, created by `scripts/provision.sh` and
entered only through `scripts/hermes-improve-as`:

- **Generator / agent (`hermes-agent`)** — the Hermes runtime, Dreaming, and
  optimizer. Writes only the candidate repository, workspaces, handoff
  directory, and its own state. Reads production through `deploy/current`
  and the production repository; has no access to the sealed holdout, the
  ledger, or any private key; cannot modify installed code, configuration,
  policy, or public ground truth.
- **Evaluator (`hermes-evaluator`)** — imports candidate bundles into its own
  Git mirror, reads TRAIN/VALIDATION/REGRESSION and the sealed holdout, signs
  reports with its private key, and appends to the ledger. It cannot write
  production, the holdout, candidates, or configuration.
- **Promoter (`hermes-promoter`)** — owns the production repository,
  releases, and the deployment pointer; verifies the signed evidence and
  fetches the evaluated commit from the handoff bundle. It has no access to
  the holdout or candidate workspaces and never runs Git in agent-owned
  repositories.

```text
agent: candidate create → candidate export ──bundle──▶ handoff/
evaluator: candidate import (mirror, re-hash) → evaluate → signed ledger rows
promoter: promote --candidate-bundle → production.git main + tag → releases/brain-vN → deploy/current
Hermes runtime (agent): reads deploy/current (read-only)
```

The boundary is enforced by the kernel through owners, modes, and POSIX ACLs;
`hermes-improve security check` and the fail-closed checks in `evaluate`,
`candidate import`, `promote`, and `rollback` verify that the layout still
matches the matrix before acting. See `security.md` for the full matrix and
the split between kernel-enforced, code-enforced, and operator-owned
controls. The Docker daemon used by the evaluator remains privileged and
trusted.

## Not yet implemented or validated

- Live installation/compatibility verification of the optional Hermes
  finalization plugin and a supported `batch_runner` adapter.
- Curated and replay-verified 20–50-task starter benchmark corpus and
  holdout-rotation/spent-generation management.
- Automatic immutable candidate creation from Dreaming proposals.
- Pinned and tested Hermes self-evolution, GEPA, or DSPy integrations.
- Live Qwen endpoint validation, production telemetry ingestion, installation
  hardening, and a complete end-to-end Hermes skill improvement/deployment
  workflow.

These are explicit gaps, not completed features. See `security.md` and
`operations.md` before attempting deployment.
