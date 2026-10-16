# Architecture and implementation status

This repository implements safety and evaluation components for a future
Hermes improvement service. It is separate from the Hermes Agent runtime and
does not yet collect live sessions or deploy changes automatically.

```text
manual JSON export, Hermes BatchRunner trajectory import, or optional Hermes on_session_finalize plugin
  → redact and queue
  → LIGHT/REM/DEEP evidence analysis and benchmark proposals
  → immutable candidate record referencing a Git commit/diff (created through
    the API or from operator-supplied content by `candidate create`)
  → paired Docker command evaluation + deterministic policy
  → signed evaluator report
  → skills-only Git promotion + atomic release pointer
  → monitoring API (observations supplied by an external collector)
  → stable or rollback
```

## Implemented components

- Session-export redaction and durable SQLite queue; an opt-in Hermes
  finalization plugin backed by the observed SessionDB message-loader API;
  benchmark eligibility and exact task fingerprint proposals.
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

## Role boundaries and remaining isolation work

- **Generator:** consumes redacted/untrusted evidence and proposes hypotheses.
  It must not read sealed benchmark files or modify production.
- **Evaluator:** runs paired tests and signs evidence. The Python holdout
  capability is process-local, so filesystem permissions and a dedicated
  evaluator identity are still essential.
- **Promoter:** verifies and deploys skill-only changes. The CLI checks its
  effective UID against configuration, but operators must provision and
  configure the separate user and protected paths.

The Docker runner isolates candidate command execution relative to the host
using Docker controls, but the Docker daemon itself remains privileged. It is
not a Hermes runtime runner and does not execute Hermes tasks unless a benchmark
provides a compatible command and pinned environment.

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
