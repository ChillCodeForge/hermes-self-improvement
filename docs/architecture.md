# Architecture and implementation status

This repository implements safety and evaluation components for a future
Hermes improvement service. It is separate from the Hermes Agent runtime and
does not yet collect live sessions or deploy changes automatically.

```text
manual JSON export, Hermes BatchRunner trajectory import, or optional Hermes on_session_finalize plugin
  → redact and queue
  → LIGHT/REM/DEEP evidence analysis and benchmark proposals
  → immutable candidate record referencing a Git commit/diff (created through
    the evolution loop, from DEEP proposals by the qwen_optimizer role,
    via the API, or by `candidate create` from operator-supplied content and either
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
- Evolution loop (`evolution_loop.py`, `hermes-improve evolve`): baseline →
  variants → SMOKE → CORE → next generation → FULL → sealed holdout for
  finalists only, with the spec's stop limits. The generator-side
  `OptimizerStrategy` (built-in `qwen_optimizer`, or pinned GEPA 0.1.4 running
  `gepa.optimize` with this evaluator as its metric) receives only filtered feedback from the evaluator-side
  `EvaluationGateway`, which alone holds the holdout grant and spent-holdout
  registry. Every variant is an immutable candidate with recorded lineage;
  variants are deduplicated by content hash and failures retained. The loop
  never promotes. Details: `dreaming.md`.
- Dreaming candidate generation (`proposal_generator.py`,
  `hermes-improve candidate generate`): the `qwen_optimizer` role writes skill
  text for up to five selected DEEP proposals; text is checked for secrets,
  protected paths and override phrases and committed only as a candidate.
  Role prompts are files under `prompts/` with mandatory untrusted-evidence
  framing; payloads are nonce-delimited escaped JSON.
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
- `hermes-agent-self-evolution` and DSPy integrations (self-evolution's
  judge-based fitness and in-optimizer holdout conflict with the role
  separation; see `dreaming.md`). The CLI `evolve` command does not run the
  sealed holdout itself; finalists are evaluated by `evaluate` under the
  evaluator identity.
- Live Qwen endpoint validation, production telemetry ingestion, installation
  hardening, and a complete end-to-end Hermes skill improvement/deployment
  workflow.

These are explicit gaps, not completed features. See `security.md` and
`operations.md` before attempting deployment.
