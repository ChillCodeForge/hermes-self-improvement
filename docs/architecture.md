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
- Experiment, candidate, run, evaluation, regression, artifact, decision, and
  release persistence with versioned schema migrations (see "Experiment
  store" below); SHA-256 content-addressed artifacts.
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

## Experiment store

`store.py` (`ExperimentStore`) is the SQLite experiment ledger. Its schema is
defined only by the numbered migrations in `src/hermes_improve/migrations/`
(`vNNNN_<name>.sql`, or `vNNNN_<name>.py` with an `upgrade(db)` function where
SQL cannot express the change). The directory sits inside the package, rather
than at the repository root as the spec layout suggests, so that the `*.sql`
files are installed as package data (`pyproject.toml`) and a root-owned
installation applies exactly the reviewed files. Like the evaluator, it is a
protected path (CODEOWNERS, `isolation.PROTECTED_REPOSITORY_PATHS`).

- `PRAGMA user_version` is the schema version. Opening a store applies every
  pending migration in order, each in its own `BEGIN IMMEDIATE` transaction
  that also appends `(version, name, sha256, applied_at)` to the append-only
  `schema_migrations` table and bumps `user_version`; a failing step rolls
  back completely. A database newer than the code is refused.
- Migrations 1-6 reproduce the layouts that existed before versioning, so a
  pre-versioning database (version 0) adopts them: tables use
  `IF NOT EXISTS` and the columns that used to be added by inline `ALTER
  TABLE` (`evaluation_runs.metrics_json/signature`, the
  `benchmark_candidates` review columns) are added only when missing.
- Migration 7 adds the spec §26 experiment fields. `candidate_commit` and
  `candidate_hash` are written when the candidate is created (for
  `create_experiment` + `add_candidate`, by the first candidate) and re-stated
  by the evaluation that decides the experiment. That deciding evaluation
  (FULL / sealed holdout, or a staged run that already proves a regression)
  also writes `model_config_digest` and `environment_digest` (the single value
  all runs shared, `composite-sha256:` over the per-task values when tasks
  differ, `NULL` when no run reported one; per-run values stay in
  `run_observations`), `seeds` (task ID to seeds), `aggregate_metrics` (the
  report metrics) and `gate_result.policy` (passed, reasons, suite, policy
  version, regression count). A recorded promotion adds `gate_result.promotion`
  from the release manifest. Experiments created before migration 7 keep
  `NULL`; nothing is back-filled.
- Migration 8 adds the `regressions` table. Each persisted evaluation writes
  one row per regression `evaluator.detect_regressions` finds on a task:
  `reported` rows for runner-counted critical/high/medium regressions and a
  `task_failure` row (severity `high`) when the baseline solved the task and
  the candidate did not, with baseline and candidate outcome summaries.
  `ExperimentStore.list_regressions` filters by experiment, candidate,
  evaluation run and severity. Rows are append-only.
- Migration 9 adds `BEFORE DELETE` triggers: experiments, candidates,
  evaluation runs, task results, run observations and regressions are never
  deleted (approvals and release events were already append-only). The
  triggers stop every delete through SQL; they are not a boundary against a
  process with write access to the database file, which could drop them
  (see the ledger limit in `security.md`).

New schema changes are new numbered files; applied migrations are never
edited.

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
- `hermes-agent-self-evolution` and DSPy integrations (self-evolution's
  judge-based fitness and in-optimizer holdout conflict with the role
  separation; see `dreaming.md`). The CLI `evolve` command does not run the
  sealed holdout itself; finalists are evaluated by `evaluate` under the
  evaluator identity.
- Live Qwen endpoint validation, production telemetry ingestion, installation
  hardening, and an end-to-end improvement driven by a live Hermes agent. The
  automated E2E test (`tests/e2e/test_full_improvement_loop.py`, spec §39)
  runs the whole loop - worse candidate rejected after screening, better
  candidate through SMOKE/CORE/FULL/sealed holdout, promotion, release tag,
  deployment switch, post-promotion observation, rollback - with real
  components on brain-consuming `skill_procedure` benchmarks, single-user,
  in Docker, and as provisioned role users. The skill executor interprets a
  declared procedure; it is not an LLM agent.

These are explicit gaps, not completed features. See `security.md` and
`operations.md` before attempting deployment.
