# Hermes Self-Improvement

**A safety-gated, evidence-driven improvement system for Hermes Agent.**

Hermes Self-Improvement is a standalone project concept for learning from real,
completed Hermes sessions and proposing measurable improvements to skills and,
in later phases, other parts of the agent. A model's claim that a change is
better is never sufficient: every candidate must be immutable, evaluated
against an unchanged baseline, checked for regressions, and promoted through a
controlled Git-backed process.

> **Project status:** Version `0.1.0` implements the complete gated loop as
> code with an automated test suite (`scripts/docker-test.sh`):
> redacted session ingestion (export, `batch_runner` trajectories, opt-in
> Hermes hooks), benchmark mining with eligibility/reproducibility scoring,
> a Pydantic/YAML benchmark schema, exact/near/MinHash clustering with
> cluster-consistent splits, sealed-holdout rotation and spent tracking,
> SMOKE/CORE/FULL/holdout suites with impact-aware selection, 15
> deterministic verifiers, a run-count policy, a blind A/B-swapped judge,
> paired statistics with effect sizes, token-per-success metrics,
> Ed25519-signed reports, Git candidates for all 12 candidate types on
> `candidate/<experiment-id>` branches, a manual signed-approval gate for
> non-skill types and for every candidate that removes a skill, policy-checked promotion with atomic deployment, release
> tags, PROVISIONAL/STABLE/ROLLED_BACK lifecycle and audited rollback,
> an iterative evolution loop (built-in Qwen strategy, pinned GEPA adapter),
> Dreaming-to-candidate generation with untrusted-evidence prompting, a
> read-only Curator connector plus mutation-to-candidate adapter,
> post-promotion monitoring with promoter-executed rollback, an
> observability report, a Laya decision-history dataset, and OS role
> isolation (`scripts/provision.sh`, `hermes-improve-as`, `security check`).
> The automated §39 E2E test rejects a deliberately worse skill and promotes
> a measurably better one through FULL and sealed holdout, then rolls back.
>
> Not verified against live external systems: the `HermesBatchRunner` and
> Hermes hooks are tested against the real CLI/schema/hook signatures with
> fixtures but not inside a live Hermes runtime; no local Qwen endpoint was
> available, so Qwen paths are tested against a local OpenAI-compatible stub;
> `hermes-agent-self-evolution`'s own `evolve()` is deliberately not called
> (see `docs/discovery.md`); the benchmark corpus is authored, not mined from
> real sessions; the E2E skill executor follows a declared procedure rather
> than an LLM agent. See `docs/gap-plan.md` for the remaining blockers.

### Quick start

Install the package and optional test dependency:

```bash
python -m pip install -e '.[test]'
hermes-improve status
hermes-improve benchmarks validate
pytest
```

The default benchmark directory is `benchmarks/`; the database defaults to
`~/.hermes-improve/experiments.sqlite3`. Override them with
`HERMES_IMPROVE_BENCHMARKS` and `HERMES_IMPROVE_HOME`. The `benchmarks/` tree holds
26 curated deterministic command benchmarks and 6 brain-consuming
`skill_procedure` benchmarks (train/validation/regression) with fixtures under
`benchmarks/fixtures/`; see `docs/benchmarking.md`.

### Implemented in this repository

- Python APIs and CLI for manual redacted session-export ingestion, queued
  LIGHT/REM/DEEP evidence analysis, benchmark proposals, and candidate records.
- Bounded source previews and persistent one-time operator triage decisions for
  mined benchmark proposals; acceptance does not automatically admit a task to
  the benchmark suite.
- Bounded import of Hermes `batch_runner` `trajectories.jsonl` as redacted,
  untrusted session evidence, including validated tool counters for recurring
  failure signals; importing never executes an agent or counts as evaluation
  evidence.
- A read-only Curator usage adapter that can contribute validated caller-
  supplied skill signals to LIGHT Dreaming; no live Hermes Curator connector is
  included.
- An opt-in Hermes entry-point plugin that reads finalized transcript messages
  from SessionDB and queues redacted exports under the active Hermes home.
- JSON/YAML benchmark records with a Pydantic schema (`benchmarks schema`),
  split/cluster leakage checks, lexical near-duplicate and MinHash similarity
  cluster checks, a nine-factor reproducibility score and eligibility
  classifier, impact-aware SMOKE/CORE/FULL/holdout suite selection with
  digest-pinned manifests, Docker replay reproducibility checks
  (`benchmarks replay`), operator-managed sealed-holdout generation rotation
  (`benchmarks holdout rotate`), SQLite experiment/evaluation/release
  metadata, and content-addressed artifacts.
- Paired baseline/candidate evaluation with a run-count policy per suite,
  15 spec-driven deterministic verifiers, a blind pairwise judge that can only
  block, paired deltas with CI95 and effect sizes, efficiency metrics,
  Ed25519-signed reports, a Docker runner for digest-pinned command benchmarks
  (network off by default, optional internal allowlist network) and a Hermes
  `batch_runner` adapter. See `docs/evaluation.md`.
- A Git promoter for brain candidates (skills auto-gated, other types behind
  signed operator approval) with signed manifests, atomic release-pointer
  switching, rollback, post-promotion monitoring with read-only shadow replay
  and promoter-executed rollback (`monitor evaluate --apply`), an
  observability report (`report`), and a Laya decision-history dataset
  (`decisions export`).
- CLI creation of immutable skill candidates in isolated Git worktrees/branches
  from operator-supplied `SKILL.md` content, with secret/path checks and atomic
  experiment persistence.
- A `brain/` tree (skills, prompts, workflows, routing, tool-descriptions, mcp,
  metadata, tools) with a baseline skill, per-type target path rules for all
  12 candidate types, and `candidate create --type T --file BRAIN_PATH=FILE`
  for any type (`docs/git-workflow.md`).
- Protected `config/*.yaml` (promotion thresholds, per-type auto/manual gates,
  run-count and evolution defaults), Ed25519-signed operator approvals for
  manually gated types, `promote <experiment-id>` with SMOKE/CORE/FULL/holdout
  evidence checks, and an append-only release audit log (`docs/promotion.md`).

Hermes hook ingestion is disabled by default and requires explicit plugin
enablement/settings. Candidate proposals are not automatically converted
into Git content. An operator may select a DEEP proposal as candidate metadata,
but candidate creation still requires explicitly supplied skill text. The
`evaluate` CLI (`--suite SMOKE|CORE|FULL`, `--runner docker|hermes|local`) needs a
Hermes install and model endpoint for the Hermes runner; CI only exercises it
with a fixture. `--runner local` runs only `skill_procedure` benchmarks, without
container isolation. Production ingestion and monitoring need
external callers. The full §39 loop is an automated test
(`tests/e2e/test_full_improvement_loop.py`): single-user, through real Docker
containers, and as provisioned agent/evaluator/promoter Unix users. It runs on
the `skill_procedure` benchmarks, not on a real Hermes task corpus or agent.

The sections below describe the target design where not explicitly marked as
implemented. A target requirement is not evidence that the feature exists.

## Why this project

An agent that changes its own skills or instructions without reliable
measurement can silently make itself worse—or learn to satisfy a test by
changing the test. This project is intended to provide the missing safety and
measurement layer around Hermes' existing runtime, Curator, session data, and
optional optimization tools.

The governing principle is:

> **Measure → compare → improve → measure again → promote only with evidence.**

The system must not improve by lowering its own standards. The optimizer
proposes hypotheses; deterministic verifiers and benchmark evidence evaluate
them; a separately controlled policy determines eligibility; Git records the
production truth; and every deployed release remains reversible.

## Target learning loops

### Improvement loop

```text
Hermes sessions and outcomes
  → collect and redact evidence after the session
  → detect recurring problems and opportunities
  → generate one or more immutable candidates
  → replay baseline and candidates on the same benchmarks
  → run deterministic verifiers, then optional blind judging
  → reject regressions or iteratively refine candidates
  → promote an eligible, measured candidate through Git
  → deploy atomically as a provisional release
  → monitor real usage, stabilize or roll back
  → feed new evidence into later analysis
```

### Benchmark-growth loop

```text
completed real task
  → sanitized session export
  → reproducibility and ground-truth assessment
  → deduplication and task clustering
  → benchmark proposal and review
  → versioned benchmark split
  → stronger future evaluations
```

Session-derived examples are evidence, not instructions. The agent's own
assertion that it succeeded is the weakest ground-truth source and must never
be sufficient for promotion.

## Architecture and trust boundaries

The three roles are distinct in code, permissions, and data access:

| Role | May do | Must not do |
|---|---|---|
| **Generator / optimizer** | Analyze permitted session evidence; form hypotheses; create and mutate candidates; use TRAIN and carefully limited VALIDATION feedback | Write production; promote; change evaluator, verifier, policies, benchmark truth, or holdout; expand its own rights |
| **Evaluator** | Run baseline and candidate; run suites and verifiers; collect metrics; identify regressions; produce reports | Change candidate contents, production, evaluation rules, or promotion policy |
| **Promoter** | Verify signed-off evidence; merge/release eligible immutable artifacts; atomically deploy or roll back | Generate candidates, execute candidate code with promoter privileges, or alter benchmarks |

The following are protected inputs or controls and must not be mutable by
Hermes, Dreaming, optimizers, or candidates: evaluator and verifier code,
promotion policy, benchmark ground truth, sealed holdout, security policies,
CI gates, experiment history, hash verification, promotion tooling, sandbox
rules, and operating-system rights. Tests for these boundaries are required;
comments or prompt instructions alone are not enforcement.

### Git is production truth

Production skills and other approved brain artifacts are authoritative in Git.
Hermes consumes a read-only checkout or atomically selected release; it does not
edit the production brain and synchronize changes back later.

```text
candidate/<experiment-id>
  → immutable candidate commit and content hash
  → isolated evaluation
  → policy decision and promotion manifest
  → merge/release tag (brain-vN)
  → atomic deployment switch
  → provisional monitoring
  → stable or rollback
```

Git worktrees are preferred for candidate isolation. The evaluated commit and
deployed diff must match exactly. Any edit after evaluation creates a new
candidate and requires a new evaluation. Rollback selects a prior release
without rewriting release history.

## Scope and candidate types

The long-term design can represent:

- `SKILL_CREATE`, `SKILL_PATCH`, `SKILL_MERGE`, `SKILL_SPLIT`
- `TOOL_DESCRIPTION`, `WORKFLOW`, `ROUTING`
- `MCP_TOOL_FILTER`, `MCP_CONFIG`, `SYSTEM_PROMPT`
- `TOOL_CREATE`, `TOOL_PATCH`

The initial automatic-promotion scope is **skills only**. Workflows,
descriptions, routing, and filters require explicit configuration or a manual
gate; MCP configuration, system prompts, and tool code require manual approval
until their risk has been addressed and the policy is deliberately changed.
Risk classes range from low (skills) through medium (workflow/routing), high
(MCP configuration and system prompt), to critical (tool code and core-agent
behavior). Riskier types must not inherit skills' promotion rights.

## Reuse Hermes; do not fork it unnecessarily

The intended integration is an adapter-based layer around Hermes, not a new
Hermes runtime. Candidate signal and ingestion sources to verify against the
actual pinned Hermes version include:

- SessionDB, session export, trajectories, and `batch_runner`
- session lifecycle hooks such as `on_session_end` / `on_session_finalize`
- tool and plugin hooks such as `pre_tool_call`
- skill lifecycle, Skill Discovery, `skill_manage`, and
  `skills.write_approval`
- Curator, MCP APIs/configuration, and MCP health checks

Curator is a source of skill-usage, lifecycle, staleness, overlap, consolidation,
backup, and audit signals where those capabilities are present and reliable.
It must not bypass the candidate path to mutate production. Curator changes
should become proposals, then candidates, then evaluated Git changes.

`hermes-agent-self-evolution`, GEPA, and DSPy are possible sources for
candidate-evolution strategies, trajectory processing, and dataset handling.
They are not trusted as promotion authorities. A pinned and tested
`SelfEvolutionAdapter` should isolate version-specific APIs. A keyword-overlap
heuristic or single aggregate score must never decide promotion.

An optional local Qwen-compatible OpenAI endpoint may support synthesis and
subjective evaluation. It is not a prerequisite for deterministic ingestion,
validation, evaluation orchestration, verifiers, policy checks, Git operations,
or reporting.

**Discovery requirement:** before committing to integration contracts, record
the actual versions/commits, APIs, reuse decisions, limitations, and risks for
Hermes, Curator, `hermes-agent-self-evolution`, DSPy, GEPA, and the local model
serving stack in `docs/discovery.md` and `docs/reuse-matrix.md`. Do not infer
that a component exists merely because it appears in this design.

## Benchmark system

Benchmarks should be versioned, provenance-aware, and divided into:

| Split | Purpose | Optimizer access |
|---|---|---|
| `TRAIN` | Candidate development and detailed feedback | Permitted |
| `VALIDATION` | Candidate selection with bounded feedback | Limited |
| `REGRESSION` | Known bugs and previous production regressions | Evaluator-controlled |
| `SEALED_HOLDOUT` | Finalist-only, unseen evaluation | No task/truth access |
| `SHADOW` | Post-promotion real-world evidence | Observation only |

Holdout tasks and ground truth must be inaccessible to the optimizer and
Dreaming process. Only finalists receive holdout evaluation; results returned
to the optimizer are bounded. If holdout feedback influences a later mutation,
the affected holdout generation is marked **spent** and must be rotated in a
later benchmark version. Similar or clustered tasks must always share a split
to prevent train/holdout leakage.

### Benchmark record

The target schema includes at least the following fields. The current package
accepts validated JSON via Python dataclasses (not Pydantic or YAML):

```yaml
schema_version: 1
id: example-task
version: 1
source:
  type: session
  session_id: sanitized-reference
  provenance: verified-source
  verified_at: "..."
task:
  title: Example task
  prompt: Sanitized task statement
taxonomy:
  categories: []
  difficulty: medium
  required_capabilities: []
environment:
  type: repository_fixture
  repository_fixture: null
  git_commit: null
  dependency_digest: null
  container_image: null
  network_policy: disabled
  input_data: null
eligibility:
  class: HARD_BENCHMARK
  reproducibility_score: 0.9
  replay_verified: true
verification:
  deterministic:
    - type: exit_code
  subjective: []
ground_truth:
  type: deterministic_test
  confidence: 1.0
benchmark:
  split: TRAIN
  suites: [SMOKE]
  cluster_id: example-cluster
  criticality: normal
security:
  sanitized: true
  contains_secrets: false
```

Session eligibility classes are `HARD_BENCHMARK`, `SOFT_BENCHMARK`,
`DREAMING_ONLY`, and `EXCLUDED`. A reproducibility score in `[0, 1]` should
account for known repository commit, available input, pinned dependencies,
known environment/container, controlled external services, tests, ground-truth
strength, secret removal, and successful replay. Only sufficiently reproducible
tasks with strong ground truth qualify as hard promotion benchmarks.

Ground-truth evidence priority is: deterministic tests/assertions; build and
integration tests; structured API/file results; later human acceptance;
multiple independent signals; LLM judge; agent self-report. Self-report is
never promotion ground truth.

The store must support exact hashing, near-duplicate detection, task
fingerprinting, semantic clustering, taxonomy, difficulty, confidence,
provenance, manifests, and versioning. Deduplication and clustering must happen
before split assignment, and cluster identity must prevent cross-split leakage.

## Evaluation and measurement

The current `PairedEvaluator` and Docker command runner pair baseline and
candidate runs on identical task IDs and seeds and require matching
environment digests. The broader target evaluation identity should also include
model configuration and limits. Current CLI evaluation only handles
digest-pinned Docker command benchmarks; no Hermes runner or Qwen-in-the-loop
evaluation is wired.

The immutable evaluation identity includes:
baseline and candidate commits, candidate hash, benchmark version, suite, model
configuration digest, environment digest, seeds, and policy version.

Suites:

- **SMOKE:** a small, impact-aware set (typically 20–50 relevant tasks)
- **CORE:** a broader representative set
- **FULL:** the complete applicable regression pool
- **SEALED_HOLDOUT:** final evaluation on controlled unseen tasks

Select tasks by candidate impact; a Docker skill should not need to run every
unrelated research benchmark for every mutation. Deterministic tasks generally
need one run. Nondeterministic LLM tasks can use 1–2 Smoke runs, 3 Core runs,
and 5 Full/final runs when warranted. Compare paired baseline/candidate deltas,
confidence intervals, and practical effect sizes; p-values alone are not a
promotion reason.

Capture per-task trajectories, tool calls and errors, input/output tokens,
latency, retries, final output, filesystem changes, Git diff, verifier outcomes,
and exit state. Large trajectories belong in a content-addressed artifact store
(SHA-256 plus path), not SQLite blobs.

Deterministic verifiers take precedence over model judging. The Docker command
runner currently supports exit-code, exact/contains/not-contains captured
output, required JSON keys, and JSON path existence/equality checks.
File-state, regex, unit/build/lint/type-check, full JSON-schema, Git-diff,
HTTP-response, and MCP-health verifiers remain planned.

When objective verification is insufficient, an optional local judge can use
blind pairwise comparison with A/B identity hidden and a second run with order
swapped. Position-dependent disagreement is `unreliable`; the judge must be
able to abstain. It can never promote a candidate by itself.

### Promotion policy

Promotion uses ordered hard constraints, never one weighted overall score:

1. Security and policy compliance
2. Deterministic correctness
3. Critical and high regressions
4. Task success and generalization
5. Category-level regression limits
6. Quality
7. Token efficiency
8. Latency

Illustrative configurable defaults from the project requirements:

| Constraint | Default target |
|---|---:|
| Critical regressions | 0 |
| High regressions | 0 |
| Medium regression rate | ≤ 1% |
| Minimum quality delta | +3 points |
| Maximum category drop | 2 points |
| Token increase | ≤ 10%, unless quality gain justifies it |
| Latency increase | ≤ 15%, subject to policy |
| Relevant delta confidence interval | Must not be clearly negative |

These are initial policy targets, not measured guarantees. Security,
correctness, or regression failure cannot be offset by quality, cost, or latency
improvements.

Track tokens per task and per successful task, input/output tokens, context
size, tool calls per success, failed tool calls, retries, and latency per
success. A cheaper failed task is not an improvement.

## Candidate evolution

An optimizer strategy interface should allow GEPA or other strategies to be
plugged in without making them trusted. Suggested initial limits:

```text
max_generations = 10
candidates_per_generation = 3
max_candidates_total = 30
stop_after_no_improvement_candidates = 8
stop_after_no_improvement_generations = 3
minimum_meaningful_delta = 0.5
```

The intended funnel is baseline → several candidates → Smoke → refine → Core →
refine → Full → sealed holdout → promote or reject. Stop when meaningful
progress stalls; do not overfit to 100% of the benchmark suite. Dreaming may
produce at most five candidates in one run and never writes production or
promotes.

**Implemented (`evolution_loop.py`, `hermes-improve evolve`):** the loop above
with those defaults (overridable through a mapping or `--settings` JSON). Every
variant is a new immutable candidate commit with recorded `parent_id` and
`lineage_id`; identical variants are deduplicated by content hash and failed
ones are retained. The generator sees detailed TRAIN feedback, aggregate
VALIDATION/REGRESSION feedback and never holdout data; holdout runs only for
finalists and yields a pass/fail bit. If that bit is fed back into a further
mutation (explicit opt-in), the lineage's holdout generation is first marked
spent. Strategies: built-in `qwen_optimizer`, and optional GEPA (`gepa.optimize`
with this evaluator as metric; `pip install '.[gepa]'`, pinned `gepa==0.1.4`). The loop only proposes;
promotion stays with the policy and the promoter. See `docs/dreaming.md`.

## Session ingestion and Dreaming

The current `ingest EXPORT.json` CLI accepts a manual session export, redacts
common credential fields and token formats, tags the result as untrusted, and
stores it in a local queue (up to 10 MiB). An optional package entry point
registers an `on_session_finalize` callback with Hermes; when explicitly
enabled, it uses Hermes' persisted-message loader and writes to a private
per-profile state directory. This adapter is based on the API observed at
Hermes commit `e06df9c2eae6901f4c88f464e0ae6e753c0428b8`; runtime installation
and compatibility across Hermes versions remain unverified. It does not
normalize every Hermes SessionDB field or extract all target metrics.
Collection should remain outside the normal user-request critical path.

Dreaming has three stages:

- **LIGHT (implemented, no LLM):** count recurring error strings and observed
  tool names from queued session exports. It does not yet calculate the full
  metric set, cluster semantic issues, or incorporate live Curator signals.
- **REM (implemented proposal API, optional local Qwen):** analyze bounded,
  explicitly untrusted session projections and validate hypotheses. It can be
  used to analyze cross-session patterns such as
  recurring errors/successes, missing or redundant skills, poor tool selection,
  unnecessary calls/tokens, deterministic reasoning that could become a tool,
  and routing issues.
- **DEEP (implemented proposal API, optional local Qwen):** turn supplied
  hypotheses into at most five path-filtered proposals. It does not write
  candidate commits or promote changes.
- **Candidate generation (implemented, optional local Qwen):**
  `hermes-improve candidate generate` lets the `qwen_optimizer` role write the
  SKILL.md text for up to five selected DEEP proposals; content must pass
  secret, protected-path and override-phrase checks and is committed only as a
  candidate branch awaiting evaluation.

All archived sessions, user and assistant text, tool output, repositories,
websites, logs, and errors are **untrusted evidence**, never instructions.
Prompt-injection content must not alter system behavior or protected policy.
Role prompts live in `src/hermes_improve/prompts/dreaming/` and `src/hermes_improve/prompts/optimizer/` and must carry
the UNTRUSTED EVIDENCE framing; every model payload is escaped JSON inside a
nonce-delimited envelope. Explicit injection tests cover promotion, config
edits and holdout reads. These safeguards reduce, but cannot eliminate, the
risk that model-written text is wrong; evaluation and policy remain the gate.

## Local model configuration

The optional Qwen adapter uses a configurable OpenAI-compatible endpoint with
loopback-only defaults and an explicit remote opt-in. `QWEN_JUDGE_MODEL` may
select a separate judge model:

```text
QWEN_BASE_URL
QWEN_API_KEY
QWEN_MODEL
QWEN_JUDGE_MODEL
QWEN_TIMEOUT
QWEN_MAX_CONCURRENCY
```

Logical roles—`qwen_solver`, `qwen_optimizer`, `qwen_dreamer`, and
`qwen_judge`—use role-specific prompts. The model variables are the ones named
in `config/models.yaml`. Every Qwen-backed command (`dream --stage rem|deep`,
`candidate generate`, `evolve`, `evaluate --judge`) persists the adapter's
per-role request, token, and latency counters to `observability.sqlite3`, also
when the command fails. GPU runtime is measured only when the operator points
`HERMES_GPU_METRICS_URL` at the model server's Prometheus endpoint and names
its cumulative seconds counters in `HERMES_GPU_RUNTIME_METRICS`; otherwise the
report says `unavailable` and why (see `docs/operations.md`).
`evaluate --judge` wires the blind pairwise judge into evaluation: each
subjective-only task is judged with A/B and swapped order, and a
position-dependent answer is `unreliable`. A judge verdict can only add
blocking reasons to the evaluation report, never make a candidate pass;
deterministic verifiers always take precedence. No live endpoint has been
validated.

## Experiment, artifact, and decision history

SQLite is the current experiment store. It persists
experiments, candidates, evaluation runs, task results, regressions, artifacts,
and releases. A `decision_history` table with a recording, ground-truth
backfill, and JSONL export API collects Laya training data (see
`docs/laya-future.md`); no Laya model exists.
An experiment records IDs and lineage; hypothesis and source sessions;
candidate type, target, and risk; baseline/candidate commits and hash;
benchmark and policy versions; model/environment digests and seeds; per-task
and aggregate metrics; gate result, decision, reason, and timestamps.

Failed and rejected experiments are retained so Dreaming can avoid repeating
known-bad approaches. Artifact payloads are content-addressed outside the DB.

Decision history should eventually capture decision type, input features,
available options, selected option, confidence, source, ground truth and its
source, correctness, model version, and timestamp. This may later provide
calibrated training data for an optional Laya decision model. Laya is not an
initial requirement and must never replace promotion gates.

## Promotion, deployment, and rollback

Promotion is available through the CLI, not an HTTP service; see `docs/promotion.md` for the exact checks.
The promoter creates and hashes an HMAC-signed manifest containing the
experiment ID, candidate and baseline commits, candidate hash, benchmark and
policy versions, Full and holdout evidence, summaries, artifact type, release
target, and timestamp.

The promoter must independently verify that the experiment is complete, the
candidate hash and evaluated/deployed diff match, all required suites and
security gates pass, no blocking regressions exist, and the artifact type is
eligible under the exact policy version. Promotion of an unevaluated candidate
or hash mismatch must be impossible.

After Git ref update and release tagging, the promoter switches the configured
deployment symlink atomically and marks the release `PROVISIONAL`. The
monitoring API and `hermes-improve monitor ingest observations.jsonl` accept
trusted collector observations; stable observation IDs make retries
idempotent. `hermes-improve monitor collect RELEASE` derives observations from
Hermes sessions already ingested (attributed by release activation time,
user corrections by documented heuristics; see `docs/operations.md`). Track
success rate, tool errors, tokens and calls per success, latency, retries,
user corrections, and regression signals. Move to `STABLE` only with adequate
evidence; otherwise roll back atomically and record the decision. Avoid
simultaneous write agents on the same task.

## Operating-system and sandbox isolation

The deployment should use separate, least-privilege identities:

| Identity | Access |
|---|---|
| `hermes-agent` | Production brain read-only; candidate workspace writable; evaluator read-only; no holdout |
| `hermes-evaluator` | Candidate read/execute; TRAIN/VALIDATION read; controlled evaluator-only holdout access; no production write |
| `hermes-promoter` | Promotion manifest read; Git release and production deployment write; never executes candidate code |
| Optimizer process | No `sudo`, production write, or holdout access |

Benchmark executions default to network disabled, no production credentials,
temporary filesystem, CPU/RAM limits, timeouts, and tool-call/turn limits.
Network access requires an explicit allowlist.

`scripts/provision.sh --root /srv/hermes-improve --install-from .` (Linux, as
root, idempotent) creates these users, the `hermes-ledger` group, the
directory layout with owners, modes, and POSIX ACLs, and the signing keys,
then runs the isolation self-check. Run every command as its role with
`scripts/hermes-improve-as <agent|evaluator|promoter> <args>`.
`hermes-improve security check [--role R]` verifies the real permissions
against the expected matrix; `evaluate`, `candidate import`, `promote`, and
`rollback` refuse to run when the boundary is violated. The permission
matrix and the split between kernel-enforced, code-enforced, and operator
controls are in [docs/security.md](docs/security.md).

## CLI target

The current command surface is:

```text
hermes-improve status
hermes-improve ingest
hermes-improve ingest-batch RUN/trajectories.jsonl
hermes-improve dream
hermes-improve candidate create --help
hermes-improve candidate generate --repository REPO --proposal-file DEEP.json --proposal-index N
hermes-improve candidates list
hermes-improve evolve <candidate-id> --repository REPO [--strategy qwen|gepa|gepa-proposer] [--max-generations N]
hermes-improve evaluate <candidate-id>
hermes-improve experiments list
hermes-improve experiments show <id>
hermes-improve benchmarks list --role <role>
hermes-improve benchmarks validate
hermes-improve benchmarks add FILE --holdout-root HOLDOUT_DIR
hermes-improve benchmarks mine <session-id>
hermes-improve benchmarks candidates
hermes-improve releases list
hermes-improve releases events [--release brain-vN]
hermes-improve approve <experiment-id> --operator NAME --rationale TEXT
hermes-improve promote <experiment-id> [--release brain-vN] [--operator-public-key KEY]
hermes-improve rollback <release>
hermes-improve curator status|propose|convert-pending
hermes-improve monitor collect|ingest|replay|evaluate [--apply]
hermes-improve report [--json]
hermes-improve decisions export|backfill
hermes-improve security check [--role agent|evaluator|promoter] [--production PATH]
hermes-improve candidate export <candidate-id> --repository REPO --out-dir HANDOFF
hermes-improve candidate import <candidate-id> --handoff-dir HANDOFF --mirror MIRROR
hermes-improve candidate sync-baseline --repository REPO --production PRODUCTION
hermes-improve promote <candidate-id> --release brain-vN --candidate-bundle HANDOFF/<candidate-id>.bundle
```

Some workflows require optional Qwen (REM/DEEP) or Docker (command evaluation).
Candidate Git creation is available for operator-supplied skill text; automatic
skill generation, a Hermes `batch_runner` execution/evaluation adapter,
experiment evolution CLI, a bundled Hermes replay runner, production
observation collection are not implemented.

## Phased delivery

Every phase must finish with tests, acceptance review, and a record of what
works, what does not, relevant measurements, and residual risks before the next
phase begins.

| Phase | Deliverable | Current repository status | Exit criteria |
|---|---|---|---|
| **0 — Discovery** | Inspect actual Hermes, Curator, Self-Evolution, DSPy/GEPA, and Qwen versions/APIs; document reuse, risks, and adapters | Initial findings are documented; several external APIs and endpoints remain unverified | Evidence-backed discovery and reuse matrix; no assumption treated as an integration |
| **1 — Evaluation foundation** | Versioned benchmark store, baseline/candidate runners, verifiers, per-task/aggregate reports, regression detection, experiment store; begin with 20–50 curated tasks | Core evaluation components, a curated replay-verified 26-task command corpus and 6 brain-consuming skill benchmarks are implemented; the E2E test rejects a deliberately worse skill and promotes a measurably better one; agent runs and real-task proof are still missing | Repeated baseline is reproducible; deliberately worse candidate is rejected; measurable improvement can be demonstrated |
| **2 — Git and promotion** | Candidate branches/worktrees, immutable commits/hashes, manifest, release tag, atomic deployment, rollback | Promotion for all candidate types (skills auto-gated, others behind signed approval), atomic deployment, release tags, audited rollback and OS role isolation via `scripts/provision.sh` are implemented and tested with real role users in the container; a real production host deployment remains operator work | Production is read-only to Hermes; unevaluated/hash-mismatched candidates cannot promote; rollback works |
| **3 — Skill evolution** | SelfEvolution/GEPA adapter where verified, optional DSPy/Qwen, iterative optimization | Iterative evolution loop (`evolve`), Qwen mutation strategy, pinned GEPA adapter and Dreaming-to-candidate generation are implemented; the E2E test drives a real skill through rejection and controlled promotion with a deterministic executor; live Qwen/GEPA runs remain unverified | At least one real skill follows candidate → evaluation → iteration → rejection or controlled promotion |
| **4 — Session ingestion** | Post-session hook/export, redaction, metadata extraction, ingestion queue | JSON and trajectory ingestion plus an opt-in finalize plugin are implemented; live Hermes session finalization remains unverified | Completed sessions enter the queue without heavy analysis in the user path |
| **5 — Benchmark miner** | Eligibility classes and reproducibility scoring | Mining proposals and operator review are implemented; proposals still require separate benchmark authoring and admission | Reproducible tasks are distinguished from Dreaming-only evidence |
| **6 — Benchmark growth** | Exact/near deduplication, clustering, taxonomy, provenance, manifests and versioning | Exact, near-duplicate and MinHash/LSH clustering with cluster-consistent splits, manifests, versioning and operator holdout rotation are implemented; clustering is lexical, so paraphrases can escape | Related tasks cannot leak across splits |
| **7 — Dreaming** | LIGHT/REM/DEEP cross-session analysis with injection defenses | LIGHT/REM/DEEP APIs are implemented; live model integration and end-to-end self-improvement remain unverified | Hypotheses/candidates are bounded and cannot directly change production |
| **8 — Curator integration** | Usage, stale, overlap, and consolidation signals through proposals | Read-only live connector to Hermes Curator files, signals, and mutation-to-candidate conversion (including archive, delete and consolidation as manually gated removal candidates) are implemented and tested against fixtures in the real format | No Curator bypass around candidate evaluation and Git |
| **9 — Post-promotion monitoring** | Provisional/stable/rollback lifecycle and shadow/canary evidence | Area-matched monitoring, read-only shadow replay through an operator runner, and audited promoter rollback via `monitor evaluate --apply` are implemented; `monitor collect` feeds ingested Hermes sessions (heuristic correction signals) and `monitor replay --source shadow-benchmarks` replays SHADOW-split tasks; a Hermes replay runner remains absent | Regression triggers an auditable rollback decision |

The design intentionally sequences reliable measurement before autonomous
learning. Later candidate types, observability expansion, and Laya are gated on
the safety and evidence foundations.

## Repository layout target

The eventual standalone package may follow this structure; it is a target, not
a claim that these directories exist today:

```text
hermes-self-improvement/
├── config/                 # promotion, scoring, taxonomy, model settings
├── src/hermes_improve/
│   ├── cli/ hermes/ curator/ self_evolution/ sessions/ dreaming/
│   ├── benchmarks/ evaluator/ verifiers/ optimizer/ candidates/
│   ├── experiments/ artifacts/ git/ promotion/ monitoring/ security/
│   └── prompts/            # dreaming, optimizer, judge role prompts (package data)
├── brain/                  # production-tracked skills/prompts/workflows/etc.
├── benchmarks/             # public splits, manifests, fixtures (not holdout)
├── tests/                  # unit, integration, security, end-to-end
├── migrations/
└── docs/
```

**Sealed holdout data must not be stored in the ordinary repository or exposed
to generator credentials.**

## Security and test requirements

The test plan must include split and duplicate-cluster leakage, holdout access
and spent handling, candidate hash and manifest mismatches, critical/high
regression blocking, deterministic-verifier precedence, judge A/B swap and
uncertainty, invalid model JSON, archived-session prompt injection, direct
production writes, Curator bypass, promotion races and authorization, atomic
deployment and rollback, candidate mutation after evaluation, Self-Evolution
version mismatch, Git-state mismatch, replay reproducibility, production
read-only enforcement, experiment lineage, and retention of failed experiments.

The end-to-end test `tests/e2e/test_full_improvement_loop.py` demonstrates a bad
candidate rejected and a measurably better candidate moving through Full and
holdout checks, manifest creation, release activation, post-promotion
evidence, and rollback. It is not a substitute for real deployment or
benchmark evidence.

## Observability goals

Reports should cover:

- **Hermes:** success rate, tokens/success, tool calls/success, failed calls,
  retry rate
- **Improvement:** candidates generated/rejected/promoted, generations,
  average measured delta, regressions caught, artifact-type breakdown
- **Benchmarks:** counts, category/difficulty/ground-truth distribution,
  reproducibility, split distribution, spent holdouts
- **Dreaming:** sessions, clusters, hypotheses, candidates, conversion rates
- **Qwen:** requests, tokens, latency, available GPU runtime, judge uncertainty
- **Production:** brain version, provisional/stable status, regressions, rollback
  count

`hermes-improve report` computes these sections from persisted state and
prints `no data` where no source exists (GPU runtime: `unavailable`; judge
uncertainty is not persisted yet). There is no dashboard.

## Initial completion criteria

The skills-only MVP is not complete until production skills are Git-authoritative
and read-only to Hermes; a useful curated benchmark set exists; baseline and
candidate results are reproducibly paired; deterministic verifiers and
per-task reports work; regressions are explicit; bad candidates are rejected;
eligible candidates can be promoted with exact hash/commit identity; release
tags and rollback work; experiment history is retained; holdout is inaccessible
to optimization; and no aggregate score can override security or correctness
constraints.

The broader system additionally requires real post-session ingestion,
reproducible benchmark mining/versioning, cross-session Dreaming, safe Curator
integration, iterative candidate evolution, post-promotion monitoring, and an
automated end-to-end workflow. Documentation alone does not satisfy these
criteria.

## Documentation plan

As implementation lands, add and keep current:

- `docs/discovery.md`, `docs/reuse-matrix.md`, `docs/architecture.md`
- `docs/security.md`, `docs/benchmarking.md`, `docs/evaluation.md`
- `docs/dreaming.md`, `docs/git-workflow.md`, `docs/promotion.md`
- `docs/operations.md`, `docs/laya-future.md`

Each document must describe verified behavior, distinguish plans from working
features, and record genuine external blockers and how they can be resolved.
