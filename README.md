# Hermes Self-Improvement

**A safety-gated, evidence-driven improvement system for Hermes Agent.**

Hermes Self-Improvement is a standalone project concept for learning from real,
completed Hermes sessions and proposing measurable improvements to skills and,
in later phases, other parts of the agent. A model's claim that a change is
better is never sufficient: every candidate must be immutable, evaluated
against an unchanged baseline, checked for regressions, and promoted through a
controlled Git-backed process.

> **Project status:** Version `0.1.0` now includes redacted session-export
> ingestion, benchmark mining proposals, LIGHT/REM/DEEP Dreaming APIs, SQLite
> experiment/evaluation persistence, benchmark split and cluster-leakage
> checks, deterministic verifiers, paired evaluation, hard-gate policy checks,
> confidence intervals, Ed25519-signed reports, and a digest-pinned Docker
> command runner (`--network=none`, read-only input/root, non-root user and
> resource limits). A Git promoter verifies signed Full+Holdout evidence and
> the exact skills-only diff, creates a signed release manifest, switches a
> deployment symlink atomically, and supports rollback. A monitoring API can
> recommend rollback or mark a release stable after enough observations. The
> package also exports an opt-in Hermes plugin for `on_session_finalize`
> ingestion; its hook API was checked against a local Hermes revision, but the
> plugin has not been exercised inside a live Hermes runtime.
>
> This is **not yet a complete autonomous Hermes self-improvement service**.
> The optional Hermes finalize hook is version-sensitive and no Hermes
> `batch_runner` adapter is wired; the Docker runner handles declared command
> benchmarks only.
> The repository still lacks the curated 20–50 real-task benchmark corpus,
> tested external GEPA/DSPy/self-evolution integrations, a live Qwen service,
> a managed Unix-user/filesystem-isolation setup, a production monitoring
> collector, and a verified end-to-end Hermes skill improvement deployment.
> Process-local holdout capabilities and UID checks do not themselves provide
> OS isolation. See `docs/security.md`, `docs/operations.md`, and
> `docs/architecture.md` for the exact boundaries.

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
`HERMES_IMPROVE_BENCHMARKS` and `HERMES_IMPROVE_HOME`. The included sample task
is a schema fixture, not one of the 20–50 curated real-task benchmarks required
for evaluation readiness.

### Implemented in this repository

- Python APIs and CLI for manual redacted session-export ingestion, queued
  LIGHT/REM/DEEP evidence analysis, benchmark proposals, and candidate records.
- An opt-in Hermes entry-point plugin that reads finalized transcript messages
  from SessionDB and queues redacted exports under the active Hermes home.
- JSON benchmark validation, split/cluster leakage checks, separate holdout
  roots, SQLite experiment/evaluation/release metadata, and content-addressed
  artifacts.
- Paired baseline/candidate evaluation, deterministic hard gates, confidence
  intervals, Ed25519-signed reports, and a Docker runner for explicit
  digest-pinned command benchmarks.
- A skills-only Git promoter with signed manifests, atomic release-pointer
  switching, rollback, and a monitoring decision API.

Hermes hook ingestion is disabled by default and requires explicit plugin
enablement/settings. Candidate proposals are not automatically converted
into Git commits. The
`evaluate` CLI accepts only command-based benchmark environments; it is not a
Hermes runtime or `batch_runner`. Production ingestion and monitoring need
external callers. The end-to-end flow has been exercised with a temporary Git
repository and local Docker image, but not against a real Hermes task corpus
or independently provisioned generator/evaluator/promoter identities.

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
  reproducibility_score: 0.0
  replay_verified: false
verification:
  deterministic: []
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
output, and JSON-output/path equality checks. File-state, regex, unit/build/
lint/type-check, full JSON-schema, Git-diff, HTTP-response, and MCP-health
verifiers remain planned.

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

All archived sessions, user and assistant text, tool output, repositories,
websites, logs, and errors are **untrusted evidence**, never instructions.
Prompt-injection content must not alter system behavior or protected policy. Current
prompt-injection safeguards are bounded projections and path filtering, not a
complete defense.

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
`qwen_judge`—use role-specific prompts. The adapter exposes in-memory request,
token, and latency counters; it does not persist them or capture GPU runtime.
Blind A/B swaps and abstention are implemented as a helper but are not part of
the deterministic promotion gate. No live endpoint has been validated.

## Experiment, artifact, and decision history

SQLite is the current experiment store. It persists
experiments, candidates, evaluation runs, task results, regressions, artifacts,
and releases. A decisions table is present but has no public recording API.
Structured decision history is future work, not an active Laya integration.
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

Skills-only V1 promotion is available through the CLI, not an HTTP service.
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
idempotent. There is not yet an automatic Hermes/production collector. Track
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
Network access requires an explicit allowlist. These operating-system controls
are part of the design and must be verified in deployment; they are not implied
by application-level role names.

## CLI target

The current command surface is:

```text
hermes-improve status
hermes-improve ingest
hermes-improve dream
hermes-improve candidates list
hermes-improve evaluate <candidate-id>
hermes-improve experiments list
hermes-improve experiments show <id>
hermes-improve benchmarks list --role <role>
hermes-improve benchmarks validate
hermes-improve benchmarks mine <session-id>
hermes-improve benchmarks candidates
hermes-improve releases list
hermes-improve promote <candidate-id> --release brain-vN
hermes-improve rollback <release>
```

Some workflows require optional Qwen (REM/DEEP) or Docker (command evaluation).
Candidate Git commit creation, experiment evolution CLI, replay management,
Hermes/Curator integrations, production observation collection, and identity
provisioning are not implemented.

## Phased delivery

Every phase must finish with tests, acceptance review, and a record of what
works, what does not, relevant measurements, and residual risks before the next
phase begins.

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0 — Discovery** | Inspect actual Hermes, Curator, Self-Evolution, DSPy/GEPA, and Qwen versions/APIs; document reuse, risks, and adapters | Evidence-backed discovery and reuse matrix; no assumption treated as an integration |
| **1 — Evaluation foundation** | Versioned benchmark store, baseline/candidate runners, verifiers, per-task/aggregate reports, regression detection, experiment store; begin with 20–50 curated tasks | Repeated baseline is reproducible; deliberately worse candidate is rejected; measurable improvement can be demonstrated |
| **2 — Git and promotion** | Candidate branches/worktrees, immutable commits/hashes, manifest, release tag, atomic deployment, rollback | Production is read-only to Hermes; unevaluated/hash-mismatched candidates cannot promote; rollback works |
| **3 — Skill evolution** | SelfEvolution/GEPA adapter where verified, optional DSPy/Qwen, iterative optimization | At least one real skill follows candidate → evaluation → iteration → rejection or controlled promotion |
| **4 — Session ingestion** | Post-session hook/export, redaction, metadata extraction, ingestion queue | Completed sessions enter the queue without heavy analysis in the user path |
| **5 — Benchmark miner** | Eligibility classes and reproducibility scoring | Reproducible tasks are distinguished from Dreaming-only evidence |
| **6 — Benchmark growth** | Exact/near deduplication, clustering, taxonomy, provenance, manifests and versioning | Related tasks cannot leak across splits |
| **7 — Dreaming** | LIGHT/REM/DEEP cross-session analysis with injection defenses | Hypotheses/candidates are bounded and cannot directly change production |
| **8 — Curator integration** | Usage, stale, overlap, and consolidation signals through proposals | No Curator bypass around candidate evaluation and Git |
| **9 — Post-promotion monitoring** | Provisional/stable/rollback lifecycle and shadow/canary evidence | Regression triggers an auditable rollback decision |

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
├── prompts/                # dreaming, optimizer, judge; separately versioned
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

An end-to-end fixture should demonstrate a bad candidate rejected and a
measurably better candidate moving through Full and holdout checks, manifest
creation, release activation, post-promotion evidence, and rollback. This
fixture is not a substitute for real deployment or benchmark evidence.

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

These are observability goals, not currently available dashboards.

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
