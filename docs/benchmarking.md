# Benchmarking

This document describes the benchmark store as implemented: record schema,
eligibility, deduplication and clustering, suites, replay, the curated
corpus, and operator-managed sealed-holdout rotation.

## Record schema

A benchmark record is a JSON (`.json`) or YAML (`.yaml`/`.yml`) object with
`schema_version: 1`. Parsing runs in two stages:

1. **Structural schema** (`hermes_improve.benchmark_schema.BenchmarkRecord`,
   Pydantic v2, strict types). It defines every section: `source{type,
   provenance, session_id, verified_at}`, `task{title, prompt}`,
   `taxonomy{categories, difficulty, required_capabilities}`,
   `environment{type, repository_fixture, git_commit, dependency_digest,
   container_image, network_policy, input_data, command, cwd,
   timeout_seconds}`, `eligibility{class, reproducibility_score,
   replay_verified, replay_evidence}`, `verification{deterministic,
   subjective}`, `ground_truth{type, confidence, expected_exit_code,
   structured_result}`, `benchmark{split, suites, cluster_id, criticality,
   impact_areas}`, `security{sanitized, contains_secrets}`. Unknown top-level
   keys are rejected; sections may carry additional runner-specific keys.
2. **Semantic validation** (`Benchmark.from_dict`) stays authoritative: hard
   benchmark thresholds, secret scanning, sanitization, label hygiene and
   safe identifiers.

Decoding rejects duplicate keys (JSON and YAML), YAML tags beyond the safe
loader, and YAML values that are not JSON types (quote dates). Schema errors
report field locations only and never echo input values.
`hermes-improve benchmarks schema` prints the JSON Schema.

The public loader reads records recursively under the benchmark root but
skips the reserved top-level directories `fixtures/` and `manifests/`.

## Taxonomy

Categories (shared with `config/taxonomy.yaml`, module constant
`benchmark_schema.TAXONOMY_CATEGORIES`): `coding`, `git`, `shell`,
`data-transform`, `docs`, `devops`, `research`, `mcp`. Difficulties: `easy`,
`medium`, `hard`. Criticalities: `low`, `normal`, `high`, `critical`.

`hermes-improve benchmarks validate` validates the public corpus and then
lints it: every record needs at least one known category, a known difficulty
and criticality; a record must not declare a stronger eligibility class or a
higher reproducibility score than the classifier derives from its own
fields; `replay_verified: true` requires `eligibility.replay_evidence`.
`BenchmarkStore.validate()` itself does not enforce the taxonomy, so other
callers (for example evaluator tests) can use their own labels.

## Eligibility and ground truth

`hermes_improve.eligibility` scores reproducibility from nine factors:

| Factor | Weight |
|---|---|
| git commit present (or repository fixture) | 0.10 |
| input present | 0.10 |
| dependencies reproducible (dependency digest) | 0.10 |
| container present | 0.10 |
| external services controllable (`network_policy` disabled/allowlist/fixture) | 0.10 |
| tests present | 0.15 |
| ground truth present | 0.15 |
| secrets removed | 0.05 |
| replay successful | 0.15 |

Ground truth is ranked: 1 deterministic tests, 2 integration tests,
3 structured API/file result, 4 human accepted, 5 multiple machine signals,
6 LLM judge, 7 agent self-report. Classification:

- secrets not removed → `EXCLUDED`;
- ground truth rank 1–3, successful replay and score ≥ 0.8 → `HARD_BENCHMARK`;
- any ground truth except self-report and score ≥ 0.5 → `SOFT_BENCHMARK`;
- otherwise → `DREAMING_ONLY`. Agent self-report is never ground truth.

The miner (`mine_session`, `mine_sessions`) applies this after redaction:
normalization → eligibility → ground truth → exact/fingerprint dedup →
similarity clustering → proposal. `benchmarks mine` clusters the session
against the public corpus, so a session similar to an existing benchmark
inherits its `cluster_id`. Mining never admits a benchmark.

### Judge benchmarks (no deterministic checks)

`Benchmark.from_dict` requires every `HARD_BENCHMARK` to have deterministic
checks, or to be a **judge benchmark**: `ground_truth.type: llm_judge`,
non-empty `verification.subjective` (the rubric the blind pairwise judge
applies) and, like every hard record, `ground_truth.confidence` ≥ 0.8 (the
author's confidence in the rubric). `ground_truth.structured_result` is
descriptive metadata only; no verifier reads it, so it no longer stands in
for checks (records that relied on it must switch to `llm_judge` or add
deterministic checks). A hard record with neither checks nor `llm_judge`
ground truth is rejected.

The classifier keeps ranking LLM-judge ground truth 6 for mined sessions
(always at most SOFT). For an authored judge benchmark, `assess_benchmark`
makes one exception: with a successful replay, score ≥ 0.8 and rubric
confidence ≥ 0.8 it may be `HARD_BENCHMARK` (reason "pairwise judge
benchmark"). The evidence is a blind, position-swapped pairwise comparison,
not an absolute grade, and the promotion policy never lets a gain resting on
judged tasks auto-promote: it needs the signed operator approval
(`promotion.md`). Without tests the score is at most 0.85.

```json
"verification": {"deterministic": [], "subjective": [
  "Starts with 'INCIDENT: <service> - <impact>'.",
  "Blameless and factual."
]},
"ground_truth": {"type": "llm_judge", "confidence": 0.9}
```

## Deduplication and clustering

`hermes_improve.clustering` provides five layers:

1. exact content hash (SHA-256 of the raw prompt);
2. task fingerprint (case-folded, whitespace-normalized SHA-256);
3. lexical near duplicates (`task_similarity`: ≥ 8 words, word-set Jaccard
   ≥ 0.85) — must share split *and* `cluster_id`;
4. similarity clusters: character 5-shingles, 64-permutation MinHash with
   16×4 LSH banding for candidates, confirmed by exact shingle Jaccard ≥ 0.5;
   connected components get the deterministic id `sem-<fingerprint prefix>`;
5. semantic clusters (optional): cosine similarity of prompt embeddings
   ≥ `scoring.yaml` `clustering.semantic_similarity_threshold` (default
   0.85). Semantic links join the layer-4 union-find, so clusters are the
   connected components of the merged lexical + semantic graph.

Validation (`BenchmarkStore.validate`, public admission and holdout rotation)
rejects duplicate id/version pairs, a declared `cluster_id` used in more than
one split, near duplicates across split/cluster, and any merged similarity
cluster spanning splits — including public/holdout leakage. Layers 1–4 are
lexical: they catch rewordings that share vocabulary but not paraphrases
with different vocabulary. Layer 5 catches those when an embedder is
configured; a leak found only through semantic links is reported as
`spans splits [semantic layer]` (`ClusterLeak.detected_by == "semantic"`).
Mining (`benchmarks mine`, `mine_sessions(..., semantic=...)`) uses the same
layer, so a paraphrased session inherits the corpus `cluster_id` and
`DedupRegistry` pins it to that cluster's split.

### Semantic layer (layer 5) configuration

The backend is pluggable (`hermes_improve.embeddings.Embedder`: `model_id`,
`sealed_holdout_trusted`, `embed(texts)`); the core has no new dependency.
The shipped backend calls an OpenAI-compatible `POST /v1/embeddings`
endpoint, for example Qwen3-Embedding served locally by vLLM
(`vllm serve Qwen/Qwen3-Embedding-0.6B --task embed`), llama.cpp
(`llama-server --embeddings`) or Ollama (`/v1/embeddings`). The
environment-variable names live in `config/models.yaml` under `embedding`:

| Variable | Meaning |
|---|---|
| `QWEN_EMBEDDING_BASE_URL` | endpoint root, e.g. `http://127.0.0.1:8001`; no default |
| `QWEN_EMBEDDING_MODEL` | model name sent to the endpoint and recorded as the model id |
| `QWEN_EMBEDDING_API_KEY` | optional bearer token |
| `QWEN_EMBEDDING_TIMEOUT` | seconds, default 60 |
| `QWEN_EMBEDDING_ALLOW_REMOTE` | `true` to allow a non-loopback endpoint (public text only) |
| `HERMES_IMPROVE_EMBEDDING_CACHE` | cache directory (default `$HERMES_IMPROVE_HOME/embedding-cache`) |

The layer is enabled only when both URL and model are set; setting just one
is an error. Embeddings are cached per model in
`embeddings-<sha256(model id)>.json` (mode `0600`, directory `0700`), keyed
by the SHA-256 of the prompt; the cache holds hashes and vectors, no text.
Repeated checks reuse cached vectors and never re-query; a different model id
uses a different cache file, so changing the model invalidates the cache.
Links depend only on cached vectors and the threshold, so a check is
reproducible from the cache file plus the recorded model id.

`benchmarks validate [--holdout-root DIR]`, `benchmarks add`, `benchmarks
mine` and `benchmarks holdout rotate` print a `semantic_layer` object
(`status`, `reason`, `model_id`, `threshold`, `embedded_texts`,
`excluded_sealed_holdout`, `semantic_links`, `cache_hits`, `cache_misses`;
never prompt text). `status` is `applied`, `partial` (sealed holdout prompts
withheld from an untrusted backend) or `not_applied` (no endpoint
configured); the latter two also print a warning on stderr, so a missing
layer is never a silent pass. Without an endpoint the checks behave exactly
as layers 1–4. Commands that only re-read an already validated corpus
(`evaluate`, `benchmarks list`, `benchmarks manifest`) run layers 1–4; the
semantic gate sits at admission, rotation and explicit validation.

Sealed holdout prompts are embedded only through a loopback endpoint (see
`docs/security.md`). With `QWEN_EMBEDDING_ALLOW_REMOTE=true` the holdout is
withheld, the status is `partial`, and paraphrase leakage into the holdout is
not checked by layer 5.

## Suites

`hermes_improve.suites.select_suite(benchmarks, suite, *, impact=None,
hard_only=True, smoke_limit=50) -> tuple[Benchmark, ...]` is a pure,
deterministic function:

- `SMOKE`/`CORE`: public, non-shadow records declaring the suite, narrowed by
  an `ImpactScope` (categories, required capabilities, or candidate target
  paths matched against `benchmark.impact_areas` globs). Critical REGRESSION
  records in the suite are always kept. SMOKE is capped at 50
  (`SMOKE_LIMIT`).
- `FULL`: all public, non-shadow records declaring FULL; impact is ignored.
- `SEALED_HOLDOUT`: every sealed record supplied by the evaluator; impact is
  ignored so holdouts cannot be cherry-picked.

## Benchmark version

`BenchmarkStore.version` binds experiments and evaluations: an experiment
records it at creation, every evaluation run signs it, and promotion only
accepts runs whose version equals the experiment's (see `promotion.md`). It is
`sha256:` over the `{id, version, digest}` entries of every public record that
an evaluation suite can select, i.e. the TRAIN, VALIDATION and REGRESSION
splits. Two splits are deliberately outside it, each with its own digest:

- `SEALED_HOLDOUT`: `holdout_version`, signed into FULL reports. The generator
  cannot read the holdout and must derive the same version as the evaluator.
- `SHADOW`: `shadow_version`. SHADOW records are post-promotion monitoring
  traffic (`monitor replay --source shadow-benchmarks`) and are never selected
  for SMOKE, CORE or FULL, so adding or editing one does not invalidate open
  experiments or their evaluation evidence.

Experiments created before SHADOW was excluded carry the older formula (all
public records including SHADOW). In a benchmark store without SHADOW records
both formulas give the same value, so nothing changes. With SHADOW records,
such an open experiment no longer matches: `evaluate` refuses before any run
and says the experiment predates the exclusion, and the experiment must be
recreated. Completed evaluations, promotions and releases keep the version
they recorded; nothing re-hashes them against the current store.

**SMOKE minimum (advisory).** The spec targets 20–50 SMOKE tasks.
`suites.SMOKE_MIN = 20` is the lower bound, but it is deliberately *not
enforced* (a documented spec deviation): a selection below it still runs, and
`suites.smoke_size_warning(suite, task_count)` produces a warning instead.
`evaluate --suite SMOKE` prints it to stderr and records it in the signed
evaluation metrics under `suite_selection.warnings` (an empty list when there
is no shortfall); `evolve` prints it to stderr once per run
(`EvaluationGateway.screening_warnings`). Reasons for not blocking:

- SMOKE is non-deciding staged evidence. Promotion is decided by FULL plus
  the sealed holdout, so a small SMOKE suite cannot let a bad candidate
  through; it only gives a weaker early signal.
- Impact-aware narrowing legitimately shrinks SMOKE to a skill candidate's
  brain area, which is often far below 20 tasks. Blocking would force
  irrelevant tasks back in or make narrowing unusable.
- The bundled corpus (SMOKE 14, below) and small lab corpora would otherwise
  be unable to run the staged pipeline at all.

Only the latest version of an id is selected; only `HARD_BENCHMARK` records
by default. `suite_manifest(...)` returns `{schema_version, suite,
benchmark_version, impact, task_count, tasks[{id, version, digest, split,
cluster_id, categories}], digest}`, where `digest` is `sha256:` over the
canonical body and `benchmark_version` is `BenchmarkStore.version` (the public
records only, so the generator, which cannot read the holdout, derives the same
version as the evaluator; the holdout has its own `holdout_version`); holdout
manifests omit task details. See "Benchmark version" below for what the
version covers. `verify_manifest` detects tampering.
`corpus_manifest(...)` describes the public corpus (counts by split,
category, difficulty, suite, eligibility; per-record digests) and refuses
holdout records. `benchmarks/manifests/corpus.json` is the committed corpus
manifest; a test fails when it no longer matches the records.

CLI: `hermes-improve benchmarks list --suite CORE --impact category:git
--impact target:skills/git-helper [--role generator]` prints the suite
manifest; `hermes-improve benchmarks manifest` prints the corpus manifest.

## Replay

`hermes-improve benchmarks replay [--id ID] --commit SHA --repository PATH
[--runs N] [runner options]` re-runs public benchmarks N times (2–20, seeds
`0..N-1`) through the runner that `evaluate` would use. It takes the same
runner options as `evaluate` (`--runner`, `--timeout`, `--max-turns`,
`--hermes-agent-dir`, `--model`, … see `evaluation.md`), and the runner
decides which records are replayed:

| `--runner` | replayed records |
|---|---|
| `docker` (default) | command benchmarks (`environment.command`), digest-pinned `DockerCommandRunner` |
| `local` | `environment.type: skill_procedure`, `LocalSkillProcedureRunner` |
| `hermes` | `environment.type: hermes_agent`, the real Hermes agent through `HermesBatchRunner` |

A benchmark is *reproducible* when every run has the identical verifier
outcome (exit code, pass/fail, success, quality, security pass, environment
digest). It is *replay verified* when it is also passing, and what passing
means depends on whether a model is in the loop:

- Deterministic runners (`criterion: identical_outcome`): every run passed the
  deterministic verifiers, so the reference solution satisfies its own ground
  truth.
- LLM-driven runners (`criterion: all_runs_pass`): every one of the N runs
  finished as a completed agent run *and* passed the deterministic verifiers.
  A timeout, a partial trajectory or a turn/tool limit fails the replay even
  when the workspace happens to be right. Token counts, latency, tool calls
  and wording vary between LLM runs and are not compared; the verdict must not
  vary. All runs must use the same model configuration, otherwise the replay
  is an integrity error. Use enough runs (the default 3, or more) for the
  evidence to mean something; one lucky run is not evidence.

Results are printed without command output and appended to the
`benchmark_replays` table of the experiment database; the command exits 1
if any benchmark is not replay verified. `ReplayResult.evidence()` is the
`eligibility.replay_evidence` block for authored records: the required
fields (`commit`, `runs`, `reproducible`, `passed`, `outcome_digest`,
`environment_digest`, `replayed_at`) plus the optional runner identity
`runner`, `llm_influenced`, `criterion` and, for Hermes, `model` and
`model_config_digest`. Older evidence without the optional fields stays
valid. This is how `hermes_agent` records obtain the replay evidence that
`HARD_BENCHMARK` and `benchmarks validate` require (lab finding F14). The
library entry point is `replay.replay_benchmark(runner, benchmark, *, commit,
runs, runner_name=None, model=None)` for any `BenchmarkRunner`. Sealed
holdout records are never replayed here: the replay reads only the public
tree, configures no holdout root, and refuses `SEALED_HOLDOUT` records.

Images are pulled by the operator in advance; the runner uses
`--pull=never` and `--network=none`.

## Curated corpus

The repository ships 26 deterministic reference-solution command benchmarks
plus six brain-consuming `skill_procedure` benchmarks (next section; public
splits only, no holdout data is in the repository). The tables describe the
26 command benchmarks:

| Split | Count |
|---|---|
| TRAIN | 10 |
| VALIDATION | 8 |
| REGRESSION | 8 |

| Category | Count | Category | Count |
|---|---|---|---|
| coding | 3 | docs | 3 |
| git | 4 | devops | 3 |
| shell | 3 | research | 3 |
| data-transform | 4 | mcp | 3 |

Difficulties: 11 easy, 13 medium, 2 hard. Suites: SMOKE 14, CORE 23,
FULL 26 (the SMOKE target of 20–50 tasks needs a larger corpus; until then
SMOKE runs emit the advisory `SMOKE_MIN` warning described under Suites).

Each task lives in `benchmarks/fixtures/<id>/`:

- `task/` — the starting workspace (absent when `setup.sh` builds it);
- `setup.sh` — optional deterministic setup (for example a Git history with
  fixed author dates);
- `solve.sh` (+ `solution/`) — the reference solution;
- `check.sh` — the deterministic check; prints one JSON line
  `{"checks": N, "failed": [...], "status": "pass"|"fail"}` and exits 0 only
  on pass. Checks verify results and guard against shortcuts (for example
  that unit tests were not edited, plus hidden assertions).

`benchmarks/fixtures/harness/run-reference.sh` copies `task/` into a scratch
directory, runs setup and the reference solution, then the check. The record
command is `["sh", "../harness/run-reference.sh"]` with
`cwd: benchmarks/fixtures/<id>`; the verifier requires exit code 0 and
`status == "pass"`, `failed == []`, `checks == N`. `environment.input_data`
is `replay.fixture_tree_digest()` of the fixture directory, and a test keeps
it in sync. Git tasks use `alpine/git` pinned by digest; all others use
`python:3.12-slim` pinned by digest.

All 26 records carry replay evidence from three Docker replays each at
commit `85b0c7d` and are `HARD_BENCHMARK` with reproducibility score 1.0.
Every reference solution was also checked to fail when the solution step is
skipped, so the checks are not vacuous.

What the reference replay proves is that the task, its ground truth and its
check are deterministic. Running the *agent* on these tasks (agent work
replacing `solve.sh`) needs the Hermes batch-runner adapter, which is not
part of this package; `hermes-improve evaluate` still runs each record's
command against the baseline and candidate commits. Because those commands
run reference solutions, their outcome does not depend on the brain under
evaluation: baseline and candidate score identically. The brain-consuming
tasks below close that gap.

## Brain-consuming `skill_procedure` benchmarks

Six further public records (TRAIN 2, VALIDATION 2, REGRESSION 2, category
`git`, impact area `skills/git-commit-hygiene`) measure the *skill text*
itself. A skill declares its procedure in exactly one fenced
` ```hermes-procedure ` block in `SKILL.md`, one operation per line from a
fixed set (`inspect`, `stage-all`, `stage-tracked`, `stage-untracked GLOB...`,
`unstage GLOB...`, `review-staged`, `subject TEMPLATE`, `body TEMPLATE`,
`trailer KEY TEMPLATE`, `commit`, `commit-no-verify`, `verify`; templates use
the task intent `{type}`, `{scope}`, `{summary}`, `{why}`, `{ticket}`).

- `benchmarks/fixtures/skill-harness/skill-executor.sh` (POSIX sh, runs in
  `alpine/git`) builds the task repository with the fixture's `setup.sh`,
  executes the procedure (arguments are data: globs go through `case`,
  templates are filled by `awk`; nothing from the skill is evaluated as
  shell), stops at the first failing or unknown operation, then runs the
  fixture's `check.sh`. Only the check's JSON line
  `{"checks", "failed", "results": {name: bool}, "status"}` is printed; the
  operation trace goes to `trace.log`.
- `skillcheck.sh` grades the commit: one new commit, exactly the intended
  paths, no secret files, no generated output, conventional subject, body
  with the reason, untracked leftovers preserved, and the `Refs:` ticket
  trailer when (and only when) the task has a ticket.
- Records use `environment.type: "skill_procedure"`, `harness`,
  `harness_digest`, `repository_fixture` (`fixtures/<id>`, relative to the
  benchmark root) and `input_data` (fixture digest). Verifiers: exit code,
  `status == "pass"`, one `json_path_equals` per named check, and a
  `git_diff` check that only `trace.log` changed, so quality is graded per
  check (11 verifiers).
- The runner (`SandboxCommandRunner`) takes only `brain/` from the evaluated
  commit; harness and fixture are copied from the evaluator's own benchmark
  root (fixtures of SEALED_HOLDOUT records from the holdout root) and must
  match the pinned digests. A candidate cannot change its yardstick.
- `benchmarks/fixtures/skill-reference/` holds the reference skill (the
  analogue of `solve.sh`). The replay evidence of all six records comes from
  three `DockerCommandRunner` replays at the deterministic reference brain
  commit `96e5cbf` built by `tests/skill_procedure_support.py`;
  `tests/integration/test_skill_procedure_corpus.py` replays them again in a local
  process sandbox and must reproduce the same outcome and environment
  digests.

The shipped `git-commit-hygiene` skill passes the three hygiene tasks and
fails only the ticket check on the three ticket tasks, so a candidate that
adds the trailer is measurably better and a candidate that stages everything
is measurably worse (`tests/e2e/test_full_improvement_loop.py`).

To run the `docker`-marked tests from a container, mount the Docker socket
and a temporary directory at the same path the daemon sees, for example
`-v /var/run/docker.sock:/var/run/docker.sock -v /tmp/hsi-dood:/tmp/hsi-dood
-e TMPDIR=/tmp/hsi-dood`, because the sandbox bind-mounts host paths.

## Admission and access

- `hermes-improve benchmarks add FILE --holdout-root HOLDOUT_DIR` admits an
  operator-authored JSON or YAML public record after validating the record
  and the complete public/holdout corpus (including similarity leakage). It
  rejects sealed holdout records and duplicate id/version pairs; an existing
  destination is never replaced.
- Generator-facing stores may contain only public roots; they expose
  TRAIN/VALIDATION only. Holdout access uses an evaluator capability that
  must be backed by OS filesystem permissions.
- `benchmarks inspect-candidate` / `review-candidate` provide bounded
  previews and immutable triage decisions for mined proposals; acceptance
  does not admit a benchmark.

## Sealed holdout rotation (operator procedure)

Holdout data never lives in this repository. The operator keeps a managed
holdout store on protected storage (owned by the operator account, readable
by the evaluator identity only, not readable by the generator/agent users):

```text
STORE_ROOT/
  .rotation.lock
  generations/<digest>/...   # immutable generations
  active -> generations/<digest>
```

1. **Stage**: write the new sealed records (SEALED_HOLDOUT split, JSON/YAML,
   regular files only, no hidden files or symlinks) into a staging directory
   outside both the public benchmark tree and the store.
2. **Rotate**: `HERMES_IMPROVE_BENCHMARKS=<public root> hermes-improve
   benchmarks holdout rotate STAGING --store-root STORE_ROOT`. Under an
   exclusive lock the command snapshots the staging files, validates every
   record (schema + semantics + SEALED_HOLDOUT split), runs the full
   public/holdout leakage checks, copies the snapshot into a private
   temporary directory, re-reads and re-verifies it, removes write bits,
   renames it to `generations/<digest>` and atomically replaces the `active`
   symlink. Any failure leaves the previous active generation unchanged and
   removes the temporary copy.
3. **Output**: only `{"generation", "record_count", "status"}` — never
   benchmark content.

`<digest>` is the sealed-manifest digest used by the spent-holdout registry
(`security.consume_holdout_generation`), so registry rows and generation
directories share one identifier. Re-rotating an unchanged manifest is
rejected; existing generations are never overwritten or re-activated, and
nothing is deleted (retention is unlimited; pruning is a manual operator
decision). The spent-holdout registry is never touched. The command creates
new directories `0700`/`0750` and files `0640` (subject to umask), only
removes write bits afterwards, and never relaxes permissions of existing
paths — filesystem permissions remain the security boundary.

**Evaluator compatibility**: pass `--holdout-root STORE_ROOT/active` to
`hermes-improve evaluate`; it resolves the symlink to the active generation.
After rotation, lineages that consumed the previous generation can be
evaluated again on the new one.

## Holdout access and spending in `evaluate`

Only finalists get holdout evaluation (spec §16). `hermes-improve evaluate
--suite FULL` adds the sealed holdout, and before it grants the evaluator
holdout capability or runs any task it requires finalist evidence
(`holdout_gate.require_holdout_finalist`):

- the ledger holds evaluator-signed SMOKE and CORE evaluation runs for the
  same candidate ID and content hash, the current benchmark version and the
  current promotion policy version; runs for another identity are ignored;
- the newest such SMOKE and the newest such CORE run each pass the per-suite
  evidence gates the promoter applies (`promotion.check_suite_evidence`: all
  security and deterministic verifiers passed, critical/high regressions
  within policy, no task the baseline solves and the candidate fails);
- every considered run verifies against the evaluator's own Ed25519 key
  (`promotion.verify_evaluation_run_signature`); a matching run with a broken
  signature is treated as tampering and refuses access.

Otherwise the command fails with `PolicyRejected` ("sealed holdout is reserved
for finalists ...") or `IntegrityError`. The order is therefore `evaluate
--suite SMOKE`, `--suite CORE`, then `--suite FULL`. This also applies to
`evolve` finalists: their screening runs happen under the agent identity and
are neither signed nor visible to the evaluator ledger, so the agent-written
`generated_variants.status = full_passed` is not accepted as finalist proof;
the evaluator re-runs SMOKE and CORE itself. (`evolve` with a holdout-capable
gateway only offers the holdout to variants that passed FULL screening.)

The holdout generation is spent when holdout tasks execute, not when access is
granted. A non-binding pre-check refuses a lineage that already spent the
active generation before any task runs. The binding spend is an atomic insert
into the spent-holdout registry (`spent_holdouts`, primary key generation +
`lineage:<lineage_id>`) made by `HoldoutSpendingRunner` immediately before the
first SEALED_HOLDOUT task is handed to the runner, and committed before that
task starts. Consequences:

- a failure before any holdout task ran (infrastructure error, crash during the
  FULL tasks, finalist refusal) leaves the generation unspent;
- once a holdout task was handed to the runner the generation stays spent,
  whether the run then passes, fails, raises or the process dies; no signed
  report is recorded for a crashed run and the lineage cannot retry until the
  operator rotates the holdout;
- of concurrent `evaluate` runs for one lineage exactly one wins the insert;
  the others abort with `AccessDenied` before executing any holdout task;
- a crash between the committed insert and the first runner call spends the
  generation without any holdout feedback (fails closed).

## Known limits

- The corpus is curated, not mined from real Hermes sessions; provenance is
  descriptive metadata, not an attestation.
- Semantic clustering (layer 5) needs a configured embedding endpoint; without
  one, clustering is lexical (character shingles) and paraphrases with
  different vocabulary can evade it and may need manual `cluster_id`s. The
  threshold is model-dependent and should be calibrated per embedding model.
  The pairwise comparison is exhaustive (O(n²) in pure Python), which is
  fine for curated corpora of a few thousand records but not for much larger
  ones.
- Agent execution on corpus tasks depends on a Hermes runtime adapter
  outside this package.
