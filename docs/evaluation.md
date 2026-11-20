# Evaluation

The evaluator compares a candidate brain commit with its baseline on identical
tasks, seeds, environment, model configuration and sandbox limits, verifies
each run deterministically, and hands per-task outcomes to `PromotionPolicy`.
It never changes candidates, production or promotion rules; its verdict can be
stricter than the policy (judge blocks), never more lenient.

## Entry points

```text
hermes-improve evaluate <candidate-id> --repository <brain-repo> --holdout-root <dir>
    [--suite SMOKE|CORE|FULL]          # default FULL; FULL also runs the sealed holdout
    [--seeds 1,2,3]                    # default: derived from the run policy
    [--runner docker|hermes|local]     # default docker; local: skill_procedure only
    [--acknowledge-no-container-isolation]   # required with --runner local
    [--timeout S] [--max-turns N] [--max-tool-calls N]
    [--network NAME --network-allow HOST:PORT ...]
    [--verifier-http-allow HOST:PORT ...]
    [--hermes-agent-dir DIR --model M --model-base-url URL [--model-api-key-env VAR]
     [--hermes-python PY] [--fixtures-root DIR]]
    [--judge]                          # qwen_judge role from QWEN_* environment
```

`hermes-improve benchmarks replay` takes the same runner options (without
`--holdout-root`) to produce `eligibility.replay_evidence` for command,
`skill_procedure` and `hermes_agent` records; see the replay section of
`benchmarking.md`.

The command must run as `HERMES_EVALUATOR_UID` and signs with
`HERMES_EVALUATOR_PRIVATE_KEY`. Every suite run is recorded as one signed
evaluation. SMOKE and CORE runs are staged evidence:
`ExperimentStore.record_evaluation(..., decides_experiment=False)` leaves the
experiment decision untouched, unless the staged run already proves a
regression (`evaluator.staged_regression`: a task the baseline solves fails
for the candidate, a security verifier fails, or a critical/high regression is
reported); then the experiment is recorded `rejected` without ever consuming
the sealed holdout. A staged run never marks an experiment `passed`. FULL
(including the sealed holdout, which is consumed per lineage) sets
`passed`/`rejected`.

Before anything runs, `evaluate` (and `candidate import`) re-derive the
candidate's changed paths from Git and refuse candidates that touch
protected paths (`benchmarks/`, `config/`, `src/`, ...), anything outside
`brain/`, or paths outside their declared target units
(`target_paths.assert_evaluable_candidate_paths`). Otherwise a candidate could
edit the fixtures or checks it is graded with.

The report's `benchmark_version` is the evaluation manifest version
(`BenchmarkStore.version`: public records without SHADOW), which the generator
also records when it creates the experiment. FULL reports additionally sign
`metrics.sealed_holdout_version` and the holdout manifest digest
(`suite_selection.holdout_manifest_digest`).

Also before anything runs, and before suite-selection decisions are recorded
or the lineage's sealed holdout is consumed, `evaluate` compares the current
`BenchmarkStore.version` with the version the experiment was bound to at
creation (`BenchmarkStore.require_bound_version`; `PairedEvaluator.evaluate`
repeats the check when it has the experiment store). A mismatch aborts with
"benchmark version changed since experiment <id> was created", naming both
versions. Promotion would reject any evaluation recorded under another version
("evaluation benchmark version differs from experiment manifest"), so running
it would only waste the runs and the holdout. Create a new experiment for the
current benchmark manifest instead.

Task selection uses `suites.select_evaluation_suite` (WP-B `select_suite`). Skill
candidates get an impact scope from their target (`brain/skills/<name>` ->
`skills/<name>` matched against `benchmark.impact_areas`, plus taxonomy
categories named by the skill); other candidate types are not narrowed. If
narrowing leaves SMOKE/CORE empty the full suite is used. FULL ignores impact
and adds every sealed holdout task.

A SMOKE selection (after narrowing) with fewer than `suites.SMOKE_MIN` (20)
tasks prints `warning: SMOKE suite has N task(s), below the advisory minimum
of 20; ...` to stderr and is recorded in the signed metrics as
`suite_selection.warnings`; the run continues. The minimum is advisory on
purpose: SMOKE does not decide promotion, impact narrowing legitimately
produces small SMOKE sets, and small corpora must still run (see
`docs/benchmarking.md`, Suites).

Programmatic use: `PairedEvaluator(benchmark_store, runner, policy,
experiment_store=None, evaluator_signing_key=None, artifact_store=None, *,
run_policy=RunPolicy(), judge=None).evaluate(candidate, baseline_commit=...,
benchmark_ids=..., suite=..., benchmark_version=..., seeds=None,
holdout_grant=None, evaluation_id=None)`.

Only `HARD_BENCHMARK` records are accepted (reproducibility score >= 0.8,
replay verified, ground-truth confidence >= 0.8, deterministic verification or
a structured ground-truth result).

## Run-count policy (`run_policy.RunPolicy`)

| Runner | SMOKE | CORE | FULL | SEALED_HOLDOUT |
|---|---|---|---|---|
| deterministic (`llm_influenced = False`, e.g. `DockerCommandRunner`) | 1 | 1 | 1 | 1 |
| LLM-influenced (`HermesBatchRunner`) | 2 (1-2 allowed) | 3 | 5 | 5 |

Runners declare `llm_influenced`; a runner that omits it is treated as
LLM-influenced. Seeds default to `base_seed + 0..n-1`. Explicit seeds are
honoured but must meet the minimum. Baseline and candidate always run the same
seed list, and the evaluator rejects pairs whose `environment_digest`,
`model_config_digest` or `limits_digest` differ.

The CLI builds the policy with `RunPolicy.from_config(config.scoring)`: each
suite runs the `max` of `scoring.yaml run_counts.llm.<SUITE>` and
deterministic tasks run `run_counts.deterministic` times. The spec floors above
still apply, so configuration can only add runs. The `PromotionPolicy` comes
from `promotion.yaml` (`PromotionPolicy.from_config`), so every recorded
evaluation carries the same `policy_version` the promoter checks. SMOKE and
CORE runs are stored with `decides_experiment=False`; the promoter picks the
newest signed run per suite (SMOKE, CORE, FULL, SEALED_HOLDOUT) as evidence.

## Deterministic verifiers (`verifiers.py`, `verifier.py`)

`verification.deterministic` entries are objects with a `type` (or a bare type
name for parameterless checks). `default_registry()` maps types to factories
that validate specs strictly (unknown fields are rejected, at most 64 checks).
An exit-code check against `ground_truth.expected_exit_code` is always added.
Quality is the percentage of passing checks; a run succeeds only if all pass.

| Type | Spec fields | Where it runs |
|---|---|---|
| `exit_code` | `expected` | task status (Hermes: agent completed within limits) |
| `command`, `build`, `lint`, `typecheck` | `argv`, `cwd`, `timeout_seconds`, `expected_exit_code`, `output_contains`, `output_regex` | sandbox executor on a fresh copy of the post-run workspace |
| `unit_test`, `integration_test` | same as `command`; a run that executed zero tests fails | sandbox executor |
| `file_exists` | `path`, `must_exist` | host read of extracted workspace; symlinks never followed |
| `file_content` | `path` + one of `contains`, `not_contains`, `equals`, `sha256` | host read |
| `unchanged_path` | `path` | pre-run source vs post-run workspace |
| `git_diff` | `allowed_paths`, `forbidden_paths`, `required_paths`, `max_changed_files` | pre/post change set |
| `json_schema` | `schema` (subset: type, properties, required, additionalProperties, items, enum, const, min/max, length, pattern, items count), `source` output/file, `path` | host read |
| `regex` | `pattern`, `source`, `path`, `must_match` | host read |
| `http_response` | `url`, `method`, `body`, `expected_status`, `body_contains`, `body_regex`, `timeout_seconds` | evaluator, only to `--verifier-http-allow` hosts, no redirects |
| `mcp_health` | `argv`, `cwd`, `timeout_seconds`, `expected_tools`, `protocol_version` | sandbox executor; JSON-RPC `initialize` + `tools/list` over stdio |
| `output_exact`, `output_contains`, `output_not_contains`, `json_output_equals`, `json_required_keys`, `json_path_exists`, `json_path_equals` | `expected`, `path` | captured task output |

Commands are never executed on the host: a verifier without a sandbox executor
raises `SandboxPolicyError`. Configuration errors (unknown types, a
non-allowlisted HTTP host) abort the evaluation instead of being recorded as
candidate failures. For deterministic benchmarks the evaluator requires every
observation to carry verifier summaries consistent with `deterministic_pass`.

`mcp_health` speaks the MCP stdio protocol directly. Hermes' own
`hermes mcp test <name>` (exit 0 ok / 1 fail / 3 unknown) could be wrapped as a
`command` verifier with `expected_exit_code: 0` when a Hermes install is part
of the sandbox image.

## Sandbox (`sandbox.py`)

`DockerSandboxExecutor` runs each command in a disposable container:
digest-pinned image, `--pull=never`, `--network=none` by default, read-only
root, user 65534, `--cap-drop=ALL`, `no-new-privileges`, PID/memory/CPU
limits, `--ipc=none`, size-bounded tmpfs for `/tmp` and `/workspace`, read-only
input bind mount, and only `PATH`/`HOME` plus the daemon-selection variables
`DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_TLS_VERIFY`, `DOCKER_CERT_PATH` and
`DOCKER_CONFIG` (when set) passed to the Docker client, so remote, rootless and
DinD daemons work. These reach the client only, never a container. If Docker
fails before creating the container (no cidfile), typically because the daemon
is unreachable, the executor raises an error naming the daemon and the CLI
output instead of a missing-snapshot error. Only `HERMES_EVAL_*` variables can
be passed into containers. The
post-run workspace is returned as a tar snapshot; the host extracts regular
files and directories only (symlinks and special files are reported, never
materialized) within size and entry limits.

Network allowlist mode: `NetworkPolicy("allowlist", network_name,
allowed_hosts)`. The operator creates a dedicated network with
`docker network create --internal <name>` and attaches only the allowlisted
fixture services (container names must match the allowlisted host names).
Before each run the evaluator inspects the network and refuses it unless it
is `Internal: true` and every attached container is allowlisted. A benchmark
opts in with `environment.network_policy: "allowlist"` and
`environment.network_allowlist: ["host:port", ...]`, which must be a subset of
the evaluator allowlist. Limits of this mode: Docker does not filter ports, so
any port of an attached service is reachable; DNS resolves only container
names on that network; there is no route to the internet or the host network.

`SandboxLimits` holds timeout, memory, CPUs, PIDs, tmpfs sizes, workspace byte
and entry limits, `max_turns` and `max_tool_calls`; its digest is recorded per
run. `LocalSandboxExecutor` (scrubbed environment, own process group, POSIX
rlimits, private copy) is for tests and development only and must be
acknowledged explicitly; it provides no kernel isolation.

## Runners

`DockerCommandRunner` (deterministic) exports the commit with `git archive`,
runs `environment.command` in the sandbox, snapshots the workspace, runs the
verifiers, and stores a run record and the unified diff in the
content-addressed `ArtifactStore`. For `environment.type: "skill_procedure"`
records it exports only `brain/` of the commit and adds the harness and task
fixture from the evaluator's benchmark roots (public, or holdout for
SEALED_HOLDOUT tasks), verified against the record's `harness_digest` and
`input_data`; the environment digest covers both (see `benchmarking.md`).
No model runs in these runners, and nothing measures token usage, so their
observations carry `tokens=0` with `token_usage_reported=False`: the zero is
not presented as a measurement. A benchmark set that only uses these runners
needs `require_token_measurement: false` in the protected `promotion.yaml` to
promote (the end-to-end test does exactly that); the shipped default blocks.

`LocalSkillProcedureRunner` (`--runner local
--acknowledge-no-container-isolation`) runs the same composition through
`LocalSandboxExecutor` for hosts without Docker and for the default test path.
It refuses every benchmark that is not `skill_procedure`, because those run
commands from the evaluated commit; for `skill_procedure` tasks the executed
program is the evaluator-owned harness and the candidate only supplies data.
It has no kernel isolation beyond the evaluator's own Unix user (scrubbed
environment, private copy, rlimits).

`HermesBatchRunner` (LLM-influenced) runs one conversation of the real Hermes
agent per task through `hermes_driver.py`, a script shipped with this package
and executed by the Hermes interpreter (`--hermes-python`) in a private temp
directory. The driver imports `AIAgent` from the unmodified Hermes checkout
(`--hermes-agent-dir`) and calls it the way `batch_runner.py` does
(`skip_context_files`, `skip_memory`, one task id, `max_iterations` = max
turns), but with explicit toolsets: `HermesModelConfig.toolsets`, default
`terminal,file,skills`.

Why not `batch_runner.py` itself (isolated lab, Hermes v0.22.2 with a local
Qwen3-4B):

- batch_runner samples toolsets only from the fixed
  `toolset_distributions.DISTRIBUTIONS`, and none of them contains `skills`.
  In every real trajectory the system prompt held no skill at all, so the
  evaluated brain could not change any outcome.
- batch_runner drops every completed prompt without reasoning, which is a
  training-data filter. Every run of an instruct model disappeared.
- batch_runner takes the API key on its command line.

The driver reuses batch_runner's helpers for tool statistics and for
registering the task's pinned `image` and workspace `cwd` as task overrides.
Hermes bind-mounts only that `cwd` override into the Docker sandbox
(`tools/terminal_tool._resolve_task_host_cwd`); with `TERMINAL_CWD` alone the
agent edited the container's private filesystem and the verified workspace
stayed unchanged.

`HERMES_HOME/skills` is copied from `brain/skills` of the evaluated commit. The
benchmark's `environment.repository_fixture` (below `--fixtures-root`) becomes
the workspace. Hermes tools run with `TERMINAL_ENV=docker`, the benchmark's
pinned image, `TERMINAL_DOCKER_NETWORK=false`, no forwarded environment or
volumes, and non-persistent containers; the host environment is not inherited
except the same Docker daemon-selection variables the sandbox forwards to its
Docker client (`DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_TLS_VERIFY`,
`DOCKER_CERT_PATH`, `DOCKER_CONFIG`, when set). The model key reaches the driver as `HERMES_EVAL_API_KEY`, never on argv.

The driver writes `out/trajectories.jsonl` (one row in batch_runner's schema)
and `out/usage.json` (Hermes' session token counters, stored in the run record
as `hermes_reported_usage`). Exit 3 means the agent failed before a trajectory
existed (`exit_state=agent_error`, never success); any other non-zero exit is an
infrastructure error. Tool-call and turn limits are enforced on the parsed
trajectory.

Timeouts keep a partial trajectory (lab finding F16). When the run exceeds
`--timeout`, the runner sends SIGTERM to the driver's process group and
SIGKILL after `HermesBatchRunner.termination_grace_seconds` (default 5 s). On
SIGTERM the driver at once writes a fallback row (`completed: false`,
`partial: true`, `terminated: "SIGTERM"`, the prompt as the only turn) and
`usage.json` with Hermes' token counters so far, then calls
`AIAgent.interrupt()` when the checkout has it. If the conversation returns
within the grace period, the row is replaced by the interrupted trajectory
with its tool statistics, still partial and not completed (driver exit 4).
Both files are replaced atomically. The runner parses that row on timeout,
so the observation keeps tool calls, API calls and `hermes_reported_usage`;
an unreadable row is dropped. The run is still classified
`exit_state=timeout` and never succeeds. An agent that ignores the interrupt
keeps only the fallback row: Hermes' token counters and API calls, but no
tool statistics. The trajectory, diff and run record are stored as artifacts. Known
gap: Hermes has no sampling seed (the seed is passed only as
`HERMES_EVAL_SEED`). Live execution requires a Hermes install and a reachable
model with at least 64K context (Hermes' floor). CI exercises the adapter with
a fake Hermes checkout (`tests/fixtures/hermes/`) that mirrors the imported
API.

### Token metering (`metering_proxy.py`)

Tokens are measured between Hermes and the model endpoint; Hermes' own counters
from `out/usage.json` are only a cross-check.

Tokens are therefore measured between Hermes and the model endpoint. For every
run `HermesBatchRunner` starts a `MeteringProxy` on `127.0.0.1` (random port)
and passes its URL to the driver as `--base-url`; the proxy forwards each
request unchanged (method, path, query, headers including `Authorization`,
body) to the configured endpoint host and relays
the response unchanged, streaming SSE chunk by chunk. From every 2xx response
to a generation path (`…/chat/completions`, `…/completions`, `…/responses`,
`…/messages`, Ollama `/api/chat` and `/api/generate`) it reads the token
counts: the top-level `usage` of JSON
responses, the final chunk of OpenAI streams (Hermes sends
`stream_options.include_usage`, `agent/chat_completion_helpers.py`), the
`response.completed` event of the Responses API, and `message_start` plus
`message_delta` of Anthropic streams, and `prompt_eval_count`/`eval_count`
of Ollama native responses (JSON or NDJSON). Paths outside the base path are
forwarded too: Hermes probes the server root (Ollama `/api/tags`,
`/api/show`) to recognise the backend, and its 64K context floor and Ollama
`num_ctx` handling depend on that probe. Only generation paths are metered. The run record artifact stores the
per-call usage under `token_usage`; headers are never stored or logged.

A run counts as measured (`token_usage_reported=True`) only when all of these
hold, otherwise it reports `tokens=0` with `token_usage_reported=False`:

- every 2xx generation response carried a valid usage (non-negative integer
  input and output counts); one response without usage, for example from a
  server that ignores `stream_options`, or a body cut off mid-stream makes the
  whole run unmeasured instead of undercounting it;
- no generation request was still open when the run ended (5 s grace);
- the proxy saw at least as many successful generation calls as the
  trajectory's `api_calls`; fewer means model traffic bypassed the proxy.

Non-2xx responses (rate limits, provider errors, refused redirects) are
relayed to Hermes and counted as failed calls; they carry no completion.
`tokens = input + output`, `input_tokens`/`output_tokens` are sums over the
run, and `context_tokens` is the largest single-call prompt. Limits: the proxy
only sees traffic to the configured endpoint, so auxiliary models that Hermes
would reach through a different provider configuration are not counted (each
run uses a fresh `HERMES_HOME` without such configuration, and the
`api_calls` check catches main-loop traffic that bypasses it). Provider detection by
hostname (for example hosted-provider URLs) sees a loopback URL instead of the
real host; detection by probing the server (local servers such as Ollama) works
through the proxy. `token_metering=False` on the
runner restores the unmetered behaviour; with the default
`require_token_measurement: true` such runs fail the policy with "token
efficiency not measured" (see `promotion.md`).

## Judge (`judge.py`, `src/hermes_improve/prompts/judge/`)

The `qwen_judge` role is used only for benchmarks with no deterministic checks
and non-empty `verification.subjective`. Evaluation only accepts
`HARD_BENCHMARK` tasks; a judge task is hard when its `ground_truth.type` is
`llm_judge` with confidence ≥ 0.8 (rule and rationale in `benchmarking.md`,
"Judge benchmarks"); `structured_result` does not qualify. If such a benchmark is selected
without `--judge`, evaluation fails before running anything (the CLI refuses
before it records any `suite_selection`/`needs_qwen` decision). Each paired run is
presented twice (A=baseline/B=candidate, then swapped) with only the task,
criteria and the two outputs. The answer must be strict JSON
`{"winner": "A|B|TIE", "confidence": 0..1, "uncertainty": 0..1, "rationale"}`.
Invalid answers or transport errors (timeouts, refused, dropped or reset
connections, malformed HTTP responses, e.g. when Ollama restarts mid-run) are
retried up to `max_attempts` per
presentation and then yield `unreliable`; contradictory positions, confidence
below 0.5 or uncertainty above 0.5 also yield `unreliable`. Verdicts across
seeds must agree. A `baseline` or `unreliable` verdict adds a blocking reason;
a `candidate` verdict changes nothing, so the judge can never promote alone.
Qwen roles share the endpoint but resolve their own model from `config/models.yaml`
(`roles.<role>.model_env`, falling back to `QWEN_MODEL`; `QWEN_JUDGE_MODEL` for the
judge), and keep separate statistics and logs (`QWEN_LOG_DIR/<role>.jsonl`, digests
only). The shared default model is the variable all roles name as their common
`fallback_model_env` (else the `qwen_solver` role's `model_env`), so no variable
name is hardcoded in the adapter. `evaluate --judge` persists the judge's
request/token/latency counters to `observability.sqlite3` (also when the
evaluation fails), and the per-task `judge_uncertainty` stored in
`task_results` feeds the report's `qwen.judge_uncertainty`. The routing rule
(deterministic checks, else subjective criteria → judge, else exit code only)
is `evaluator.judge_route`; each evaluate run records it per selected task as
a `needs_qwen` decision (sealed holdout tasks without their ID), and records
the impact-aware selection as a `suite_selection` decision (see
`laya-future.md`). `evaluate --runner hermes` defaults the solver to the `qwen_solver` role
(`QWEN_BASE_URL`/v1) when `--model`/`--model-base-url` are omitted.

## Report contents

`EvaluationReport.metrics` keeps the policy metrics and adds (schema version
`evaluation_schema_version: 2`, signed with the report):

- `suite`, `run_policy` (counts, `llm_influenced`, `runs_per_task`);
- `suite_selection`: the `suite_manifest` digest of the selected tasks, the
  holdout manifest digest (FULL only, ids/versions/digests), the applied impact
  scope and the task IDs;
- `paired_statistics` for quality, success rate, tokens, latency and tool
  calls: per-task paired mean delta, CI95, SD, Cohen's d_z, Hedges' g_z and a
  magnitude label. No p-values are produced;
- `efficiency` per arm: tokens per task/success, input/output tokens, context
  size, tool calls per success, failed tool calls, retries, turns, API calls,
  latency per success, filesystem changes and exit states, plus
  `cheaper_failures` and efficiency deltas that are only computed when the
  candidate's success rate is not lower;
- `judge` configuration and per-task verdicts. Each verdict keeps its
  per-seed, per-order `presentations` (`run`, `seed`, `order`, the parsed
  `judgment` or the `error`) so position bias can be audited; at most 20
  entries per task are kept, rationales and errors are cut to 500 characters,
  and `presentations_omitted` counts entries dropped by the cap.

`RunObservation` v2 fields (all optional with defaults): `input_tokens`,
`output_tokens`, `token_usage_reported`, `context_chars`, `turns`,
`api_calls`, `exit_state`, `model_config_digest`, `limits_digest`,
`verifier_results`, `filesystem_changes`, `trajectory_digest`, `diff_digest`.
`TaskOutcome` adds `run_count`, `judge_result`, `judge_confidence`,
`judge_uncertainty` and `token_usage_reported` (true only when every paired
run of both arms reported token usage; defaults to false). The signed top-level payload shape is unchanged, so
reports written before version 2 still verify.

An evaluation report is not sufficient on its own to establish operational
safety: benchmark quality, evaluator key custody, filesystem isolation,
independent role identities, and the runner's environment must also be trusted.
