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
    [--runner docker|hermes]           # default docker
    [--timeout S] [--max-turns N] [--max-tool-calls N]
    [--network NAME --network-allow HOST:PORT ...]
    [--verifier-http-allow HOST:PORT ...]
    [--hermes-agent-dir DIR --model M --model-base-url URL [--model-api-key-env VAR]
     [--hermes-python PY] [--fixtures-root DIR]]
    [--judge]                          # qwen_judge role from QWEN_* environment
```

The command must run as `HERMES_EVALUATOR_UID` and signs with
`HERMES_EVALUATOR_PRIVATE_KEY`. Every suite run is recorded as one signed
evaluation. SMOKE and CORE runs are staged evidence:
`ExperimentStore.record_evaluation(..., decides_experiment=False)` leaves the
experiment decision untouched; FULL (including the sealed holdout, which is
consumed per lineage) sets `passed`/`rejected`.

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
input bind mount, and only `PATH`/`HOME`/`DOCKER_CONFIG` passed to the Docker
client. Only `HERMES_EVAL_*` variables can be passed into containers. The
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
content-addressed `ArtifactStore`.

`HermesBatchRunner` (LLM-influenced) runs Hermes' unmodified `batch_runner.py`
(`--dataset_file --batch_size=1 --run_name --distribution --model --base_url
--max_turns --num_workers=1 [--reasoning_effort] [--api_key]`) in a private
temp directory. `HERMES_HOME/skills` is copied from `brain/skills` of the
evaluated commit; the benchmark's `environment.repository_fixture` (below
`--fixtures-root`) becomes the agent's `TERMINAL_CWD`. Hermes tools run with
`TERMINAL_ENV=docker`, the benchmark's pinned image, `TERMINAL_DOCKER_NETWORK=false`,
no forwarded environment or volumes and non-persistent containers; the host
environment is not inherited. `data/<run>/trajectories.jsonl` is parsed:
a missing row means the prompt failed (`exit_state=agent_error`, never success),
a non-zero batch_runner exit is an infrastructure error. Tool-call and turn
limits are enforced on the parsed trajectory. The trajectory, diff and run
record are stored as artifacts. Known gaps: batch_runner has no sampling seed
(the seed is passed only as `HERMES_EVAL_SEED`), and trajectories carry no
token usage, so these runs report `token_usage_reported=False` and token
metrics are shown as unavailable rather than zero. Live execution requires a
Hermes install (`--hermes-agent-dir`) and a reachable model endpoint; CI
exercises the adapter with a fixture that mimics the real CLI flags and row
schema.

## Judge (`judge.py`, `prompts/judge/`)

The `qwen_judge` role is used only for benchmarks with no deterministic checks
and non-empty `verification.subjective`; if such a benchmark is selected
without `--judge`, evaluation fails before running anything. Each paired run is
presented twice (A=baseline/B=candidate, then swapped) with only the task,
criteria and the two outputs. The answer must be strict JSON
`{"winner": "A|B|TIE", "confidence": 0..1, "uncertainty": 0..1, "rationale"}`.
Invalid answers or transport errors are retried up to `max_attempts` per
presentation and then yield `unreliable`; contradictory positions, confidence
below 0.5 or uncertainty above 0.5 also yield `unreliable`. Verdicts across
seeds must agree. A `baseline` or `unreliable` verdict adds a blocking reason;
a `candidate` verdict changes nothing, so the judge can never promote alone.
Qwen roles share the endpoint but have separate models (`QWEN_JUDGE_MODEL`),
statistics and logs (`QWEN_LOG_DIR/<role>.jsonl`, digests only).

## Report contents

`EvaluationReport.metrics` keeps the policy metrics and adds (schema version
`evaluation_schema_version: 2`, signed with the report):

- `suite`, `run_policy` (counts, `llm_influenced`, `runs_per_task`);
- `paired_statistics` for quality, success rate, tokens, latency and tool
  calls: per-task paired mean delta, CI95, SD, Cohen's d_z, Hedges' g_z and a
  magnitude label. No p-values are produced;
- `efficiency` per arm: tokens per task/success, input/output tokens, context
  size, tool calls per success, failed tool calls, retries, turns, API calls,
  latency per success, filesystem changes and exit states, plus
  `cheaper_failures` and efficiency deltas that are only computed when the
  candidate's success rate is not lower;
- `judge` configuration and per-task verdicts.

`RunObservation` v2 fields (all optional with defaults): `input_tokens`,
`output_tokens`, `token_usage_reported`, `context_chars`, `turns`,
`api_calls`, `exit_state`, `model_config_digest`, `limits_digest`,
`verifier_results`, `filesystem_changes`, `trajectory_digest`, `diff_digest`.
`TaskOutcome` adds `run_count`, `judge_result`, `judge_confidence`,
`judge_uncertainty`. The signed top-level payload shape is unchanged, so
reports written before version 2 still verify.

An evaluation report is not sufficient on its own to establish operational
safety: benchmark quality, evaluator key custody, filesystem isolation,
independent role identities, and the runner's environment must also be trusted.
