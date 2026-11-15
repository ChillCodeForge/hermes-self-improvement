# Discovery and verified integration status

Status date: 2026-11-15 (WP-0). Every fact below carries its evidence. "Verified"
means read in source or observed on disk/network at the stated revision;
"executed" means code was actually run. Anything else is listed under
"Not verified".

## Sources inspected

| Source | Revision / version | How |
|---|---|---|
| Hermes Agent checkout (`%LOCALAPPDATA%\hermes\hermes-agent`) | `d923a5eb487a17c0a68801bba9ef0f56c81040a8`, tag `v0.22.2`, clean tree | Source reading (read-only) |
| Hermes home (`%LOCALAPPDATA%\hermes`) | live state on the dev host | Structure only; no secrets, `auth.json` not read |
| `NousResearch/hermes-agent-self-evolution` | `main` @ `0a929e3aa20e15cf04dc7c28492a7d41a5139125` (2026-06-17), package `0.1.0` | `gh api` contents |
| GEPA (PyPI `gepa`) | `0.1.4` (2026-07-15), MIT | PyPI JSON + `inspect.signature` in `python:3.12-slim` container |
| DSPy (PyPI `dspy`) | `3.5.0` (2026-11-10), MIT, requires `gepa>=0.1.4,<0.2.0` | Same |

An earlier phase inspected a different checkout (`/home/dev/.hermes/hermes-agent`
@ `e06df9c2`) and loaded this package's entry point through the real
`PluginManager` with a mocked transcript loader. That result is not re-run here.

## Hermes Agent (v0.22.2 @ d923a5eb) — verified by source reading

Paths are relative to the Hermes checkout; `HERMES_HOME` is `~/.hermes` on POSIX
(`%LOCALAPPDATA%\hermes` on this Windows host, resolved by
`hermes_constants.get_hermes_home()`).

### Plugin system

- Entry-point group `hermes_agent.plugins` (`hermes_cli/plugins_discovery.py:29`,
  docstring `hermes_cli/plugins.py:1-8`). Plugin exposes `register(ctx)`;
  `ctx.register_hook(name, callback)` (`plugins.py:933`),
  `ctx.get_config(key, default)` (`plugins.py:266`). Our `pyproject.toml`
  already uses this group.
- Hook names are validated against `VALID_HOOKS` (`plugins.py:109-135`); all four
  spec hooks exist.

| Hook | Fire site | Kwargs | Return |
|---|---|---|---|
| `on_session_end` | `agent/turn_finalizer.py:796-807` (per turn, skipped when persistence disabled); also `hermes_cli/cli_shutdown.py:190`, `tui_gateway/session_lifecycle.py:389` with `interrupted=True` | `session_id, task_id, turn_id, completed, failed, interrupted, turn_exit_reason, model, platform` (sample payload `hermes_cli/hooks.py:127-131`) | ignored (observer) |
| `on_session_finalize` | `hermes_cli/lifecycle.py:69-85` via `cli_session_mixin.py:415-431` and `tui_gateway/session_lifecycle.py:399` | `session_id, platform, reason` (`"session_boundary"`) | `str` or `{"message": str}` shown out-of-band (`lifecycle.py:57-66`) |
| `on_skill_lifecycle` | `tools/skill_usage.py:470-482` | `action, skill_name, provenance, task_id, session_id, use_count, reused, reuse_after_patch`; `action` ∈ `loaded, patched, edited, created, installed, archived, stale, restored` (`skill_usage.py:508-569`) | ignored |
| `pre_tool_call` | `hermes_cli/plugins.py:1971-1991` | `tool_name, args, task_id, session_id, tool_call_id, turn_id, api_request_id, middleware_trace` | `{"action":"block","message"}` (veto) > `{"action":"approve",...}`; also `modify` args (`plugins.py:2083-2091`) |

`on_session_end` is per turn, not per session; `on_session_finalize` is the
session boundary and is what `hermes_plugin.py` uses. No hook carries the
transcript; it must be read from SessionDB.

### SessionDB and export

- `hermes_state.SessionDB(db_path=None, read_only=False)` (`hermes_state.py:460,579`);
  default `HERMES_HOME/state.db` (`hermes_state.py:177`), SQLite WAL + FTS5,
  `SCHEMA_VERSION = 31` (`hermes_state_common.py:283`). Local `state.db` exists (8.6 MB).
- Read API: `get_session(id)` (`hermes_state_sessions.py:895`),
  `search_sessions(source, limit, offset, workspace_key)` (`:1578`),
  `get_messages(...)` (`hermes_state_messages.py:1392`),
  `get_messages_as_conversation(...)` → OpenAI-format messages (`:1520`).
  Convenience loader `agent/trace_upload.py:226 load_session_messages(session_id, db_path)`
  → `(messages, meta)`.
- Export: `export_session(id, include_compacted, include_inactive)` → session row
  dict + `messages` + `timings` (`hermes_state_portability.py:292`),
  `export_session_lineage` (`:302`), `export_all(source)` (`:319`). CLI
  `hermes sessions export <output|-> [--source S] [--session-id ID]` writes JSONL,
  one session per line (`hermes_cli/console_engine.py:660-684,818`).
- Session row columns include `model, model_config, system_prompt_hash, started_at,
  ended_at, end_reason, message_count, tool_call_count, input_tokens, output_tokens,
  cache_read_tokens, cache_write_tokens, reasoning_tokens, api_call_count, cwd,
  git_branch, git_repo_root, estimated_cost_usd, tool_names`
  (`hermes_state_common.py:382-447`); message columns include `role, content,
  tool_call_id, tool_calls, tool_name, timestamp, token_count, finish_reason,
  reasoning, active, compacted` (`:449-480`).

### batch_runner.py

- CLI via `fire.Fire(main)` (`batch_runner.py:867-1020`):
  `python batch_runner.py --dataset_file=F.jsonl --batch_size=N --run_name=R
  [--model=M --base_url=URL --api_key=K --max_turns=10 --num_workers=4
  --distribution=default --resume --ephemeral_system_prompt=... --reasoning_effort=...
  --reasoning_disabled --prefill_messages_file=... --max_samples=N]`.
  Defaults: `model="anthropic/claude-sonnet-4.6"`, `base_url="https://openrouter.ai/api/v1"`.
- Input row: `{"prompt": str, "image"|"docker_image"?: str}` (`:232-240,187`); the
  image is verified/pulled when `TERMINAL_ENV=docker`.
- Output dir `data/<run_name>/` relative to CWD (`:467-470`): `batch_<n>.jsonl`,
  `checkpoint.json`, `statistics.json`, combined `trajectories.jsonl` (`:710-756`).
- `trajectories.jsonl` row (`:358-367`):
  `{prompt_index:int, conversations:[{from:"system"|"human"|"gpt"|"tool", value:str}],
  metadata:{batch_num, timestamp, model}, completed:bool, partial:bool, api_calls:int,
  toolsets_used:[str], tool_stats:{tool:{count,success,failure}}, tool_error_counts:{tool:int}}`.
  Tool calls are inside `gpt` values as `<tool_call>{json}</tool_call>`, results in
  `tool` values as `<tool_response>` blocks (`agent/agent_runtime_helpers.py:100-145,169-195`).
  Failed prompts are not written (retried on `--resume`); `discarded` tombstones and
  rows with unknown tool names are dropped from the combined file.
- Each prompt runs `AIAgent(..., skip_context_files=True, skip_memory=True,
  platform="batch")` (`:249-262`): no SOUL.md/AGENTS.md, no memory.
- **No token counts in the output rows.** `run_conversation` returns
  `input_tokens, output_tokens, cache_*_tokens, reasoning_tokens, prompt_tokens,
  completion_tokens, total_tokens` (`agent/turn_finalizer.py:33-36,687-720`) but
  batch_runner drops them. Alternative per-task path with tokens:
  `hermes -z PROMPT --usage-file PATH [-m MODEL --provider P]` writes a JSON usage
  report (tokens, cost, model, api_calls) even on failure (`hermes_cli/_parser.py:148-158`).

### Skills, write approval, discovery

- `skill_manage(action, name, content, category, file_path, file_content,
  old_string, new_string, replace_all, absorbed_into, task_id, session_id, operations)`
  (`tools/skill_manager_tool.py:768`); actions `create, edit, patch, delete,
  write_file, remove_file` (`:722-728`). Every success writes a ledger entry and
  bumps usage telemetry (`:731-765`).
- `skills.write_approval` (also `memory.write_approval`), default off
  (`tools/write_approval.py:38-52`). When on, skill writes are staged under
  `HERMES_HOME/pending/skills/<id>.json` instead of committed
  (`write_approval.py:1-10,74`; gate `skill_manager_tool.py:657-690`).
- Search roots and precedence (`agent/skill_utils.py:380-410`): trusted project
  `.hermes/skills` / `.agents/skills` (`:496`) > `HERMES_HOME/skills` >
  `skills.create_dir` > `skills.external_dirs`. **External-dir skills are
  read-only to the Curator and never curation-eligible**
  (`tools/skill_usage.py:274-288`).

### Curator

| Capability | Exists | Evidence | On-disk location |
|---|---|---|---|
| Usage tracking | yes: `use_count, view_count, patch_count, patch_generation, last_used_at, last_viewed_at, last_patched_at, created_by, created_at, state, pinned, archived_at, first_seen_at` | `tools/skill_usage.py:348-353` | `HERMES_HOME/skills/.usage.json` (+ `.lock`) |
| Lifecycle active → stale → archived, pin | yes; defaults stale 14 d, archive 30 d, interval 7 d, min idle 2 h | `agent/curator.py:30-34,209-260`; `skill_usage.py:35,550-569` | archive dir `skills/.archive/` |
| Consolidation / overlap | LLM fork (opt-in, `curator.consolidate`, default off) judges overlap on content and calls `skill_manage` | `agent/curator.py:146-148,268-470,881` | — |
| Patching | via `skill_manage` from the fork; writes **directly** into `HERMES_HOME/skills` | `skill_manager_tool.py:731-765` | local ledger shows 26 of 33 entries by actor `curator` (19 `patch`) |
| Backups | tar.gz before every real run | `agent/curator_backup.py:1-7,153`; `curator.py:938` | `skills/.curator_backups/<utc-iso>/{manifest.json,skills.tar.gz}` |
| Audit | per-mutation JSONL, any actor (`curator, agent, user`) | `tools/skill_ledger.py:1-8,38,298-324` | `skills/.curator_ledger.jsonl`; blobs `HERMES_HOME/.curator_backups/blobs/` |
| Rollback | whole-tree `curator_backup.rollback(id)` and per-entry `skill_ledger.rollback_entry(id)` | `curator_backup.py:406`; `skill_ledger.py:569` | — |
| Scheduler state | `last_run_at, last_run_duration_seconds, last_run_summary, last_report_path, paused, run_count` | `agent/curator.py:40-57` | `skills/.curator_state` (JSON) |
| Run reports | `run.json` (counts, archived, consolidated, pruned, state_transitions, tool_calls) + `REPORT.md` | `agent/curator.py:471-480,699-742` | `HERMES_HOME/logs/curator/<YYYYMMDD-HHMMSS>/` |

CLI `hermes curator {status, usage [--json], run [--dry-run --consolidate],
pause, resume, pin, unpin, list-unmanaged, adopt, prune, restore, list-archived,
archive, backup, rollback [entry_id|--id], ledger, purge}` (`hermes_cli/curator.py:621-705`).
Ledger entry: `{id, ts, actor, action, skill, evidence, before:[{path,sha256}], after:[...]}`;
backup manifest keys `id, created_at, reason, archive, archive_bytes, skill_files,
cron_jobs` (both confirmed on local files).

### MCP health

- `hermes mcp test <name>`: exit 0 connected, 1 failed, 3 unknown server
  (`hermes_cli/mcp_config.py:836-842`). In-process: `MCPServerHealthMixin.ensure_healthy`
  probes with `ping`, falling back to `list_tools` (`tools/mcp_tool_health.py:169-245`).

## Local Hermes home — verified (dev host)

- `config.yaml`: `model.provider: nous`, `model.base_url:
  https://inference-api.nousresearch.com/v1`, `model.default` empty. No custom
  provider, no local/Qwen endpoint; no `QWEN_*` or OpenAI-base-URL variables in
  `.env` or the process environment. `skills.write_approval` is not set (off).
- Desktop API server configured at `127.0.0.1:8642`; `GET /v1/models` returns 401
  (Hermes gateway, API key required). This is Hermes itself, not a model server.
- Probed `GET http://127.0.0.1:{8000,8001,8080,11434,1234,5000,30000}/v1/models`
  (3 s timeout): no listener. **No local Qwen endpoint is reachable.**
- Curator is active locally: 8 runs, last 2026-11-09; 65 usage records, 33 ledger
  entries, 1 snapshot, 5 report dirs.

## hermes-agent-self-evolution @ 0a929e3a — verified by source reading

- License: MIT per `pyproject.toml` and README; the repo has no LICENSE file
  (GitHub reports `license: null`). Not on PyPI (404). Deps `dspy>=3.0.0,
  openai>=1.0.0, pyyaml, click, rich`. Only Phase 1 (skills) is implemented.
- Python surface: `evolution.skills.evolve_skill.evolve(skill_name, iterations,
  eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests,
  dry_run)` (CLI `python -m evolution.skills.evolve_skill`);
  `evolution.skills.skill_module.{load_skill, find_skill, reassemble_skill, SkillModule}`;
  `evolution.core.constraints.ConstraintValidator(config).validate_all(text,
  artifact_type, baseline_text=None)` (size 15 kB, growth ≤ 20 %, non-empty,
  skill structure); `evolution.core.dataset_builder.{EvalExample, EvalDataset,
  GoldenDatasetLoader, SyntheticDatasetBuilder}` (`train/val/holdout.jsonl`,
  example fields `task_input, expected_behavior, difficulty, category, source`);
  `evolution.core.external_importers.build_dataset_from_external`.
- Defects that block direct use (code reading, not executed against an LLM):
  1. `dspy.GEPA(metric=..., max_steps=iterations)`: `max_steps` is not a parameter
     of `dspy.GEPA` in DSPy 3.5.0 (signature verified), so it raises and the code
     silently falls back to MIPROv2.
  2. `SkillModule` passes the skill text as an `InputField`; an optimizer mutates
     signature instructions, so `optimized_module.skill_text` stays the baseline text.
  3. `skill_fitness_metric` scores keyword overlap (`fitness.py:107-136`), which §7 forbids
     as a decision signal.
  4. `evolve()` itself splits and scores a holdout, and `HermesSessionImporter` reads
     hard-coded `~/.hermes/sessions/*.json` (ignores `HERMES_HOME`, not SessionDB).

## GEPA 0.1.4 / DSPy 3.5.0 — verified (signatures inspected in a container)

- `gepa.optimize(seed_candidate: dict[str,str], trainset, valset=None, adapter=None,
  task_lm=None, evaluator=None, reflection_lm=None, candidate_selection_strategy="pareto",
  reflection_minibatch_size=None, perfect_score=1.0, use_merge=False,
  max_metric_calls=None, stop_callbacks=None, run_dir=None, seed=0,
  raise_on_exception=True, acceptance_criterion="strict_improvement", ...) -> GEPAResult`.
- Custom evaluation via `gepa.GEPAAdapter` with `evaluate(batch, candidate,
  capture_traces=False) -> EvaluationBatch(outputs, scores, trajectories,
  objective_scores, num_metric_calls)` and `make_reflective_dataset(candidate,
  eval_batch, components_to_update)`. `reflection_lm` accepts a model string or a
  callable `str -> str`. `GEPAResult` exposes `candidates, parents,
  val_aggregate_scores, val_subscores, total_metric_calls, ...`.
- `dspy.GEPA(metric, *, auto|max_full_evals|max_metric_calls, reflection_lm, ...)`.

## Not verified

- No Hermes code was executed in this WP: no live session finalize, no SessionDB
  read through the API, no `batch_runner` run, no `hermes curator` command.
- Hook dispatch into this package's plugin at v0.22.2 (only the earlier e06df9c2 run).
- Any Qwen/OpenAI-compatible model server (none configured or reachable).
- self-evolution was not installed or run; GEPA was not run end-to-end.

## Adapter decisions

| Component | Wraps | Decision |
|---|---|---|
| Ingestion hook (`hermes_plugin.py`) | `on_session_finalize` + `trace_upload.load_session_messages` / `SessionDB(read_only=True)` | Keep. `on_session_end` (per turn) and `on_skill_lifecycle` are optional observer signals only; `pre_tool_call` is not used to change runtime behavior. |
| `BatchRunnerAdapter` (WP-C) | `python batch_runner.py` subprocess with `--dataset_file --run_name --model --base_url --max_turns --num_workers=1`, reads `data/<run>/trajectories.jsonl` | Tokens are absent from rows: capture usage per task with `hermes -z ... --usage-file`, or record tokens as unavailable. Never mark a missing row as success. |
| `CuratorAdapter` (WP-E) | read-only reads of `.usage.json`, `.curator_state`, `.curator_ledger.jsonl`, `logs/curator/*/run.json`, or `hermes curator usage --json` | Curator patches skills directly, so each new `actor: curator` ledger entry becomes a candidate (before/after blobs give the diff). Production brain is mounted via `skills.external_dirs`, which the Curator treats as read-only. |
| Skill write path | `skills.write_approval: true` | Recommended operator setting; staged writes in `pending/skills/` become candidate input, never production. |
| `SelfEvolutionAdapter` (WP-D) | `hermes-agent-self-evolution` pinned to `0a929e3aa20e15cf04dc7c28492a7d41a5139125` | Reuse only `ConstraintValidator`, `load_skill/reassemble_skill`, and `EvalExample` format. Do not call `evolve()` (defects above, holdout use, keyword metric). |
| `GepaStrategy` (WP-D) | `gepa==0.1.4` `gepa.optimize` with our own `GEPAAdapter` | Seed `{"skill": <SKILL.md body>}`; trainset = TRAIN, valset = VALIDATION only; scores come from our evaluator; `reflection_lm` = Qwen optimizer role callable. Optional extra; DSPy `3.5.0` not required. |
| `MCPHealthVerifier` (WP-C) | `hermes mcp test <name>` exit code | 0 = healthy, 1 = failed, 3 = unknown server. |
| Qwen client | `QWEN_*` env | No endpoint exists on the dev host; live tests stay opt-in, CI uses a local stub. |
