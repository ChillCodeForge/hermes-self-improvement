# Operations

## Local development

```bash
python -m pip install -e '.[test]'
pytest                          # POSIX only; on Windows use scripts/docker-test.sh
hermes-improve status
hermes-improve benchmarks validate
hermes-improve benchmarks list --role generator
hermes-improve experiments list
```

`status` reads the ledger read-only and reports the active brain version (from the
deployment link given by `--deployment-link` or the provisioned layout, else the release
ledger, with any disagreement reported as `mismatch`), the active release and its status,
and counts of experiments by decision, candidates, evaluation runs, releases by status,
rollbacks, spent holdouts, queued sessions and public benchmarks.

`HERMES_IMPROVE_HOME` selects the local state root (`experiments.sqlite3`,
`sessions.sqlite3`, `monitoring.sqlite3`, `observability.sqlite3`,
`hook_signals.sqlite3`, `artifacts/`); `HERMES_IMPROVE_BENCHMARKS` selects the
non-holdout benchmark directory. The repository's example benchmark is a
schema fixture, not a curated real-task evaluation.

## Daily operations

A typical operator day uses only these commands. Every command prints JSON
unless noted.

| When | Command | Effect |
|---|---|---|
| Morning check | `hermes-improve report` (or `--json`) | Six-section report (Hermes, Improvement, Benchmarks, Dreaming, Qwen, Production). Read-only; never creates state. |
| Curator review | `hermes-improve curator status [--hermes-home PATH]` | Reads the live Hermes Curator state read-only, lists signals, records each signal in the decision history. |
| Evidence intake | `hermes-improve ingest EXPORT.json` / `ingest-batch RUN/trajectories.jsonl` | Queues redacted, untrusted session evidence. |
| Outcome evidence | `hermes-improve sessions label SESSION_ID --outcome correct\|incorrect\|partial --source verifier\|user\|monitor [--note TEXT]` / `sessions import-labels LABELS.jsonl` | Attaches the checked result of a queued session (latest label wins). Without labels Dreaming cannot tell a completed-but-wrong session from a correct one; see `dreaming.md`. |
| Analysis | `hermes-improve dream --stage light [--curator-hermes-home PATH]` | LIGHT analysis; REM/DEEP need a loopback Qwen endpoint. Run summary and Qwen usage go to `observability.sqlite3`. |
| Curator change wanted | `hermes-improve curator propose ...` / `curator convert-pending ID ...` | Turns a Curator mutation into an immutable candidate; never writes production. |
| Candidate work | `candidate create`, `evaluate`, `promote` | Unchanged gated path (see below). |
| Post-promotion | `monitor collect RELEASE [--apply ...]`, `monitor ingest FILE`, `monitor replay RELEASE ...`, `monitor evaluate RELEASE [--apply ...]` | Collects evidence (ingested Hermes sessions, collector files, shadow replay) and stabilizes or rolls back a provisional release. |
| Training data | `decisions export [--labeled-only] [--output FILE]`, `decisions backfill ID ...` | Laya decision dataset (see `laya-future.md`). |

## Hermes plugin (opt-in observers)

Install this package into the same Python environment as Hermes Agent and
explicitly enable the entry-point plugin in Hermes `config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-self-improvement
  entries:
    hermes-self-improvement:
      settings:
        auto_ingest: true               # on_session_finalize transcript queueing
        observe_skill_lifecycle: true   # on_skill_lifecycle facts
        observe_tool_calls: true        # pre_tool_call name counter
        state_dir: /absolute/path/to/hermes-improvement-state
```

Every setting defaults to off and must be exactly `true`. Point
`HERMES_IMPROVE_HOME` at the same directory to see the collected data in
`curator status` (`hook_signals`). In multi-profile installations configure a
distinct `state_dir` per profile; the directory must not be accessible by group
or other users.

Hook contracts were verified in the local Hermes source (v0.22.2) at commit
`d923a5eb487a17c0a68801bba9ef0f56c81040a8`:

- `on_session_finalize`: reads finalized messages through
  `agent.trace_upload.load_session_messages`, redacts them, and queues them.
- `on_skill_lifecycle`: listed in `VALID_HOOKS` (`hermes_cli/plugins.py:135`)
  and fired by `tools/skill_usage.py:477` (`_emit_skill_lifecycle`) with
  `action, skill_name, provenance, task_id, session_id, use_count, reused,
  reuse_after_patch`. Actions emitted: `loaded`, `patched`, `edited`,
  `created`, `installed`, `archived`, `stale`, `restored`. The observer stores
  these facts in `hook_signals.sqlite3`.
- `pre_tool_call`: listed in `VALID_HOOKS` (`hermes_cli/plugins.py:110`) and
  invoked at `hermes_cli/plugins.py:1988` with `tool_name, args, task_id,
  session_id, tool_call_id, turn_id, api_request_id, middleware_trace`. A
  `{"action": "block"}` return would veto the tool, so this observer always
  returns `None`, stores only a per-day count per tool name (never the
  arguments), uses a 250 ms SQLite timeout, and swallows its own errors. It is
  registered only when enabled, so a disabled observer adds no work.

The hooks were exercised with the exact keyword arguments above in tests,
not inside a live Hermes runtime.

## Curator integration

`curator status` parses these real Hermes Curator files under the Hermes home
(`--hermes-home`, else `HERMES_HOME`, else `~/.hermes`). Paths and formats are
from the Hermes source at the commit above:

| File | Source | Used for |
|---|---|---|
| `skills/.usage.json` | `tools/skill_usage.py:51`, record shape `:348` | usage counters, `active/stale/archived` state, `pinned`, `created_by` |
| `skills/.curator_state` | `agent/curator.py:40` | last run, run count, paused |
| `logs/curator/<YYYYMMDD-HHMMSS>/run.json` | `agent/curator.py:473`, `:737` | consolidations (Curator's overlap outcome), prunings, archives, state transitions |
| `skills/.curator_ledger.jsonl` | `tools/skill_ledger.py:84`, `:298` | mutation audit counts by action/actor, ledger `rollback` entries (`:628`) |
| `skills/.curator_backups/<id>/manifest.json` | `agent/curator_backup.py:67`, `:133` | snapshot backups available for `hermes curator rollback` |
| `pending/skills/<id>.json` | `tools/write_approval.py:65`, `:74` | skill writes staged by `skills.write_approval` |

Hermes has no separate overlap detector: overlap is judged by the LLM
consolidation pass and recorded as `consolidated` entries in `run.json`.

Files are opened without following symbolic links, size-bounded, and
validated. Malformed records are listed under `rejected_records` rather than
dropped or trusted. The connector never writes the Hermes home.

Signals produced: stale skills, archived skills that had recorded use,
repeated in-place patching (patch churn), consolidations (as SKILL_MERGE
review signals), ledger rollbacks, and pending staged writes.

### Keeping the Curator off the production path

Hermes' Curator (checked at Hermes v0.22.2) writes skills directly through
`skill_manage` unless `skills.write_approval: true` is set. Two Hermes
settings keep it off the production brain:

- Mount the deployed brain checkout (`--deployment-link` target) through
  `skills.external_dirs`. Hermes treats external skill directories as
  read-only to the Curator (`tools/skill_usage.py:278`, `is_curation_eligible`),
  so lifecycle transitions and consolidation cannot touch them.
- Set `skills.write_approval: true` so remaining skill writes are staged
  instead of applied.

With the gate on, every skill write
(foreground or background review, including the Curator fork) is staged as
`pending/skills/<id>.json` (`tools/skill_manager_tool.py:657`,
`tools/write_approval.py:201`). Convert staged writes instead of approving
them in place:

```bash
hermes-improve curator convert-pending 1a2b3c4d --repository REPO \
  --experiment-id exp-curator-1 --candidate-id cand-curator-1
hermes-improve curator propose --kind patch --skill NAME --rationale TEXT \
  --old-string-file OLD --new-string-file NEW --repository REPO \
  --experiment-id ID --candidate-id ID
```

Conversion rules (`CuratorAdapter`):

- `create` → `SKILL_CREATE`; `edit` or full-content `patch` → `SKILL_PATCH`.
- Targeted `patch` (old/new string) is applied to the production `SKILL.md`
  read from `main`; the match must be unique unless `replace_all` is set.
- `consolidate` (and a staged `delete` with `absorbed_into`) → `SKILL_MERGE`
  that writes the umbrella `SKILL.md` and deletes every tracked file of the
  absorbed skill. Without new umbrella content (the staged `delete`
  payload carries none unless it sets `content`) or with content equal to
  production, only the removal remains and the candidate is the
  `archive` form below.
- `archive` (and a staged `delete` without `absorbed_into`) → `SKILL_PATCH`
  that deletes every tracked file of `brain/skills/<name>`. The skill must
  exist on `main` with an exact-case `SKILL.md` and contain only regular,
  candidate-legal files; otherwise conversion is refused and nothing is
  recorded.
- Every candidate that removes a skill is created with risk `HIGH` and needs
  a signed operator approval before promotion, even though skill types are
  auto-gated: `hermes-improve approve EXP --operator NAME --rationale TEXT
  --repository REPO` (the repository is needed to read the removal from the
  candidate diff). See `docs/promotion.md`.
- Supporting-file writes and skill names that are not a safe `brain/skills/`
  directory component are refused with an explicit error.

The converted candidate still needs `evaluate` and `promote`. The adapter
never writes production or the Hermes pending record.

## Candidate, evaluation and promotion workflows

- `benchmarks mine SESSION_ID` creates a benchmark proposal; proposals require
  review (`benchmarks inspect-candidate`, `benchmarks review-candidate`).
  `benchmarks add FILE --holdout-root HOLDOUT_DIR` admits an operator-authored
  public record after schema and complete-corpus validation.
- `candidate create --repository REPO --experiment-id ID --candidate-id ID
  --content FILE` (with `--type/--target/--hypothesis` or
  `--proposal-file/--proposal-index`) creates an immutable `SKILL.md`
  candidate commit on `candidate/<experiment-id>`. Candidates created from a
  DEEP proposal are counted for the Dreaming conversion rate.
- `evaluate CANDIDATE_ID` runs paired command benchmarks and signs evidence;
  it requires a digest-pinned Docker benchmark, a separate holdout root, the
  evaluator private key, and the evaluator UID.
- `promote CANDIDATE_ID --release brain-vN` deploys a skills-only candidate as
  a `PROVISIONAL` release. `rollback brain-vN` restores a prior release. Both
  require the promoter UID, evaluator public key, manifest key, and
  repository/release/deployment paths.

## Post-promotion monitoring

Releases move `CANDIDATE → PROVISIONAL → STABLE`, or `PROVISIONAL/STABLE →
ROLLED_BACK`. `ExperimentStore.set_release_status` enforces these transitions.

Evidence sources:

1. `monitor ingest observations.jsonl` — trusted external collectors:

   ```json
   {"observation_id":"gateway-event-123","release_id":"brain-v7","task_id":"task-42","success":true,"tool_errors":0,"tokens":820,"tool_calls":3,"latency_ms":1420.5,"user_corrections":0,"retries":0,"critical_regression":false,"skills":["demo"],"source":"production"}
   ```

   Required: `observation_id, release_id, task_id, success, tool_errors,
   tokens, tool_calls, latency_ms`; `tokens` may be `null` when the collector
   has no token count (stored as unknown, never as 0). Optional:
   `user_corrections, retries, critical_regression, skills, source`
   (`production`, `hermes_session`, `shadow_replay`, `historical_replay`). IDs
   are idempotent; the whole file is validated before anything is recorded
   (10 MiB / 10,000 rows).
2. `monitor replay RELEASE --runner-command '["/abs/path/replay"]' [--limit N]
   [--source sessions|shadow-benchmarks]` — read-only shadow replay. With the
   default `--source sessions`, prior sessions in the queue whose content names
   a skill changed by the release are replayed; with `--source
   shadow-benchmarks`, benchmarks of the `SHADOW` split (spec §11) under
   `HERMES_IMPROVE_BENCHMARKS` whose task or taxonomy names a changed skill (all
   of them if the release changed no skill) are replayed instead. Other splits
   are never replayed here. Replays run through the runner against the
   release's deployment checkout, which must be a real directory without write
   bits (the promoter installs it as `0555`). The runner gets one JSON object
   on stdin (`session_id`, `prompt`, `release_checkout`,
   `evidence_trust: "untrusted"`, plus `benchmark_id` as `ID@vVERSION` for
   benchmark replays), a reduced environment (`PATH`, `HOME`,
   `HERMES_BRAIN_CHECKOUT`), and must print `success, tool_errors, tokens,
   tool_calls, latency_ms` (optional `retries, critical_regression`). The
   runner's own `success` is recorded; the benchmark's verifiers are not run by
   this command. Runner failures are reported, not recorded as evidence. No
   Hermes replay wrapper ships with this package; the operator supplies one.
3. `monitor collect RELEASE [--limit N] [--switch-margin SECONDS] [--apply ...]`
   — real Hermes sessions already in `sessions.sqlite3` (from the
   `on_session_finalize` plugin hook or `hermes-improve ingest` of a session
   export) become observations with source `hermes_session`
   (`hermes_improve/session_feed.py`). The queue is only read.

   *Attribution.* Release activation times come from the append-only
   `release_events` (`PROMOTED` and `REACTIVATED`; a `STABLE` status change
   does not switch the brain). A session's span is the minimum and maximum of
   its message `timestamp` values (Hermes persists one per message), widened by
   `started_at` / `ended_at` when the export carries them. The session counts
   for the release active over the whole span. It is excluded and counted by
   reason, never guessed, when: any message lacks a usable timestamp (epoch
   seconds, or ISO-8601 with a timezone) (`unknown_time`), `ended_at` is
   present but null (`session_not_ended`), the span crosses an activation
   (`spans_release_switch`) or lies within `--switch-margin` seconds of one
   (default 60, `within_switch_margin`), it predates every recorded activation
   (`before_first_recorded_release`), it lies in the future
   (`implausible_time`), it has no transcript (`no_transcript`), or it is a
   batch-runner trajectory rather than a production session
   (`not_production_session`). Sessions active under another release are
   counted as `other_release`. A release without any activation event (for
   example one written by the legacy `record_release`) cannot be collected.

   *Per-session derivation* (deterministic, `heuristics_version
   session-feed-1`):

   | Field | Rule |
   |---|---|
   | `tool_calls` | entries in assistant `tool_calls` lists |
   | `tool_errors` | tool results starting with `Error`, or a JSON object with non-zero `exit_code`, `success: false` or a non-empty `error` (the persisted part of Hermes' `_detect_tool_failure`); cancelled calls are not errors |
   | `retries` | tool calls repeating the exact name and arguments of an earlier call whose result was an error |
   | `tokens` | `input_tokens + output_tokens` of the session row (or a `usage` object) when both are integers and the sum is positive; otherwise unknown. Hermes defaults both columns to 0, and the plugin export carries no token counts, so plugin sessions are token-unknown |
   | `latency_ms` | agent-active time: per user turn, from the user message to the last assistant/tool message before the next user message (human think time excluded) |
   | `success` | `final_status.failed`/`interrupted` true is a failure, `completed` true a success; otherwise the session must end with an assistant answer with text, no pending tool calls, a finish reason other than `length`, `content_filter`, `error`, `tool_calls`, and no user interruption |
   | `skills` | brain skill names from `skill_view` calls (last path segment, validated); used for affected-skill matching |
   | `critical_regression` | never derived from sessions (always false) |

   `user_corrections` is the sum of four heuristics, recorded per name in the
   observation's `derivation` (with `heuristics_version`, the success rule,
   whether tokens were reported, and `evidence_trust: "untrusted"`):
   `rewind` (the session row's `rewind_count`, Hermes' explicit transcript
   undo), `explicit_undo_retry` (a user message answering an assistant turn
   that starts with `/undo`, `/retry`, `undo`, `revert`, `retry`, `try again`,
   `roll back`, `rückgängig`, `nochmal` ...), `negation_after_assistant` (a
   user message answering an assistant turn that starts with `no`, `nope`,
   `wrong`, `that's not right/what`, `it doesn't work`, `you forgot/missed/broke`,
   `stop`, `nein`, `falsch`, `das ist falsch/nicht richtig` ...; `no problem`
   and similar are excluded), and `interrupted_tool_call` (tool results Hermes
   wrote as `[Tool execution cancelled`, or an `interrupted` final status).
   Only the first 200 characters of a message are matched and only at its
   start; the opening task prompt never counts. These are heuristics: an
   imperative such as "Revert the formatting change" is counted as a
   correction, and corrections phrased differently are missed.

   Session content is untrusted (spec §28): it is parsed, counted and
   pattern-matched only. The release, observation ID
   (`hermes_session:` + hash of release and session ID), source label and
   status never come from session content; fields such as `release_id`,
   `critical_regression` or `success` inside a session are ignored. Forged
   timestamps or texts can only shift counts. Re-running `collect` is
   idempotent (`duplicates`); a session already recorded with different
   derived values (a future heuristics version) is reported under `conflicts`
   and keeps its first record. The output contains the feed report and the
   assessment; nothing changes status without `--apply`, which takes the same
   promoter options and identity checks as `monitor evaluate --apply`.

Observations are matched to the release's affected area: the skills under
`brain/skills/` in the signed manifest's `changed_paths`. An observation that
names skills counts only if one of them changed; an observation without skill
attribution counts for the whole release.

Metrics: success rate, tool error rate, tokens/success, tool calls/success,
latency/success, retry rate, user-correction rate, and critical regression
signals. Tokens/success uses only observations that reported tokens
(`tokens_reported_samples`); if none did, the metric is `null` and listed
under `unavailable_metrics` instead of being compared. Each metric is compared
with the manifest's `baseline_metrics` only when that baseline exists; missing
baselines are listed under `unavailable_baselines` instead of being assumed. Any critical signal means
immediate rollback; a success-rate or tool-error regression is `high`
severity, any other regression is `medium`.

```bash
hermes-improve monitor evaluate brain-v7          # assessment only, no change
HERMES_PROMOTER_UID=... HERMES_IMPROVE_MANIFEST_KEY=... \
hermes-improve monitor evaluate brain-v7 --apply --repository REPO \
  --release-root RELEASES --deployment-link CURRENT --evaluator-public-key KEY
```

`--apply` runs only as the promoter UID. A healthy release becomes `STABLE`.
On regression it finds the release whose commit was the regressing release's
baseline, checks that the regressing release is the active `main`, and calls
`GitPromoter.rollback` (atomic link switch, release tags kept). Every applied
or blocked action is audited in `monitoring_actions` (release, action, target,
previous commit, reasons, metrics, actor UID). A regression without an
earlier restorable release is recorded as `ROLLBACK_BLOCKED` and needs an
operator. `monitor ingest` and `monitor replay` still mark a release `STABLE`
automatically once policy passes, but never roll back by themselves;
`monitor collect` changes nothing without `--apply`.

## Observability report

`hermes-improve report [--json]` reads, never writes:

- **Hermes**: headline metrics from `source=production` observations only
  (success rate, tokens/success, tool calls/success, failed tool calls, retry
  rate; `no data` without real production traffic), queued-session counts, and
  `observations_by_source`: the same metrics plus `critical_observations` for
  every recorded source (`production`, `hermes_session`, `shadow_replay`,
  `historical_replay`), so lab runs fed only by shadow replay still show their
  evidence.
- **Improvement**: candidates generated/rejected/promoted, experiments by
  decision, generations (longest parent chain), mean `quality_delta_mean`,
  failed evaluations with regression reasons.
- **Benchmarks**: public benchmarks only (count, categories, difficulty,
  ground-truth types, reproducibility, splits) and spent holdout generations.
- **Dreaming**: runs, sessions processed, clusters, hypotheses, and
  proposal-to-candidate conversion rate from the CLI's observability log.
- **Qwen**: requests, tokens, average latency per request, and usage by role,
  persisted by every Qwen-backed command (`dream --stage rem|deep`,
  `candidate generate`, `evolve`, `evaluate --judge`), including failed runs.
  `judge_uncertainty` aggregates the per-task judge uncertainty stored in the
  experiment store's `task_results` (judged tasks, mean, max, and the number of
  `unreliable` verdicts); `no data` until a judged evaluation exists.
  `gpu_runtime` is described below.
- **Production**: active brain version and status, releases by status,
  critical post-promotion observations (all sources, with
  `critical_observations_by_source` matching `hermes.observations_by_source`),
  releases with rollback decisions, rollback count.

Missing sources print `no data`.

### GPU runtime

There is no built-in GPU source. The operator can configure one through the
environment of the process that runs the Qwen-backed command:

```text
HERMES_GPU_METRICS_URL=http://127.0.0.1:8080/metrics
HERMES_GPU_RUNTIME_METRICS=llamacpp:prompt_seconds_total,llamacpp:tokens_predicted_seconds_total
HERMES_GPU_METRICS_ALLOW_REMOTE=true   # only if the endpoint is not loopback
```

`HERMES_GPU_METRICS_URL` is the model server's Prometheus text endpoint
(llama.cpp `--metrics`, vLLM `/metrics`); `HERMES_GPU_RUNTIME_METRICS` lists
cumulative counters measured in seconds that the server actually exports (the
llama.cpp names above are an example; for vLLM pick the matching `_sum` series
of its request-time histograms). Both must be set together. The command scrapes
the endpoint before and after its Qwen work and stores the difference of the
summed counters per command. The value therefore includes any other traffic
the server handled in that window. A failed scrape, a missing counter, or a
counter that went backwards (server restart) is stored as an unmeasured sample
with its reason and never changes the command's result. The report shows
`gpu_runtime` as total seconds, measured/unmeasured commands, and seconds per
command; with no samples it shows `unavailable` and `gpu_runtime_reason`.

## Limits

The CLI does not generate skill text, create users, establish filesystem
permissions, rotate holdout generations, or run Hermes. The Docker command
runner is not a Hermes runner and no production collector is wired. Never run
generator/evaluator/promoter roles as one shared UID for a production rollout.
Read `security.md` before configuring keys, Docker, or deployment paths.
