# Operations

## Local development

```bash
python -m pip install -e '.[test]'
pytest
hermes-improve status
hermes-improve benchmarks validate
hermes-improve benchmarks list --role generator
hermes-improve experiments list
```

`HERMES_IMPROVE_HOME` selects the local database/artifact state root;
`HERMES_IMPROVE_BENCHMARKS` selects the non-holdout benchmark directory. The
repository's example benchmark is a schema fixture, not a curated real-task
evaluation.

## Optional Hermes session hook

Install this package into the same Python environment as Hermes Agent and
explicitly enable the entry-point plugin. For example, in Hermes `config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-self-improvement
  entries:
    hermes-self-improvement:
      settings:
        auto_ingest: true
        state_dir: /absolute/path/to/hermes-improvement-state
```

The plugin reads finalized transcript messages through Hermes'
`agent.trace_upload.load_session_messages` API, redacts them, and queues them
under `state_dir`. Keep this path private and point
`HERMES_IMPROVE_HOME` at the same directory when running this package's CLI.
In multi-profile Hermes installations, configure a distinct state directory
per profile. The hook is disabled unless `auto_ingest` is explicitly `true`.
This integration was checked against the API present at the Hermes revision
recorded in `discovery.md`. The package was installed into the local Hermes
virtual environment, and an entry-point/PluginManager hook smoke test queued a
mocked transcript. It has not yet been verified with a real finalized Hermes
session or SessionDB transcript.

## Available operator workflows

- `ingest EXPORT.json` redacts and queues a session export.
- `ingest-batch RUN/trajectories.jsonl` imports Hermes `batch_runner` output;
  discarded tombstones are skipped, and all imported conversations remain
  redacted, untrusted evidence. Bounded tool counters can feed recurring
  failure signals. This imports previous run data only—it does not run Hermes
  or produce candidate evaluation results.
- `dream --stage light|rem` analyzes queued evidence; REM requires a configured
  loopback Qwen endpoint. `dream --stage deep --hypotheses FILE` produces
  filtered proposals and also requires Qwen.
- `benchmarks mine SESSION_ID` creates a benchmark proposal; proposals require
  review before they are admitted to a benchmark suite.
- `candidate create --repository REPO --experiment-id ID --candidate-id ID
  --type SKILL_CREATE|SKILL_PATCH --target brain/skills/NAME --content FILE
  --hypothesis TEXT` creates an immutable candidate commit in a separate
  `candidate/<experiment-id>` branch/worktree. Candidate content must be supplied
  explicitly; the target must be a canonical directory path under
  `brain/skills/` (not a `SKILL.md` file path). The command validates the
  target, content size, and credential-like strings, and does not write
  production. Only `SKILL.md` candidates are currently supported.
- `evaluate CANDIDATE_ID` runs paired command benchmarks and signs persisted
  evidence. It requires a digest-pinned Docker-compatible benchmark, separate
  holdout root, evaluator private key, and execution under the configured
  evaluator effective UID. It consumes the current sealed-manifest generation
  once per candidate evolution lineage before starting; even a failed attempt
  requires a rotated holdout generation before retrying that lineage.
- `promote CANDIDATE_ID --release brain-vN` verifies evidence and deploys a
  skills-only Git candidate. `rollback brain-vN` restores a prior release.
  Both require a configured promoter UID, evaluator public key, manifest key,
  and repository/release/deployment paths.

The CLI does not generate skill text, create users, establish filesystem
permissions, rotate holdout generations, or connect a Hermes session/telemetry
service. Read `security.md` before configuring keys, Docker, or deployment
paths. Never run generator/evaluator/promoter roles as one shared UID for a
production rollout.

## Monitoring and deployment limits

`ReleaseMonitor` accepts observations through both its Python API and a
bounded JSONL import for trusted external collectors:

```bash
hermes-improve monitor ingest observations.jsonl
```

Example record:

```json
{"observation_id":"gateway-event-123","release_id":"brain-v7","task_id":"task-42","success":true,"tool_errors":0,"tokens":820,"tool_calls":3,"latency_ms":1420.5,"user_corrections":0,"retries":0,"critical_regression":false}
```

Each row requires an `observation_id`, `release_id`, `task_id`, `success`,
`tool_errors`, `tokens`, `tool_calls`, and `latency_ms`; corrections, retries,
and critical-regression status are optional. IDs are idempotent: replaying the
same ID and same content does not increase the sample count, while reusing an
ID for different content is rejected. The import validates the entire file
before recording any row and is limited to 10 MiB / 10,000 records. No
production collector or Hermes hook is connected automatically.

The monitor recommends rollback on observed regressions and changes a release
to `STABLE` only after the configured sample count passes its policy.
Recommendation does not itself perform rollback.

The Docker command runner is not a general Hermes runner. Full project
acceptance requires real benchmark tasks, a Hermes runtime adapter, independent
role identities and filesystem controls, production observation wiring, and a
rehearsed end-to-end deployment.
