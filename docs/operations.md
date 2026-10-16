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
- `dream --stage light|rem` analyzes queued evidence; REM requires a configured
  loopback Qwen endpoint. `dream --stage deep --hypotheses FILE` produces
  filtered proposals and also requires Qwen.
- `benchmarks mine SESSION_ID` creates a benchmark proposal; proposals require
  review before they are admitted to a benchmark suite.
- `evaluate CANDIDATE_ID` runs paired command benchmarks and signs persisted
  evidence. It requires a digest-pinned Docker-compatible benchmark, separate
  holdout root, evaluator private key, and execution under the configured
  evaluator effective UID.
- `promote CANDIDATE_ID --release brain-vN` verifies evidence and deploys a
  skills-only Git candidate. `rollback brain-vN` restores a prior release.
  Both require a configured promoter UID, evaluator public key, manifest key,
  and repository/release/deployment paths.

The CLI does not create candidate commits or users, establish filesystem
permissions, rotate holdout generations, or connect a Hermes session/telemetry
service. Read `security.md` before configuring keys, Docker, or deployment
paths. Never run generator/evaluator/promoter roles as one shared UID for a
production rollout.

## Monitoring and deployment limits

`ReleaseMonitor` is a Python API. A trusted external collector must supply
observations; no daemon or Hermes hook currently does so. It recommends
rollback on observed regressions and changes a release to `STABLE` only after
the configured sample count passes its policy. Recommendation does not itself
perform rollback.

The Docker command runner is not a general Hermes runner. Full project
acceptance requires real benchmark tasks, a Hermes runtime adapter, independent
role identities and filesystem controls, production observation wiring, and a
rehearsed end-to-end deployment.
