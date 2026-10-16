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

Set `HERMES_IMPROVE_HOME` to change the SQLite database/artifact state root and
`HERMES_IMPROVE_BENCHMARKS` to choose the benchmark directory.

## Current limitations

No session ingestion, Dreaming scheduler, Qwen endpoint, production brain
checkout, evaluator sandbox, Git candidate worktree, promotion, atomic
deployment, or rollback command is available yet. Do not use this version to
promote or deploy changes.
