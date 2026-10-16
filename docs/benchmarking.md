# Benchmarking

Benchmark JSON records are parsed and validated by `Benchmark.from_dict`.
`BenchmarkStore.validate()` checks duplicate ID/version pairs, required
metadata, sanitization flags, separate storage of sealed holdout, and that a
cluster is assigned to only one split. Its manifest version is a deterministic
hash of the validated records.

Generator-facing stores may contain only public benchmark roots. They expose
TRAIN/VALIDATION while evaluator-only suites remain hidden. Evaluator holdout
access uses a process-local capability and must be backed by OS filesystem
permissions; the capability alone is not a security boundary.

## Available benchmark tools

- `hermes-improve benchmarks validate` validates the public set.
- `hermes-improve benchmarks list --role ROLE` lists visible records.
- `hermes-improve benchmarks mine SESSION_ID` creates a deduplicated proposal
  from a redacted session. The miner scores reproducibility and ground-truth
  source; it does not establish truth or automatically admit the proposal.
- `hermes-improve evaluate` selects declared FULL tasks plus sealed holdout
  tasks, then uses a digest-pinned Docker command runner.

The runner requires `environment.container_image` pinned by SHA-256 digest,
`network_policy: disabled`, an argv-array command, and an expected exit code.
It exports the selected Git commit, rejects symlinks and oversized source
trees, and records redacted output plus per-check results as a
content-addressed artifact.

`verification.deterministic` supports up to 64 declarative checks:

```json
[
  {"type": "exit_code"},
  {"type": "json_path_equals", "path": ["status"], "expected": "ok"},
  {"type": "json_required_keys", "expected": ["status", "result"]},
  {"type": "output_contains", "expected": "completed"}
]
```

Supported types are `exit_code`, `output_exact`, `output_contains`,
`output_not_contains`, `json_output_equals`, `json_required_keys`,
`json_path_exists`, and `json_path_equals`. The exit code check defaults to
`ground_truth.expected_exit_code` and is added automatically when omitted. JSON
equality is type-sensitive (for example, `true` does not equal `1`).
Unsupported or malformed checks fail closed rather than passing on a
successful process exit. Quality is the percentage of deterministic checks
passed; overall task success still requires every check.

## Still required

The repository includes only a schema fixture, not the 20–50 curated
real-session tasks required for meaningful readiness. Manual curation and
replay verification are still required. Semantic deduplication, human review
workflow, benchmark provenance attestation, automatic holdout-generation
rotation, and impact-aware suite selection are not implemented. Evaluation
records a durable spent marker keyed by the sealed benchmark manifest and
candidate evolution lineage before consuming holdout feedback; failed attempts
also consume that generation, so operators must rotate the holdout before
retrying that lineage.
