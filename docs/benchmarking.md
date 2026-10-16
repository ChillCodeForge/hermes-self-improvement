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
trees, and records redacted output as a content-addressed artifact.

## Still required

The repository includes only a schema fixture, not the 20–50 curated
real-session tasks required for meaningful readiness. Manual curation and
replay verification are still required. Semantic deduplication, human review
workflow, benchmark provenance attestation, holdout-generation rotation and
spent-state persistence, and impact-aware suite selection are not implemented.
