# Benchmarking

Benchmark JSON records are parsed and validated by `Benchmark.from_dict`.
`BenchmarkStore.validate()` checks duplicate ID/version pairs, required
metadata, non-empty source type/provenance, taxonomy labels and difficulty,
and a finite ground-truth confidence in `[0, 1]`. These provenance fields are
descriptive metadata, not a signed or independently attested source claim.
Validation also checks sanitization flags, separate storage of sealed holdout,
and that a cluster is assigned to only one split. It rejects normalized exact
duplicates and likely lexical near duplicates that do not share both a split
and cluster: prompts with at least eight words and word-set Jaccard similarity
of at least 0.85. This conservative rule is a deterministic leakage guard, not
semantic clustering. Its manifest version is a deterministic hash of the
validated records.

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
- `hermes-improve benchmarks inspect-candidate FINGERPRINT` shows a bounded
  preview of up to three redacted source sessions. `review-candidate` records
  one immutable operator accept/reject decision with reviewer and rationale;
  the source-session set is frozen at that point. Acceptance is triage only: it
  does not create or admit an executable benchmark; benchmark authoring and
  validation remain separate.
- `hermes-improve evaluate` accepts only hard benchmarks with successful replay,
  reproducibility score and ground-truth confidence of at least `0.8`, and
  deterministic or structured ground truth. It selects declared FULL tasks
  plus sealed holdout tasks, then uses a digest-pinned Docker command runner.

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
replay verification are still required. The human proposal-review record does
not replace benchmark authoring or admission controls. Semantic deduplication
and clustering, benchmark provenance attestation, automatic holdout-generation
rotation, and impact-aware suite selection are not implemented. The lexical
near-duplicate guard can miss paraphrases and can require manual cluster
assignment for similar prompts. Evaluation
records a durable spent marker keyed by the sealed benchmark manifest and
candidate evolution lineage before consuming holdout feedback; failed attempts
also consume that generation, so operators must rotate the holdout before
retrying that lineage.
