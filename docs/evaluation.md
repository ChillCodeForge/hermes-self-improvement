# Evaluation

`PairedEvaluator` accepts a `BenchmarkRunner` and executes baseline/candidate
runs for matching task IDs and seeds. It verifies the runner's commit/task/seed
identity and equal environment digest, aggregates per-task measurements, and
applies `PromotionPolicy` hard gates. Persisted evaluations require an
Ed25519 private key; the signature covers the report and per-run observations.
Artifacts are stored separately by content hash.

The included `DockerCommandRunner` handles declared command benchmarks in
digest-pinned, no-network containers with read-only source input, non-root
execution, dropped capabilities, resource limits, and bounded temporary
filesystems. It does not implement a Hermes `batch_runner` or general
interactive-agent environment.

The policy blocks security/deterministic failure, critical or high regressions,
excess medium regressions, missing Full/Holdout evidence, insufficient quality
gain, uncertainty intervals containing regression, category degradation,
excess token/latency growth, and final-task failure. Skills are the only
default auto-promotable candidate class.

An evaluation report is not sufficient on its own to establish operational
safety: benchmark quality, evaluator key custody, filesystem isolation,
independent role identities, and the runner's environment must also be trusted.
