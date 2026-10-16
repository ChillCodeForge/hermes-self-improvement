# Benchmarking foundation

Benchmark JSON records are validated by `Benchmark.from_dict`. Current checks
cover schema version, required objects, task title/prompt, enumerated split and
eligibility, reproducibility range, sanitized/no-secrets flags, and cluster
identity. `BenchmarkStore.validate()` rejects duplicate ID/version pairs and
any cluster assigned to more than one split.

`BenchmarkStore` restricts generator reads to TRAIN/VALIDATION and only permits
SEALED_HOLDOUT access using an evaluator capability. This control is
process-local; OS-level holdout isolation remains required before running
untrusted optimizers. The sample under `benchmarks/train/` is a schema fixture,
not a curated evaluation corpus.

Not implemented yet: benchmark mining, redaction, exact/near deduplication,
semantic clustering, holdout generation rotation/spent persistence, impact-aware
suite selection, benchmark provenance verification, and the 20–50 real-task
starter set.
