# Reuse matrix

| Capability | Existing source | Decision | Current state |
|---|---|---|---|
| Hermes runtime/session lifecycle | Hermes Agent | Keep as external runtime; use lifecycle hook and persisted-message API | Optional `on_session_finalize` entry-point plugin exists; version compatibility and live installation remain unverified |
| Hermes batch-run evidence | Hermes `batch_runner` | Import existing trajectories only; never execute agent tasks during ingestion | Bounded `trajectories.jsonl` importer queues redacted untrusted evidence; no runtime execution/evaluation adapter exists |
| Curator usage/lifecycle evidence | Hermes Curator / skill usage | Consume as read-only signals; never mutate production directly | Bounded JSON usage report and validated adapter can feed LIGHT/REM Dreaming; no live Hermes API adapter |
| Curator backup/audit/rollback | Hermes Curator | Reuse only after exact behavior and APIs are verified | Not called |
| Candidate evolution | `hermes-agent-self-evolution` | Encapsulate behind pinned `SelfEvolutionAdapter` | Version-gated interface and generic optimizer API exist; external package not integrated |
| Prompt/optimizer evolution | GEPA, optionally DSPy | Optional strategy adapters; never trusted as evaluator or promotion authority | Not integrated |
| Local model inference | OpenAI-compatible Qwen endpoint | Optional role-specific prompts; deterministic policy remains authoritative | Adapter, loopback default, optional judge model, and in-memory usage metrics exist; no live endpoint or evaluator integration tested |
| Experiment metadata | New SQLite schema | Implement locally; keep large artifacts external | Implemented foundation |
| Benchmark integrity | New validators/access controller | Deterministic; holdout separated from generator API | Implemented foundation |
| Production Git brain | Git repository | Sole production truth; separate promoter identity | Skills-only CLI promotion, signed manifests, atomic release pointer, and rollback exist; OS identities/permissions remain operator-provisioned |

No external library or integration is represented as available until its
version, API, and behavior have been inspected and tested.
