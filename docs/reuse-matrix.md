# Reuse matrix

| Capability | Existing source | Decision | Current state |
|---|---|---|---|
| Hermes runtime/session lifecycle | Hermes Agent | Keep as external runtime; integrate via supported hooks after API/commit discovery | Not wired |
| Curator usage/lifecycle evidence | Hermes Curator / skill usage | Consume as read-only signals; never mutate production directly | Generic adapter contract present |
| Curator backup/audit/rollback | Hermes Curator | Reuse only after exact behavior and APIs are verified | Not called |
| Candidate evolution | `hermes-agent-self-evolution` | Encapsulate behind pinned `SelfEvolutionAdapter` | Not integrated |
| Prompt/optimizer evolution | GEPA, optionally DSPy | Optional strategy adapters; never trusted as evaluator or promotion authority | Not integrated |
| Local model inference | OpenAI-compatible Qwen endpoint | Optional roles with separate prompts/config/logs | Not integrated |
| Experiment metadata | New SQLite schema | Implement locally; keep large artifacts external | Implemented foundation |
| Benchmark integrity | New validators/access controller | Deterministic; holdout separated from generator API | Implemented foundation |
| Production Git brain | Git repository | Sole production truth; separate promoter identity | Not implemented |

No external library or integration is represented as available until its
version, API, and behavior have been inspected and tested.
