# Reuse matrix

Evidence for every row is in `docs/discovery.md` (Hermes `v0.22.2` @ `d923a5eb`,
self-evolution @ `0a929e3a`, GEPA `0.1.4`, DSPy `3.5.0`).

| Capability | Existing source (verified) | Decision | Current state |
|---|---|---|---|
| Session lifecycle | Hermes hooks `on_session_finalize` (session boundary), `on_session_end` (per turn), entry-point group `hermes_agent.plugins` | Use `on_session_finalize` for ingestion; other hooks are observer-only | `on_session_finalize` plugin exists; not re-executed against v0.22.2 |
| Transcript / export | `SessionDB` (`state.db`, schema 31), `load_session_messages`, `export_session`, `hermes sessions export` | Read-only reads; treat content as untrusted evidence | Plugin uses the loader; SessionDB not exercised end-to-end |
| Skill lifecycle signals | `on_skill_lifecycle` hook, `skills/.usage.json` | Read-only signal input for Dreaming | Not wired (JSON report import only) |
| Batch execution | `batch_runner.py` (fire CLI, `data/<run>/trajectories.jsonl`, ShareGPT `from/value`) | Wrap as evaluation runner subprocess; tokens via `hermes -z --usage-file` because batch rows lack them | Importer for existing `trajectories.jsonl` exists; runner adapter not built |
| Curator usage/lifecycle | `.usage.json`, `.curator_state`, `logs/curator/*/run.json`, `hermes curator usage --json` | Consume read-only; never call `run`, `archive`, `prune`, `rollback` | Caller-supplied JSON report adapter exists; no live reader |
| Curator mutations | Curator patches skills directly via `skill_manage`; audit in `.curator_ledger.jsonl` with content-addressed blobs | `CuratorAdapter` turns each `actor: curator` ledger entry into a candidate; production brain served from `skills.external_dirs` (Curator read-only) | Not built |
| Curator backup/rollback | `curator_backup` snapshots, `skill_ledger.rollback_entry` | Do not reuse for production rollback; Git releases stay authoritative | Not called |
| Skill write gate | `skills.write_approval` → `pending/skills/<id>.json` | Recommend `true`; staged writes are candidate input | Not wired |
| MCP health | `hermes mcp test <name>` (exit 0/1/3) | `MCPHealthVerifier` wraps the exit code | Not built |
| Candidate evolution | `hermes-agent-self-evolution` @ `0a929e3a` (MIT, not on PyPI) | Pin SHA; reuse `ConstraintValidator`, skill load/reassemble, `EvalExample` format only. `evolve()` not used: wrong `dspy.GEPA` args, skill text never mutated, keyword-overlap metric, own holdout | Version-gated `SelfEvolutionAdapter` interface exists; external code not integrated |
| Prompt/skill optimization | `gepa==0.1.4` `gepa.optimize` + custom `GEPAAdapter` | Optional strategy; our evaluator supplies scores; TRAIN/VALIDATION only | Not integrated |
| DSPy | `dspy==3.5.0` | Not needed for GEPA on raw text; do not depend on it | Not integrated |
| Local model inference | OpenAI-compatible Qwen via `QWEN_*` | Role-separated client; live use opt-in | Client exists; no endpoint configured or reachable on the dev host |
| Experiment metadata | New SQLite schema | Implement locally; large artifacts external | Implemented foundation |
| Benchmark integrity | New validators/access controller | Deterministic; holdout separated from generator API | Implemented foundation |
| Production Git brain | Git repository | Sole production truth; separate promoter identity | Skills-only CLI promotion, signed manifests, atomic release pointer, rollback exist; OS identities operator-provisioned |

No external library or integration is represented as available until its
version, API, and behavior have been inspected and tested.
