# Discovery and verified integration status

## Current repository state

This is a standalone repository, separate from the Hermes Agent checkout. At
project start it contained only a README. A small earlier prototype existed in
the local Hermes Agent worktree; its hash-gated candidate idea and Curator signal
adapter informed this foundation, but its JSON ledger and direct promotion
method were not carried forward as production-safe mechanisms.

## Verified facts and limits

| System | Verified in this work | Status in this repository |
|---|---|---|
| Hermes Agent | Local checkout at `/home/dev/.hermes/hermes-agent`, HEAD `e06df9c2eae6901f4c88f464e0ae6e753c0428b8`; `PluginContext.register_hook`, `on_session_finalize`, and `agent.trace_upload.load_session_messages` were inspected | Optional entry-point plugin registers the finalize hook and reads persisted messages; operator opt-in required. Unit tested, but no live Hermes installation test or `batch_runner` integration |
| Curator | Existing Hermes checkout provides Curator-related modules and skill usage surfaces | Read-only signal adapter contract exists; there is no direct production mutation path here |
| `hermes-agent-self-evolution` | Not inspected/pinned during this phase | Not integrated |
| DSPy | Not inspected/pinned during this phase | Not integrated |
| GEPA | Not inspected/pinned during this phase | Not integrated |
| Local Qwen serving | Endpoint/configuration was specified by project requirements, but no endpoint was exercised | Not integrated; no model/version is hardcoded |

This document deliberately does not claim version verification for external
projects or a live Qwen server.

## Reuse and adapter decisions

- Hermes remains the runtime; this project is a separate package.
- Curator is treated as a read-only signal source. Changes require proposals,
  candidates, evaluation, and Git promotion.
- Self-evolution, GEPA, and DSPy must be isolated behind version-pinned adapters
  if adopted after discovery.
- Qwen remains optional and outside deterministic evaluation/promotion rules.
- Benchmark validation, access checks, hashes, storage, verifiers, and policy
  decisions must work without any LLM dependency.

## Next discovery gate

Before wiring runtime integration, record exact Hermes and Curator commit/API
details, inspect the self-evolution/GEPA/DSPy revisions proposed for use, and
exercise the configured Qwen endpoint. Pin tested versions and document
permission, prompt-injection, data-retention, and API-compatibility risks.
