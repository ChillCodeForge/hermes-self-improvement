# Security status and boundaries

The package contains enforceable safeguards for its supported flows, but it is
not a hardened multi-user service. Operators must provide separate OS accounts,
file ownership, and deployment controls; environment variables are configuration
checks, not privilege separation.

## Implemented safeguards

- Benchmark records validate required metadata, sanitized/no-secrets flags,
  split declarations, and cluster separation. Sealed holdout records must be
  stored in a distinct root; the evaluator CLI rejects overlapping roots.
- Generator-facing benchmark APIs deny regression and holdout reads. Holdout
  access requires an evaluator capability, but this is an in-process guard
  only.
- Session exports are redacted for common credential fields and token formats
  before persistence, tagged as untrusted evidence, and rejected if a session
  ID is reused with different content.
- The Hermes finalize observer is disabled unless
  `plugins.entries.hermes-self-improvement.settings.auto_ingest` is explicitly
  `true`. It writes into a private (`0700` directory, `0600` database) state
  root and logs ingestion failures through Hermes' plugin logger.
- Dreaming uses a bounded projection of session data as untrusted evidence.
  DEEP proposals targeting known protected areas are filtered; this is not a
  general prompt-injection defense.
- Candidates are content-hashed and store writes verify candidate/evaluation
  identity. Run artifacts are content-addressed and hash-checked on read.
- Evaluation pairs baseline and candidate by benchmark and seed, checks commit,
  task, seed, and environment identity, applies deterministic hard gates, and
  can sign the full report and run-level evidence with Ed25519.
- The Docker command runner requires digest-pinned images and disabled network,
  and applies read-only container root/input, non-root user, dropped
  capabilities, no-new-privileges, PID/memory/CPU limits, and bounded temporary
  filesystems. Candidate trees reject symlinks and oversized workspaces.
- Qwen endpoints must be loopback unless `QWEN_ALLOW_REMOTE=true` is explicitly
  set. Model calls are optional and do not decide promotion.
- The promoter verifies the evaluator signature, policy/benchmark versions,
  Full and Holdout evidence, immutable candidate hash, candidate branch,
  production baseline, and actual Git diff. Automatic promotion is limited to
  skill paths. It exports Git archives without executing candidate code,
  writes an HMAC-signed manifest, switches an atomic symlink, and supports
  rollback.

## Required deployment controls not supplied by this package

- Separate generator, evaluator, and promoter Unix users with restrictive
  ownership and permissions for the holdout root, database, keys, Docker
  socket, repository, and release directories. CLI UID checks only compare the
  current effective UID with configured values; they do not create users or
  verify filesystem ownership.
- Docker daemon isolation. Access to a host Docker socket is highly privileged;
  container flags do not protect the host from a compromised daemon/kernel.
- A live Hermes runtime/session adapter, production monitoring collector, and
  a reviewed installation/deployment service for release symlinks.
- Secret-management and rotation procedures for evaluator private keys and
  manifest HMAC keys.

Do not treat the process-local holdout capability as protection from another
process or direct filesystem reads. Do not place real sealed tasks on a
generator-readable filesystem. Session content and model output remain
untrusted; redaction is best-effort and cannot guarantee removal of every
secret or prompt injection. No curated real-task benchmark corpus or full
Hermes deployment workflow has yet been validated.
