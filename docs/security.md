# Security status

## Enforced in the current foundation

- Benchmark documents must declare sanitization and no secrets.
- Benchmark clusters cannot span data splits.
- Generator API access to regression/shadow/holdout is denied; holdout requires
  an evaluator-only process-local capability.
- Candidate evaluation records must match the candidate's immutable content
  hash in the SQLite store.
- Artifact bytes are checked against their SHA-256 address on read.
- Failed security, deterministic, critical, or high-regression gates block the
  evaluation report.
- Non-skill candidate types are not auto-promotable by the default policy.

## Not yet enforced

The current capability object is an in-process guard, not a security boundary
against another process or direct filesystem access. Unix-user separation,
holdout filesystem isolation, sandboxing, network restrictions, resource limits,
cryptographic signing, promotion authorization, Git state validation, atomic
deployment, and rollback are not implemented yet. Do not use this package to
run untrusted candidate code or to deploy production changes.

All session-derived text must be treated as untrusted evidence. Dreaming and
prompt-injection defenses are not implemented yet.
