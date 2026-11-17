# Security status and boundaries

The package separates Generator (agent/optimizer), Evaluator, and Promoter
with operating-system identities, a provisioned filesystem layout, and code
checks that refuse to run when that layout has drifted. This document lists
what is enforced by the kernel, what is enforced by code, and what remains an
operator responsibility.

## Operating-system isolation (provisioned)

`scripts/provision.sh` (Linux, run as root, idempotent) creates the system
users `hermes-agent`, `hermes-evaluator`, `hermes-promoter` (each with its own
primary group, no login shell, no home), the group `hermes-ledger`
(evaluator and promoter only), and this layout under `--root`
(default `/srv/hermes-improve`):

| Path | Owner:group mode (+ACL) | agent | evaluator | promoter |
|---|---|---|---|---|
| `install/` (evaluator/verifier/promotion code) | root:root 0755 | r-x | r-x | r-x |
| `config/` (policy, `layout.json`, role env files) | root:root 0755 | r-x | r-x | r-x |
| `keys/evaluator/signing.pem` | root:hermes-evaluator 0440 | – | r | – |
| `keys/evaluator.pub` | root:root 0644 | r | r | r |
| `keys/promoter/manifest.key` | root:hermes-promoter 0440 | – | – | r |
| `brain/production.git` (production `main`, release tags) | promoter 0750 +ACL agent,evaluator rX | r-x | r-x | rwx |
| `brain/candidates.git` (candidate branches) | agent 0750 +ACL evaluator rX | rwx | r-x | – |
| `workspaces/` (candidate worktrees, agent TMPDIR) | agent 0750 +ACL evaluator rX | rwx | r-x | – |
| `handoff/` (candidate bundles) | agent 0750 +ACL evaluator,promoter rX | rwx | r-x | r-x |
| `releases/`, `deploy/current` (production checkout) | promoter 0755, release files 0444 | r-x | r-x | rwx |
| `benchmarks/public/` (TRAIN/VALIDATION/REGRESSION ground truth) | root:root 0755 | r-x | r-x | r-x |
| `holdout/` (sealed holdout) | root:hermes-evaluator 0750, files 0640 | – | r-x | – |
| `ledger/` (experiment DB, spent holdouts, artifacts) | root:hermes-ledger 3770, DB files root 0660 | – | rwx | rwx |
| `agent/`, `evaluator/` (role state, evaluator Git mirror), `home/<role>` | role 0700 | own only | own only | own only |

Default ACLs make new objects in the shared Git repositories and handoff
directory inherit the reader entries. The ledger directory is setgid and
sticky: both roles can append, neither can delete or replace the root-owned
database files. Provisioning fails if any role user is in `sudo`, `wheel`,
`admin`, `adm`, `disk`, `shadow`, `lxd`, `root`, or another role's group, has a
sudoers entry, or (except the evaluator, see below) is in `docker`.

`scripts/hermes-improve-as <role> <args>` is the only supported entry point.
It reads the root-owned `config/roles/<role>.env`, discards the caller's
environment, and drops to the role with `setpriv` (`--init-groups`,
`--no-new-privs`, empty inheritable and bounding capability sets). Each role
has a subcommand allowlist: the agent cannot invoke `evaluate`,
`candidate import`, `promote`, or `rollback`; the promoter cannot create or
evaluate candidates.

### Candidate handoff without trusting another user's repository

Git executes configuration from the repository it operates on, so the
evaluator and promoter never run Git inside the agent-owned candidate
repository. The agent runs `candidate export`, which writes a Git bundle and
metadata into `handoff/`. The evaluator runs `candidate import`, which
fetches the bundle into its own mirror, recomputes the diff and content hash,
and only then records the candidate in the ledger. The promoter's
`promote --candidate-bundle` fetches exactly the evaluated commit into the
production repository; `GitPromoter` then re-verifies signature, hashes, and
diff. The agent's `candidate sync-baseline` fetches production `main` through
`upload-pack`, trusting only the more privileged promoter repository.

## Enforcement levels

**Kernel-enforced (by the provisioned layout)** — proven by
`tests/security/test_os_isolation.py`, which runs as the real users:

- The agent cannot write production (repository refs, releases, the
  deployment symlink), cannot read the sealed holdout, the ledger, or either
  private key, and cannot modify configuration, policy, public ground truth,
  the layout description, role environments, or installed code.
- The evaluator cannot write production, the holdout, candidate repositories,
  or configuration; it can read holdout and candidates.
- The promoter cannot read the holdout or candidate workspaces.
- An agent that calls `hermes-improve promote` directly with spoofed
  `HERMES_PROMOTER_UID` and the isolation check disabled still fails, because
  it cannot open the ledger or write production.
- `no_new_privs` and empty capability sets for every launched role; no role
  holds sudo rights or privileged groups.

**Code-enforced (fail-closed checks)**

- `hermes-improve security check [--role R] [--production PATH]` recomputes
  each role's effective access from owner, group, mode bits, POSIX ACL
  entries, and traversal along the whole path; checks that no role owns an
  entry it must not write (an owner can chmod), that no ancestor lets a role
  replace the layout root, that `deploy/current` resolves inside `releases/`,
  that every file of the deployed checkout is unwritable by agent and
  evaluator, that role users are distinct and unprivileged, and, with
  `--role`, that the process UID/GID/groups match the role and that the
  kernel's own `access()` decisions match the matrix. Exit status 1 on any
  violation. Entries the checking identity cannot inspect are listed as
  `unverified`; running the check as root (as `provision.sh` does) verifies
  everything.
- `evaluate`, `candidate import`, `promote`, `rollback` and the generator command
  `evolve` (agent role: candidate repository or `workspaces/` only, public
  benchmarks, agent state) run that check for
  their role and refuse on violations, and refuse arguments that point outside
  the provisioned layout (holdout, public benchmarks, ledger, mirror,
  production repository, release root, deployment link, public key, handoff).
  Without a layout file they refuse unless the operator sets
  `HERMES_IMPROVE_ISOLATION=unprovisioned` (development only; prints a
  warning and gives no separation guarantee). This check detects
  misconfiguration; it is not the barrier against a hostile caller — the
  kernel permissions above are.
- Existing application safeguards remain: the generator-facing benchmark API
  denies regression and holdout reads; holdout generations are marked spent
  per lineage before evaluation; evaluation reports are Ed25519-signed and the
  promoter verifies signature, policy and benchmark versions, Full and
  holdout evidence, immutable candidate hash, branch tip, production baseline,
  and actual Git diff; automatic promotion is restricted to `brain/skills/`;
  release manifests are HMAC-signed; the deployment pointer switches
  atomically. Candidate targets are validated, session content is redacted and
  treated as untrusted evidence, and the Docker runner requires digest-pinned
  images, disabled network, read-only root, non-root user, dropped
  capabilities, and resource limits.
- CI: `.github/CODEOWNERS` assigns owners to evaluator, verifier, policy,
  promotion, security, benchmark ground truth, scripts, CI, and these docs; a
  unit test keeps it in sync with `isolation.PROTECTED_REPOSITORY_PATHS`. The
  workflow runs the suite on a non-root runner and as root in the test
  container, where skipped root tests fail the job.

**Operator responsibility**

- Keep branch protection on `main` (enabled 2026-11-17): pull requests only,
  "Require review from Code Owners" (team `@ChillCodeForge/chillcode`, which
  needs write access to the repository), both Tests jobs as required checks,
  no force pushes or branch deletion. CODEOWNERS has no effect without it.
  Admins are not subject to the rules, so an admin merge (`gh pr merge
  --admin`) stays possible; candidate, Dreaming, optimizer and Hermes
  identities must never hold admin rights.
- Running every role through `hermes-improve-as` (or an equivalent
  systemd unit), installing the CLI root-owned (`--install-from`), placing real
  sealed holdout files in `holdout/` before re-running `provision.sh`, and
  re-running `provision.sh` or `security check` after any manual change.
- The Hermes runtime runs as `hermes-agent` and reads the brain only through
  `deploy/current` (for example via Hermes `skills.external_dirs`, which
  Hermes' Curator treats as read-only). Hermes' own writable skill directory
  is not production and must be routed through the Curator adapter.
- The evaluator's `docker` membership (needed by the Docker runner) is
  root-equivalent and reported as a warning; use rootless Docker or a
  dedicated sandbox host where that matters. The Docker daemon, kernel, and
  host remain trusted.
- Key rotation for the evaluator Ed25519 key and the manifest HMAC key,
  backups of the ledger, and secret management.

## Embedding endpoint for semantic leakage checks

Layer 5 of the leakage checks (`hermes_improve.clustering`,
`hermes_improve.embeddings`) sends benchmark prompts to an embedding
endpoint. During `benchmarks holdout rotate`, `benchmarks add` and
`benchmarks validate --holdout-root` that includes sealed holdout prompts,
so the endpoint is treated as part of the evaluator/operator trust boundary:

- There is no default endpoint and no fallback to the chat endpoint; the
  layer runs only when `QWEN_EMBEDDING_BASE_URL` and `QWEN_EMBEDDING_MODEL`
  are both set (names fixed in the protected `config/models.yaml`).
- The URL must be loopback (`localhost`, `127.0.0.0/8`, `::1`) unless
  `QWEN_EMBEDDING_ALLOW_REMOTE=true` is set; `QWEN_ALLOW_REMOTE` for the chat
  model does not widen it. Credentials, query strings and fragments in the
  URL are rejected.
- Only a loopback endpoint is trusted with SEALED_HOLDOUT text
  (`Embedder.sealed_holdout_trusted`). With an explicitly allowed remote
  endpoint, holdout prompts are never sent; the report status is `partial`
  and names the number of withheld records. A custom `Embedder` must set
  `sealed_holdout_trusted` only if it keeps holdout text inside the
  evaluator boundary (for example an in-process local model).
- The loopback endpoint must itself run on the operator/evaluator host and
  must not forward requests elsewhere; the code cannot verify that.
- Vectors of holdout prompts are holdout-derived data (embeddings can be
  partially inverted). The cache stores only content hashes and vectors,
  in a `0700` directory with `0600` files, under the invoking role's private
  state (`$HERMES_IMPROVE_HOME/embedding-cache`, overridable with
  `HERMES_IMPROVE_EMBEDDING_CACHE`). Keep it out of the ledger and out of any
  directory the agent or promoter can read.
- Reports and CLI output carry the model id, threshold and counts, never
  prompt text.

## Known limits

- The evaluator and promoter can both write the ledger database (SQLite
  cannot be append-only via file permissions). Evaluation evidence integrity
  rests on the evaluator signature; a compromised evaluator can forge
  evidence because it holds the signing key.
- The agent controls the candidate metadata it exports, including the
  lineage ID used for holdout-spent accounting.
- The promoter can technically execute files it can read; "never executes
  candidate code" is enforced by code (`git archive` extraction, no hooks),
  not by the kernel.
- Recursive checks are conservative: a writable bit on an entry below a
  directory the role cannot traverse is still reported.
- Redaction is best-effort, and the process-local holdout capability is not a
  defense against another process; the filesystem permissions above are.
