# Promotion and rollback

Implemented in `promotion.py` (`GitPromoter`), `approvals.py`, `policy.py`,
`config.py` and the `ExperimentStore` release helpers.

## Protected configuration

The policy comes from `config/promotion.yaml` (plus `scoring.yaml`,
`taxonomy.yaml`, `models.yaml`). The CLI reads the directory named by
`HERMES_IMPROVE_CONFIG`; without it the built-in defaults apply. The working
directory is never searched, so a command started inside a candidate worktree
cannot pick up candidate files, and candidate diffs touching `config/` are
rejected anyway. The loader rejects unknown or duplicate keys, symlinks and
group/world-writable files. The shipped files equal the built-in defaults:

| Key | Default |
|---|---|
| `policy_version` | `"1"` |
| `thresholds.max_critical_regressions` / `max_high_regressions` | 0 / 0 |
| `thresholds.max_medium_regression_rate` | 0.01 |
| `thresholds.min_quality_delta` | 3.0 |
| `thresholds.max_category_drop` | 2.0 |
| `thresholds.max_token_increase_ratio` | 0.10 |
| `thresholds.max_latency_increase_ratio` | 0.15 |
| `promotion_gates` | skills `auto`; every other type `manual` |
| `required_evidence_suites` | `SMOKE, CORE, FULL, SEALED_HOLDOUT` (FULL and SEALED_HOLDOUT cannot be removed) |

`scoring.yaml` also carries run-count defaults (deterministic 1; LLM SMOKE 1-2,
CORE 3, FULL 5, SEALED_HOLDOUT 5) and evolution limits (10 generations, 3 per
generation, 30 total, stop after 8 candidates / 3 generations without
improvement, minimum delta 0.5) as data for other components;
`overall_score.dashboard_only` must stay `true`. `PromotionPolicy.from_config()`
builds the policy; a config digest is recorded in each release manifest.

Unlocking auto-promotion for a type means changing its gate in the protected
config; bumping `policy_version` at the same time invalidates evaluations and
approvals made under the old policy.

## `hermes-improve promote <experiment-id>`

Runs as the promoter UID (`HERMES_PROMOTER_UID`) with
`HERMES_IMPROVE_MANIFEST_KEY`, `--repository`, `--release-root`,
`--deployment-link`, `--evaluator-public-key`, optional
`--operator-public-key` (repeatable) and optional `--release brain-vN`
(default: next free number). A candidate ID is still accepted; an identifier
that names both an experiment and a different candidate is rejected as
ambiguous. All promotions and rollbacks on one release root are serialized by
an exclusive file lock.

Checks, in order. Any failure aborts before Git or the deployment changes:

1. Candidate metadata matches its content hash; experiment baseline equals the
   candidate base commit.
2. Experiment complete: decision `passed` and the latest evaluation passed.
3. Every evaluation run used as evidence is signature-verified with the
   evaluator Ed25519 key. The latest run must match the candidate hash, the
   experiment's benchmark version and the current policy version. Earlier runs
   count only if they match the same identity.
4. Experiment policy version equals the current policy version.
5. Evidence: the latest run contains FULL and SEALED_HOLDOUT rows; every
   required suite has rows (newest run containing that suite). Each suite's
   rows must pass security and deterministic verifiers, stay within the
   critical/high regression limits and show no task-success regression.
   SEALED_HOLDOUT rows must come from the SEALED_HOLDOUT split.
6. Regression policy: the latest FULL and holdout rows are re-assessed with
   the current policy and must pass.
7. Type gate: auto types pass; manual types need a valid operator approval.
8. Git: `candidate/<experiment-id>` still points to the evaluated commit,
   `main` still equals the baseline, the baseline is an ancestor, the binary
   diff hash equals the evaluated one, no protected paths, only `brain/`
   paths, changed paths equal the declared target units, no case collisions,
   tag and release directory do not exist yet.

Then the promoter extracts the commit via `git archive` into a read-only
release directory (no symlinks, 100 MiB limit), writes an HMAC-signed
manifest, moves `main` with a compare-and-swap `update-ref`, re-checks that
the deployed diff (`baseline..main`) equals the evaluated diff, creates the
tag, switches the deployment symlink atomically, and records the release as
`PROVISIONAL` together with a `PROMOTED` audit event in one transaction. Any
error undoes the symlink, tag, `main` and release directory.

When two promotions run concurrently against the same baseline, one wins; the
other fails with "production Git ref has changed since evaluation" and leaves
no tag, release row or release directory (tested).

## Manual gate approvals

```text
HERMES_OPERATOR_PRIVATE_KEY=operator.pem \
  hermes-improve approve <experiment-id> --operator alice --rationale "..."
```

Only experiments with a passing evaluation and a manually gated type can be
approved. The Ed25519-signed payload binds `experiment_id`, `candidate_id`,
`candidate_hash`, `candidate_type`, `policy_version`, operator, rationale,
key fingerprint and time. Approvals are stored in `promotion_approvals`, which
rejects UPDATE and DELETE with triggers. The promoter accepts only approvals
signed by a key passed with `--operator-public-key`; that key must differ from
the evaluator key. Approvals for another candidate hash or policy version, or
with an edited payload, do not count. The accepted approval is copied into the
release manifest.

## Release statuses

`ReleaseStatus`: `CANDIDATE`, `PROVISIONAL`, `STABLE`, `ROLLED_BACK`. Allowed
transitions (`RELEASE_TRANSITIONS`): CANDIDATE to PROVISIONAL, PROVISIONAL to
STABLE or ROLLED_BACK, STABLE to ROLLED_BACK, ROLLED_BACK to PROVISIONAL.
Promotion writes `PROVISIONAL`. `ExperimentStore.transition_release_status(
release_id, new_status, expected_status=..., reason=...)` is a compare-and-set
along these transitions and appends a `STATUS_CHANGED` event. The `release_events`
table is append-only; `hermes-improve releases events [--release brain-vN]`
lists it. (`update_release_status`, used by the monitoring API, does not write
events.)

## `hermes-improve rollback <release>`

Activates the given earlier release: verifies its signed manifest, requires
its commit to be an ancestor of current `main`, moves `main` back with
compare-and-swap, switches the deployment symlink atomically and, in one
transaction, marks the release owning the previous commit `ROLLED_BACK` and the
target `PROVISIONAL`, with `ROLLED_BACK` and `REACTIVATED` audit events. Tags,
release directories and manifests are left untouched. Failures restore the
symlink and `main`.

## Current limitations

- The `evaluate` CLI produces FULL plus holdout evidence only. With the default
  `required_evidence_suites`, SMOKE and CORE evidence must currently come from
  `PairedEvaluator.evaluate(suite="SMOKE"|"CORE")` until the evaluator CLI adds
  suite selection; otherwise promotion is rejected with "required SMOKE
  evidence is missing".
- OS-level separation of operator, evaluator and promoter identities is not
  enforced here; it depends on deployment provisioning.
