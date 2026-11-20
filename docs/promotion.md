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
| `thresholds.min_holdout_quality_delta` (optional) | 0.0 |
| `thresholds.token_exception_min_quality_ci_lower` (optional, `null` disables) | 3.0 |
| `thresholds.latency_exception_min_quality_ci_lower` (optional, `null` disables) | 3.0 |
| `require_token_measurement` (optional) | `true` |
| `promotion_gates` | skills `auto`; every other type `manual`; a candidate that removes a skill is always manual |
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

The optional keys fall back to the defaults above when omitted; the defaults
are the strict settings. A provisioned layout (`scripts/provision.sh`) points
every role's `HERMES_IMPROVE_CONFIG` at `<root>/config` once all four YAML
files exist there (root-owned like the rest of that directory); otherwise the
roles run with the built-in defaults.

## Regression policy (`PromotionPolicy.assess`)

There is no overall score. The hard constraints of spec §23 are evaluated in
this order (`policy.CONSTRAINT_ORDER`); `reasons` are sorted by it, and
`metrics.primary_reason` is the first failed one, `metrics.failed_constraints`
lists the failed constraint names in order. "Final rows" are FULL rows plus
SEALED_HOLDOUT rows.

1. **Security**: any row with `security_pass=False` fails.
2. **Deterministic correctness**: any row with `deterministic_pass=False` fails.
3. **Critical regressions** and 4. **high regressions**: summed over every row,
   must not exceed 0 / 0. The **medium regression rate** of the final rows
   must not exceed 1% (same severity group, evaluated right after).
   Then the **evidence** precondition: FULL rows and SEALED_HOLDOUT rows must
   both be present.
5. **Task success must not regress**: any task (any suite) that the baseline
   solved and the candidate did not blocks, listed by task ID. A task that both
   arms fail is not a regression and is left to the quality gate. The previous
   rule ("candidate must succeed on every final task") was stricter than the
   spec and blocked candidates that fixed some hard tasks but not all; the
   per-task no-loss rule keeps the safety intent (a gain elsewhere can never
   offset a lost task).
6. **Generalization**: the sealed holdout on its own. Its mean quality delta
   must be at least `min_holdout_quality_delta` (default 0.0), and its
   candidate success count must not fall below the baseline's. FULL rows can
   therefore not carry a holdout regression through the pooled mean.
7. **Category regressions**: per category, the pooled mean delta of final rows
   must not drop by more than `max_category_drop` (2.0).
8. **Quality**: pooled final-row mean delta at least `min_quality_delta` (3.0)
   and the 95% CI lower bound not below 0.
9. **Token efficiency**: if any final row has `token_usage_reported=False`,
   the constraint cannot be evaluated. With `require_token_measurement: true`
   (default) that is a blocking reason "token efficiency not measured: runner
   reported no token usage"; with `false` the check is recorded as skipped in
   `metrics.token_check`. A missing measurement is blocking rather than a
   manual-gate trigger because the signed operator approval does not supply the
   missing number, and the manual gate only covers the type risk. When
   measured, summed candidate tokens above baseline × 1.10 fail (a measured
   zero baseline is not skipped) unless the significant-gain exception applies.
   `HermesBatchRunner` measures tokens through its loopback metering proxy
   (`evaluation.md`, "Token metering"); the Docker and local command runners
   run no model and always report unmeasured.
10. **Latency**: the 95% CI lower bound of the per-task excess over
    baseline × 1.15 must not be above 0, unless the significant-gain exception
    applies.

Significant-gain exception (§23 "unless significant quality gain" for tokens,
"unless justified" for latency, defined the same way): at least two final
rows, and the pooled 95% CI lower bound of the quality delta strictly above
`token_exception_min_quality_ci_lower` / `latency_exception_min_quality_ci_lower`
(both default 3.0, i.e. the whole interval clears the minimum quality delta).
`metrics.token_exception_applied` and `metrics.latency_exception_applied` record
its use; `null` disables an exception.

A stored task row that predates `TaskOutcome.token_usage_reported` is read as
unmeasured (the field defaults to `false`), so re-assessment at promotion time
fails closed under `require_token_measurement: true`.

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
7. Git: `candidate/<experiment-id>` still points to the evaluated commit,
   `main` still equals the baseline, the baseline is an ancestor, the binary
   diff hash equals the evaluated one, no protected paths, only `brain/`
   paths, changed paths equal the declared target units, no case collisions.
8. Type gate: auto types pass; manual types need a valid operator approval.
   A candidate whose verified diff deletes a `brain/skills/<name>/SKILL.md`
   (`git diff --no-renames --diff-filter=D`) removes a skill and needs the
   approval as well, whatever its type gate says (see "Skill removal").
   Finally the release tag and release directory must not exist yet.

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

Only experiments with a passing evaluation and a manually gated type, or a
candidate that removes a skill (pass `--repository REPO`), can be approved. The Ed25519-signed payload binds `experiment_id`, `candidate_id`,
`candidate_hash`, `candidate_type`, `policy_version`, operator, rationale,
key fingerprint and time. Approvals are stored in `promotion_approvals`, which
rejects UPDATE and DELETE with triggers. The promoter accepts only approvals
signed by a key passed with `--operator-public-key`; that key must differ from
the evaluator key. Approvals for another candidate hash or policy version, or
with an edited payload, do not count. The accepted approval is copied into the
release manifest.

## Skill removal (Curator archive, delete, consolidate)

Removing a skill is expressed with the existing candidate types and target
rules; there is no removal type and no archive area in `brain/`:

- `archive`, and a staged `delete` without `absorbed_into`, become a
  `SKILL_PATCH` whose diff deletes every tracked file of
  `brain/skills/<name>`.
- `consolidate`, and a staged `delete` with `absorbed_into` plus new umbrella
  content, become a `SKILL_MERGE` that rewrites the umbrella `SKILL.md` and
  deletes every tracked file of the absorbed skill. If the umbrella would not
  change, only the removal remains and the archive form is used, because a
  `SKILL_MERGE` must touch both units.

Why not move the skill into an archive directory, the way Hermes' Curator
moves it to `skills/.archive/`? Every brain area is loaded by Hermes or owned
by one candidate type; a new `brain/` archive area would need a 13th target
rule or a cross-area type, and a hidden `.archive` component is rejected by
the path rules on purpose. Git already is the archive: the removed files stay
in the baseline commit, every earlier release directory and its tag, and
`hermes-improve rollback brain-vN` reactivates them. Restoring a single skill
later is an ordinary `SKILL_CREATE` candidate.

Safety properties:

- **Never auto-promoted.** `PromotionPolicy.requires_manual_approval` returns
  true for any candidate whose diff deletes a `brain/skills/<name>/SKILL.md`,
  independent of `promotion_gates` in `config/promotion.yaml` (where skill
  types stay `auto` for ordinary patches). The promoter derives the removal
  set from the verified Git diff with rename detection off, so a merge or a
  hand-built `candidate create --delete` cannot slip a removal through as an
  auto-promoted rename. `hermes-improve approve` needs `--repository` for
  such candidates because the removal is read from the candidate diff; the
  signed payload is the same as for manually gated types. The release
  manifest lists `removed_skills`.
- **Benchmarks still run.** Removal candidates go through the same SMOKE,
  CORE, FULL and sealed-holdout evaluation. Impact narrowing keeps every unit
  of a multi-unit target in scope, so SMOKE/CORE tasks of the removed skill
  are selected; FULL is never narrowed. A task that passed on the baseline
  and fails without the skill is a task-success regression and blocks the
  evaluation and promotion.
- **Path safety.** Deleted paths come from `main`'s tree for exactly
  `brain/skills/<name>/` (exact case; a case variant is "does not exist"),
  must all be regular files that pass the `SKILL_*` path rules, and the
  builder refuses symlinked entries. Nothing outside the unit is touched.
- **Rollback** of the release restores the skill (tested in
  `tests/integration/test_curator_removal.py`).

## Release statuses

`ReleaseStatus`: `CANDIDATE`, `PROVISIONAL`, `STABLE`, `ROLLED_BACK`. Allowed
transitions (`RELEASE_TRANSITIONS`): CANDIDATE to PROVISIONAL, PROVISIONAL to
STABLE or ROLLED_BACK, STABLE to ROLLED_BACK, ROLLED_BACK to PROVISIONAL.
Promotion writes `PROVISIONAL`. `ExperimentStore.transition_release_status(
release_id, new_status, expected_status=..., reason=...)` is a compare-and-set
along these transitions and appends a `STATUS_CHANGED` event. The `release_events`
table is append-only; `hermes-improve releases events [--release brain-vN]`
lists it. (`update_release_status`, used by the monitoring API, does not write
events.) Post-promotion monitoring uses the `PROMOTED` and `REACTIVATED` events
as the release activation timeline when it attributes ingested Hermes sessions
to a release (`monitor collect`, see `operations.md`).

## Monitoring epochs

Post-promotion monitoring (`monitor evaluate`, `monitor collect`, `monitor
replay`, `monitor ingest`) assesses a release only on evidence from its
**current activation**. An activation starts with the release's latest
`PROMOTED` or `REACTIVATED` event; a rollback that re-activates an earlier
release therefore starts a new monitoring epoch for it. Observations recorded
during an earlier activation describe a deployment that already ended: they
stay in the monitoring database (and in the all-time observability counts per
source) but do not count towards the minimum sample, the metrics, critical
regression signals or the STABLE/ROLLBACK decision of the new epoch. Without
this, a release that regressed before, was superseded and was later restored
by a rollback would immediately be recommended for rollback again from its
old evidence, and `--apply` could roll back a healthy release.

- Each observation stores the ID of the activation event that was current
  when it was recorded (`production_observations.activation_id`). The event ID
  comes from the append-only `release_events` table, so the epoch boundary
  does not depend on wall-clock comparisons.
- Rows recorded before this column existed carry `NULL`; they count for the
  current epoch only when their record time is at or after the current
  activation event's time. Releases recorded without any activation event
  (direct `record_release` calls) keep a single epoch covering all their rows.
- `monitor collect` attributes sessions by their own timestamps. A session
  that ran during an earlier activation of the requested release is excluded
  with the reason `earlier_activation` instead of being counted for the
  current one.
- Shadow replay observation IDs are deterministic per release and task. In a
  re-activated epoch they additionally include the activation event ID, so a
  replay of the same session or SHADOW benchmark is recorded as new evidence
  instead of colliding with the earlier epoch's record. The first activation
  keeps the original ID format.
- Operator-supplied observation IDs (`monitor ingest`) are global: re-sending
  an ID from an earlier epoch is reported as a duplicate and adds no evidence
  to the current epoch; use new IDs for new observations.

## `hermes-improve rollback <release>`

Activates the given earlier release: verifies its signed manifest, requires
its commit to be an ancestor of current `main`, moves `main` back with
compare-and-swap, switches the deployment symlink atomically and, in one
transaction, marks the release owning the previous commit `ROLLED_BACK` and the
target `PROVISIONAL`, with `ROLLED_BACK` and `REACTIVATED` audit events. Tags,
release directories and manifests are left untouched. Failures restore the
symlink and `main`.

## Current limitations

- With the default `required_evidence_suites`, run `evaluate --suite SMOKE`,
  `--suite CORE` and `--suite FULL` (which adds the sealed holdout) before
  promoting; otherwise promotion is rejected with "required SMOKE evidence is
  missing". `tests/e2e/test_full_improvement_loop.py` exercises this sequence
  including promotion, rollback and the role-user variant.
- The experiment benchmark version is the evaluation manifest version
  (`BenchmarkStore.version`: public records without SHADOW), so candidates
  created by the generator, which cannot read the sealed holdout, bind to the
  same version the evaluator signs, and SHADOW monitoring benchmarks added
  after experiment creation do not invalidate it (see `benchmarking.md`,
  "Benchmark version"). `evaluate` refuses a changed version before any run,
  so the mismatch no longer surfaces only here at promotion.
- OS-level separation of operator, evaluator and promoter identities is not
  enforced here; it depends on deployment provisioning.
