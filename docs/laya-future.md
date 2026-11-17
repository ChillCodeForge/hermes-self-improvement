# Laya: decision history foundation

Laya is an optional, later decision model that could learn to imitate routine
decisions of the improvement system. This repository only collects the data
such a model would need. **Laya never replaces a gate.** Promotion, rollback,
holdout access, security checks, and evaluator verdicts stay deterministic and
are never read from or overridden by the decision history.

## What exists

`hermes_improve.decisions` stores decisions in the `decision_history` table of
`experiments.sqlite3` (separate from the experiment-bound `decisions` table).
Each row has the spec fields:

| Field | Meaning |
|---|---|
| `id` | SHA-256 prefix of type, features, options, selection, source, model version; repeating the same decision on the same evidence is idempotent |
| `decision_type` | `benchmark_eligibility`, `risk`, `needs_qwen`, `regression_severity`, `suite_selection`, `experiment_priority`, `curator_signal` |
| `input_features` | JSON object (max 64 KiB, finite numbers, no credential-like material) |
| `available_options` / `selected_option` | 2–64 labels; the selection must be one of them |
| `confidence` | optional, 0–1 |
| `source` / `model_version` | producing component and rule/model version |
| `ground_truth` / `ground_truth_source` / `correct` | backfilled later; immutable once set |
| `timestamp` | first recording time (UTC) |

APIs:

- `record_decision(database, **fields)` or `DecisionRecorder(path).record(...)`
  — callable from any package.
- `DecisionRecorder.backfill_ground_truth(id, ground_truth=..., ground_truth_source=...)`
  — sets `correct = (ground_truth == selected_option)`.
- `DecisionRecorder.export_jsonl(stream, decision_type=None, labeled_only=False)`.

CLI:

```bash
hermes-improve decisions export --labeled-only --output dataset.jsonl
hermes-improve decisions export --type regression_severity
hermes-improve decisions backfill DECISION_ID --ground-truth high --source operator-review
```

Recorded today:

- `regression_severity` — every `monitor evaluate` that reaches a verdict
  (`none/medium/high/critical`), with metrics, baseline, affected skills and
  evidence sources as features (`source=release-monitor`,
  `model_version=monitoring-policy-1`).
- `curator_signal` — every Curator signal shown by `curator status`
  (`no_action/propose_candidate`, confidence = signal impact score).
- `benchmark_eligibility` — every `benchmarks mine` (`source=benchmark-miner`,
  features: task fingerprint, reproducibility score, ground-truth type,
  reasons) and every `benchmarks add` admission (`source=benchmark-admission`,
  features additionally: benchmark ID/version and declared class). Options are
  `EXCLUDED/DREAMING_ONLY/SOFT_BENCHMARK/HARD_BENCHMARK`; confidence is the
  fixed confidence of the ranked ground-truth source, null for pre-screen
  exclusions (`model_version=eligibility-rules-1`).
- `risk` — every `candidate create --file/--delete`, the configured
  `taxonomy.candidate_risk` lookup (`LOW/MEDIUM/HIGH/CRITICAL`; features:
  candidate type, target, file counts, experiment ID;
  `source=taxonomy-candidate-risk`, `model_version=taxonomy-<config digest>`).
- `needs_qwen` — every selected task of an `evaluate` run: the route from
  `evaluator.judge_route` (`deterministic_verification/qwen_judge/exit_code_only`;
  features: split, suite, deterministic check count, subjective criteria count,
  whether `--judge` was given, and task ID/version for public tasks only).
- `suite_selection` — every `evaluate` run (`impact_narrowed/unnarrowed`;
  features: suite, candidate type, requested and applied impact scope, whether
  narrowing fell back, selected and holdout task counts, suite manifest digest).
- `experiment_priority` — every finalist ranking in `evolve`: each improved
  variant ordered by CORE score is `finalist` (top `max_finalists`) or
  `deferred` (features: run, lineage, generation, candidate, SMOKE/CORE
  scores, rank, pending count, `max_finalists`; `source=evolution-loop`).

Every record is written after the deterministic rule has decided, from the
same values the rule used; nothing reads the history back, so recording cannot
change a decision. Rule-based decisions without a calibrated confidence record
`confidence = null` rather than an invented number. A recording error (for
example a database failure) fails the command instead of being swallowed.

## What future fine-tuning would need

1. **Labels.** Ground truth is only present where an operator or a later
   outcome backfills it. A useful set needs a routine: for example
   `regression_severity` labeled from the final release outcome (rolled back
   or stable after a longer window) and `curator_signal` labeled from whether
   the resulting candidate was promoted.
2. **Volume and balance.** Hundreds of labeled examples per decision type,
   with rare classes (critical regressions) represented, before any training is
   meaningful.
3. **Richer features.** The recorded features are the ones the rules saw;
   a learned model may need more context (for example task text embeddings),
   which must not include sealed-holdout content.
4. **Leakage control.** Features must never contain sealed-holdout task
   content or results; exports would need the same split discipline as
   benchmarks.
5. **Evaluation.** A trained model would be compared offline against the
   recorded rule decisions and labels, and could at most be shown as a
   suggestion next to the deterministic decision.

None of this is implemented: there is no training pipeline, model, or runtime
use of Laya in this repository.
