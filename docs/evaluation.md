# Evaluation foundation

`PairedEvaluator` accepts an injected `BenchmarkRunner` contract and executes
baseline/candidate runs with matching task IDs, seeds, and environment digest.
The runner must return commit/task/seed identity with every observation; a
mismatch fails closed. The evaluator aggregates paired outcomes per task and
applies `PromotionPolicy` hard gates. The store rejects a report whose candidate
ID/hash differs from the immutable stored candidate.

This repository does not yet include a concrete Hermes runtime runner, a
candidate sandbox, full trajectory persistence, confidence intervals, a
multi-run statistical analysis, or a subjective Qwen judge. The injected runner
must not be treated as safe for executing untrusted candidate code until those
controls exist.

The policy currently blocks security or deterministic failure, any critical or
high regression, excess medium regression, missing Full/Holdout evidence,
inadequate quality improvement, excessive category degradation, excessive
token increase, latency regression, and final-task failure. Skill types are
the only default auto-promotable types; a complete Git promoter is not yet
implemented.
