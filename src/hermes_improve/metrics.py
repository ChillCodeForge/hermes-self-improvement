"""Efficiency and outcome metrics aggregated from paired run observations (spec §24).

Efficiency is only comparable when the candidate does not succeed less often
than the baseline: a cheaper failed task is never an improvement. The report
therefore lists "cheaper failures" explicitly and only computes efficiency
deltas when success is not lower.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Sequence

from .models import RunObservation


def _per_success(total: float, successes: int) -> float | None:
    return total / successes if successes else None


def arm_metrics(runs: Sequence[RunObservation]) -> dict[str, Any]:
    if not runs:
        raise ValueError("arm metrics require at least one run")
    count = len(runs)
    successes = sum(run.success for run in runs)
    reported = all(run.token_usage_reported for run in runs)
    tokens = sum(run.tokens for run in runs)
    tool_calls = sum(run.tool_calls for run in runs)
    latency = sum(run.latency_ms for run in runs)
    return {
        "runs": count,
        "successes": successes,
        "success_rate": successes / count,
        "token_usage_reported": reported,
        "tokens_per_task": tokens / count if reported else None,
        "tokens_per_success": _per_success(tokens, successes) if reported else None,
        "input_tokens": sum(run.input_tokens for run in runs) if reported else None,
        "output_tokens": sum(run.output_tokens for run in runs) if reported else None,
        "context_tokens_mean": mean(run.context_tokens for run in runs) if reported else None,
        "context_chars_mean": mean(run.context_chars for run in runs),
        "tool_calls": tool_calls,
        "tool_calls_per_task": tool_calls / count,
        "tool_calls_per_success": _per_success(tool_calls, successes),
        "failed_tool_calls": sum(run.failed_tool_calls for run in runs),
        "retry_count": sum(run.retries for run in runs),
        "turns_mean": mean(run.turns for run in runs),
        "api_calls": sum(run.api_calls for run in runs),
        "latency_per_task_ms": latency / count,
        "latency_per_success_ms": _per_success(latency, successes),
        "filesystem_changes_mean": mean(len(run.filesystem_changes) for run in runs),
        "exit_states": dict(sorted(Counter(run.exit_state or "unspecified" for run in runs).items())),
    }


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline


def efficiency_report(pairs: Sequence[tuple[RunObservation, RunObservation]]) -> dict[str, Any]:
    """Compare paired (baseline, candidate) runs that share task, seed and environment."""
    if not pairs:
        raise ValueError("efficiency report requires paired runs")
    for baseline, candidate in pairs:
        if (baseline.task_id, baseline.seed) != (candidate.task_id, candidate.seed):
            raise ValueError("efficiency pairs must share task and seed")
    baseline_metrics = arm_metrics([pair[0] for pair in pairs])
    candidate_metrics = arm_metrics([pair[1] for pair in pairs])
    cheaper_failures = [
        {"task_id": baseline.task_id, "seed": baseline.seed}
        for baseline, candidate in pairs
        if baseline.success and not candidate.success and (
            candidate.tokens < baseline.tokens
            or candidate.latency_ms < baseline.latency_ms
            or candidate.tool_calls < baseline.tool_calls
        )
    ]
    comparable = candidate_metrics["success_rate"] >= baseline_metrics["success_rate"]
    deltas = {
        "tokens_per_success_ratio": _ratio(candidate_metrics["tokens_per_success"], baseline_metrics["tokens_per_success"]),
        "tool_calls_per_success_ratio": _ratio(
            candidate_metrics["tool_calls_per_success"], baseline_metrics["tool_calls_per_success"]
        ),
        "latency_per_success_ratio": _ratio(
            candidate_metrics["latency_per_success_ms"], baseline_metrics["latency_per_success_ms"]
        ),
    } if comparable else {
        "tokens_per_success_ratio": None,
        "tool_calls_per_success_ratio": None,
        "latency_per_success_ratio": None,
    }
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "success_not_lower": comparable,
        "cheaper_failures": cheaper_failures,
        "efficiency_deltas": deltas,
        "efficiency_note": (
            "efficiency deltas are only reported when candidate success is not lower; "
            "cheaper failed runs are listed and never count as improvement"
        ),
    }
