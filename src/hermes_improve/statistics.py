from __future__ import annotations

from math import isfinite, sqrt
from statistics import mean, stdev
from typing import Sequence


_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def paired_mean_confidence_interval(deltas: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Student-t interval for paired per-task deltas (95% default)."""
    if not deltas:
        raise ValueError("at least one paired delta is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if confidence != 0.95:
        raise ValueError("only 95% confidence is currently supported")
    center = mean(deltas)
    if len(deltas) < 2:
        return center, center
    degrees = len(deltas) - 1
    critical = _T_975.get(degrees, 1.96 if degrees > 30 else 2.042)
    margin = critical * stdev(deltas) / sqrt(len(deltas))
    return center - margin, center + margin


def cohens_dz(deltas: Sequence[float]) -> float | None:
    """Standardized mean of paired deltas; ``None`` when undefined (n < 2 or zero spread)."""
    if len(deltas) < 2:
        return None
    spread = stdev(deltas)
    if spread == 0:
        return None
    return mean(deltas) / spread


def hedges_gz(deltas: Sequence[float]) -> float | None:
    """Small-sample bias-corrected ``cohens_dz`` (Hedges' correction J = 1 - 3/(4 df - 1))."""
    effect = cohens_dz(deltas)
    if effect is None:
        return None
    degrees = len(deltas) - 1
    return effect * (1 - 3 / (4 * degrees - 1))


def effect_magnitude(effect: float | None, deltas: Sequence[float]) -> str:
    """Conventional magnitude label; constant non-zero deltas are reported as 'constant_shift'."""
    if effect is None:
        if deltas and all(delta == deltas[0] for delta in deltas):
            return "none" if deltas[0] == 0 else "constant_shift"
        return "undetermined"
    size = abs(effect)
    if size < 0.2:
        return "negligible"
    if size < 0.5:
        return "small"
    if size < 0.8:
        return "medium"
    return "large"


def paired_summary(baseline: Sequence[float], candidate: Sequence[float]) -> dict[str, object]:
    """Paired delta statistics (candidate - baseline) for reports.

    Reports the mean delta, its 95% Student-t interval and a practical effect
    size. No p-value is produced: promotion decisions rely on hard gates,
    interval bounds and effect sizes, never on significance alone.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired statistics require equally many baseline and candidate values")
    if not baseline:
        raise ValueError("at least one paired value is required")
    values = [*baseline, *candidate]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in values):
        raise ValueError("paired statistics require finite numeric values")
    deltas = [float(after) - float(before) for before, after in zip(baseline, candidate)]
    low, high = paired_mean_confidence_interval(deltas)
    effect = hedges_gz(deltas)
    return {
        "n": len(deltas),
        "baseline_mean": mean(baseline),
        "candidate_mean": mean(candidate),
        "mean_delta": mean(deltas),
        "delta_ci95": [low, high],
        "delta_sd": stdev(deltas) if len(deltas) > 1 else None,
        "cohens_dz": cohens_dz(deltas),
        "hedges_gz": effect,
        "magnitude": effect_magnitude(effect, deltas),
    }
