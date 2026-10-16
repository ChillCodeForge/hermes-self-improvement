from __future__ import annotations

from math import sqrt
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
