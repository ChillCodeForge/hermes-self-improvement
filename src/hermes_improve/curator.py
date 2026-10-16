from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from .ingestion import contains_secret_material
from .models import ImprovementSignal


_MAX_COUNTER = (1 << 63) - 1


class CuratorAdapter:
    """Consumes caller-supplied Curator data; never writes production skills."""

    def __init__(self, usage_report: Callable[[], Iterable[Mapping[str, Any]]]) -> None:
        self._usage_report = usage_report

    def signals(self) -> list[ImprovementSignal]:
        result = []
        for row in self._usage_report():
            if not isinstance(row, Mapping):
                raise ValueError("Curator usage rows must be objects")
            name = row.get("name")
            if (
                not isinstance(name, str) or not name.strip() or name != name.strip()
                or len(name) > 128 or any(ord(character) < 32 for character in name)
                or contains_secret_material(name)
            ):
                raise ValueError("Curator skill names must be bounded, clean, and credential-free")
            use_count = _counter(row.get("use_count", 0), "use_count")
            patch_count = _counter(row.get("patch_count", 0), "patch_count")
            activity = _counter(row.get("activity_count", 0), "activity_count")
            if use_count == patch_count == activity == 0:
                continue
            result.append(ImprovementSignal(
                source="curator",
                summary=f"Review usage and maintenance signals for {name[:100]}",
                issue=f"Assess whether {name[:100]} needs a proposal, consolidation, or documentation improvement",
                impact_score=min(1.0, max(0.0, max(use_count, patch_count) / max(1, activity + 1))),
                evidence={
                    "name": name,
                    "use_count": use_count,
                    "patch_count": patch_count,
                    "activity_count": activity,
                },
            ))
        return result


def _counter(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if (
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value > _MAX_COUNTER
    ):
        raise ValueError(f"Curator {field_name} must be a non-negative bounded integer")
    return value
