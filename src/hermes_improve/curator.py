from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class ImprovementSignal:
    source: str
    summary: str
    issue: str
    impact_score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


class CuratorAdapter:
    """Consumes caller-supplied Curator data; never writes production skills."""

    def __init__(self, usage_report: Callable[[], Iterable[Mapping[str, Any]]]) -> None:
        self._usage_report = usage_report

    def signals(self) -> list[ImprovementSignal]:
        result = []
        for row in self._usage_report():
            name = str(row.get("name") or "unnamed-skill")
            use_count = int(row.get("use_count") or 0)
            patch_count = int(row.get("patch_count") or 0)
            activity = int(row.get("activity_count") or 0)
            if use_count == patch_count == activity == 0:
                continue
            result.append(ImprovementSignal(
                source="curator",
                summary=f"Review usage and maintenance signals for {name}",
                issue=f"Assess whether {name} needs a proposal, consolidation, or documentation improvement",
                impact_score=min(1.0, max(0.0, max(use_count, patch_count) / max(1, activity + 1))),
                evidence={"name": name, "use_count": use_count, "patch_count": patch_count, "activity_count": activity},
            ))
        return result
