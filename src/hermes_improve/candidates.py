from __future__ import annotations

import uuid

from .models import Candidate, CandidateType, ImprovementSignal
from .store import ExperimentStore


class Generator:
    """Creates immutable proposals; it has no production or promotion API."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def create_candidate(
        self,
        experiment_id: str,
        signal: ImprovementSignal,
        *,
        candidate_type: CandidateType,
        target: str,
        base_commit: str,
        candidate_commit: str,
        diff: str,
    ) -> Candidate:
        if not all((experiment_id, target, base_commit, candidate_commit, diff.strip())):
            raise ValueError("candidate requires experiment, target, commits, and a non-empty diff")
        candidate = Candidate(
            candidate_id=f"cand-{uuid.uuid4().hex}",
            experiment_id=experiment_id,
            candidate_type=candidate_type,
            target=target,
            base_commit=base_commit,
            commit=candidate_commit,
            diff=diff,
        )
        self.store.add_candidate(candidate)
        return candidate
