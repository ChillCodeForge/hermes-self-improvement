"""Benchmark candidate mining from redacted session exports (spec §14).

Pipeline per session (hook, export and redaction happen upstream in
`ingestion`): normalization -> eligibility -> ground-truth analysis -> dedup
(exact / fingerprint) -> similarity clustering -> benchmark candidate. Mining
never admits a benchmark; it only proposes one for operator review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .clustering import SimilarityIndex, task_fingerprint
from .eligibility import assess_session
from .embeddings import SemanticLayer
from .ingestion import contains_secret_material
from .models import Benchmark, Eligibility

__all__ = [
    "BenchmarkCandidate",
    "DedupRegistry",
    "corpus_similarity_index",
    "mine_session",
    "mine_sessions",
    "task_fingerprint",
]


@dataclass(frozen=True)
class BenchmarkCandidate:
    candidate_id: str
    task_fingerprint: str
    cluster_id: str
    eligibility: Eligibility
    reproducibility_score: float
    ground_truth_type: str
    reasons: tuple[str, ...]


def mine_session(
    session: Mapping[str, Any], *, cluster_index: SimilarityIndex | None = None,
) -> BenchmarkCandidate:
    """Assess one redacted session.

    Without `cluster_index` the cluster id is the task fingerprint. With an
    index, the task joins the cluster of the most similar indexed task (for
    example an existing corpus benchmark) and is added to the index.
    """
    if contains_secret_material(session):
        return _excluded(session, "credential-like material detected")
    task = session.get("task")
    if not isinstance(task, Mapping):
        return _excluded(session, "task statement unavailable")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _excluded(session, "task prompt unavailable")
    decision = assess_session(session)
    fingerprint = task_fingerprint(prompt)
    candidate_id = hashlib.sha256(
        f"{session.get('session_id', '')}:{fingerprint}".encode()
    ).hexdigest()[:24]
    if cluster_index is None:
        cluster_id = fingerprint
    else:
        key = f"session:{candidate_id}"
        cluster_id = (
            cluster_index.cluster_of(key) if key in cluster_index
            else cluster_index.add(key, prompt, cluster_id=fingerprint)
        )
    return BenchmarkCandidate(
        candidate_id=candidate_id,
        task_fingerprint=fingerprint,
        cluster_id=cluster_id,
        eligibility=decision.eligibility,
        reproducibility_score=decision.reproducibility_score,
        ground_truth_type=decision.ground_truth.value,
        reasons=decision.reasons,
    )


def mine_sessions(
    sessions: Iterable[Mapping[str, Any]], *, corpus: Iterable[Benchmark] = (),
    semantic: SemanticLayer | None = None,
) -> list[BenchmarkCandidate]:
    """Mine a batch; similar sessions share a cluster, and tasks similar to an
    existing corpus benchmark inherit that benchmark's declared cluster_id.

    With ``semantic`` (layer 5), paraphrases with little shared vocabulary
    also join the cluster, so `DedupRegistry` pins them to the same split."""
    index = corpus_similarity_index(corpus, semantic=semantic)
    return [mine_session(session, cluster_index=index) for session in sessions]


def corpus_similarity_index(
    corpus: Iterable[Benchmark], *, semantic: SemanticLayer | None = None,
) -> SimilarityIndex:
    index = SimilarityIndex(semantic=semantic)
    rows = sorted(corpus, key=lambda item: (item.id, item.version))
    if semantic is not None:
        # One batched (cached) embedding request for the whole corpus.
        semantic.vectors([str(item.task["prompt"]) for item in rows if str(item.task["prompt"]).strip()])
    for benchmark in rows:
        index.add(
            f"benchmark:{benchmark.id}:{benchmark.version}",
            str(benchmark.task["prompt"]),
            cluster_id=benchmark.cluster_id,
        )
    return index


def _excluded(session: Mapping[str, Any], reason: str) -> BenchmarkCandidate:
    identity = str(session.get("session_id", "unknown"))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return BenchmarkCandidate(
        candidate_id=digest[:24], task_fingerprint=digest, cluster_id=digest,
        eligibility=Eligibility.EXCLUDED, reproducibility_score=0.0,
        ground_truth_type="unknown", reasons=(reason,),
    )


class DedupRegistry:
    """Pins each cluster to the first split it was assigned to.

    Cluster ids come from `mine_session`: the normalized task fingerprint, or a
    similarity cluster when the candidate was mined with a `SimilarityIndex`.
    """

    def __init__(self) -> None:
        self._clusters: dict[str, str] = {}

    def assign(self, candidate: BenchmarkCandidate, split: str) -> tuple[str, bool]:
        existing = self._clusters.get(candidate.cluster_id)
        if existing is not None and existing != split:
            return existing, False
        self._clusters[candidate.cluster_id] = split
        return split, True
