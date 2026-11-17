"""Task deduplication and similarity clustering without heavy dependencies (spec §15).

Five layers, from strictest to broadest:

1. exact content hash: SHA-256 of the raw prompt bytes;
2. task fingerprint: SHA-256 of the case-folded, whitespace-normalized prompt;
3. near duplicate: word-set Jaccard >= 0.85 for prompts of at least 8 words
   (`task_similarity.are_near_duplicate_tasks`);
4. similarity clustering: character 5-shingles, 64-permutation MinHash with
   16x4 LSH banding for candidate pairs, confirmed by exact shingle Jaccard
   >= 0.5. Connected components form clusters.

5. semantic clustering (optional): cosine similarity of prompt embeddings
   >= ``scoring.yaml clustering.semantic_similarity_threshold``, computed by a
   pluggable :class:`~hermes_improve.embeddings.Embedder` (default backend: a
   local OpenAI-compatible ``/v1/embeddings`` endpoint configured through
   ``models.yaml embedding``). Semantic links are added to the same
   union-find as layer 4, so clusters are connected components of the merged
   lexical + semantic graph and split consistency is checked on them.

Layer 4 is lexical: it groups reworded and lightly edited prompts but cannot
recognise paraphrases that share little vocabulary; layer 5 closes that gap
when an embedder is configured. Without one, behaviour is exactly layers 1-4
and every leakage report says explicitly that layer 5 was not applied
(:class:`SemanticReport` status ``not_applied``). SEALED_HOLDOUT prompts are
embedded only by a backend marked ``sealed_holdout_trusted`` (a loopback
endpoint); otherwise they are excluded and the status is ``partial``.

Layers 1-4 are deterministic (fixed hash salts, sorted iteration), so cluster
ids are reproducible across processes and machines. Layer 5 is reproducible
because vectors are cached by (model id, content hash) and the model id is
recorded in the report; links depend only on cached vectors and the threshold.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from .embeddings import Vector, cosine
from .models import Benchmark, Split

if TYPE_CHECKING:
    from .embeddings import SemanticLayer

SHINGLE_SIZE = 5
NUM_PERMUTATIONS = 64
LSH_BANDS = 16
LSH_ROWS = NUM_PERMUTATIONS // LSH_BANDS
SIMILARITY_THRESHOLD = 0.5
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _permutation(index: int) -> tuple[int, int]:
    seed = hashlib.sha256(f"hermes-minhash-permutation:{index}".encode()).digest()
    a = int.from_bytes(seed[:8], "big") % (_MERSENNE_PRIME - 1) + 1
    b = int.from_bytes(seed[8:16], "big") % _MERSENNE_PRIME
    return a, b


_PERMUTATIONS: tuple[tuple[int, int], ...] = tuple(_permutation(i) for i in range(NUM_PERMUTATIONS))


def exact_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_fingerprint(task: str) -> str:
    normalized = re.sub(r"\s+", " ", task.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_for_shingles(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(part for part in _NON_WORD.split(folded) if part)


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    normalized = normalize_for_shingles(text)
    if not normalized:
        return frozenset()
    if len(normalized) <= size:
        return frozenset({normalized})
    return frozenset(normalized[index:index + size] for index in range(len(normalized) - size + 1))


def minhash_signature(items: frozenset[str]) -> tuple[int, ...]:
    if not items:
        return tuple(_MAX_HASH for _ in range(NUM_PERMUTATIONS))
    hashed = [
        int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big")
        for item in sorted(items)
    ]
    return tuple(
        min(((a * value + b) % _MERSENNE_PRIME) for value in hashed)
        for a, b in _PERMUTATIONS
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def similarity(left: str, right: str) -> float:
    return jaccard(shingles(left), shingles(right))


def are_similar_tasks(left: str, right: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    return similarity(left, right) >= threshold


@dataclass(frozen=True)
class _Entry:
    key: str
    fingerprint: str
    shingles: frozenset[str]
    bands: tuple[str, ...]


def _bands(signature: Sequence[int]) -> tuple[str, ...]:
    return tuple(
        f"{band}:" + ",".join(str(value) for value in signature[band * LSH_ROWS:(band + 1) * LSH_ROWS])
        for band in range(LSH_BANDS)
    )


class SimilarityIndex:
    """Incremental MinHash/LSH index assigning deterministic cluster ids."""

    def __init__(
        self, threshold: float = SIMILARITY_THRESHOLD, *, semantic: "SemanticLayer | None" = None,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("similarity threshold must be in (0, 1]")
        self.threshold = threshold
        self.semantic = semantic
        self._vectors: dict[str, Vector] = {}
        self._entries: dict[str, _Entry] = {}
        self._buckets: dict[str, list[str]] = defaultdict(list)
        self._cluster_of: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def similar_keys(self, text: str) -> tuple[str, ...]:
        """Return indexed keys whose confirmed similarity meets the threshold."""
        candidate = self._entry("", text)
        matches = self._matches(candidate) or self._semantic_matches(text)
        return tuple(key for key, _score in matches)

    def add(self, key: str, text: str, *, cluster_id: str | None = None) -> str:
        """Index `text` under `key`; return the cluster it joins.

        A text joins the cluster of its most similar indexed entry (ties broken
        by key). Otherwise it starts a cluster named `cluster_id`, or
        `sem-<fingerprint prefix>` when none is supplied.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("similarity index keys must be non-empty strings")
        if key in self._entries:
            raise ValueError(f"similarity index key already present: {key!r}")
        entry = self._entry(key, text)
        matches = self._matches(entry) or self._semantic_matches(text)
        if matches:
            assigned = self._cluster_of[matches[0][0]]
        else:
            assigned = cluster_id or f"sem-{entry.fingerprint[:16]}"
        self._entries[key] = entry
        self._cluster_of[key] = assigned
        for band in entry.bands:
            self._buckets[band].append(key)
        if self.semantic is not None and text.strip():
            self._vectors[key] = self.semantic.vectors([text])[0]
        return assigned

    def cluster_of(self, key: str) -> str:
        return self._cluster_of[key]

    def _entry(self, key: str, text: str) -> _Entry:
        if not isinstance(text, str):
            raise ValueError("similarity index text must be a string")
        items = shingles(text)
        return _Entry(key, task_fingerprint(text), items, _bands(minhash_signature(items)))

    def _semantic_matches(self, text: str) -> list[tuple[str, float]]:
        """Layer 5: indexed keys whose embedding cosine meets the semantic threshold."""
        if self.semantic is None or not self._vectors or not text.strip():
            return []
        vector = self.semantic.vectors([text])[0]
        scored = []
        for key, other in self._vectors.items():
            score = cosine(vector, other)
            if score >= self.semantic.threshold:
                scored.append((key, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def _matches(self, entry: _Entry) -> list[tuple[str, float]]:
        if not entry.shingles:
            return []
        candidates: set[str] = set()
        for band in entry.bands:
            candidates.update(self._buckets.get(band, ()))
        for key, other in self._entries.items():
            if other.fingerprint == entry.fingerprint:
                candidates.add(key)
        scored = []
        for key in candidates:
            score = jaccard(entry.shingles, self._entries[key].shingles)
            if self._entries[key].fingerprint == entry.fingerprint:
                score = 1.0
            if score >= self.threshold:
                scored.append((key, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored


@dataclass(frozen=True)
class SemanticReport:
    """Whether layer 5 ran; always part of a leakage report (never a silent pass)."""

    status: str  # "applied" | "partial" | "not_applied"
    reason: str | None = None
    model_id: str | None = None
    threshold: float | None = None
    embedded_texts: int = 0
    excluded_sealed_holdout: int = 0
    semantic_links: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "embedded_texts": self.embedded_texts,
            "excluded_sealed_holdout": self.excluded_sealed_holdout,
            "semantic_links": self.semantic_links,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def warning(self) -> str | None:
        if self.status == "applied":
            return None
        if self.status == "partial":
            return f"semantic clustering layer applied partially: {self.reason}"
        return f"semantic clustering layer NOT applied: {self.reason}"


NOT_CONFIGURED = SemanticReport(
    "not_applied",
    reason="no embedding endpoint configured (models.yaml embedding); "
    "paraphrases sharing little vocabulary are not checked",
)


def _lexical_pairs(entries: Mapping[str, _Entry], threshold: float) -> list[tuple[str, str]]:
    keys = sorted(entries)
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        for band in entries[key].bands:
            buckets[band].append(key)
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_fingerprint[entries[key].fingerprint].append(key)
    pairs: set[tuple[str, str]] = set()
    for members in list(buckets.values()) + list(by_fingerprint.values()):
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                pairs.add((min(left, right), max(left, right)))
    confirmed = []
    for left, right in sorted(pairs):
        a, b = entries[left], entries[right]
        if a.fingerprint == b.fingerprint or (
            a.shingles and b.shingles and jaccard(a.shingles, b.shingles) >= threshold
        ):
            confirmed.append((left, right))
    return confirmed


def semantic_pairs(
    texts: Mapping[str, str], semantic: "SemanticLayer",
) -> list[tuple[str, str]]:
    """Layer 5: all key pairs whose embedding cosine meets the semantic threshold.

    Exhaustive O(n^2) comparison of unit vectors in sorted key order, so the
    result depends only on the (cached) vectors and the threshold.
    """
    keys = sorted(key for key in texts if texts[key].strip())
    if not keys:
        return []
    vectors = dict(zip(keys, semantic.vectors([texts[key] for key in keys])))
    pairs = []
    for index, left in enumerate(keys):
        left_vector = vectors[left]
        for right in keys[index + 1:]:
            if cosine(left_vector, vectors[right]) >= semantic.threshold:
                pairs.append((left, right))
    return pairs


def _components(
    keys: Sequence[str], entries: Mapping[str, _Entry], pairs: Iterable[tuple[str, str]],
) -> dict[str, str]:
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)
    components: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        components[find(key)].append(key)
    result: dict[str, str] = {}
    for members in components.values():
        cluster = "sem-" + min(entries[key].fingerprint for key in members)[:16]
        for key in members:
            result[key] = cluster
    return result


def _entries(texts: Mapping[str, str], threshold: float) -> dict[str, _Entry]:
    builder = SimilarityIndex(threshold)
    return {key: builder._entry(key, texts[key]) for key in sorted(texts)}


def cluster_texts(
    texts: Mapping[str, str], threshold: float = SIMILARITY_THRESHOLD,
    *, semantic: "SemanticLayer | None" = None,
) -> dict[str, str]:
    """Cluster all texts at once (order independent).

    Connected components of the confirmed-similarity graph (layer 4, plus
    layer-5 semantic links when ``semantic`` is given) form clusters; a cluster
    id is `sem-` plus the prefix of the smallest member fingerprint.
    """
    entries = _entries(texts, threshold)
    pairs = _lexical_pairs(entries, threshold)
    if semantic is not None:
        pairs += semantic_pairs(texts, semantic)
    return _components(sorted(texts), entries, pairs)


@dataclass(frozen=True)
class ClusterLeak:
    cluster: str
    members: tuple[tuple[str, int, str], ...]
    # "lexical" when layers 1-4 alone already put members of different splits
    # in one cluster; "semantic" when only layer-5 links connect them.
    detected_by: str = "lexical"

    def describe(self) -> str:
        rendered = ", ".join(f"{bid} v{version} ({split})" for bid, version, split in self.members)
        layer = " [semantic layer]" if self.detected_by == "semantic" else ""
        return f"similarity cluster {self.cluster} spans splits{layer}: {rendered}"


@dataclass(frozen=True)
class LeakageReport:
    leaks: tuple[ClusterLeak, ...]
    semantic: SemanticReport
    # "<id>\x00<version>" -> merged (lexical + semantic) cluster id.
    clusters: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "leaks": [
                {"cluster": leak.cluster, "detected_by": leak.detected_by,
                 "members": [list(member) for member in leak.members]}
                for leak in self.leaks
            ],
            "semantic_layer": self.semantic.to_dict(),
        }


def similarity_leakage_report(
    benchmarks: Iterable[Benchmark], threshold: float = SIMILARITY_THRESHOLD,
    *, semantic: "SemanticLayer | None" = None,
) -> LeakageReport:
    """Cluster benchmark prompts (layers 4 and 5) and report clusters spanning splits.

    SEALED_HOLDOUT prompts are passed to the embedder only when it is
    ``sealed_holdout_trusted``; otherwise they keep their lexical links only and
    the semantic report status is ``partial``.
    """
    rows = list(benchmarks)
    keyed = {f"{item.id}\x00{item.version}": item for item in rows}
    texts = {key: str(item.task["prompt"]) for key, item in keyed.items()}
    keys = sorted(texts)
    entries = _entries(texts, threshold)
    lexical = _lexical_pairs(entries, threshold)
    lexical_clusters = _components(keys, entries, lexical)
    if semantic is None:
        report = NOT_CONFIGURED
        clusters = lexical_clusters
    else:
        trusted = getattr(semantic.embedder, "sealed_holdout_trusted", False) is True
        excluded = (
            set() if trusted
            else {key for key, item in keyed.items() if item.split is Split.SEALED_HOLDOUT}
        )
        hits, misses = semantic.cache_hits, semantic.cache_misses
        eligible = {key: text for key, text in texts.items() if key not in excluded}
        links = semantic_pairs(eligible, semantic)
        clusters = _components(keys, entries, lexical + links)
        report = SemanticReport(
            "partial" if excluded else "applied",
            reason=(
                f"{len(excluded)} SEALED_HOLDOUT prompt(s) not embedded: the embedding "
                "backend is not trusted with sealed holdout text (endpoint is not loopback)"
                if excluded else None
            ),
            model_id=semantic.model_id,
            threshold=float(semantic.threshold),
            embedded_texts=sum(1 for text in eligible.values() if text.strip()),
            excluded_sealed_holdout=len(excluded),
            semantic_links=len(links),
            cache_hits=semantic.cache_hits - hits,
            cache_misses=semantic.cache_misses - misses,
        )
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        grouped[clusters[key]].append(key)
    leaks = []
    for cluster in sorted(grouped):
        members = grouped[cluster]
        if len({keyed[key].split for key in members}) <= 1:
            continue
        lexical_spans = any(
            len({keyed[key].split for key in members if lexical_clusters[key] == sub}) > 1
            for sub in {lexical_clusters[key] for key in members}
        )
        leaks.append(ClusterLeak(
            cluster,
            tuple(sorted(
                (keyed[key].id, keyed[key].version, keyed[key].split.value) for key in members
            )),
            "lexical" if lexical_spans else "semantic",
        ))
    return LeakageReport(tuple(leaks), report, MappingProxyType(dict(clusters)))


def find_similarity_split_leaks(
    benchmarks: Iterable[Benchmark], threshold: float = SIMILARITY_THRESHOLD,
    *, semantic: "SemanticLayer | None" = None,
) -> list[ClusterLeak]:
    """Report similarity clusters whose members are assigned to different splits."""
    return list(similarity_leakage_report(benchmarks, threshold, semantic=semantic).leaks)
