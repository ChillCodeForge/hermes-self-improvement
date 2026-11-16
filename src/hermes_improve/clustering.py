"""Task deduplication and similarity clustering without heavy dependencies (spec §15).

Four layers, from strictest to broadest:

1. exact content hash: SHA-256 of the raw prompt bytes;
2. task fingerprint: SHA-256 of the case-folded, whitespace-normalized prompt;
3. near duplicate: word-set Jaccard >= 0.85 for prompts of at least 8 words
   (`task_similarity.are_near_duplicate_tasks`);
4. similarity clustering: character 5-shingles, 64-permutation MinHash with
   16x4 LSH banding for candidate pairs, confirmed by exact shingle Jaccard
   >= 0.5. Connected components form clusters.

Layer 4 is a lexical approximation of semantic similarity: it groups reworded
and lightly edited prompts, but cannot recognise paraphrases that share
little vocabulary. Every step is deterministic (fixed hash salts, sorted
iteration), so cluster ids are reproducible across processes and machines.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .models import Benchmark

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

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("similarity threshold must be in (0, 1]")
        self.threshold = threshold
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
        return tuple(key for key, _score in self._matches(candidate))

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
        matches = self._matches(entry)
        if matches:
            assigned = self._cluster_of[matches[0][0]]
        else:
            assigned = cluster_id or f"sem-{entry.fingerprint[:16]}"
        self._entries[key] = entry
        self._cluster_of[key] = assigned
        for band in entry.bands:
            self._buckets[band].append(key)
        return assigned

    def cluster_of(self, key: str) -> str:
        return self._cluster_of[key]

    def _entry(self, key: str, text: str) -> _Entry:
        if not isinstance(text, str):
            raise ValueError("similarity index text must be a string")
        items = shingles(text)
        return _Entry(key, task_fingerprint(text), items, _bands(minhash_signature(items)))

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


def cluster_texts(
    texts: Mapping[str, str], threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Cluster all texts at once (order independent).

    Connected components of the confirmed-similarity graph form clusters; a
    cluster id is `sem-` plus the prefix of the smallest member fingerprint.
    """
    keys = sorted(texts)
    entries = {key: SimilarityIndex(threshold)._entry(key, texts[key]) for key in keys}
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

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
    for left, right in sorted(pairs):
        a, b = entries[left], entries[right]
        if a.fingerprint == b.fingerprint or (
            a.shingles and b.shingles and jaccard(a.shingles, b.shingles) >= threshold
        ):
            union(left, right)
    components: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        components[find(key)].append(key)
    result: dict[str, str] = {}
    for members in components.values():
        cluster = "sem-" + min(entries[key].fingerprint for key in members)[:16]
        for key in members:
            result[key] = cluster
    return result


@dataclass(frozen=True)
class ClusterLeak:
    cluster: str
    members: tuple[tuple[str, int, str], ...]

    def describe(self) -> str:
        rendered = ", ".join(f"{bid} v{version} ({split})" for bid, version, split in self.members)
        return f"similarity cluster {self.cluster} spans splits: {rendered}"


def find_similarity_split_leaks(
    benchmarks: Iterable[Benchmark], threshold: float = SIMILARITY_THRESHOLD,
) -> list[ClusterLeak]:
    """Report similarity clusters whose members are assigned to different splits."""
    rows = list(benchmarks)
    keyed = {f"{item.id}\x00{item.version}": item for item in rows}
    clusters = cluster_texts({key: str(item.task["prompt"]) for key, item in keyed.items()}, threshold)
    grouped: dict[str, list[Benchmark]] = defaultdict(list)
    for key, cluster in clusters.items():
        grouped[cluster].append(keyed[key])
    leaks = []
    for cluster in sorted(grouped):
        members = grouped[cluster]
        if len({item.split for item in members}) > 1:
            leaks.append(ClusterLeak(cluster, tuple(sorted(
                (item.id, item.version, item.split.value) for item in members
            ))))
    return leaks
