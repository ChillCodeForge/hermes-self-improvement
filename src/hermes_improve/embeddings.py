"""Pluggable text embedders for the semantic clustering layer (layer 5).

The core stays dependency-free: an :class:`Embedder` is any object with a
``model_id``, a ``sealed_holdout_trusted`` flag and ``embed(texts)``. The
shipped backend is :class:`OpenAICompatibleEmbedder`, which calls
``POST {base_url}/v1/embeddings`` on a local server (vLLM, llama.cpp
``--embeddings`` or Ollama serving a Qwen embedding model).

Security decisions (see docs/security.md):

- There is no default endpoint. The layer is enabled only when both
  ``QWEN_EMBEDDING_BASE_URL`` and ``QWEN_EMBEDDING_MODEL`` (names configurable
  in ``config/models.yaml`` ``embedding``) are set.
- The endpoint must be loopback unless ``QWEN_EMBEDDING_ALLOW_REMOTE=true``.
  The chat-model ``QWEN_ALLOW_REMOTE`` switch does not widen it.
- Only a loopback endpoint is trusted with SEALED_HOLDOUT text. With an
  explicitly allowed remote endpoint, holdout records are never sent; the
  leakage report then says the semantic layer was applied only partially.

Embeddings are cached on disk keyed by (model id, SHA-256 of the text), so
repeated leakage checks are reproducible and do not re-query the endpoint.
The cache stores hashes and vectors only, never text, in a ``0700``
directory with ``0600`` files; because vectors of holdout text are
holdout-derived material, the cache must live in the operator/evaluator's
private state (default ``$HERMES_IMPROVE_HOME/embedding-cache``).
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .config import HermesConfig

_MAX_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_CACHE_BYTES = 512 * 1024 * 1024
DEFAULT_BATCH_SIZE = 32
CACHE_SCHEMA_VERSION = 1

Vector = tuple[float, ...]


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to fixed-length vectors; deterministic for a given ``model_id``."""

    model_id: str
    sealed_holdout_trusted: bool

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(vector: Sequence[float]) -> Vector:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding vectors must be non-empty and finite")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0:
        raise ValueError("embedding vector must not be all zeros")
    return tuple(value / norm for value in values)


_sumprod = getattr(math, "sumprod", None)


def cosine(left: Vector, right: Vector) -> float:
    """Dot product of two unit vectors (cosine similarity)."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    if _sumprod is not None:
        return _sumprod(left, right)
    return math.fsum(a * b for a, b in zip(left, right))


def is_loopback_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return hostname == "localhost"


@dataclass(frozen=True)
class EmbeddingEndpointConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60
    allow_remote: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE

    def validate(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("embedding base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("embedding base URL must not contain credentials, query, or fragment")
        if not is_loopback_url(self.base_url) and not self.allow_remote:
            raise ValueError(
                "embedding endpoint must be loopback unless QWEN_EMBEDDING_ALLOW_REMOTE=true "
                "is explicitly set"
            )
        if not self.model.strip():
            raise ValueError("embedding model must be a non-empty model name")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.batch_size < 1:
            raise ValueError("embedding timeout and batch size must be positive")


class OpenAICompatibleEmbedder:
    """``POST /v1/embeddings`` client for local OpenAI-compatible servers."""

    def __init__(self, config: EmbeddingEndpointConfig) -> None:
        config.validate()
        self.config = config
        self.model_id = config.model
        self.sealed_holdout_trusted = is_loopback_url(config.base_url)
        self.requests = 0

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        size = self.config.batch_size
        for start in range(0, len(texts), size):
            vectors.extend(self._request(list(texts[start:start + size])))
        return vectors

    def _request(self, batch: list[str]) -> list[Vector]:
        if any(not isinstance(text, str) for text in batch):
            raise TypeError("embedding inputs must be strings")
        body = json.dumps({"model": self.config.model, "input": batch}).encode("utf-8")
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("embedding request exceeds the 4 MiB payload limit")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            f"{self.config.base_url}/v1/embeddings", data=body, headers=headers, method="POST",
        )
        self.requests += 1
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("embedding response exceeds the 64 MiB limit")
        try:
            payload = json.loads(raw)
            rows = payload["data"]
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("embedding response must contain one vector per input")
            ordered = sorted(rows, key=lambda row: row["index"])
            if [row["index"] for row in ordered] != list(range(len(batch))):
                raise ValueError("embedding response indexes must cover every input once")
            vectors = [row["embedding"] for row in ordered]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("embedding response has an invalid shape") from exc
        if any(
            not isinstance(vector, list)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector)
            for vector in vectors
        ):
            raise ValueError("embedding vectors must be numeric arrays")
        return [tuple(float(value) for value in vector) for vector in vectors]


class EmbeddingCache:
    """On-disk vector cache keyed by (model id, content hash); one file per model."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def path_for(self, model_id: str) -> Path:
        digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:32]
        return self.directory / f"embeddings-{digest}.json"

    def load(self, model_id: str) -> dict[str, Vector]:
        path = self.path_for(model_id)
        if not path.exists():
            return {}
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"embedding cache must be a regular file: {path}")
        with path.open("rb") as source:
            raw = source.read(_MAX_CACHE_BYTES + 1)
        if len(raw) > _MAX_CACHE_BYTES:
            raise ValueError("embedding cache exceeds 512 MiB")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"embedding cache is not valid JSON: {path}") from exc
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != CACHE_SCHEMA_VERSION
            or document.get("model_id") != model_id
            or not isinstance(document.get("vectors"), dict)
        ):
            # Foreign or outdated file: treat as empty, it is rewritten on store.
            return {}
        return {
            key: tuple(float(value) for value in vector)
            for key, vector in document["vectors"].items()
        }

    def store(self, model_id: str, vectors: Mapping[str, Vector]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError("embedding cache directory must not be a symlink")
        path = self.path_for(model_id)
        document = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_id": model_id,
            "vectors": {key: list(vectors[key]) for key in sorted(vectors)},
        }
        temporary = self.directory / f".embeddings-{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                json.dump(document, destination, sort_keys=True, separators=(",", ":"))
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise


@dataclass
class SemanticLayer:
    """Layer-5 configuration: an embedder, a cosine threshold and an optional cache."""

    embedder: Embedder
    threshold: float
    cache: EmbeddingCache | None = None
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)
    _memory: dict[str, Vector] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.embedder, Embedder):
            raise TypeError("embedder must implement the Embedder protocol")
        if (
            isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold) or not 0 < self.threshold <= 1
        ):
            raise ValueError("semantic similarity threshold must be in (0, 1]")
        if not isinstance(self.embedder.model_id, str) or not self.embedder.model_id.strip():
            raise ValueError("embedder model_id must be a non-empty string")

    @property
    def model_id(self) -> str:
        return self.embedder.model_id

    def vectors(self, texts: Sequence[str]) -> list[Vector]:
        """Unit vectors for ``texts``; cached vectors are reused, misses fetched once."""
        model_id = self.model_id
        if self._memory is None:
            self._memory = self.cache.load(model_id) if self.cache is not None else {}
        cached = self._memory
        hashes = [content_hash(text) for text in texts]
        missing: dict[str, str] = {}
        for digest, text in zip(hashes, texts):
            if digest not in cached and digest not in missing:
                missing[digest] = text
        self.cache_hits += sum(1 for digest in dict.fromkeys(hashes) if digest in cached)
        self.cache_misses += len(missing)
        if missing:
            order = sorted(missing)
            fetched = self.embedder.embed([missing[digest] for digest in order])
            if len(fetched) != len(order):
                raise ValueError("embedder returned a different number of vectors than inputs")
            for digest, vector in zip(order, fetched):
                cached[digest] = normalize(vector)
            if self.cache is not None:
                self.cache.store(model_id, cached)
        result = [normalize(cached[digest]) for digest in hashes]
        if len({len(vector) for vector in result}) > 1:
            raise ValueError("embedding dimensions differ within one model; clear the cache")
        return result


def semantic_layer_from_environment(
    config: "HermesConfig | None" = None,
    environ: Mapping[str, str] | None = None,
    *,
    cache_dir: Path | None = None,
) -> SemanticLayer | None:
    """Build the layer from ``config/models.yaml`` ``embedding`` env names, or ``None``.

    ``None`` means "not configured"; callers must report that layer 5 was not
    applied. A partially configured endpoint (only one of URL/model) is an error.
    """
    if config is None:
        from .config import config_from_environment

        config = config_from_environment()
    environ = os.environ if environ is None else environ
    names = config.models.embedding
    base_url = environ.get(names.base_url_env, "").strip().rstrip("/")
    model = environ.get(names.model_env, "").strip()
    if not base_url and not model:
        return None
    if not base_url or not model:
        raise ValueError(
            f"{names.base_url_env} and {names.model_env} must both be set to enable the "
            "semantic clustering layer"
        )
    try:
        timeout = float(environ.get(names.timeout_env, "60"))
    except ValueError as exc:
        raise ValueError(f"{names.timeout_env} must be a number") from exc
    endpoint = EmbeddingEndpointConfig(
        base_url=base_url,
        model=model,
        api_key=environ.get(names.api_key_env) or None,
        timeout=timeout,
        allow_remote=environ.get(names.allow_remote_env, "").lower() == "true",
    )
    if cache_dir is None:
        explicit = environ.get("HERMES_IMPROVE_EMBEDDING_CACHE")
        if explicit:
            cache_dir = Path(explicit)
        else:
            home = Path(environ.get("HERMES_IMPROVE_HOME", Path.home() / ".hermes-improve"))
            cache_dir = home / "embedding-cache"
    return SemanticLayer(
        OpenAICompatibleEmbedder(endpoint),
        config.scoring.clustering.semantic_similarity_threshold,
        EmbeddingCache(cache_dir),
    )
