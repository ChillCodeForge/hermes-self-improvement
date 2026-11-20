"""Layer 5 (semantic clustering): fake embedder, cache, HTTP backend, reporting, config."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

from benchmark_factory import record
from openai_stub import EmbeddingStub
from hermes_improve.benchmarks import BenchmarkStore, validate_benchmark_set
from hermes_improve.clustering import (
    cluster_texts,
    similarity,
    similarity_leakage_report,
    task_fingerprint,
)
from hermes_improve.config import ConfigError, default_config, load_config
from hermes_improve.embeddings import (
    EmbeddingCache,
    EmbeddingEndpointConfig,
    OpenAICompatibleEmbedder,
    SemanticLayer,
    semantic_layer_from_environment,
)
from hermes_improve.miner import DedupRegistry, mine_sessions
from hermes_improve.models import Benchmark
from hermes_improve.task_similarity import are_near_duplicate_tasks

HOLDOUT = "Convert the inventory CSV export into a JSON array sorted by SKU with typed numeric fields"
REWORDED = "Convert the inventory CSV export to a JSON array, sorted by SKU, with typed numeric values"
# Same task, almost no shared vocabulary: layers 1-4 cannot link it to HOLDOUT.
PARAPHRASE = "Transform our stock spreadsheet dump to ordered records keyed by product code, keeping quantities as numbers"
UNRELATED = "Revert the faulty commit that broke the greeting script while keeping later history"

# Deterministic "embedding model": words map to concepts; paraphrases share concepts.
_CONCEPTS = {
    "convert": "convert", "transform": "convert", "turn": "convert",
    "inventory": "inventory", "stock": "inventory",
    "csv": "table", "spreadsheet": "table",
    "export": "export", "dump": "export",
    "json": "records", "array": "records", "records": "records",
    "sorted": "ordered", "ordered": "ordered",
    "sku": "product-id", "product": "product-id", "code": "product-id",
    "typed": "numeric", "numeric": "numeric", "numbers": "numeric", "quantities": "numeric",
    "fields": "field", "values": "field", "keyed": "field",
    "revert": "revert", "faulty": "broken", "broke": "broken", "commit": "commit",
    "greeting": "greeting", "script": "script", "history": "history", "later": "history",
}
_DIMENSIONS = 64


def fake_vector(text: str) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    for word in re.findall(r"[a-z]+", text.lower()):
        concept = _CONCEPTS.get(word)
        if concept is not None:
            vector[hashlib.sha256(concept.encode()).digest()[0] % _DIMENSIONS] += 1.0
    vector[-1] += 0.05  # never all-zero
    return vector


class FakeEmbedder:
    def __init__(self, model_id: str = "fake-concepts-v1", *, trusted: bool = True) -> None:
        self.model_id = model_id
        self.sealed_holdout_trusted = trusted
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [fake_vector(text) for text in texts]


def layer(tmp_path: Path | None = None, *, model_id: str = "fake-concepts-v1", trusted: bool = True,
          threshold: float = 0.85) -> SemanticLayer:
    cache = EmbeddingCache(tmp_path / "cache") if tmp_path is not None else None
    return SemanticLayer(FakeEmbedder(model_id, trusted=trusted), threshold, cache)


def rows(*items: tuple[str, str, str]) -> list[Benchmark]:
    return [Benchmark.from_dict(record(bid, split, prompt=prompt)) for bid, split, prompt in items]


def test_fake_embedder_links_paraphrase_that_lexical_layers_miss():
    assert task_fingerprint(HOLDOUT) != task_fingerprint(PARAPHRASE)  # layers 1-2
    assert not are_near_duplicate_tasks(HOLDOUT, PARAPHRASE)          # layer 3
    assert similarity(HOLDOUT, PARAPHRASE) < 0.2                       # layer 4
    lexical = cluster_texts({"h": HOLDOUT, "p": PARAPHRASE, "u": UNRELATED})
    assert lexical["h"] != lexical["p"]
    merged = cluster_texts({"h": HOLDOUT, "p": PARAPHRASE, "u": UNRELATED}, semantic=layer())
    assert merged["h"] == merged["p"] != merged["u"]


def test_paraphrased_holdout_leaking_into_train_is_caught_only_by_layer_5():
    corpus = rows(("inv-holdout", "SEALED_HOLDOUT", HOLDOUT), ("inv-train", "TRAIN", PARAPHRASE),
                  ("git-task", "TRAIN", UNRELATED))
    # Layers 1-4 (ids, declared clusters, near duplicates, MinHash) pass the leak.
    lexical_only = validate_benchmark_set(corpus)
    assert lexical_only.leaks == ()
    assert lexical_only.semantic.status == "not_applied"
    # Layer 5 rejects it and names the layer and the model.
    report = similarity_leakage_report(corpus, semantic=layer())
    assert [(leak.detected_by, {m[0] for m in leak.members}) for leak in report.leaks] == [
        ("semantic", {"inv-holdout", "inv-train"}),
    ]
    assert report.semantic.status == "applied"
    assert report.semantic.model_id == "fake-concepts-v1"
    assert report.semantic.semantic_links == 1
    with pytest.raises(ValueError, match=r"spans splits \[semantic layer\].*SEALED_HOLDOUT"):
        validate_benchmark_set(corpus, semantic=layer())


def test_store_validation_rejects_semantic_holdout_leak(tmp_path):
    public = tmp_path / "public" / "train"
    public.mkdir(parents=True)
    (public / "a.json").write_text(json.dumps(record("inv-train", "TRAIN", prompt=PARAPHRASE)))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "h.json").write_text(json.dumps(record("inv-holdout", "SEALED_HOLDOUT", prompt=HOLDOUT)))
    store = BenchmarkStore(tmp_path / "public", holdout_root=holdout)
    store.validate()
    assert store.last_leakage_report.semantic.status == "not_applied"
    with pytest.raises(ValueError, match="semantic layer"):
        BenchmarkStore(tmp_path / "public", holdout_root=holdout, semantic=layer(tmp_path)).validate()


def test_clusters_are_connected_components_of_merged_lexical_and_semantic_links():
    # REWORDED-HOLDOUT is a lexical link, HOLDOUT-PARAPHRASE a semantic one.
    corpus = rows(("a", "TRAIN", REWORDED), ("b", "TRAIN", HOLDOUT), ("c", "VALIDATION", PARAPHRASE))
    assert validate_benchmark_set(corpus).leaks == ()
    report = similarity_leakage_report(corpus, semantic=layer())
    (leak,) = report.leaks
    assert {member[0] for member in leak.members} == {"a", "b", "c"}
    assert len(set(report.clusters.values())) == 1
    # A lexical leak stays attributed to the lexical layers.
    lexical = similarity_leakage_report(
        rows(("a", "TRAIN", REWORDED), ("b", "VALIDATION", HOLDOUT)), semantic=layer(),
    )
    assert [leak.detected_by for leak in lexical.leaks] == ["lexical"]


def test_mined_paraphrase_inherits_corpus_cluster_so_split_assignment_is_consistent():
    corpus = [Benchmark.from_dict(record("inventory", "TRAIN", "inventory-cluster", prompt=HOLDOUT))]
    sessions = [{"session_id": "s1", "task": {"prompt": PARAPHRASE}}]
    (without,) = mine_sessions(sessions, corpus=corpus)
    assert without.cluster_id != "inventory-cluster"
    reworded, mined = mine_sessions(
        [{"session_id": "s0", "task": {"prompt": REWORDED}}, *sessions], corpus=corpus, semantic=layer(),
    )
    assert reworded.cluster_id == mined.cluster_id == "inventory-cluster"
    registry = DedupRegistry()
    assert registry.assign(reworded, "TRAIN") == ("TRAIN", True)
    assert registry.assign(mined, "SEALED_HOLDOUT") == ("TRAIN", False)


def test_cache_hit_miss_and_reproducible_results(tmp_path):
    corpus = rows(("h", "SEALED_HOLDOUT", HOLDOUT), ("p", "TRAIN", PARAPHRASE), ("u", "TRAIN", UNRELATED))
    first_layer = layer(tmp_path)
    first = similarity_leakage_report(corpus, semantic=first_layer)
    assert (first.semantic.cache_misses, first.semantic.cache_hits) == (3, 0)
    assert len(first_layer.embedder.calls) == 1

    second_layer = layer(tmp_path)  # new process: only the on-disk cache is shared
    second = similarity_leakage_report(corpus, semantic=second_layer)
    assert (second.semantic.cache_misses, second.semantic.cache_hits) == (0, 3)
    assert second_layer.embedder.calls == []  # no re-query
    assert second.leaks == first.leaks and dict(second.clusters) == dict(first.clusters)

    # A new task only embeds the missing text.
    third_layer = layer(tmp_path)
    similarity_leakage_report([*corpus, *rows(("r", "TRAIN", REWORDED))], semantic=third_layer)
    assert third_layer.embedder.calls == [[REWORDED]]

    (cache_file,) = (tmp_path / "cache").iterdir()
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "cache").stat().st_mode) & 0o077 == 0
    content = cache_file.read_text()
    assert "inventory" not in content and "stock" not in content  # hashes and vectors only
    document = json.loads(content)
    assert document["model_id"] == "fake-concepts-v1"
    assert hashlib.sha256(HOLDOUT.encode()).hexdigest() in document["vectors"]


def test_model_id_change_invalidates_the_cache(tmp_path):
    texts = [HOLDOUT, PARAPHRASE]
    layer(tmp_path).vectors(texts)
    changed = layer(tmp_path, model_id="fake-concepts-v2")
    changed.vectors(texts)
    assert (changed.cache_misses, changed.cache_hits) == (2, 0)
    assert changed.embedder.calls == [sorted(texts, key=lambda t: hashlib.sha256(t.encode()).hexdigest())]
    assert len(list((tmp_path / "cache").iterdir())) == 2
    # A cache file claiming another model id is ignored, not trusted.
    cache = EmbeddingCache(tmp_path / "cache")
    path = cache.path_for("fake-concepts-v1")
    document = json.loads(path.read_text())
    document["model_id"] = "someone-else"
    path.write_text(json.dumps(document))
    assert cache.load("fake-concepts-v1") == {}


def test_untrusted_backend_never_receives_sealed_holdout_text_and_reports_partial():
    untrusted = layer(trusted=False)
    corpus = rows(("h", "SEALED_HOLDOUT", HOLDOUT), ("p", "TRAIN", PARAPHRASE), ("u", "VALIDATION", UNRELATED))
    report = similarity_leakage_report(corpus, semantic=untrusted)
    sent = [text for call in untrusted.embedder.calls for text in call]
    assert HOLDOUT not in sent and sorted(sent) == sorted([PARAPHRASE, UNRELATED])
    assert report.semantic.status == "partial"
    assert report.semantic.excluded_sealed_holdout == 1
    assert "SEALED_HOLDOUT" in report.semantic.warning()
    assert report.leaks == ()  # the leak is not detectable without embedding the holdout


def test_http_backend_against_local_stub(tmp_path):
    with EmbeddingStub(fake_vector, reverse=True) as stub:
        embedder = OpenAICompatibleEmbedder(EmbeddingEndpointConfig(
            base_url=stub.base_url, model="Qwen3-Embedding-0.6B", api_key="local-test-key", batch_size=2,
        ))
        assert embedder.sealed_holdout_trusted
        semantic = SemanticLayer(embedder, 0.85, EmbeddingCache(tmp_path / "cache"))
        corpus = rows(("h", "SEALED_HOLDOUT", HOLDOUT), ("p", "TRAIN", PARAPHRASE), ("u", "TRAIN", UNRELATED))
        report = similarity_leakage_report(corpus, semantic=semantic)
        assert [leak.detected_by for leak in report.leaks] == ["semantic"]
        assert report.semantic.model_id == "Qwen3-Embedding-0.6B"
        assert len(stub.requests) == 2  # 3 texts in batches of 2
        assert all(request["model"] == "Qwen3-Embedding-0.6B" for request in stub.requests)
        assert stub.headers[0]["Authorization"] == "Bearer local-test-key"
        assert sorted(stub.inputs()) == sorted([HOLDOUT, PARAPHRASE, UNRELATED])
        # Second run is served from the cache.
        again = SemanticLayer(OpenAICompatibleEmbedder(embedder.config), 0.85, EmbeddingCache(tmp_path / "cache"))
        assert similarity_leakage_report(corpus, semantic=again).leaks == report.leaks
        assert len(stub.requests) == 2


def test_http_backend_rejects_bad_responses():
    with EmbeddingStub(fake_vector, failure=(200, json.dumps({"data": [{"index": 0}]}))) as stub:
        embedder = OpenAICompatibleEmbedder(EmbeddingEndpointConfig(stub.base_url, "m"))
        with pytest.raises(ValueError, match="invalid shape"):
            embedder.embed(["x"])
    with EmbeddingStub(fake_vector, failure=(500, "boom")) as stub:
        embedder = OpenAICompatibleEmbedder(EmbeddingEndpointConfig(stub.base_url, "m"))
        with pytest.raises(RuntimeError, match="embedding request failed"):
            embedder.embed(["x"])
    with EmbeddingStub(lambda text: [0.0, 0.0]) as stub:
        semantic = SemanticLayer(OpenAICompatibleEmbedder(EmbeddingEndpointConfig(stub.base_url, "m")), 0.9)
        with pytest.raises(ValueError, match="all zeros"):
            semantic.vectors(["x"])


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:8000", "https://embeddings.example.com", "http://user:pw@localhost:8000",
    "http://localhost:8000?x=1", "ftp://localhost", "localhost:8000",
])
def test_endpoint_must_be_local_and_well_formed(url):
    with pytest.raises(ValueError):
        OpenAICompatibleEmbedder(EmbeddingEndpointConfig(url, "m"))


def test_remote_endpoint_needs_its_own_explicit_switch_and_is_never_holdout_trusted(monkeypatch):
    monkeypatch.setenv("QWEN_ALLOW_REMOTE", "true")  # chat switch must not widen embeddings
    environ = {"QWEN_EMBEDDING_BASE_URL": "https://gpu.internal:8443", "QWEN_EMBEDDING_MODEL": "m",
               "QWEN_ALLOW_REMOTE": "true", "HERMES_IMPROVE_HOME": "/nonexistent"}
    with pytest.raises(ValueError, match="QWEN_EMBEDDING_ALLOW_REMOTE"):
        semantic_layer_from_environment(default_config(), environ)
    semantic = semantic_layer_from_environment(
        default_config(), environ | {"QWEN_EMBEDDING_ALLOW_REMOTE": "true"},
    )
    assert semantic is not None and semantic.embedder.sealed_holdout_trusted is False


def test_not_configured_is_none_and_half_configured_is_an_error(tmp_path):
    assert semantic_layer_from_environment(default_config(), {}) is None
    with pytest.raises(ValueError, match="must both be set"):
        semantic_layer_from_environment(default_config(), {"QWEN_EMBEDDING_MODEL": "m"})
    semantic = semantic_layer_from_environment(default_config(), {
        "QWEN_EMBEDDING_BASE_URL": "http://127.0.0.1:9/", "QWEN_EMBEDDING_MODEL": "m",
        "HERMES_IMPROVE_HOME": str(tmp_path),
    })
    assert semantic.threshold == 0.85 and semantic.model_id == "m"
    assert semantic.cache.directory == tmp_path / "embedding-cache"
    assert semantic.embedder.config.base_url == "http://127.0.0.1:9"


def _write_corpus(tmp_path: Path) -> tuple[Path, Path]:
    public = tmp_path / "public"
    (public / "train").mkdir(parents=True)
    (public / "train" / "a.json").write_text(json.dumps(record("inv-train", "TRAIN", prompt=PARAPHRASE)))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "h.json").write_text(json.dumps(record("inv-holdout", "SEALED_HOLDOUT", prompt=HOLDOUT)))
    return public, holdout


def _cli(monkeypatch, *argv: str) -> int:
    from hermes_improve import cli

    # Factory records are not curated-corpus quality; the lint is not under test here.
    monkeypatch.setattr(cli, "lint_public_corpus", lambda rows: None)
    monkeypatch.setattr(sys, "argv", ["hermes-improve", *argv])
    return cli.run()


def test_cli_validate_reports_semantic_layer_not_applied(tmp_path, monkeypatch, capsys):
    public, holdout = _write_corpus(tmp_path)
    for name in ("QWEN_EMBEDDING_BASE_URL", "QWEN_EMBEDDING_MODEL", "HERMES_IMPROVE_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    assert _cli(monkeypatch, "benchmarks", "validate", "--holdout-root", str(holdout)) in (0, None)
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["valid_benchmarks"] == 2
    assert output["semantic_layer"]["status"] == "not_applied"
    assert "semantic clustering layer NOT applied" in captured.err


def test_cli_validate_with_configured_endpoint_rejects_paraphrase_leak(tmp_path, monkeypatch, capsys):
    public, holdout = _write_corpus(tmp_path)
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    with EmbeddingStub(fake_vector) as stub:
        monkeypatch.setenv("QWEN_EMBEDDING_BASE_URL", stub.base_url)
        monkeypatch.setenv("QWEN_EMBEDDING_MODEL", "Qwen3-Embedding-0.6B")
        with pytest.raises(ValueError, match="semantic layer"):
            _cli(monkeypatch, "benchmarks", "validate", "--holdout-root", str(holdout))
        assert (tmp_path / "home" / "embedding-cache").is_dir()
        # Public-only validation passes and records the model id.
        assert _cli(monkeypatch, "benchmarks", "validate") in (0, None)
        output = json.loads(capsys.readouterr().out)
        assert output["semantic_layer"]["status"] == "applied"
        assert output["semantic_layer"]["model_id"] == "Qwen3-Embedding-0.6B"


SHIPPED = Path(__file__).parents[2] / "config"


def _config_copy(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(SHIPPED, target)
    for path in [target, *target.iterdir()]:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    return target


@pytest.mark.parametrize(("filename", "old", "new", "message"), [
    ("scoring.yaml", "semantic_similarity_threshold: 0.85", "semantic_similarity_threshold: 0", r"\(0, 1\]"),
    ("scoring.yaml", "semantic_similarity_threshold: 0.85", "semantic_similarity_threshold: 1.5", r"\(0, 1\]"),
    ("scoring.yaml", "semantic_similarity_threshold: 0.85", "semantic_similarity_threshold: true", r"\(0, 1\]"),
    ("scoring.yaml", "semantic_similarity_threshold: 0.85", "semantic_similarity_threshold: .nan", r"\(0, 1\]"),
    ("scoring.yaml", "semantic_similarity_threshold: 0.85",
     "semantic_similarity_threshold: 0.85\n  extra: 1", "unknown extra"),
    ("models.yaml", "model_env: QWEN_EMBEDDING_MODEL", "model_env: Qwen3-Embedding-8B", "not a model name"),
    ("models.yaml", "base_url_env: QWEN_EMBEDDING_BASE_URL", "base_url_env: http://localhost:8000", "QWEN_"),
    ("models.yaml", "  allow_remote_env: QWEN_EMBEDDING_ALLOW_REMOTE\n", "", "missing allow_remote_env"),
])
def test_invalid_semantic_config_is_rejected(tmp_path, filename, old, new, message):
    root = _config_copy(tmp_path)
    path = root / filename
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(root)


def test_semantic_config_sections_are_optional_and_threshold_flows_through(tmp_path):
    root = _config_copy(tmp_path)
    scoring = (root / "scoring.yaml").read_text(encoding="utf-8")
    (root / "scoring.yaml").write_text(
        scoring.replace("semantic_similarity_threshold: 0.85", "semantic_similarity_threshold: 0.9"),
        encoding="utf-8",
    )
    loaded = load_config(root)
    assert loaded.scoring.clustering.semantic_similarity_threshold == 0.9
    semantic = semantic_layer_from_environment(loaded, {
        "QWEN_EMBEDDING_BASE_URL": "http://localhost:1", "QWEN_EMBEDDING_MODEL": "m",
        "HERMES_IMPROVE_EMBEDDING_CACHE": str(tmp_path / "c"),
    })
    assert semantic.threshold == 0.9 and semantic.cache.directory == tmp_path / "c"

    legacy = _config_copy(tmp_path / "legacy")
    for name, marker in (("scoring.yaml", "# Semantic clustering layer"),
                         ("models.yaml", "# Optional embedding endpoint")):
        text = (legacy / name).read_text(encoding="utf-8")
        (legacy / name).write_text(text[:text.index(marker)], encoding="utf-8")
    old = load_config(legacy)
    assert old.scoring == default_config().scoring and old.models == default_config().models


@pytest.mark.parametrize("threshold", [0, -0.1, 1.01, float("nan"), True])
def test_semantic_layer_validates_threshold_and_embedder(threshold):
    with pytest.raises(ValueError):
        SemanticLayer(FakeEmbedder(), threshold)
    with pytest.raises(TypeError):
        SemanticLayer(object(), 0.85)
