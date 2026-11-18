"""LIGHT dreaming groups similar error messages deterministically (no LLM, no embeddings)."""

from __future__ import annotations

import itertools

from hermes_improve.dreaming import DreamingEngine, ErrorCluster, normalize_error


def _sessions(*errors: list[str]) -> list[dict]:
    return [{"errors": list(batch)} for batch in errors]


def test_normalize_error_masks_paths_numbers_and_ids():
    assert normalize_error("FileNotFoundError: /tmp/run-42/out.json") == normalize_error(
        "FileNotFoundError: C:\\work\\x\\result.json"
    )
    assert normalize_error("Timeout after 30s on port 8080") == normalize_error("timeout after 5s on port 9000")
    assert normalize_error("job 3f2a9c1e7b4d5a60 failed") == normalize_error("job 0aa1bb2cc3dd4ee5 failed")
    assert normalize_error(
        "session 123e4567-e89b-12d3-a456-426614174000 lost"
    ) == normalize_error("session 00000000-0000-0000-0000-000000000000 lost")


def test_similar_errors_merge_into_one_cluster():
    report = DreamingEngine().light(_sessions(
        ["FileNotFoundError: /tmp/run-1/out.json", "FileNotFoundError: /tmp/run-2/out.json"],
        ["FileNotFoundError: /home/u/project/data/other.csv"],
        ["Command timed out after 30 seconds", "Command timed out after 120 seconds"],
    ))
    assert report.clusters["errors"] == 2
    assert [cluster.count for cluster in report.error_clusters] == [3, 2]
    file_cluster = report.error_clusters[0]
    assert file_cluster.variants == 3
    assert file_cluster.representative.startswith("filenotfounderror: ")
    assert report.signals[0].issue == file_cluster.representative
    assert report.signals[0].evidence == {
        "occurrences": 3, "cluster_id": file_cluster.cluster_id, "variants": 3,
    }


def test_lexically_similar_wording_merges_without_masking():
    report = DreamingEngine().light(_sessions(
        ["connection reset by peer while reading model response"],
        ["connection reset by peer while reading the model response"],
    ))
    assert report.clusters["errors"] == 1
    assert report.error_clusters[0].count == 2
    assert report.error_clusters[0].variants == 2


def test_distinct_errors_stay_separate():
    report = DreamingEngine().light(_sessions(
        ["permission denied writing lockfile", "permission denied writing lockfile"],
        ["connection refused by upstream model endpoint"],
        ["json decode error in tool response payload"],
    ))
    assert report.clusters["errors"] == 3
    assert sorted(cluster.count for cluster in report.error_clusters) == [1, 1, 2]
    assert len({cluster.cluster_id for cluster in report.error_clusters}) == 3
    assert len(report.signals) == 1  # only the recurring cluster becomes a signal


def test_tool_failures_are_keyed_by_tool_name_and_never_merged():
    report = DreamingEngine().light([{"tool_stats": {
        "read_file": {"count": 2, "failure": 2},
        "write_file": {"count": 3, "failure": 2},
    }}])
    issues = sorted(signal.issue for signal in report.signals)
    assert issues == ["tool read_file failed", "tool write_file failed"]
    assert report.clusters == {"errors": 2, "tools": 2}


def test_clustering_is_deterministic_across_input_order():
    batches = [
        ["FileNotFoundError: /tmp/a.json", "Timeout after 3s"],
        ["FileNotFoundError: /var/b/c.json"],
        ["Timeout after 10s", "disk quota exceeded"],
        ["FileNotFoundError: /tmp/a.json"],
    ]
    reference = DreamingEngine().light(_sessions(*batches))
    for permutation in itertools.permutations(batches):
        shuffled = [list(reversed(batch)) for batch in permutation]
        report = DreamingEngine().light(_sessions(*shuffled))
        assert report.error_clusters == reference.error_clusters
        assert report.signals == reference.signals
        assert report.clusters == reference.clusters
    for cluster in reference.error_clusters:
        assert cluster.cluster_id.startswith("err-")
    # The most frequent raw variant represents the cluster.
    assert reference.error_clusters[0].representative == "filenotfounderror: /tmp/a.json"


def test_empty_input_yields_empty_report():
    report = DreamingEngine().light([])
    assert report.sessions_processed == 0
    assert report.clusters == {"errors": 0, "tools": 0}
    assert report.error_clusters == ()
    assert report.signals == ()


def test_error_cluster_is_serialisable():
    cluster = ErrorCluster("err-0123456789abcdef", "boom", 2, 1)
    assert cluster.to_dict() == {
        "cluster_id": "err-0123456789abcdef", "representative": "boom", "count": 2, "variants": 1,
    }
