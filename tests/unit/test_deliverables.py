"""F24: subjective criteria can declare deliverable files that the judge sees."""

from __future__ import annotations

import json
import os

import pytest

from benchmark_factory import record
from hermes_improve.benchmark_schema import benchmark_json_schema, parse_benchmark_document
from hermes_improve.deliverables import (
    MAX_DELIVERABLE_BYTES,
    collect_deliverables,
    declared_deliverables,
    validate_deliverables,
)
from hermes_improve.judge import PairwiseJudge
from hermes_improve.models import Benchmark


def judge_record(subjective):
    data = record("file-deliverable", categories=["docs"])
    data["verification"] = {"deterministic": [], "subjective": subjective}
    data["ground_truth"] = {"type": "llm_judge", "confidence": 0.9}
    return data


def test_criterion_objects_declare_deliverable_files():
    benchmark = parse_benchmark_document(judge_record([
        {"criterion": "Has an INCIDENT headline.", "files": ["incident.md"]},
        {"criterion": "Lists follow-ups.", "files": ["incident.md", "docs/follow-up.md"]},
        "Blameless.",
    ]))
    assert declared_deliverables(benchmark.verification) == ("incident.md", "docs/follow-up.md")


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (["../escape.md"], "escape"),
        (["/etc/passwd"], "escape"),
        (["a\\b.md"], "relative POSIX"),
        (["./incident.md"], "normalized"),
        (["incident.md", "incident.md"], "unique"),
        ([], "1-8"),
        ("incident.md", "1-8"),
        ([f"f{i}.md" for i in range(9)], "1-8"),
        ([3], "relative POSIX"),
    ],
)
def test_invalid_deliverable_paths_are_rejected_at_validation(files, message):
    with pytest.raises(ValueError, match=message):
        Benchmark.from_dict(judge_record([{"criterion": "Good summary.", "files": files}]))


def test_empty_or_malformed_criteria_are_rejected_at_validation():
    for criterion in ("  ", {"files": ["a.md"]}, 7):
        with pytest.raises(ValueError, match="subjective criteri"):
            Benchmark.from_dict(judge_record([criterion]))


def test_json_schema_exports_the_criterion_object():
    definitions = benchmark_json_schema()["$defs"]
    assert {"criterion", "files"} <= set(definitions["SubjectiveCriterion"]["properties"])


def test_collect_reads_bounded_utf8_files_inside_the_workspace_only(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "incident.md").write_text("INCIDENT: api - 503\n", encoding="utf-8")
    (workspace / "docs" / "big.md").write_bytes(b"x" * (MAX_DELIVERABLE_BYTES + 1))
    (workspace / "binary.bin").write_bytes(b"\xff\xfe\x00")
    (workspace / "dir").mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    os.symlink(secret, workspace / "link.md")
    rows = {
        row["path"]: row for row in collect_deliverables(
            ["incident.md", "docs/big.md", "binary.bin", "dir", "link.md", "absent.md"], workspace,
        )
    }
    assert rows["incident.md"]["status"] == "ok"
    assert rows["incident.md"]["content"] == "INCIDENT: api - 503\n"
    assert len(rows["incident.md"]["sha256"]) == 64
    assert rows["docs/big.md"]["status"] == "too_large" and rows["docs/big.md"]["content"] == ""
    assert rows["binary.bin"]["status"] == "not_utf8" and rows["binary.bin"]["content"] == ""
    assert rows["dir"]["status"] == "not_a_file"
    assert rows["link.md"]["status"] == "not_a_file" and rows["link.md"]["content"] == ""
    assert rows["absent.md"]["status"] == "missing"
    assert validate_deliverables(tuple(rows.values()))
    with pytest.raises(ValueError):
        collect_deliverables(["../secret.txt"], workspace)


def test_collect_redacts_credentials(tmp_path):
    (tmp_path / "notes.md").write_text("token ghp_" + "a" * 36 + "\n")
    [row] = collect_deliverables(["notes.md"], tmp_path)
    assert "ghp_" + "a" * 36 not in row["content"]


def test_validate_deliverables_rejects_malformed_rows():
    good = {"path": "a.md", "status": "ok", "bytes": 1, "sha256": "", "content": "a"}
    assert validate_deliverables((good,))
    assert not validate_deliverables([good])
    assert not validate_deliverables((good, good))
    assert not validate_deliverables(({**good, "status": "weird"},))
    assert not validate_deliverables(({**good, "status": "missing"},))
    assert not validate_deliverables(({**good, "extra": 1},))


class RecordingModel:
    def __init__(self):
        self.payloads = []

    def complete(self, *, role, system_prompt, user_payload, json_object=False):
        payload = json.loads(user_payload)
        self.payloads.append(payload)
        winner = "A" if "good" in json.dumps(payload.get("files_A", [])) else "B"
        return json.dumps({"winner": winner, "confidence": 0.9, "uncertainty": 0.1, "rationale": "r"})


def test_judge_receives_deliverable_files_of_both_arms_in_both_orders():
    model = RecordingModel()
    judge = PairwiseJudge(model)
    base = ({"path": "incident.md", "status": "ok", "bytes": 3, "sha256": "", "content": "bad"},)
    cand = ({"path": "incident.md", "status": "ok", "bytes": 4, "sha256": "", "content": "good"},)
    verdict = judge.compare(
        task_prompt="Write incident.md", criteria=["headline"], baseline_output="done", candidate_output="done",
        baseline_files=base, candidate_files=cand,
    )
    assert verdict.result == "candidate"
    first, second = model.payloads
    assert first["response_A"] == "done" and first["files_A"][0]["content"] == "bad"
    assert first["files_B"][0]["content"] == "good" and second["files_A"][0]["content"] == "good"
    assert set(first["files_A"][0]) == {"path", "status", "content"}


def test_judge_payload_is_unchanged_without_deliverables():
    model = RecordingModel()
    PairwiseJudge(model).compare(task_prompt="t", criteria=["c"], baseline_output="a", candidate_output="b")
    assert set(model.payloads[0]) == {"task", "criteria", "response_A", "response_B"}


def test_judge_system_prompt_frames_files_as_untrusted_evidence():
    prompt = PairwiseJudge(RecordingModel()).system_prompt
    assert '"files_A"' in prompt and "untrusted" in prompt
