"""Brain-consuming (``skill_procedure``) benchmarks: records, trusted inputs, replay and runners."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from hermes_improve.artifacts import ArtifactStore
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.docker_runner import (
    DockerCommandRunner,
    LocalSkillProcedureRunner,
    is_skill_procedure_benchmark,
)
from hermes_improve.models import Benchmark, Eligibility, Split
from hermes_improve.replay import fixture_tree_digest, replay_benchmark
from hermes_improve.sandbox import SandboxPolicyError
from skill_procedure_support import (
    BENCHMARKS,
    CHECKS,
    HARNESS,
    IMAGE,
    REPO_ROOT,
    SKILL_TARGET,
    deterministic_git_env,
    reference_repository,
)


def skill_records():
    return [item for item in BenchmarkStore(BENCHMARKS).validate() if is_skill_procedure_benchmark(item)]


def local_runner(repository, tmp_path, **kwargs):
    return LocalSkillProcedureRunner(
        repository, ArtifactStore(tmp_path / "artifacts"), benchmark_root=BENCHMARKS,
        acknowledge_no_isolation=True, **kwargs,
    )


def brain_repository(tmp_path, skill_text, *, extra_files=None):
    repository = tmp_path / "brain-repo"
    repository.mkdir()
    env = deterministic_git_env()
    subprocess.run(["git", "-C", str(repository), "init", "-q", "-b", "main"], check=True, env=env)
    (repository / SKILL_TARGET).mkdir(parents=True)
    (repository / SKILL_TARGET / "SKILL.md").write_text(skill_text, encoding="utf-8")
    for path, content in (extra_files or {}).items():
        (repository / path).parent.mkdir(parents=True, exist_ok=True)
        (repository / path).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(repository), "commit", "-q", "-m", "brain"], check=True, env=env)
    commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True, env=env).stdout.strip()
    return repository, commit


def test_skill_procedure_records_cover_public_splits_and_pin_trusted_inputs(tmp_path):
    records = skill_records()
    assert len(records) == 6
    assert {item.split for item in records} == {Split.TRAIN, Split.VALIDATION, Split.REGRESSION}
    _, reference_commit = reference_repository(tmp_path / "reference")
    harness_digest = fixture_tree_digest(BENCHMARKS / HARNESS)
    for item in records:
        environment = item.environment
        fixture = BENCHMARKS / environment["repository_fixture"]
        assert environment["input_data"] == fixture_tree_digest(fixture), item.id
        assert environment["harness_digest"] == harness_digest, item.id
        assert environment["command"] == ["sh", "harness/skill-executor.sh", "git-commit-hygiene"]
        assert environment["container_image"] == IMAGE and environment["network_policy"] == "disabled"
        for name in ("setup.sh", "check.sh", "intent", "expected-paths", "strays"):
            assert (fixture / name).is_file(), f"{item.id}: missing {name}"
        assert Eligibility(item.eligibility["class"]) is Eligibility.HARD_BENCHMARK
        evidence = item.eligibility["replay_evidence"]
        assert evidence["commit"] == reference_commit, item.id
        assert evidence["passed"] and evidence["reproducible"] and evidence["runs"] >= 3
        assert item.benchmark["impact_areas"] == ["skills/git-commit-hygiene"]
        checks = [entry for entry in item.verification["deterministic"] if entry.get("path", [None])[0] == "results"]
        assert [entry["path"][1] for entry in checks] == list(CHECKS)


def test_local_replay_reproduces_the_committed_docker_replay_evidence(tmp_path):
    repository, commit = reference_repository(tmp_path / "reference")
    runner = local_runner(repository, tmp_path)
    for item in skill_records():
        result = replay_benchmark(runner, item, commit=commit, runs=2)
        evidence = item.eligibility["replay_evidence"]
        assert result.replay_verified, item.id
        assert result.environment_digest == evidence["environment_digest"], item.id
        assert result.outcome_digest == evidence["outcome_digest"], item.id


def _results(observation):
    return json.loads(observation.final_output)["results"]


def test_outcomes_depend_on_the_skill_text(tmp_path):
    """The repository's baseline skill misses only the ticket trailer; a broken procedure regresses."""
    baseline_text = (REPO_ROOT / SKILL_TARGET / "SKILL.md").read_text(encoding="utf-8")
    repository, baseline = brain_repository(tmp_path, baseline_text)
    runner = local_runner(repository, tmp_path)
    for item in skill_records():
        observation = runner.run(baseline, item, 0)
        results = _results(observation)
        ticket_task = "ticket" in item.id
        assert observation.success is (not ticket_task), item.id
        assert [name for name, ok in results.items() if not ok] == (["ticket_reference"] if ticket_task else [])
        assert observation.quality == pytest.approx(100.0 if not ticket_task else 100.0 * 8 / 11)
    trace = json.loads(ArtifactStore(tmp_path / "artifacts").get(observation.artifact_digests[0]))
    assert trace["isolation"] == "local-process"
    assert trace["filesystem_changes"] == ["A trace.log"]


def test_missing_or_malformed_procedure_executes_nothing(tmp_path):
    text = "---\nname: git-commit-hygiene\ndescription: prose only\n---\n\nCommit carefully.\n"
    repository, commit = brain_repository(tmp_path, text)
    item = next(item for item in skill_records() if item.id == "skill-commit-config-fix")
    observation = local_runner(repository, tmp_path).run(commit, item, 0)
    assert not observation.success
    assert _results(observation)["one_new_commit"] is False


def test_candidate_commit_cannot_replace_the_harness_or_fixture(tmp_path):
    """Only brain/ is taken from the evaluated commit; harness and fixture come from the evaluator root."""
    forged = 'printf \'{"checks": 8, "failed": [], "results": {}, "status": "pass"}\\n\'\n'
    repository, commit = brain_repository(
        tmp_path, "---\nname: git-commit-hygiene\ndescription: empty\n---\n",
        extra_files={
            "harness/skill-executor.sh": forged,
            "benchmarks/fixtures/skill-harness/skill-executor.sh": forged,
            "fixture/check.sh": forged,
        },
    )
    item = next(item for item in skill_records() if item.id == "skill-commit-config-fix")
    observation = local_runner(repository, tmp_path).run(commit, item, 0)
    assert not observation.success
    assert json.loads(observation.final_output)["status"] == "fail"


def test_tampered_trusted_fixture_is_refused(tmp_path):
    root = tmp_path / "benchmarks"
    shutil.copytree(BENCHMARKS / "fixtures" / "skill-harness", root / "fixtures" / "skill-harness")
    shutil.copytree(BENCHMARKS / "fixtures" / "skill-commit-config-fix", root / "fixtures" / "skill-commit-config-fix")
    (root / "fixtures" / "skill-commit-config-fix" / "check.sh").write_text("echo pass\n")
    repository, commit = reference_repository(tmp_path / "reference")
    item = next(item for item in skill_records() if item.id == "skill-commit-config-fix")
    runner = LocalSkillProcedureRunner(
        repository, ArtifactStore(tmp_path / "artifacts"), benchmark_root=root, acknowledge_no_isolation=True,
    )
    with pytest.raises(SandboxPolicyError, match="input_data"):
        runner.run(commit, item, 0)


def test_local_runner_refuses_candidate_commands_and_requires_acknowledgement(tmp_path):
    repository, commit = reference_repository(tmp_path / "reference")
    command = next(item for item in BenchmarkStore(BENCHMARKS).validate() if not is_skill_procedure_benchmark(item))
    with pytest.raises(SandboxPolicyError, match="only skill_procedure"):
        local_runner(repository, tmp_path).run(commit, command, 0)
    with pytest.raises(SandboxPolicyError, match="no container isolation"):
        LocalSkillProcedureRunner(repository, ArtifactStore(tmp_path / "a"), benchmark_root=BENCHMARKS)


def test_holdout_fixtures_resolve_only_against_the_holdout_root(tmp_path):
    from skill_procedure_support import write_holdout_task

    holdout = tmp_path / "holdout"
    record = write_holdout_task(
        holdout, "holdout-commit-cache-cleanup", harness_digest=fixture_tree_digest(BENCHMARKS / HARNESS),
    )
    item = Benchmark.from_dict(record)
    repository, commit = reference_repository(tmp_path / "reference")
    with pytest.raises(ValueError, match="repository_fixture"):
        local_runner(repository, tmp_path).run(commit, item, 0)  # no holdout root configured
    observation = local_runner(repository, tmp_path, holdout_root=holdout).run(commit, item, 0)
    assert observation.success


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode == 0


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="Docker and the pinned alpine/git image are required")
def test_docker_runner_matches_local_runner_outcomes(tmp_path):
    repository, commit = reference_repository(tmp_path / "reference")
    docker = DockerCommandRunner(repository, ArtifactStore(tmp_path / "docker-artifacts"), benchmark_root=BENCHMARKS)
    local = local_runner(repository, tmp_path)
    for item in skill_records():
        in_docker, in_process = docker.run(commit, item, 0), local.run(commit, item, 0)
        assert in_docker.success and in_process.success, item.id
        assert in_docker.environment_digest == in_process.environment_digest
        assert in_docker.final_output == in_process.final_output
