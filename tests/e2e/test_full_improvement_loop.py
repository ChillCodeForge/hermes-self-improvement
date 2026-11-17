"""Automated end-to-end improvement loop (spec §39) with real components only.

Every step runs the production code paths: the ``hermes-improve`` CLI
(``candidate create``, ``evaluate``, ``promote``, ``monitor ingest``,
``releases events``, ``rollback``), real Git repositories, the real
``ExperimentStore``/``ArtifactStore``, real Ed25519 evaluator keys and HMAC
manifest keys, ``PairedEvaluator`` with ``PromotionPolicy``/``RunPolicy`` from
the shipped ``config/`` files (with ``require_token_measurement: false``, see
``install_policy_config``), ``GitPromoter``, ``ReleaseMonitor`` and rollback.

Outcomes come from the brain-consuming ``skill_procedure`` benchmarks: the
evaluator-owned harness executes the candidate skill's declared procedure
against each task fixture, and the fixture's deterministic check grades the
resulting repository. Nothing is stubbed; the skill text is the only variable.

Story. Production starts with a legacy ``git-commit-hygiene`` skill whose
procedure only excluded editor temp files. Phase 0 promotes the repository's
baseline skill (``brain/skills/git-commit-hygiene``) as ``brain-v1`` against
the hygiene tasks. Then the operator adds ticket-trailer tasks (benchmark
version changes): the baseline misses the ``Refs:`` trailer there. Phase 1 is
the §39 sequence: a deliberately broken candidate is screened, its regression
detected and rejected without touching the sealed holdout; a better candidate
passes SMOKE, CORE, FULL and the sealed holdout, is promoted as ``brain-v2``,
observed post-promotion, and rolled back to ``brain-v1``. (A previous release
must exist for a rollback, and with deterministic tasks a promoted brain passes
every final task of its benchmark version, so a second promotion needs the
benchmark set to grow - which is how the story is staged.)

Variants:

* ``test_full_improvement_loop_single_user`` - default path. The CLI runs in
  process with ``HERMES_IMPROVE_ISOLATION=unprovisioned`` (needed because no
  provisioned role layout exists for a non-root test run) and
  ``evaluate --runner local``: the same harness in ``LocalSandboxExecutor``.
* ``test_full_improvement_loop_in_docker`` - identical flow with
  ``evaluate --runner docker`` (the real ``DockerCommandRunner``); skipped when
  Docker or the pinned ``alpine/git`` image is unavailable.
* ``test_full_improvement_loop_with_role_users`` (``root_required``) - the
  flow through ``scripts/provision.sh`` and ``scripts/hermes-improve-as`` as
  the real ``hermes-agent``/``hermes-evaluator``/``hermes-promoter`` users,
  with candidate export/import through the handoff directory. It uses
  ``--runner local`` because the test container has no Docker daemon.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve import cli
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.models import Split, canonical_json
from hermes_improve.replay import fixture_tree_digest
from hermes_improve.sandbox import LocalSandboxExecutor
from hermes_improve.store import ExperimentStore
from hermes_improve.suites import select_evaluation_suite
from skill_procedure_support import (
    BENCHMARKS,
    HARNESS,
    IMAGE,
    REPO_ROOT,
    SKILL,
    SKILL_TARGET,
    write_holdout_task,
)

HYGIENE_TASKS = ("skill-commit-config-fix", "skill-commit-docs-update", "skill-commit-leaked-credentials")
TICKET_TASKS = ("skill-commit-ticket-feature", "skill-commit-ticket-cli", "skill-commit-ticket-hotfix")
HYGIENE_HOLDOUT = ("holdout-commit-cache-cleanup",)
TICKET_HOLDOUT = ("holdout-commit-billing-ticket", "holdout-commit-metrics-ticket")
LOCAL_RUNNER = ("--runner", "local", "--acknowledge-no-container-isolation")
QUALITY_BASELINE_TICKET = 100.0 * 8 / 11  # 11 verifiers; the baseline fails exit, status, ticket_reference


# ----------------------------------------------------------------- skill texts

_BLOCK = re.compile(r"```hermes-procedure\n(.*?)```\n", re.S)


def procedure(text: str) -> list[str]:
    match = _BLOCK.search(text)
    assert match, "skill has no hermes-procedure block"
    return match.group(1).splitlines()


def with_procedure(text: str, lines: list[str]) -> str:
    return _BLOCK.sub(lambda _: "```hermes-procedure\n" + "\n".join(lines) + "\n```\n", text, count=1)


def skill_variants() -> dict[str, str]:
    baseline = (REPO_ROOT / SKILL_TARGET / "SKILL.md").read_text(encoding="utf-8")
    steps = procedure(baseline)
    unstage = next(index for index, line in enumerate(steps) if line.startswith("unstage "))
    legacy = list(steps)
    legacy[unstage] = "unstage *.tmp *.swp"  # the legacy skill only knew editor temp files
    better = list(steps)
    better.insert(better.index("commit"), "trailer ticket Refs: {ticket}")
    # Deliberately worse: "simplified" to stage everything and skip the body.
    worse = ["stage-all", "subject {type}({scope}): {summary}", "commit"]
    return {
        "baseline": baseline,
        "legacy": with_procedure(baseline, legacy),
        "better": with_procedure(baseline, better),
        "worse": with_procedure(baseline, worse),
    }


# ----------------------------------------------------------------- benchmarks

def install_public(root: Path, identifiers) -> None:
    shutil.copytree(BENCHMARKS / HARNESS, root / HARNESS, dirs_exist_ok=True)
    for identifier in identifiers:
        source = next(BENCHMARKS.glob(f"*/{identifier}-v1.json"))
        destination = root / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        shutil.copytree(BENCHMARKS / "fixtures" / identifier, root / "fixtures" / identifier)


def install_holdout(root: Path, identifiers) -> None:
    digest = fixture_tree_digest(BENCHMARKS / HARNESS)
    for identifier in identifiers:
        write_holdout_task(root, identifier, harness_digest=digest)


def git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def brain_tree(skill_text: str) -> dict[str, str]:
    """Production brain files: the repository's brain/ READMEs plus the given skill text."""
    files = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "brain").rglob("*") if path.is_file()
    }
    files[f"{SKILL_TARGET}/SKILL.md"] = skill_text
    return files


def install_policy_config(target: Path) -> None:
    """Shipped ``config/*.yaml`` with the one change this deterministic story needs.

    The ``skill_procedure`` runners execute no model, so they report token
    usage as unmeasured. With the shipped ``require_token_measurement: true``
    every final evaluation would (correctly) fail "token efficiency not
    measured". The operator of a model-free benchmark set opts out explicitly
    in protected config; nothing else differs from the shipped files.
    """
    target.mkdir(parents=True, exist_ok=True)
    for name in ("promotion.yaml", "scoring.yaml", "taxonomy.yaml", "models.yaml"):
        text = (REPO_ROOT / "config" / name).read_text(encoding="utf-8")
        if name == "promotion.yaml":
            assert "\nrequire_token_measurement: true\n" in text
            text = text.replace("\nrequire_token_measurement: true\n", "\nrequire_token_measurement: false\n")
        (target / name).write_text(text, encoding="utf-8")
        (target / name).chmod(0o644)


# ----------------------------------------------------------------- deployments

class SingleUserDeployment:
    """All roles in this process via ``cli.main``; one repository for candidates and production."""

    def __init__(self, tmp: Path, monkeypatch, capsys, runner_args) -> None:
        self.tmp, self.monkeypatch, self.capsys, self.runner_args = tmp, monkeypatch, capsys, runner_args
        self.public = tmp / "benchmarks"
        self.holdout = tmp / "holdout"
        self.home = tmp / "ledger"
        self.repository = tmp / "production"
        self.release_root = tmp / "releases"
        self.deploy_link = tmp / "deploy" / "current"
        for directory in (self.public, self.holdout, self.home):
            directory.mkdir()
        key = Ed25519PrivateKey.generate()
        self.private_key = tmp / "evaluator.pem"
        self.private_key.write_bytes(key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
        ))
        self.public_key = tmp / "evaluator.pub"
        self.public_key.write_bytes(key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
        self.manifest_key = os.urandom(32).hex()
        self.config = tmp / "config"
        install_policy_config(self.config)
        self.config.chmod(0o755)

    @property
    def ledger(self) -> Path:
        return self.home / "experiments.sqlite3"

    def cli(self, role: str, *args) -> dict:
        env = {
            "HERMES_IMPROVE_HOME": str(self.home),
            "HERMES_IMPROVE_BENCHMARKS": str(self.public),
            "HERMES_IMPROVE_CONFIG": str(self.config),
            "HERMES_IMPROVE_LAYOUT": str(self.tmp / "no-layout.json"),
            "HERMES_IMPROVE_ISOLATION": "unprovisioned",
        }
        for name in ("HERMES_EVALUATOR_UID", "HERMES_PROMOTER_UID", "HERMES_EVALUATOR_PRIVATE_KEY",
                     "HERMES_IMPROVE_MANIFEST_KEY"):
            self.monkeypatch.delenv(name, raising=False)
        if role == "evaluator":
            env |= {"HERMES_EVALUATOR_UID": str(os.geteuid()), "HERMES_EVALUATOR_PRIVATE_KEY": str(self.private_key)}
        elif role == "promoter":
            env |= {"HERMES_PROMOTER_UID": str(os.geteuid()), "HERMES_IMPROVE_MANIFEST_KEY": self.manifest_key}
        for name, value in env.items():
            self.monkeypatch.setenv(name, value)
        self.monkeypatch.setattr(sys, "argv", ["hermes-improve", *map(str, args)])
        self.capsys.readouterr()
        assert cli.main() == 0
        return json.loads(self.capsys.readouterr().out)

    def seed_production(self, files: dict[str, str]) -> str:
        self.repository.mkdir()
        git_output(self.repository, "init", "-q", "-b", "main")
        for path, content in files.items():
            (self.repository / path).parent.mkdir(parents=True, exist_ok=True)
            (self.repository / path).write_text(content, encoding="utf-8")
        git_output(self.repository, "add", "-A")
        git_output(self.repository, "-c", "user.name=Operator", "-c", "user.email=op@example.invalid",
                   "commit", "-q", "-m", "Production brain")
        return git_output(self.repository, "rev-parse", "main")

    def add_benchmarks(self, public, holdout) -> None:
        install_public(self.public, public)
        install_holdout(self.holdout, holdout)

    def create_candidate(self, experiment: str, candidate: str, content: str, hypothesis: str) -> dict:
        path = self.tmp / f"{candidate}.md"
        path.write_text(content, encoding="utf-8")
        return self.cli(
            "agent", "candidate", "create", "--repository", self.repository, "--experiment-id", experiment,
            "--candidate-id", candidate, "--type", "SKILL_PATCH", "--target", SKILL_TARGET,
            "--content", path, "--hypothesis", hypothesis,
        )

    def evaluate(self, candidate: str, suite: str) -> dict:
        return self.cli(
            "evaluator", "evaluate", candidate, "--repository", self.repository,
            "--holdout-root", self.holdout, "--suite", suite, *self.runner_args,
        )

    def _release_args(self):
        return ("--repository", self.repository, "--release-root", self.release_root,
                "--deployment-link", self.deploy_link, "--evaluator-public-key", self.public_key)

    def promote(self, experiment: str, candidate: str, release: str) -> dict:
        return self.cli("promoter", "promote", experiment, "--release", release, *self._release_args())

    def after_promotion(self) -> None:
        """Single repository: candidates already branch from the new production main."""

    def rollback(self, release: str) -> dict:
        return self.cli("promoter", "rollback", release, *self._release_args())

    def ingest_observations(self, lines: list[dict]) -> dict:
        path = self.tmp / "observations.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        return self.cli("promoter", "monitor", "ingest", path)

    def assert_release_read_only(self, release_dir: Path) -> None:
        skill = release_dir / SKILL_TARGET / "SKILL.md"
        assert release_dir.stat().st_mode & 0o222 == 0 and skill.stat().st_mode & 0o222 == 0


# --------------------------------------------------------------- shared flow

def suite_rows(report: dict, suite: str) -> dict[str, dict]:
    return {row["task_id"]: row for row in report["outcomes"] if row["suite"] == suite}


def shadow_observation(deployment, release: str, task: str) -> dict:
    """Run the deployed (read-only) release through the harness once, as post-promotion evidence."""
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "input"
        shutil.copytree(deployment.deploy_link.resolve() / "brain", source / "brain")
        shutil.copytree(deployment.public / HARNESS, source / "harness")
        shutil.copytree(deployment.public / "fixtures" / task, source / "fixture")
        result = LocalSandboxExecutor(acknowledge_no_isolation=True).execute(
            ["sh", "harness/skill-executor.sh", SKILL], input_root=source,
        )
    verdict = json.loads(result.output)
    return {
        "observation_id": f"shadow-{release}-{task}", "release_id": release, "task_id": task,
        "success": verdict["status"] == "pass", "tool_errors": 0, "tokens": 0, "tool_calls": 0,
        "latency_ms": round(result.duration_ms, 3), "skills": [SKILL], "source": "shadow_replay",
    }


def run_full_improvement_loop(deployment) -> dict:
    texts = skill_variants()
    evidence: dict = {}

    # Phase 0: production runs the legacy skill; promote the repository baseline as brain-v1.
    deployment.add_benchmarks(HYGIENE_TASKS, HYGIENE_HOLDOUT)
    legacy_commit = deployment.seed_production(brain_tree(texts["legacy"]))
    deployment.create_candidate("exp-baseline", "cand-baseline", texts["baseline"],
                                "exclude secrets and generated output from commits")
    for suite in ("SMOKE", "CORE", "FULL"):
        report = deployment.evaluate("cand-baseline", suite)
    assert report["passed"], report["reasons"]
    v1 = deployment.promote("exp-baseline", "cand-baseline", "brain-v1")
    deployment.after_promotion()
    assert v1["baseline_commit"] == legacy_commit
    v1_commit = v1["candidate_commit"]

    # The operator adds ticket-trailer tasks (public) and new sealed holdout tasks.
    deployment.add_benchmarks(TICKET_TASKS, TICKET_HOLDOUT)

    # Step 1: load the baseline skill from the deployed production brain.
    deployed_skill = (deployment.deploy_link / SKILL_TARGET / "SKILL.md").read_text(encoding="utf-8")
    assert deployed_skill == texts["baseline"]
    assert "trailer ticket Refs: {ticket}" not in procedure(deployed_skill)

    # Step 2: load the benchmark tasks (evaluator view: public + sealed holdout).
    store = BenchmarkStore(deployment.public, holdout_root=deployment.holdout)
    tasks = store.validate()
    splits = {split: sorted(item.id for item in tasks if item.split is split) for split in Split}
    assert {split.value: len(ids) for split, ids in splits.items() if ids} == {
        "TRAIN": 2, "VALIDATION": 2, "REGRESSION": 2, "SEALED_HOLDOUT": 3,
    }
    assert all(item.environment["type"] == "skill_procedure" for item in tasks)
    smoke, _, _ = select_evaluation_suite(tasks, "SMOKE")
    assert sorted(item.id for item in smoke) == sorted(
        ["skill-commit-config-fix", "skill-commit-ticket-feature",
         "skill-commit-leaked-credentials", "skill-commit-ticket-hotfix"])
    assert BenchmarkStore(deployment.public).version == store.version  # generator view binds the same version

    # Step 3: a deliberately worse candidate (SKILL_PATCH that breaks the procedure).
    worse = deployment.create_candidate("exp-worse", "cand-worse", texts["worse"],
                                        "simplify the commit procedure to stage everything")
    assert worse["base_commit"] == v1_commit and worse["branch"] == "candidate/exp-worse"

    # Step 4: evaluate (screening only - a non-finalist never sees the sealed holdout).
    worse_report = deployment.evaluate("cand-worse", "SMOKE")

    # Step 5: regression detected against the baseline on identical tasks and seeds.
    rows = suite_rows(worse_report, "SMOKE")
    regressions = sorted(task for task, row in rows.items() if row["baseline_success"] and not row["candidate_success"])
    assert regressions == ["skill-commit-config-fix", "skill-commit-leaked-credentials"]
    assert all(row["candidate_quality"] < row["baseline_quality"] for row in rows.values())
    worse_quality = worse_report["metrics"]["paired_statistics"]["quality"]
    assert worse_quality["mean_delta"] < 0
    assert not worse_report["passed"] and "deterministic verifier failed" in worse_report["reasons"]
    evidence["worse_quality_delta"] = worse_quality["mean_delta"]

    # Step 6: rejected; the failed experiment and its evidence are retained.
    ledger = ExperimentStore(deployment.ledger)
    experiment = ledger.show_experiment("exp-worse")
    assert experiment["decision"] == "rejected"
    assert experiment["candidates"][0]["id"] == "cand-worse"
    assert len(ledger.candidate_evaluations("cand-worse")) == 1
    with sqlite3.connect(deployment.ledger) as db:
        spent = {row[0] for row in db.execute("SELECT candidate_id FROM spent_holdouts")}
    assert "lineage:exp-worse" not in spent

    # Step 7: a better candidate (adds the ticket trailer the baseline misses).
    deployment.create_candidate("exp-better", "cand-better", texts["better"],
                                "reference the tracker ticket in a Refs trailer")

    # Step 8: compare baseline vs candidate (paired, staged SMOKE then CORE).
    for suite in ("SMOKE", "CORE"):
        staged = deployment.evaluate("cand-better", suite)
        rows = suite_rows(staged, suite)
        for task, row in rows.items():
            assert row["candidate_success"], task
            delta = row["candidate_quality"] - row["baseline_quality"]
            assert delta == pytest.approx(100.0 - QUALITY_BASELINE_TICKET if "ticket" in task else 0.0), task
        assert staged["metrics"]["paired_statistics"]["quality"]["mean_delta"] > 3.0
    assert len(rows) == 6  # CORE covers every public task of the impacted skill
    assert ledger.show_experiment("exp-better")["decision"] == "pending"

    # Steps 9 + 10: FULL plus the sealed holdout (the finalist's evaluator holdout grant).
    full = deployment.evaluate("cand-better", "FULL")
    full_rows, holdout_rows = suite_rows(full, "FULL"), suite_rows(full, "SEALED_HOLDOUT")
    assert sorted(full_rows) == sorted(HYGIENE_TASKS + TICKET_TASKS)
    assert sorted(holdout_rows) == sorted(HYGIENE_HOLDOUT + TICKET_HOLDOUT)
    assert all(row["split"] == "SEALED_HOLDOUT" for row in holdout_rows.values())
    metrics = full["metrics"]
    assert full["passed"], full["reasons"]
    assert metrics["final_tasks"] == 9
    assert metrics["baseline_success_rate"] == pytest.approx(4 / 9)
    assert metrics["candidate_success_rate"] == 1.0
    assert metrics["quality_delta_mean"] == pytest.approx(5 * (100.0 - QUALITY_BASELINE_TICKET) / 9)
    assert metrics["quality_delta_mean"] > 3.0 and metrics["quality_delta_ci95"][0] > 0
    assert metrics["sealed_holdout_version"] == store.holdout_version
    assert metrics["suite_selection"]["holdout_manifest_digest"].startswith("sha256:")
    assert metrics["run_policy"]["llm_influenced"] is False
    with sqlite3.connect(deployment.ledger) as db:
        spent = {row[0] for row in db.execute("SELECT candidate_id FROM spent_holdouts")}
    assert {"lineage:exp-better", "lineage:exp-baseline"} <= spent and "lineage:exp-worse" not in spent
    assert ledger.show_experiment("exp-better")["decision"] == "passed"
    evidence.update({
        "baseline_success_rate": metrics["baseline_success_rate"],
        "candidate_success_rate": metrics["candidate_success_rate"],
        "quality_delta_mean": metrics["quality_delta_mean"],
        "quality_delta_ci95": metrics["quality_delta_ci95"],
    })

    # Steps 11 + 12: promotion manifest and promotion through the promoter CLI.
    manifest = deployment.promote("exp-better", "cand-better", "brain-v2")
    deployment.after_promotion()
    assert manifest["release_target"] == "brain-v2" and manifest["baseline_commit"] == v1_commit
    assert set(manifest["evidence_runs"]) == {"SMOKE", "CORE", "FULL", "SEALED_HOLDOUT"}
    assert manifest["full_suite_passed"] and manifest["holdout_passed"] and manifest["security_passed"]
    assert manifest["changed_paths"] == [f"{SKILL_TARGET}/SKILL.md"] and manifest["promotion_gate"] == "auto"
    assert manifest["baseline_metrics"]["success_rate"] == pytest.approx(4 / 9)
    v2_commit = manifest["candidate_commit"]
    stored = json.loads((deployment.release_root / "brain-v2.manifest.json").read_text())
    body = {key: value for key, value in stored.items() if key not in {"manifest_sha256", "signature_hmac_sha256"}}
    assert stored["manifest_sha256"] == hashlib.sha256(canonical_json(body).encode()).hexdigest()
    if hasattr(deployment, "manifest_key"):
        expected = hmac.new(deployment.manifest_key.encode(), canonical_json(body).encode(), hashlib.sha256)
        assert stored["signature_hmac_sha256"] == expected.hexdigest()

    # Step 13: release tags.
    tags = {tag: deployment.rev_parse(tag) for tag in ("brain-v1", "brain-v2")}
    assert tags == {"brain-v1": v1_commit, "brain-v2": v2_commit}
    assert deployment.rev_parse("main") == v2_commit

    # Step 14: the deployment link points at the new read-only release checkout.
    assert os.readlink(deployment.deploy_link) == str(deployment.release_root / "brain-v2")
    deployed = (deployment.deploy_link / SKILL_TARGET / "SKILL.md").read_text(encoding="utf-8")
    assert deployed == texts["better"]
    deployment.assert_release_read_only(deployment.release_root / "brain-v2")

    # Step 15: post-promotion metric from the deployed release, recorded as PROVISIONAL.
    observation = shadow_observation(deployment, "brain-v2", "skill-commit-ticket-cli")
    assert observation["success"] is True
    ingested = deployment.ingest_observations([observation])["observations"]
    assert ingested[0]["decision"] == "PROVISIONAL" and ingested[0]["sample_count"] == 1
    releases = {row["id"]: row for row in ledger.list_releases()}
    assert releases["brain-v2"]["status"] == "PROVISIONAL"
    with sqlite3.connect(deployment.home / "monitoring.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM production_observations WHERE release_id='brain-v2'").fetchone()[0] == 1

    # Step 16: roll back to the previous release; history is unchanged and audited.
    rolled = deployment.rollback("brain-v1")
    assert rolled["previous_commit"] == v2_commit
    assert deployment.rev_parse("main") == v1_commit
    assert os.readlink(deployment.deploy_link) == str(deployment.release_root / "brain-v1")
    assert (deployment.deploy_link / SKILL_TARGET / "SKILL.md").read_text(encoding="utf-8") == texts["baseline"]
    assert {tag: deployment.rev_parse(tag) for tag in tags} == tags
    assert (deployment.release_root / "brain-v2").is_dir()
    releases = {row["id"]: row for row in ledger.list_releases()}
    assert (releases["brain-v1"]["status"], releases["brain-v2"]["status"]) == ("PROVISIONAL", "ROLLED_BACK")
    events = deployment.cli("promoter", "releases", "events")
    assert [(row["release_id"], row["event"]) for row in events] == [
        ("brain-v1", "PROMOTED"), ("brain-v2", "PROMOTED"),
        ("brain-v2", "ROLLED_BACK"), ("brain-v1", "REACTIVATED"),
    ]
    evidence["release_events"] = [row["event"] for row in events]
    print("E2E evidence:", json.dumps(evidence, sort_keys=True))
    return evidence


def _rev_parse(deployment, name: str) -> str:
    return git_output(deployment.repository, "rev-parse", f"{name}^{{commit}}")


SingleUserDeployment.rev_parse = _rev_parse


# ------------------------------------------------------------------- variants

def test_full_improvement_loop_single_user(tmp_path, monkeypatch, capsys):
    evidence = run_full_improvement_loop(SingleUserDeployment(tmp_path, monkeypatch, capsys, LOCAL_RUNNER))
    assert evidence["quality_delta_mean"] > 3.0 > 0 > evidence["worse_quality_delta"]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode == 0


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="Docker and the pinned alpine/git image are required")
def test_full_improvement_loop_in_docker(tmp_path, monkeypatch, capsys):
    evidence = run_full_improvement_loop(
        SingleUserDeployment(tmp_path, monkeypatch, capsys, ("--runner", "docker"))
    )
    assert evidence["candidate_success_rate"] == 1.0


# ------------------------------------------------------------- role users

PROVISION = REPO_ROOT / "scripts" / "provision.sh"
LAUNCHER = REPO_ROOT / "scripts" / "hermes-improve-as"
CLI = Path(sys.executable).parent / "hermes-improve"
SAFE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
AGENT, EVALUATOR, PROMOTER = "hermes-agent", "hermes-evaluator", "hermes-promoter"


def _run(*command, check=True):
    result = subprocess.run([str(part) for part in command], capture_output=True, env=SAFE_ENV, cwd="/", check=False)
    if check and result.returncode:
        raise AssertionError(f"{command} failed ({result.returncode}):\n{result.stdout.decode()}\n{result.stderr.decode()}")
    return result


def _as_user(user, *command, check=True):
    return _run("setpriv", f"--reuid={user}", f"--regid={user}", "--init-groups", "--", *command, check=check)


class RoleUserDeployment:
    """The same flow through provision.sh and hermes-improve-as as the real role users."""

    def __init__(self, base: Path) -> None:
        self.root = base / "srv"
        _run("bash", PROVISION, "--root", self.root, "--cli", CLI, "--python", sys.executable)
        # Protected policy config in the root-owned config directory; a second
        # provisioning run points every role's HERMES_IMPROVE_CONFIG at it.
        install_policy_config(self.root / "config")
        _run("bash", PROVISION, "--root", self.root, "--cli", CLI, "--python", sys.executable)
        assert (self.root / "config" / "roles" / "evaluator.env").read_text().count(
            f"HERMES_IMPROVE_CONFIG={self.root}/config\n") == 1
        self.public = self.root / "benchmarks" / "public"
        self.holdout = self.root / "holdout"
        self.home = self.root / "ledger"
        self.repository = self.root / "brain" / "production.git"
        self.candidates = self.root / "brain" / "candidates.git"
        self.mirror = self.root / "evaluator" / "mirror.git"
        self.handoff = self.root / "handoff"
        self.release_root = self.root / "releases"
        self.deploy_link = self.root / "deploy" / "current"

    @property
    def ledger(self) -> Path:
        return self.home / "experiments.sqlite3"

    def cli(self, role: str, *args) -> dict:
        result = _run("bash", LAUNCHER, "--root", self.root, role, *args)
        return json.loads(result.stdout)

    def rev_parse(self, name: str) -> str:
        return git_output(self.repository, "rev-parse", f"{name}^{{commit}}")

    def seed_production(self, files: dict[str, str]) -> str:
        seed = Path(tempfile.mkdtemp(prefix="seed-", dir=self.root / "home" / "promoter"))
        for path, content in files.items():
            (seed / path).parent.mkdir(parents=True, exist_ok=True)
            (seed / path).write_text(content, encoding="utf-8")
        _run("chown", "-R", PROMOTER, seed)
        git = ["git", "-c", "core.hooksPath=/dev/null", "-C", seed]
        _as_user(PROMOTER, "env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", *git, "init", "-q", "-b", "main")
        _as_user(PROMOTER, "env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", *git, "add", "-A")
        _as_user(PROMOTER, "env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", *git,
                 "-c", "user.name=Operator", "-c", "user.email=op@example.invalid", "commit", "-q", "-m", "Production brain")
        _as_user(PROMOTER, "env", "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL=/dev/null", *git,
                 "push", "-q", self.repository, "main")
        self.after_promotion()
        return self.rev_parse("main")

    def add_benchmarks(self, public, holdout) -> None:
        install_public(self.public, public)
        install_holdout(self.holdout, holdout)
        _run("bash", PROVISION, "--root", self.root, "--cli", CLI, "--python", sys.executable)  # normalize perms

    def create_candidate(self, experiment: str, candidate: str, content: str, hypothesis: str) -> dict:
        path = self.root / "agent" / f"{candidate}.md"
        path.write_text(content, encoding="utf-8")
        _run("chown", AGENT, path)
        created = self.cli(
            "agent", "candidate", "create", "--repository", self.candidates, "--experiment-id", experiment,
            "--candidate-id", candidate, "--type", "SKILL_PATCH", "--target", SKILL_TARGET,
            "--content", path, "--hypothesis", hypothesis,
        )
        self.cli("agent", "candidate", "export", candidate, "--repository", self.candidates, "--out-dir", self.handoff)
        imported = self.cli("evaluator", "candidate", "import", candidate, "--handoff-dir", self.handoff,
                            "--mirror", self.mirror)
        assert imported["status"] == "imported_awaiting_evaluation"
        return created

    def evaluate(self, candidate: str, suite: str) -> dict:
        return self.cli("evaluator", "evaluate", candidate, "--repository", self.mirror,
                        "--holdout-root", self.holdout, "--suite", suite, *LOCAL_RUNNER)

    def _release_args(self):
        return ("--repository", self.repository, "--release-root", self.release_root,
                "--deployment-link", self.deploy_link, "--evaluator-public-key", self.root / "keys" / "evaluator.pub")

    def promote(self, experiment: str, candidate: str, release: str) -> dict:
        return self.cli("promoter", "promote", experiment, "--release", release,
                        "--candidate-bundle", self.handoff / f"{candidate}.bundle", *self._release_args())

    def after_promotion(self) -> None:
        self.cli("agent", "candidate", "sync-baseline", "--repository", self.candidates, "--production", self.repository)

    def rollback(self, release: str) -> dict:
        return self.cli("promoter", "rollback", release, *self._release_args())

    def ingest_observations(self, lines: list[dict]) -> dict:
        path = self.root / "home" / "promoter" / "observations.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        _run("chown", PROMOTER, path)
        return self.cli("promoter", "monitor", "ingest", path)

    def assert_release_read_only(self, release_dir: Path) -> None:
        skill = release_dir / SKILL_TARGET / "SKILL.md"
        for user in (AGENT, EVALUATOR):
            assert _as_user(user, "cat", skill).returncode == 0
            denied = _as_user(user, "sh", "-c", f"echo tampered >> {skill}", check=False)
            assert denied.returncode != 0 and b"Permission denied" in denied.stderr
        # The agent cannot reach the sealed holdout it was never evaluated against.
        assert _as_user(AGENT, "ls", self.holdout, check=False).returncode != 0


@pytest.mark.root_required
def test_full_improvement_loop_with_role_users():
    base = Path(tempfile.mkdtemp(prefix="hsi-e2e-", dir="/var/tmp"))
    base.chmod(0o755)
    try:
        evidence = run_full_improvement_loop(RoleUserDeployment(base))
        assert evidence["release_events"] == ["PROMOTED", "PROMOTED", "ROLLED_BACK", "REACTIVATED"]
    finally:
        shutil.rmtree(base)
