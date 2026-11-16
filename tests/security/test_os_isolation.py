"""Real operating-system isolation tests.

These tests run as root (inside the Linux test container), provision the
layout with scripts/provision.sh into a temporary root, and then act as the
real role users through setpriv / the hermes-improve-as launcher. Every
"cannot" assertion is a kernel permission decision, not a mocked check.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.root_required

REPO = Path(__file__).resolve().parents[2]
PROVISION = REPO / "scripts" / "provision.sh"
LAUNCHER = REPO / "scripts" / "hermes-improve-as"
CLI = Path(sys.executable).parent / "hermes-improve"
AGENT, EVALUATOR, PROMOTER = "hermes-agent", "hermes-evaluator", "hermes-promoter"
SAFE_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}


def run(*command, check=True, env=None, input=None):
    result = subprocess.run(
        [str(part) for part in command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env or SAFE_ENV, input=input, check=False, cwd="/",
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"{command} failed ({result.returncode}):\n{result.stdout.decode()}\n{result.stderr.decode()}"
        )
    return result


def as_user(user, *command, env=None, check=False):
    """Run a command as ``user`` with only that user's own groups."""
    return run("setpriv", f"--reuid={user}", f"--regid={user}", "--init-groups", "--",
               *command, env=env, check=check)


def denied(result):
    output = (result.stdout + result.stderr).decode(errors="replace")
    return result.returncode != 0 and ("Permission denied" in output or "permission" in output.lower())


def provision(root):
    return run("bash", PROVISION, "--root", root, "--cli", CLI, "--python", sys.executable)


def launch(root, role, *args, check=False):
    return run("bash", LAUNCHER, "--root", root, role, *args, check=check)


def git(*args, user=None, check=True):
    command = ["git", "-c", "core.hooksPath=/dev/null", *args]
    env = SAFE_ENV | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    if user is None:
        return run(*command, env=env, check=check)
    return as_user(user, "env", *[f"{key}={value}" for key, value in env.items()], *command, check=check)


def snapshot(root):
    """Owner, group, mode, and ACL for every provisioned path (idempotency proof)."""
    state = {}
    for directory, subdirs, files in os.walk(root):
        for name in [".", *subdirs, *files]:
            path = Path(directory) / name if name != "." else Path(directory)
            if path.is_symlink():
                continue
            info = path.stat()
            acl = run("getfacl", "-n", "-p", "--omit-header", path).stdout.decode()
            state[str(path)] = (info.st_uid, info.st_gid, oct(info.st_mode), acl)
    return state


@pytest.fixture(scope="module")
def layout_root():
    base = Path(tempfile.mkdtemp(prefix="hsi-iso-", dir="/var/tmp"))
    base.chmod(0o755)
    root = base / "srv"
    provision(root)
    (root / "holdout" / "sealed-task.json").write_text('{"secret": "holdout"}\n')
    (root / "config" / "promotion.yaml").write_text("min_quality_delta: 3\n")
    (root / "benchmarks" / "public" / "train").mkdir()
    shutil.copy(REPO / "benchmarks" / "train" / "example-smoke.json", root / "benchmarks" / "public" / "train")
    provision(root)  # normalizes the newly added files
    yield root
    shutil.rmtree(base)


@pytest.fixture
def broken_root():
    base = Path(tempfile.mkdtemp(prefix="hsi-iso-broken-", dir="/var/tmp"))
    base.chmod(0o755)
    root = base / "srv"
    provision(root)
    yield root
    shutil.rmtree(base)


def test_provision_is_idempotent_and_every_role_passes_its_self_check(layout_root):
    before = snapshot(layout_root)
    output = provision(layout_root).stdout.decode()
    assert "provisioned" in output
    assert snapshot(layout_root) == before
    report = json.loads(run(CLI, "security", "check", "--layout", layout_root / "config/layout.json").stdout)
    assert report["ok"] and report["violations"] == [] and report["unverified"] == []
    for role in ("agent", "evaluator", "promoter"):
        result = launch(layout_root, role, "security", "check", "--role", role)
        assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
        assert json.loads(result.stdout)["ok"]


def test_role_users_have_no_sudo_and_no_privileged_groups(layout_root):
    for user in (AGENT, EVALUATOR, PROMOTER):
        groups = set(run("id", "-nG", user).stdout.decode().split())
        assert not groups & {"root", "sudo", "wheel", "admin", "adm", "disk", "shadow", "docker"}
    assert "hermes-ledger" not in run("id", "-nG", AGENT).stdout.decode().split()
    assert shutil.which("sudo") is None or as_user(AGENT, "sudo", "-n", "true").returncode != 0
    # The launcher sets no_new_privs, so even a setuid binary cannot raise privileges.
    status = launch(layout_root, "agent", "status", check=True)
    assert json.loads(status.stdout)["version"]


def test_agent_cannot_write_production(layout_root):
    production = layout_root / "brain" / "production.git"
    assert denied(as_user(AGENT, "touch", layout_root / "releases" / "brain-v99"))
    assert denied(as_user(AGENT, "touch", production / "refs" / "heads" / "main"))
    assert denied(as_user(AGENT, "sh", "-c", f"echo x > {production / 'HEAD'}"))
    assert denied(as_user(AGENT, "ln", "-s", layout_root / "workspaces", layout_root / "deploy" / "current"))
    assert denied(as_user(AGENT, "mv", layout_root / "releases", layout_root / "workspaces" / "stolen"))
    # Read-only access to the production repository still works.
    assert as_user(AGENT, "cat", production / "HEAD").returncode == 0


def test_agent_cannot_read_holdout_ledger_or_keys(layout_root):
    assert denied(as_user(AGENT, "cat", layout_root / "holdout" / "sealed-task.json"))
    assert denied(as_user(AGENT, "ls", layout_root / "holdout"))
    assert denied(as_user(AGENT, "cat", layout_root / "ledger" / "experiments.sqlite3"))
    assert denied(as_user(AGENT, "cat", layout_root / "keys" / "evaluator" / "signing.pem"))
    assert denied(as_user(AGENT, "cat", layout_root / "keys" / "promoter" / "manifest.key"))
    assert denied(as_user(PROMOTER, "cat", layout_root / "holdout" / "sealed-task.json"))
    assert denied(as_user(PROMOTER, "ls", layout_root / "workspaces"))


def test_agent_cannot_tamper_with_config_policy_ground_truth_or_evaluator_code(layout_root):
    policy = layout_root / "config" / "promotion.yaml"
    for target in (
        policy, layout_root / "config" / "layout.json", layout_root / "config" / "roles" / "promoter.env",
        layout_root / "benchmarks" / "public" / "train" / "example-smoke.json",
    ):
        assert denied(as_user(AGENT, "sh", "-c", f"echo tampered >> {target}"))
        assert denied(as_user(AGENT, "chmod", "666", target))
        assert denied(as_user(AGENT, "rm", "-f", target))
    assert denied(as_user(AGENT, "touch", layout_root / "install" / "evaluator.py"))
    assert denied(as_user(AGENT, "mv", layout_root / "config", layout_root / "workspaces" / "config"))
    assert denied(as_user(EVALUATOR, "sh", "-c", f"echo tampered >> {policy}"))
    # The installed evaluator/verifier/promoter code is not writable either.
    package = Path(run(sys.executable, "-c", "import hermes_improve,os;print(os.path.dirname(hermes_improve.__file__))").stdout.decode().strip())
    assert denied(as_user(AGENT, "sh", "-c", f"echo x >> {package / 'policy.py'}"))
    assert policy.read_text() == "min_quality_delta: 3\n"


def test_evaluator_reads_holdout_but_cannot_write_production_holdout_or_candidates(layout_root):
    assert as_user(EVALUATOR, "cat", layout_root / "holdout" / "sealed-task.json").returncode == 0
    assert denied(as_user(EVALUATOR, "sh", "-c", f"echo x >> {layout_root / 'holdout' / 'sealed-task.json'}"))
    assert denied(as_user(EVALUATOR, "touch", layout_root / "holdout" / "new.json"))
    assert denied(as_user(EVALUATOR, "touch", layout_root / "releases" / "brain-v99"))
    assert denied(as_user(EVALUATOR, "touch", layout_root / "brain" / "production.git" / "refs" / "heads" / "x"))
    assert denied(as_user(EVALUATOR, "touch", layout_root / "workspaces" / "x"))
    assert denied(as_user(EVALUATOR, "touch", layout_root / "brain" / "candidates.git" / "refs" / "heads" / "x"))
    assert as_user(EVALUATOR, "ls", layout_root / "workspaces").returncode == 0


def _seed_production(root):
    seed = Path(tempfile.mkdtemp(prefix="seed-", dir=root / "home" / "promoter"))
    run("chown", PROMOTER, seed)
    skill = seed / "brain" / "skills" / "demo"
    as_user(PROMOTER, "mkdir", "-p", skill, check=True)
    as_user(PROMOTER, "sh", "-c", f"echo original > {skill / 'SKILL.md'}", check=True)
    git("-C", seed, "init", "-q", "-b", "main", user=PROMOTER)
    git("-C", seed, "add", ".", user=PROMOTER)
    git("-C", seed, "-c", "user.name=op", "-c", "user.email=op@example.invalid",
        "commit", "-q", "-m", "baseline", user=PROMOTER)
    git("-C", seed, "push", "-q", root / "brain" / "production.git", "main", user=PROMOTER)
    return git("-C", root / "brain" / "production.git", "rev-parse", "main", user=PROMOTER).stdout.decode().strip()


_RECORD_EVALUATION = r"""
import os, sys
from hermes_improve.evaluation_signing import sign_report
from hermes_improve.models import Split, TaskOutcome
from hermes_improve.policy import PromotionPolicy
from hermes_improve.store import ExperimentStore
store = ExperimentStore(os.path.join(os.environ["HERMES_IMPROVE_HOME"], "experiments.sqlite3"))
candidate = store.get_candidate(sys.argv[1])
outcomes = tuple(
    TaskOutcome(f"task-{suite}", split, suite, True, True, 5, 10, 100, 100, categories=("skills",),
                baseline_latency_ms=100, candidate_latency_ms=100, baseline_tool_calls=5, candidate_tool_calls=5)
    for suite, split in (
        ("SMOKE", Split.REGRESSION), ("CORE", Split.REGRESSION),
        ("FULL", Split.REGRESSION), ("SEALED_HOLDOUT", Split.SEALED_HOLDOUT),
    )
)
benchmark_version = store.show_experiment(candidate["experiment_id"])["benchmark_version"]
report = PromotionPolicy().assess(sys.argv[1], candidate["content_hash"], benchmark_version, outcomes)
assert report.passed, report.reasons
key = open(os.environ["HERMES_EVALUATOR_PRIVATE_KEY"], "rb").read()
store.record_evaluation(f"eval-{sys.argv[1]}", sys.argv[1], report, signature=sign_report(report, key))
"""


def test_role_separated_candidate_flow_and_promoter_switches_deployment(layout_root):
    root = layout_root
    baseline = _seed_production(root)
    candidates = root / "brain" / "candidates.git"
    handoff = root / "handoff"

    synced = launch(root, "agent", "candidate", "sync-baseline", "--repository", candidates,
                    "--production", root / "brain" / "production.git", check=True)
    assert json.loads(synced.stdout)["main"] == baseline
    content = root / "agent" / "improved.md"
    as_user(AGENT, "sh", "-c", f"echo improved > {content}", check=True)
    launch(root, "agent", "candidate", "create", "--repository", candidates, "--experiment-id", "exp-iso",
           "--candidate-id", "cand-iso", "--type", "SKILL_PATCH", "--target", "brain/skills/demo",
           "--content", content, "--hypothesis", "clearer skill", check=True)
    launch(root, "agent", "candidate", "export", "cand-iso", "--repository", candidates,
           "--out-dir", handoff, check=True)
    # The agent may not import into the ledger or promote, even through the launcher.
    refused = launch(root, "agent", "candidate", "import", "cand-iso", "--handoff-dir", handoff,
                     "--mirror", root / "evaluator" / "mirror.git")
    assert refused.returncode == 2 and b"may not run" in refused.stderr

    imported = launch(root, "evaluator", "candidate", "import", "cand-iso", "--handoff-dir", handoff,
                      "--mirror", root / "evaluator" / "mirror.git", check=True)
    assert json.loads(imported.stdout)["status"] == "imported_awaiting_evaluation"
    # Stand-in for the Docker-backed evaluation run: the evaluator identity
    # signs a passing report with its own key and writes it to the ledger.
    evaluator_env = dict(
        line.split("=", 1) for line in (root / "config/roles/evaluator.env").read_text().splitlines()
    )
    as_user(EVALUATOR, "env", "-i", *(f"{k}={v}" for k, v in (SAFE_ENV | evaluator_env).items()),
            sys.executable, "-c", _RECORD_EVALUATION, "cand-iso", check=True)

    promote_args = [
        "--repository", root / "brain" / "production.git", "--release-root", root / "releases",
        "--deployment-link", root / "deploy" / "current",
        "--evaluator-public-key", root / "keys" / "evaluator.pub",
    ]
    refused = launch(root, "agent", "promote", "cand-iso", "--release", "brain-v1", *promote_args)
    assert refused.returncode == 2 and b"may not run" in refused.stderr

    # Unauthorized promotion: the agent invokes the CLI directly with spoofed
    # role variables. The isolation check refuses it ...
    agent_uid = run("id", "-u", AGENT).stdout.decode().strip()
    spoofed = SAFE_ENV | {
        "HERMES_PROMOTER_UID": agent_uid, "HERMES_IMPROVE_MANIFEST_KEY": "k" * 48,
        "HERMES_IMPROVE_HOME": str(root / "ledger"), "HERMES_IMPROVE_LAYOUT": str(root / "config/layout.json"),
        "HOME": str(root / "home" / "agent"),
    }
    attempt = as_user(AGENT, "env", "-i", *(f"{k}={v}" for k, v in spoofed.items()),
                      CLI, "promote", "cand-iso", "--release", "brain-v1",
                      "--candidate-bundle", handoff / "cand-iso.bundle", *promote_args)
    assert attempt.returncode != 0 and b"isolation boundary violated" in attempt.stderr
    # ... and with the check disabled the kernel still blocks it.
    bypass = spoofed | {"HERMES_IMPROVE_LAYOUT": "/nonexistent/layout.json", "HERMES_IMPROVE_ISOLATION": "unprovisioned"}
    attempt = as_user(AGENT, "env", "-i", *(f"{k}={v}" for k, v in bypass.items()),
                      CLI, "promote", "cand-iso", "--release", "brain-v1",
                      "--candidate-bundle", handoff / "cand-iso.bundle", *promote_args)
    assert attempt.returncode != 0
    assert b"unable to open database" in attempt.stderr or b"Permission denied" in attempt.stderr
    assert git("-C", root / "brain/production.git", "rev-parse", "main", user=PROMOTER).stdout.decode().strip() == baseline
    assert not os.path.lexists(root / "deploy" / "current")

    promoted = launch(root, "promoter", "promote", "cand-iso", "--release", "brain-v1",
                      "--candidate-bundle", handoff / "cand-iso.bundle", *promote_args, check=True)
    manifest = json.loads(promoted.stdout)
    assert manifest["release_target"] == "brain-v1"
    assert os.readlink(root / "deploy" / "current") == str(root / "releases" / "brain-v1")
    deployed = root / "deploy" / "current" / "brain" / "skills" / "demo" / "SKILL.md"
    assert as_user(AGENT, "cat", deployed, check=True).stdout == b"improved\n"
    assert denied(as_user(AGENT, "sh", "-c", f"echo hacked > {deployed}"))
    assert denied(as_user(EVALUATOR, "sh", "-c", f"echo hacked > {deployed}"))
    report = launch(root, "agent", "security", "check", "--role", "agent", check=True)
    assert json.loads(report.stdout)["ok"]


def test_boundary_violation_is_detected_and_blocks_promotion_and_evaluation(broken_root):
    root = broken_root
    layout = root / "config" / "layout.json"
    run("chmod", "o+w", root / "releases")
    run("setfacl", "-m", f"u:{AGENT}:r-x", root / "holdout")
    report = json.loads(run(CLI, "security", "check", "--layout", layout, check=False).stdout)
    assert not report["ok"]
    joined = "\n".join(report["violations"])
    assert "agent (hermes-agent) has rwx on releases" in joined
    assert "agent (hermes-agent) has rx on holdout" in joined

    promote = launch(root, "promoter", "promote", "cand-x", "--release", "brain-v1",
                     "--repository", root / "brain/production.git", "--release-root", root / "releases",
                     "--deployment-link", root / "deploy/current",
                     "--evaluator-public-key", root / "keys/evaluator.pub")
    assert promote.returncode != 0 and b"isolation boundary violated" in promote.stderr
    evaluate = launch(root, "evaluator", "evaluate", "cand-x", "--repository", root / "evaluator/mirror.git",
                      "--holdout-root", root / "holdout")
    assert evaluate.returncode != 0 and b"isolation boundary violated" in evaluate.stderr

    # Re-provisioning repairs the drift.
    provision(root)
    assert json.loads(run(CLI, "security", "check", "--layout", layout).stdout)["ok"]


def test_promoter_rejects_paths_outside_the_layout(layout_root):
    elsewhere = Path(tempfile.mkdtemp(prefix="hsi-elsewhere-", dir="/var/tmp"))
    try:
        result = launch(layout_root, "promoter", "promote", "cand-x", "--release", "brain-v1",
                        "--repository", elsewhere, "--release-root", layout_root / "releases",
                        "--deployment-link", layout_root / "deploy/current",
                        "--evaluator-public-key", layout_root / "keys/evaluator.pub")
        assert result.returncode != 0 and b"production_repo must be the provisioned path" in result.stderr
    finally:
        shutil.rmtree(elsewhere)
