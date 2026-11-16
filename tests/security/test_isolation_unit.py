"""Unprivileged tests for the isolation model, layout loading, and role handoff."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from hermes_improve.candidate_builder import create_skill_candidate
from hermes_improve.errors import IntegrityError
from hermes_improve.isolation import (
    PROTECTED_REPOSITORY_PATHS,
    check_isolation,
    enforce_isolation,
    load_layout,
    parse_posix_acl,
    permission_bits,
)
from hermes_improve.models import CandidateType
from hermes_improve.role_handoff import (
    export_candidate,
    import_bundle_ref,
    import_candidate,
    sync_production_baseline,
)
from hermes_improve.store import ExperimentStore

REPO = Path(__file__).resolve().parents[2]


def _acl(*entries):
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


ACL = [
    (0x01, 7, 0xFFFFFFFF),  # user::rwx
    (0x02, 7, 1001),        # user:1001:rwx (masked)
    (0x04, 5, 0xFFFFFFFF),  # group::r-x
    (0x08, 2, 3000),        # group:3000:-w-
    (0x10, 5, 0xFFFFFFFF),  # mask::r-x
    (0x20, 0, 0xFFFFFFFF),  # other::---
]


def test_posix_acl_xattr_is_decoded_and_malformed_data_rejected():
    assert parse_posix_acl(_acl(*ACL)) == ACL
    with pytest.raises(ValueError, match="malformed"):
        parse_posix_acl(b"\x02\x00\x00\x00\x01")
    with pytest.raises(ValueError, match="version"):
        parse_posix_acl(struct.pack("<I", 1))


def test_permission_bits_follow_posix_1e_order_and_mask():
    mode, owner, group = 0o40750, 1000, 2000
    assert permission_bits(mode, owner, group, ACL, 1000, [1000]) == 7
    assert permission_bits(mode, owner, group, ACL, 1001, [1001]) == 5  # rwx & mask r-x
    assert permission_bits(mode, owner, group, ACL, 1002, [2000]) == 5
    assert permission_bits(mode, owner, group, ACL, 1002, [3000]) == 0  # -w- & r-x
    assert permission_bits(mode, owner, group, ACL, 1002, [2000, 3000]) == 5
    assert permission_bits(mode, owner, group, ACL, 1003, [4000]) == 0
    assert permission_bits(0o100644, owner, group, None, 1003, [4000]) == 4
    assert permission_bits(0o100664, owner, group, None, 1003, [2000]) == 6
    assert permission_bits(0o100600, owner, group, None, 0, [0]) == 7


def _write_layout(tmp_path, **overrides):
    document = {
        "schema_version": 1,
        "root": str(tmp_path / "srv"),
        "users": {"agent": "hsi-test-agent", "evaluator": "hsi-test-eval", "promoter": "hsi-test-prom"},
        "ledger_group": "hsi-test-ledger",
        "cli": "/usr/bin/hermes-improve",
    } | overrides
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(document))
    path.chmod(0o644)
    return path


def test_layout_file_must_be_trusted_and_well_formed(tmp_path):
    path = _write_layout(tmp_path)
    layout = load_layout(path, required_owner=os.getuid())
    assert layout.path("holdout") == tmp_path / "srv" / "holdout"
    assert layout.deployment_link == tmp_path / "srv" / "deploy" / "current"
    path.chmod(0o666)
    with pytest.raises(PermissionError, match="not group/other writable"):
        load_layout(path, required_owner=os.getuid())
    with pytest.raises(PermissionError, match="owned by uid"):
        load_layout(_write_layout(tmp_path), required_owner=os.getuid() + 1)
    duplicate = {"agent": "same", "evaluator": "same", "promoter": "other"}
    with pytest.raises(ValueError, match="distinct"):
        load_layout(_write_layout(tmp_path, users=duplicate), required_owner=os.getuid())
    with pytest.raises(ValueError, match="safe absolute"):
        load_layout(_write_layout(tmp_path, root="/srv/../etc"), required_owner=os.getuid())
    with pytest.raises(ValueError, match="unknown fields"):
        load_layout(_write_layout(tmp_path, extra=True), required_owner=os.getuid())


def test_check_reports_missing_users_and_layout_as_violations(tmp_path):
    layout = load_layout(_write_layout(tmp_path), required_owner=os.getuid())
    report = check_isolation(layout)
    assert not report.ok
    assert any("hsi-test-agent" in item and "does not exist" in item for item in report.violations)
    assert any(item.startswith("holdout is missing") for item in report.violations)


def test_layout_rejects_cli_paths_outside_the_layout(tmp_path):
    layout = load_layout(_write_layout(tmp_path), required_owner=os.getuid())
    layout.require_paths({"holdout": tmp_path / "srv" / "holdout"})
    with pytest.raises(PermissionError, match="holdout must be the provisioned path"):
        layout.require_paths({"holdout": tmp_path / "elsewhere"})
    with pytest.raises(PermissionError, match="deployment_link"):
        layout.require_paths({"deployment_link": tmp_path / "current"})


def test_privileged_commands_fail_closed_without_a_layout(tmp_path, capsys):
    environ = {"HERMES_IMPROVE_LAYOUT": str(tmp_path / "absent.json")}
    with pytest.raises(PermissionError, match="no provisioned isolation layout"):
        enforce_isolation("promoter", environ)
    assert enforce_isolation("promoter", environ | {"HERMES_IMPROVE_ISOLATION": "unprovisioned"}) is None
    assert "no operating-system role separation is verified" in capsys.readouterr().err
    # Unprivileged: the layout is not root-owned. As root: the role users do not
    # exist and the directories are missing. Either way the command is refused.
    expected = "isolation boundary violated" if os.geteuid() == 0 else "owned by uid 0"
    with pytest.raises(PermissionError, match=expected):
        enforce_isolation("evaluator", {"HERMES_IMPROVE_LAYOUT": str(_write_layout(tmp_path))})


def test_codeowners_protects_every_protected_repository_path():
    rules = {}
    for line in (REPO / ".github" / "CODEOWNERS").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pattern, *owners = line.split()
            rules[pattern] = owners
    for path in PROTECTED_REPOSITORY_PATHS:
        assert rules.get(path), f"CODEOWNERS lacks an owner rule for {path}"


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()


@pytest.fixture
def handoff_setup(tmp_path):
    production = tmp_path / "production.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(production))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    (seed / "brain" / "skills" / "demo").mkdir(parents=True)
    (seed / "brain" / "skills" / "demo" / "SKILL.md").write_text("original\n")
    git(seed, "add", ".")
    git(seed, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "base")
    git(seed, "push", "-q", str(production), "main")
    candidates = tmp_path / "candidates.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(candidates))
    assert sync_production_baseline(candidates, production) == git(production, "rev-parse", "main")
    agent_store = ExperimentStore(tmp_path / "agent.sqlite3")
    candidate = create_skill_candidate(
        candidates, agent_store, experiment_id="exp-h", candidate_id="cand-h",
        candidate_type=CandidateType.SKILL_PATCH, target="brain/skills/demo",
        content="improved\n", hypothesis="clearer", benchmark_version="bench-v1",
    )
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    export_candidate(agent_store, candidates, "cand-h", handoff)
    mirror = tmp_path / "mirror.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(mirror))
    return {
        "tmp": tmp_path, "production": production, "candidates": candidates, "candidate": candidate,
        "handoff": handoff, "mirror": mirror, "ledger": ExperimentStore(tmp_path / "ledger.sqlite3"),
    }


def test_exported_candidate_is_reverified_into_the_ledger_and_production(handoff_setup):
    setup = handoff_setup
    imported = import_candidate(setup["ledger"], setup["handoff"], "cand-h", setup["mirror"])
    assert imported.content_hash == setup["candidate"].content_hash
    assert setup["ledger"].get_candidate("cand-h")["commit_id"] == setup["candidate"].commit
    with pytest.raises(ValueError, match="already exported"):
        export_candidate(ExperimentStore(setup["tmp"] / "agent.sqlite3"), setup["candidates"], "cand-h", setup["handoff"])
    branch = import_bundle_ref(setup["production"], setup["handoff"] / "cand-h.bundle", "exp-h", setup["candidate"].commit)
    assert git(setup["production"], "rev-parse", branch) == setup["candidate"].commit
    with pytest.raises(IntegrityError, match="exactly the evaluated candidate branch"):
        import_bundle_ref(setup["production"], setup["handoff"] / "cand-h.bundle", "exp-h", "0" * 40)


def test_tampered_handoff_metadata_or_bundle_is_rejected(handoff_setup):
    setup = handoff_setup
    metadata_path = setup["handoff"] / "cand-h.candidate.json"
    original = json.loads(metadata_path.read_text())
    metadata_path.chmod(0o644)
    tampered = json.loads(json.dumps(original))
    tampered["candidate"]["target"] = "brain/skills/other"
    metadata_path.write_text(json.dumps(tampered))
    with pytest.raises(IntegrityError, match="content hash"):
        import_candidate(setup["ledger"], setup["handoff"], "cand-h", setup["mirror"])
    tampered = json.loads(json.dumps(original))
    tampered["bundle_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(tampered))
    with pytest.raises(IntegrityError, match="bundle digest"):
        import_candidate(setup["ledger"], setup["handoff"], "cand-h", setup["mirror"])
    metadata_path.write_text(json.dumps(original | {"extra": 1}))
    with pytest.raises(IntegrityError, match="unknown fields"):
        import_candidate(setup["ledger"], setup["handoff"], "cand-h", setup["mirror"])
    with pytest.raises(KeyError):
        setup["ledger"].get_candidate("cand-h")


def test_bundle_cannot_move_an_existing_candidate_branch(handoff_setup):
    setup = handoff_setup
    production = setup["production"]
    baseline = git(production, "rev-parse", "main")
    git(production, "update-ref", "refs/heads/candidate/exp-h", baseline)
    with pytest.raises(IntegrityError, match="different commit"):
        import_bundle_ref(production, setup["handoff"] / "cand-h.bundle", "exp-h", setup["candidate"].commit)
    assert git(production, "rev-parse", "refs/heads/candidate/exp-h") == baseline
