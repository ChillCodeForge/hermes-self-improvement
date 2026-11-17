"""Shared helpers for brain-consuming (``skill_procedure``) benchmarks in tests.

The committed public records under ``benchmarks/{train,validation,regression}``
are produced by :func:`skill_record`; sealed holdout records and fixtures are
only ever written by tests into temporary holdout roots
(:func:`write_holdout_task`). :func:`reference_repository` builds the
deterministic reference brain commit whose replay evidence the public records
carry (the reference solution of these tasks, analogous to ``solve.sh``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"
HARNESS = "fixtures/skill-harness"
SKILL = "git-commit-hygiene"
SKILL_TARGET = f"brain/skills/{SKILL}"
REFERENCE_SKILL = BENCHMARKS / "fixtures" / "skill-reference" / SKILL / "SKILL.md"
IMAGE = "alpine/git:latest@sha256:2306e2b05ee0ef99f5d25b0779f645b24ae9f91ba187107503592fcdfe78da3a"
CHECKS = (
    "one_new_commit", "intended_paths_committed", "no_secret_committed", "no_generated_committed",
    "subject_follows_convention", "body_explains_why", "strays_preserved", "ticket_reference",
)
_DATE = "2024-01-01T00:00:00+0000"
_CHECK_SH = '. "$SKILL_HARNESS/skillcheck.sh"\nstandard_checks\nfinish\n'


def deterministic_git_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C",
        "GIT_AUTHOR_NAME": "Hermes Benchmarks", "GIT_AUTHOR_EMAIL": "benchmarks@example.invalid",
        "GIT_COMMITTER_NAME": "Hermes Benchmarks", "GIT_COMMITTER_EMAIL": "benchmarks@example.invalid",
        "GIT_AUTHOR_DATE": _DATE, "GIT_COMMITTER_DATE": _DATE,
    }


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
        env=deterministic_git_env(),
    ).stdout.strip()


def reference_repository(destination: Path) -> tuple[Path, str]:
    """A brain repository whose only file is the reference skill; the commit ID is deterministic."""
    destination.mkdir(parents=True)
    _git(destination, "init", "-q", "-b", "main")
    skill = destination / SKILL_TARGET / "SKILL.md"
    skill.parent.mkdir(parents=True)
    shutil.copyfile(REFERENCE_SKILL, skill)
    _git(destination, "add", "-A")
    _git(destination, "commit", "-q", "-m", "Reference brain for skill_procedure benchmarks")
    return destination, _git(destination, "rev-parse", "HEAD")


def verification() -> dict:
    return {
        "deterministic": [
            {"type": "exit_code"},
            {"type": "json_path_equals", "path": ["status"], "expected": "pass"},
            *({"type": "json_path_equals", "path": ["results", name], "expected": True} for name in CHECKS),
            {"type": "git_diff", "allowed_paths": ["trace.log"], "required_paths": ["trace.log"]},
        ],
        "subjective": [],
    }


def skill_record(
    identifier: str, *, split: str, suites: list[str], title: str, prompt: str,
    input_data: str, harness_digest: str, criticality: str = "normal", difficulty: str = "easy",
    evidence: dict | None = None, verified_at: str = "2026-11-17T00:00:00+00:00",
) -> dict:
    return {
        "schema_version": 1,
        "id": identifier,
        "version": 1,
        "source": {
            "type": "manual",
            "provenance": "WP-G brain-consuming corpus; fixture, reference skill and deterministic check authored together",
            "verified_at": verified_at,
        },
        "task": {"title": title, "prompt": prompt},
        "taxonomy": {"categories": ["git"], "difficulty": difficulty,
                     "required_capabilities": ["git-cli", "commit-hygiene"]},
        "environment": {
            "type": "skill_procedure",
            "repository_fixture": f"fixtures/{identifier}",
            "harness": HARNESS,
            "harness_digest": harness_digest,
            "input_data": input_data,
            "command": ["sh", "harness/skill-executor.sh", SKILL],
            "cwd": ".",
            "container_image": IMAGE,
            "dependency_digest": "sha256:" + IMAGE.rsplit("@sha256:", 1)[1],
            "network_policy": "disabled",
            "timeout_seconds": 60,
        },
        "eligibility": {
            "class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True,
            **({"replay_evidence": evidence} if evidence is not None else {}),
        },
        "verification": verification(),
        "ground_truth": {"type": "deterministic_tests", "confidence": 1.0, "expected_exit_code": 0},
        "benchmark": {
            "split": split, "suites": suites, "cluster_id": identifier, "criticality": criticality,
            "impact_areas": [f"skills/{SKILL}"],
        },
        "security": {"sanitized": True, "contains_secrets": False},
    }


def write_fixture(directory: Path, *, setup: str, intent: dict[str, str], expected: list[str], strays: list[str]) -> None:
    directory.mkdir(parents=True)
    (directory / "setup.sh").write_text(setup, encoding="utf-8")
    (directory / "check.sh").write_text(_CHECK_SH, encoding="utf-8")
    (directory / "intent").write_text("".join(f"{key}={value}\n" for key, value in intent.items()), encoding="utf-8")
    (directory / "expected-paths").write_text("".join(f"{path}\n" for path in expected), encoding="utf-8")
    (directory / "strays").write_text("".join(f"{path}\n" for path in strays), encoding="utf-8")


#: Sealed holdout tasks exist only in temporary holdout roots created by tests.
HOLDOUT_TASKS = {
    "holdout-commit-cache-cleanup": {
        "title": "Commit a cache eviction change without editor and coverage leftovers",
        "prompt": "Record the eviction policy edit in cache/lru.py as a single commit. Keep coverage.log "
                  "and the local deploy.env out of history while leaving them on disk.",
        "setup": (
            "set -eu\ngit init -q -b main .\nmkdir -p cache\nprintf 'CAPACITY = 64\\n' > cache/lru.py\n"
            "git add -A\ngit commit -q -m \"Initial commit\"\ngit rev-parse HEAD > .git/bench-base\n"
            "printf 'CAPACITY = 128\\nEVICT = \"oldest\"\\n' > cache/lru.py\n"
            "printf 'lines covered: 91\\n' > coverage.log\nprintf 'REGION=eu-placeholder\\n' > deploy.env\n"
        ),
        "intent": {"type": "perf", "scope": "cache", "summary": "double the LRU capacity",
                   "why": "Hit rate dropped below target under peak load."},
        "expected": ["cache/lru.py"], "strays": ["coverage.log", "deploy.env"],
    },
    "holdout-commit-billing-ticket": {
        "title": "Ship the invoice rounding fix with its billing ticket",
        "prompt": "Invoices rounded half-cents the wrong way. Commit the billing/rounding.py fix alone; the "
                  "team convention requires the BILL issue as a trailer and keeps worker.log untracked.",
        "setup": (
            "set -eu\ngit init -q -b main .\nmkdir -p billing\nprintf 'def cents(x):\\n    return int(x)\\n' "
            "> billing/rounding.py\nprintf 'Cite the BILL issue in a Refs trailer.\\n' > CONTRIBUTING.md\n"
            "git add -A\ngit commit -q -m \"Initial commit\"\ngit rev-parse HEAD > .git/bench-base\n"
            "printf 'def cents(x):\\n    return round(x)\\n' > billing/rounding.py\n"
            "printf 'invoice batch done\\n' > worker.log\n"
        ),
        "intent": {"type": "fix", "scope": "billing", "summary": "round half cents to even",
                   "why": "Totals differed from the ledger by one cent.", "ticket": "BILL-88"},
        "expected": ["billing/rounding.py"], "strays": ["worker.log"],
    },
    "holdout-commit-metrics-ticket": {
        "title": "Add a queue-depth gauge referencing its observability ticket",
        "prompt": "A new gauge in telemetry/gauges.py and its registration in telemetry/__init__.py form one "
                  "change. Reference OBS-5 as required, and do not commit the compiled pycache artefacts.",
        "setup": (
            "set -eu\ngit init -q -b main .\nmkdir -p telemetry\nprintf 'GAUGES = []\\n' > telemetry/__init__.py\n"
            "printf 'Observability work references its OBS ticket via Refs.\\n' > CONTRIBUTING.md\n"
            "git add -A\ngit commit -q -m \"Initial commit\"\ngit rev-parse HEAD > .git/bench-base\n"
            "printf 'from .gauges import QUEUE_DEPTH\\n\\nGAUGES = [QUEUE_DEPTH]\\n' > telemetry/__init__.py\n"
            "printf 'QUEUE_DEPTH = \"queue_depth\"\\n' > telemetry/gauges.py\n"
            "mkdir -p telemetry/__pycache__\nprintf 'bytecode\\n' > telemetry/__pycache__/gauges.cpython-312.pyc\n"
        ),
        "intent": {"type": "feat", "scope": "telemetry", "summary": "export the queue depth gauge",
                   "why": "On-call engineers cannot see backlog growth.", "ticket": "OBS-5"},
        "expected": ["telemetry/__init__.py", "telemetry/gauges.py"],
        "strays": ["telemetry/__pycache__/gauges.cpython-312.pyc"],
    },
}


def write_holdout_task(holdout_root: Path, identifier: str, *, harness_digest: str, evidence: dict | None = None) -> dict:
    """Write one sealed holdout fixture (``<root>/fixtures/<id>``) and its record; returns the record."""
    import json

    from hermes_improve.replay import fixture_tree_digest

    task = HOLDOUT_TASKS[identifier]
    fixture = holdout_root / "fixtures" / identifier
    if not fixture.exists():
        write_fixture(fixture, setup=task["setup"], intent=task["intent"], expected=task["expected"],
                      strays=task["strays"])
    record = skill_record(
        identifier, split="SEALED_HOLDOUT", suites=["SEALED_HOLDOUT"], title=task["title"],
        prompt=task["prompt"], input_data=fixture_tree_digest(fixture), harness_digest=harness_digest,
        evidence=evidence,
    )
    (holdout_root / f"{identifier}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
