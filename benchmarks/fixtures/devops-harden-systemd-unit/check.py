import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

REQUIRED = {
    "NoNewPrivileges": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "PrivateTmp": "yes",
    "ReadWritePaths": "/var/lib/report-worker",
}


def sections(text):
    result, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            result.setdefault(current, [])
        else:
            key, _, value = line.partition("=")
            result[current].append((key, value))
    return result


def body(checks):
    unit = sections(read("report-worker.service"))
    service = unit.get("Service", [])
    keys = [key for key, _ in service]
    values = dict(service)
    for key, value in REQUIRED.items():
        checks.check(f"service:{key}", values.get(key) == value)
    checks.check("no_duplicate_keys", len(keys) == len(set(keys)))
    checks.check("exec_unchanged", values.get("ExecStart") == "/usr/local/bin/report-worker --queue nightly")
    checks.check("user_unchanged", values.get("User") == "report" and values.get("Restart") == "on-failure")
    checks.check("other_sections_unchanged", unit.get("Unit") == [
        ("Description", "Nightly report worker"), ("After", "network-online.target"),
    ] and unit.get("Install") == [("WantedBy", "multi-user.target")])


run(body)
