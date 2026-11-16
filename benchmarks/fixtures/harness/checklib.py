"""Shared helpers for corpus check.py scripts: one-line JSON report, exit 0 iff all pass."""

import hashlib
import json
import sys
from pathlib import Path


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool]] = []

    def check(self, name: str, condition: object) -> None:
        self.results.append((name, bool(condition)))

    def finish(self) -> None:
        failed = [name for name, passed in self.results if not passed]
        report = {"checks": len(self.results), "failed": failed, "status": "fail" if failed else "pass"}
        print(json.dumps(report, sort_keys=True))
        sys.exit(1 if failed else 0)


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run(body) -> None:
    checks = Checks()
    try:
        body(checks)
    except Exception as exc:  # report, never crash without a JSON verdict
        checks.check(f"error:{type(exc).__name__}", False)
    checks.finish()
