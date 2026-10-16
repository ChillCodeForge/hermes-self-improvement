from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VerificationResult:
    name: str
    passed: bool
    detail: str


class Verifier(Protocol):
    def verify(self, subject: str | Path) -> VerificationResult: ...


@dataclass(frozen=True)
class ExitCodeVerifier:
    command: tuple[str, ...]
    timeout_seconds: float = 60

    def verify(self, subject: str | Path) -> VerificationResult:
        try:
            result = subprocess.run(
                self.command,
                cwd=Path(subject),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(type(self).__name__, False, str(exc))
        return VerificationResult(type(self).__name__, result.returncode == 0, result.stdout[-4000:])


@dataclass(frozen=True)
class FileExistsVerifier:
    relative_path: str

    def verify(self, subject: str | Path) -> VerificationResult:
        root = Path(subject).resolve()
        target = (root / self.relative_path).resolve()
        if target != root and root not in target.parents:
            return VerificationResult(type(self).__name__, False, "path escapes subject root")
        return VerificationResult(type(self).__name__, target.exists(), str(target))


@dataclass(frozen=True)
class FileContentVerifier:
    relative_path: str
    expected: str

    def verify(self, subject: str | Path) -> VerificationResult:
        root = Path(subject).resolve()
        target = (root / self.relative_path).resolve()
        if target != root and root not in target.parents:
            return VerificationResult(type(self).__name__, False, "path escapes subject root")
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return VerificationResult(type(self).__name__, False, str(exc))
        return VerificationResult(type(self).__name__, self.expected in content, "expected content present" if self.expected in content else "expected content absent")


@dataclass(frozen=True)
class RegexVerifier:
    pattern: str

    def verify(self, subject: str | Path) -> VerificationResult:
        matched = re.search(self.pattern, str(subject)) is not None
        return VerificationResult(type(self).__name__, matched, "matched" if matched else "did not match")


@dataclass(frozen=True)
class JsonSchemaVerifier:
    required_keys: tuple[str, ...]

    def verify(self, subject: str | Path) -> VerificationResult:
        try:
            value = json.loads(str(subject))
        except json.JSONDecodeError as exc:
            return VerificationResult(type(self).__name__, False, str(exc))
        passed = isinstance(value, dict) and set(self.required_keys).issubset(value)
        return VerificationResult(type(self).__name__, passed, "required keys present" if passed else "required keys missing")


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, name: str, verifier: Verifier) -> None:
        if not name or name in self._verifiers:
            raise ValueError(f"verifier name is empty or already registered: {name!r}")
        self._verifiers[name] = verifier

    def run(self, subject: str | Path) -> list[VerificationResult]:
        return [verifier.verify(subject) for verifier in self._verifiers.values()]
