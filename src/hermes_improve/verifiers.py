"""Deterministic verifier plugins created from declarative benchmark specs.

Benchmarks declare checks in ``verification.deterministic`` as objects with a
``type`` (or a bare type string for parameterless checks). ``VerifierRegistry``
maps each type to a factory that validates the spec strictly and returns a
verifier. Verifiers inspect a ``VerificationContext``:

* host-side checks only *read* the extracted post-run workspace, the pre-run
  source tree and the captured task output;
* command-style checks (command, unit/integration tests, build, lint,
  typecheck, MCP health) never run on the host: they are executed through the
  context's ``SandboxExecutor`` against a fresh copy of the post-run workspace;
* network checks (HTTP) are refused unless the target ``host:port`` is in the
  evaluator-configured allowlist and never follow redirects.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Callable, ClassVar, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .models import canonical_json
from .sandbox import (
    SandboxExecutor,
    SandboxPolicyError,
    WorkspaceChanges,
    compute_changes,
    glob_match,
    validate_argv,
    validate_relative_path,
)


MAX_CHECKS = 64
_MAX_DETAIL = 2000
_MAX_INSPECT_BYTES = 1024 * 1024
_MAX_PATTERN = 512
_MAX_HTTP_TIMEOUT = 30.0
_MAX_SCHEMA_DEPTH = 32
_VACUOUS_TEST_RUN = re.compile(r"\bno tests ran\b|\bcollected 0 items\b|\bRan 0 tests\b", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationResult:
    name: str
    passed: bool
    detail: str
    label: str = ""

    def summary(self) -> dict[str, Any]:
        row: dict[str, Any] = {"type": self.name, "passed": self.passed}
        if self.label:
            row["label"] = self.label
        return row


@dataclass(frozen=True)
class VerificationContext:
    """Evidence a verifier may inspect. Paths are host copies, never executed."""

    exit_code: int
    output: str
    workspace: Path | None = None
    source: Path | None = None
    executor: SandboxExecutor | None = None
    http_allowlist: frozenset[str] = frozenset()
    special_entries: tuple[str, ...] = ()
    command_timeout: float | None = None

    @cached_property
    def changes(self) -> WorkspaceChanges:
        workspace, source = self.require_workspace("git_diff"), self.require_source("git_diff")
        return compute_changes(source, workspace, self.special_entries)

    def require_workspace(self, verifier: str) -> Path:
        if self.workspace is None:
            raise ValueError(f"{verifier} verifier requires a captured post-run workspace")
        return self.workspace

    def require_source(self, verifier: str) -> Path:
        if self.source is None:
            raise ValueError(f"{verifier} verifier requires the pre-run source tree")
        return self.source

    def require_executor(self, verifier: str) -> SandboxExecutor:
        if self.executor is None:
            raise SandboxPolicyError(f"{verifier} verifier requires a sandbox executor; host execution is not allowed")
        return self.executor


class Verifier(Protocol):
    type_name: ClassVar[str]

    def verify(self, context: VerificationContext) -> VerificationResult: ...


VerifierFactory = Callable[[Mapping[str, Any]], Verifier]


def _detail(text: str) -> str:
    return text[-_MAX_DETAIL:]


def _check_keys(spec: Mapping[str, Any], type_name: str, required: set[str], optional: set[str]) -> None:
    keys = set(spec) - {"type", "label"}
    missing = required - keys
    if missing:
        raise ValueError(f"{type_name} verifier requires: {', '.join(sorted(missing))}")
    unknown = keys - required - optional
    if unknown:
        raise ValueError(f"{type_name} verifier has unsupported fields: {', '.join(sorted(unknown))}")
    label = spec.get("label", "")
    if not isinstance(label, str) or len(label) > 128:
        raise ValueError("verifier label must be a string of at most 128 characters")


def _label(spec: Mapping[str, Any]) -> str:
    return str(spec.get("label", ""))


def _int(spec: Mapping[str, Any], key: str, default: int | None = None, *, minimum: int | None = None) -> int:
    value = spec.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        raise ValueError(f"verifier field {key!r} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _timeout(spec: Mapping[str, Any], *, maximum: float | None = None) -> float | None:
    if "timeout_seconds" not in spec:
        return None
    value = spec["timeout_seconds"]
    if (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        or value <= 0 or (maximum is not None and value > maximum)
    ):
        raise ValueError("verifier timeout_seconds must be a finite positive number within limits")
    return float(value)


def _pattern(value: Any, key: str = "pattern") -> re.Pattern[str]:
    if not isinstance(value, str) or not value or len(value) > _MAX_PATTERN:
        raise ValueError(f"verifier {key} must be a non-empty regex of at most {_MAX_PATTERN} characters")
    try:
        return re.compile(value)
    except re.error as exc:
        raise ValueError(f"verifier {key} is not a valid regex: {exc}") from exc


def _read_workspace_file(context: VerificationContext, path: PurePosixPath, verifier: str) -> bytes | None:
    root = context.require_workspace(verifier).resolve()
    target = root.joinpath(*path.parts)
    if target.is_symlink():
        return None
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    with resolved.open("rb") as stream:
        data = stream.read(_MAX_INSPECT_BYTES + 1)
    if len(data) > _MAX_INSPECT_BYTES:
        raise ValueError(f"{verifier} verifier input exceeds the 1 MiB inspection limit")
    return data


# --------------------------------------------------------------------------- task status


@dataclass(frozen=True)
class ExitCodeVerifier:
    """Checks the benchmark task's own exit code (agent completion for Hermes runs)."""

    type_name: ClassVar[str] = "exit_code"
    expected: int
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "ExitCodeVerifier":
        _check_keys(spec, cls.type_name, set(), {"expected"})
        if "expected" in spec and (isinstance(spec["expected"], bool) or not isinstance(spec["expected"], int)):
            raise ValueError("exit_code verifier expected value must be an integer")
        return cls(int(spec.get("expected", 0)), _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        passed = context.exit_code == self.expected
        return VerificationResult(
            self.type_name, passed, f"exit code {context.exit_code}, expected {self.expected}", self.label
        )


# --------------------------------------------------------------------------- output checks


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_json(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return False, None


def _resolve_path(value: Any, path: Sequence[Any]) -> tuple[bool, Any]:
    for part in path:
        if isinstance(value, Mapping) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and isinstance(part, int) and not isinstance(part, bool) and 0 <= part < len(value):
            value = value[part]
        else:
            return False, None
    return True, value


def _json_path(spec: Mapping[str, Any], type_name: str) -> tuple[Any, ...]:
    path = spec.get("path")
    if not isinstance(path, list) or not path:
        raise ValueError(f"{type_name} verifier requires a non-empty path array")
    if any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in path):
        raise ValueError(f"{type_name} path elements must be strings or integers")
    return tuple(path)


@dataclass(frozen=True)
class OutputCheckVerifier:
    """Checks on the captured task output (exact/contains/JSON value checks)."""

    type_name: ClassVar[str] = "output"
    kind: str
    expected: Any = None
    path: tuple[Any, ...] = ()
    label: str = ""

    KINDS: ClassVar[frozenset[str]] = frozenset({
        "output_exact", "output_contains", "output_not_contains", "json_output_equals",
        "json_required_keys", "json_path_exists", "json_path_equals",
    })

    @classmethod
    def factory(cls, kind: str) -> VerifierFactory:
        def build(spec: Mapping[str, Any]) -> "OutputCheckVerifier":
            return cls.from_spec(kind, spec)

        return build

    @classmethod
    def from_spec(cls, kind: str, spec: Mapping[str, Any]) -> "OutputCheckVerifier":
        if kind not in cls.KINDS:
            raise ValueError(f"unsupported deterministic verifier: {kind!r}")
        expected = spec.get("expected")
        path: tuple[Any, ...] = ()
        if kind in {"output_exact", "output_contains", "output_not_contains"}:
            _check_keys(spec, kind, set(), {"expected"})
            if not isinstance(expected, str):
                raise ValueError(f"{kind} verifier expected value must be a string")
        elif kind == "json_output_equals":
            _check_keys(spec, kind, set(), {"expected"})
            if "expected" not in spec:
                raise ValueError("json_output_equals verifier requires an expected value")
        elif kind == "json_required_keys":
            _check_keys(spec, kind, set(), {"expected"})
            if (
                not isinstance(expected, list) or not expected or len(expected) > MAX_CHECKS
                or not all(isinstance(key, str) and key for key in expected)
                or len(set(expected)) != len(expected)
            ):
                raise ValueError("json_required_keys verifier requires a non-empty array of unique strings")
            expected = tuple(expected)
        elif kind == "json_path_exists":
            _check_keys(spec, kind, {"path"}, set())
            path = _json_path(spec, kind)
        else:
            _check_keys(spec, kind, set(), {"path", "expected"})
            path = _json_path(spec, kind)
            if "expected" not in spec:
                raise ValueError("json_path_equals verifier requires an expected value")
        return cls(kind, expected, path, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        output = context.output
        kind = self.kind
        if kind == "output_exact":
            passed = output == self.expected
        elif kind == "output_contains":
            passed = self.expected in output
        elif kind == "output_not_contains":
            passed = self.expected not in output
        else:
            parsed_ok, parsed = _parse_json(output)
            if not parsed_ok:
                return VerificationResult(kind, False, "output is not standard JSON", self.label)
            if kind == "json_output_equals":
                passed = canonical_json(parsed) == canonical_json(self.expected)
            elif kind == "json_required_keys":
                passed = isinstance(parsed, Mapping) and all(key in parsed for key in self.expected)
            elif kind == "json_path_exists":
                passed, _value = _resolve_path(parsed, self.path)
            else:
                found, value = _resolve_path(parsed, self.path)
                passed = found and canonical_json(value) == canonical_json(self.expected)
        return VerificationResult(kind, passed, "matched" if passed else "did not match", self.label)


# --------------------------------------------------------------------------- sandboxed commands


@dataclass(frozen=True)
class CommandVerifier:
    """Runs a command against a fresh sandbox copy of the post-run workspace."""

    type_name: ClassVar[str] = "command"
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: float | None = None
    expected_exit_code: int = 0
    output_contains: str | None = None
    output_regex: str | None = None
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]):
        _check_keys(
            spec, cls.type_name, {"argv"},
            {"cwd", "timeout_seconds", "expected_exit_code", "output_contains", "output_regex"},
        )
        argv = spec["argv"]
        if not isinstance(argv, list):
            raise ValueError(f"{cls.type_name} verifier argv must be an array")
        cwd = spec.get("cwd", ".")
        validate_relative_path(cwd, label=f"{cls.type_name} verifier cwd")
        contains = spec.get("output_contains")
        if contains is not None and (not isinstance(contains, str) or not contains):
            raise ValueError(f"{cls.type_name} verifier output_contains must be a non-empty string")
        regex = spec.get("output_regex")
        if regex is not None:
            _pattern(regex, "output_regex")
        return cls(
            validate_argv(argv), cwd, _timeout(spec),
            _int(spec, "expected_exit_code", 0), contains, regex, _label(spec),
        )

    def verify(self, context: VerificationContext) -> VerificationResult:
        executor = context.require_executor(self.type_name)
        workspace = context.require_workspace(self.type_name)
        result = executor.execute(
            self.argv, input_root=workspace, cwd=self.cwd,
            timeout=self.timeout_seconds or context.command_timeout,
        )
        problems = []
        if result.timed_out:
            problems.append("command timed out")
        if result.exit_code != self.expected_exit_code:
            problems.append(f"exit code {result.exit_code}, expected {self.expected_exit_code}")
        if self.output_contains is not None and self.output_contains not in result.output:
            problems.append("expected output text absent")
        if self.output_regex is not None and re.search(self.output_regex, result.output) is None:
            problems.append("output did not match output_regex")
        problems.extend(self._extra_problems(result.output))
        detail = "; ".join(problems) if problems else "passed"
        return VerificationResult(self.type_name, not problems, _detail(f"{detail}\n{result.output}"), self.label)

    def _extra_problems(self, output: str) -> list[str]:
        return []


@dataclass(frozen=True)
class UnitTestVerifier(CommandVerifier):
    """Test command; a run that executed zero tests is a failure, not a pass."""

    type_name: ClassVar[str] = "unit_test"

    def _extra_problems(self, output: str) -> list[str]:
        return ["test command executed no tests"] if _VACUOUS_TEST_RUN.search(output) else []


@dataclass(frozen=True)
class IntegrationTestVerifier(UnitTestVerifier):
    type_name: ClassVar[str] = "integration_test"


@dataclass(frozen=True)
class BuildVerifier(CommandVerifier):
    type_name: ClassVar[str] = "build"


@dataclass(frozen=True)
class LintVerifier(CommandVerifier):
    type_name: ClassVar[str] = "lint"


@dataclass(frozen=True)
class TypecheckVerifier(CommandVerifier):
    type_name: ClassVar[str] = "typecheck"


# --------------------------------------------------------------------------- workspace inspection


@dataclass(frozen=True)
class FileExistsVerifier:
    type_name: ClassVar[str] = "file_exists"
    path: str
    must_exist: bool = True
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "FileExistsVerifier":
        _check_keys(spec, cls.type_name, {"path"}, {"must_exist"})
        validate_relative_path(spec["path"], label="file_exists path", allow_root=False)
        must_exist = spec.get("must_exist", True)
        if not isinstance(must_exist, bool):
            raise ValueError("file_exists must_exist must be a boolean")
        return cls(spec["path"], must_exist, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        root = context.require_workspace(self.type_name).resolve()
        target = root.joinpath(*PurePosixPath(self.path).parts)
        exists = not target.is_symlink() and target.exists()
        if exists:
            resolved = target.resolve()
            exists = resolved == root or root in resolved.parents
        passed = exists == self.must_exist
        state = "exists" if exists else "absent"
        return VerificationResult(self.type_name, passed, f"{self.path} {state}", self.label)


@dataclass(frozen=True)
class FileContentVerifier:
    type_name: ClassVar[str] = "file_content"
    path: str
    mode: str
    value: str
    label: str = ""

    MODES: ClassVar[tuple[str, ...]] = ("contains", "not_contains", "equals", "sha256")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "FileContentVerifier":
        _check_keys(spec, cls.type_name, {"path"}, set(cls.MODES))
        validate_relative_path(spec["path"], label="file_content path", allow_root=False)
        modes = [mode for mode in cls.MODES if mode in spec]
        if len(modes) != 1:
            raise ValueError("file_content verifier requires exactly one of contains, not_contains, equals, sha256")
        value = spec[modes[0]]
        if not isinstance(value, str) or (modes[0] != "equals" and not value):
            raise ValueError(f"file_content {modes[0]} must be a string")
        if modes[0] == "sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("file_content sha256 must be a lowercase SHA-256 hex digest")
        return cls(spec["path"], modes[0], value, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        data = _read_workspace_file(context, PurePosixPath(self.path), self.type_name)
        if data is None:
            return VerificationResult(self.type_name, False, f"{self.path} is not a regular workspace file", self.label)
        if self.mode == "sha256":
            passed = hashlib.sha256(data).hexdigest() == self.value
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return VerificationResult(self.type_name, False, f"{self.path} is not UTF-8 text", self.label)
            if self.mode == "contains":
                passed = self.value in text
            elif self.mode == "not_contains":
                passed = self.value not in text
            else:
                passed = text == self.value
        return VerificationResult(
            self.type_name, passed, f"{self.mode} check {'passed' if passed else 'failed'} for {self.path}", self.label
        )


@dataclass(frozen=True)
class UnchangedPathVerifier:
    """Fails if any file at or below ``path`` was added, modified, or deleted."""

    type_name: ClassVar[str] = "unchanged_path"
    path: str
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "UnchangedPathVerifier":
        _check_keys(spec, cls.type_name, {"path"}, set())
        validate_relative_path(spec["path"], label="unchanged_path path")
        return cls(spec["path"], _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        prefix = PurePosixPath(self.path).as_posix()
        changed = [
            path for path in context.changes.changed_paths
            if prefix == "." or path == prefix or path.startswith(prefix + "/")
        ]
        detail = "unchanged" if not changed else "changed: " + ", ".join(changed[:20])
        return VerificationResult(self.type_name, not changed, detail, self.label)


@dataclass(frozen=True)
class GitDiffVerifier:
    """Constrains the candidate's workspace diff (allowed/forbidden/required paths, size)."""

    type_name: ClassVar[str] = "git_diff"
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    max_changed_files: int | None = None
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "GitDiffVerifier":
        fields = {"allowed_paths", "forbidden_paths", "required_paths", "max_changed_files"}
        _check_keys(spec, cls.type_name, set(), fields)
        if not fields & set(spec):
            raise ValueError("git_diff verifier requires at least one constraint")
        lists = {}
        for key in ("allowed_paths", "forbidden_paths", "required_paths"):
            value = spec.get(key, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item and not item.startswith("/") and ".." not in item.split("/")
                for item in value
            ):
                raise ValueError(f"git_diff {key} must be an array of relative path patterns")
            lists[key] = tuple(value)
        maximum = _int(spec, "max_changed_files", minimum=0) if "max_changed_files" in spec else None
        return cls(lists["allowed_paths"], lists["forbidden_paths"], lists["required_paths"], maximum, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        changed = context.changes.changed_paths
        problems = []
        if self.allowed_paths:
            outside = [path for path in changed if not glob_match(path, self.allowed_paths)]
            if outside:
                problems.append("changes outside allowed paths: " + ", ".join(outside[:20]))
        forbidden = [path for path in changed if glob_match(path, self.forbidden_paths)]
        if forbidden:
            problems.append("forbidden paths changed: " + ", ".join(forbidden[:20]))
        missing = [pattern for pattern in self.required_paths if not any(glob_match(path, (pattern,)) for path in changed)]
        if missing:
            problems.append("required changes missing: " + ", ".join(missing))
        if self.max_changed_files is not None and len(changed) > self.max_changed_files:
            problems.append(f"{len(changed)} changed files exceed limit {self.max_changed_files}")
        return VerificationResult(
            self.type_name, not problems, "; ".join(problems) or f"{len(changed)} changed files within policy", self.label
        )


# --------------------------------------------------------------------------- structured content


_SCHEMA_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_SCHEMA_KEYWORDS = {
    "type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
    "pattern", "minItems", "maxItems", "description", "title",
}


def _validate_schema(schema: Any, depth: int = 0) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise ValueError("json_schema is nested too deeply")
    if not isinstance(schema, Mapping):
        raise ValueError("json_schema schemas must be objects")
    unknown = set(schema) - _SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"json_schema uses unsupported keywords: {', '.join(sorted(unknown))}")
    kind = schema.get("type")
    if kind is not None:
        kinds = kind if isinstance(kind, list) else [kind]
        if not kinds or not all(isinstance(item, str) and item in _SCHEMA_TYPES for item in kinds):
            raise ValueError("json_schema type must name JSON types")
    for key in ("properties",):
        if key in schema:
            if not isinstance(schema[key], Mapping):
                raise ValueError("json_schema properties must be an object")
            for child in schema[key].values():
                _validate_schema(child, depth + 1)
    if "items" in schema:
        _validate_schema(schema["items"], depth + 1)
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, bool):
        _validate_schema(additional, depth + 1)
    if "required" in schema and (
        not isinstance(schema["required"], list) or not all(isinstance(item, str) for item in schema["required"])
    ):
        raise ValueError("json_schema required must be an array of strings")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("json_schema enum must be a non-empty array")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], (int, float))):
            raise ValueError(f"json_schema {key} must be a number")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], int) or schema[key] < 0):
            raise ValueError(f"json_schema {key} must be a non-negative integer")
    if "pattern" in schema:
        _pattern(schema["pattern"])


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def schema_errors(value: Any, schema: Mapping[str, Any], location: str = "$") -> list[str]:
    errors: list[str] = []
    kind = schema.get("type")
    if kind is not None:
        kinds = set(kind if isinstance(kind, list) else [kind])
        actual = _json_type(value)
        if not (actual in kinds or (actual == "integer" and "number" in kinds)):
            return [f"{location}: expected {sorted(kinds)}, got {actual}"]
    if "const" in schema and canonical_json(value) != canonical_json(schema["const"]):
        errors.append(f"{location}: value differs from const")
    if "enum" in schema and canonical_json(value) not in {canonical_json(item) for item in schema["enum"]}:
        errors.append(f"{location}: value not in enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{location}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{location}: above maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{location}: not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{location}: not below exclusiveMaximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{location}: shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{location}: longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{location}: does not match pattern")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{location}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{location}: more than maxItems")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema["items"], f"{location}[{index}]"))
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{location}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                errors.extend(schema_errors(item, properties[key], f"{location}.{key}"))
            elif additional is False:
                errors.append(f"{location}: unexpected property {key!r}")
            elif isinstance(additional, Mapping):
                errors.extend(schema_errors(item, additional, f"{location}.{key}"))
    return errors


def _source_text(spec: Mapping[str, Any], type_name: str) -> tuple[str, str | None]:
    source = spec.get("source", "output")
    if source not in {"output", "file"}:
        raise ValueError(f"{type_name} source must be 'output' or 'file'")
    path = spec.get("path")
    if source == "file":
        validate_relative_path(path, label=f"{type_name} path", allow_root=False)
    elif path is not None:
        raise ValueError(f"{type_name} path is only valid with source 'file'")
    return source, path


def _load_subject(context: VerificationContext, source: str, path: str | None, verifier: str) -> str | None:
    if source == "output":
        return context.output[-_MAX_INSPECT_BYTES:]
    data = _read_workspace_file(context, PurePosixPath(path or ""), verifier)
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass(frozen=True)
class JsonSchemaVerifier:
    """Validates task output or a workspace file against a JSON Schema subset."""

    type_name: ClassVar[str] = "json_schema"
    schema: Mapping[str, Any]
    source: str = "output"
    path: str | None = None
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "JsonSchemaVerifier":
        _check_keys(spec, cls.type_name, {"schema"}, {"source", "path"})
        _validate_schema(spec["schema"])
        source, path = _source_text(spec, cls.type_name)
        return cls(json.loads(canonical_json(spec["schema"])), source, path, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        text = _load_subject(context, self.source, self.path, self.type_name)
        if text is None:
            return VerificationResult(self.type_name, False, f"{self.path} is not a readable UTF-8 file", self.label)
        parsed_ok, value = _parse_json(text)
        if not parsed_ok:
            return VerificationResult(self.type_name, False, "subject is not standard JSON", self.label)
        errors = schema_errors(value, self.schema)
        return VerificationResult(self.type_name, not errors, "; ".join(errors[:20]) or "schema satisfied", self.label)


@dataclass(frozen=True)
class RegexVerifier:
    type_name: ClassVar[str] = "regex"
    pattern: str
    source: str = "output"
    path: str | None = None
    must_match: bool = True
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "RegexVerifier":
        _check_keys(spec, cls.type_name, {"pattern"}, {"source", "path", "must_match"})
        _pattern(spec["pattern"])
        source, path = _source_text(spec, cls.type_name)
        must_match = spec.get("must_match", True)
        if not isinstance(must_match, bool):
            raise ValueError("regex must_match must be a boolean")
        return cls(spec["pattern"], source, path, must_match, _label(spec))

    def verify(self, context: VerificationContext) -> VerificationResult:
        text = _load_subject(context, self.source, self.path, self.type_name)
        if text is None:
            return VerificationResult(self.type_name, False, f"{self.path} is not a readable UTF-8 file", self.label)
        matched = re.search(self.pattern, text, re.MULTILINE) is not None
        passed = matched == self.must_match
        return VerificationResult(self.type_name, passed, "matched" if matched else "did not match", self.label)


# --------------------------------------------------------------------------- network / services


@dataclass(frozen=True)
class HTTPResponseVerifier:
    """Requests an allowlisted HTTP endpoint without following redirects."""

    type_name: ClassVar[str] = "http_response"
    url: str
    method: str = "GET"
    body: str | None = None
    expected_status: int = 200
    body_contains: str | None = None
    body_regex: str | None = None
    timeout_seconds: float = 10.0
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "HTTPResponseVerifier":
        _check_keys(
            spec, cls.type_name, {"url"},
            {"method", "body", "expected_status", "body_contains", "body_regex", "timeout_seconds"},
        )
        url = spec["url"]
        parts = urlsplit(url) if isinstance(url, str) else None
        if (
            parts is None or parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username or parts.password or parts.fragment
        ):
            raise ValueError("http_response url must be an absolute http(s) URL without credentials")
        method = spec.get("method", "GET")
        if method not in {"GET", "HEAD", "POST"}:
            raise ValueError("http_response method must be GET, HEAD or POST")
        body = spec.get("body")
        if body is not None and (method != "POST" or not isinstance(body, str) or len(body) > 64 * 1024):
            raise ValueError("http_response body is only allowed for POST and must be a short string")
        contains = spec.get("body_contains")
        if contains is not None and (not isinstance(contains, str) or not contains):
            raise ValueError("http_response body_contains must be a non-empty string")
        regex = spec.get("body_regex")
        if regex is not None:
            _pattern(regex, "body_regex")
        status = _int(spec, "expected_status", 200, minimum=100)
        timeout = _timeout(spec, maximum=_MAX_HTTP_TIMEOUT) or 10.0
        return cls(url, method, body, status, contains, regex, timeout, _label(spec))

    @property
    def authority(self) -> str:
        parts = urlsplit(self.url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return f"{parts.hostname}:{port}"

    def verify(self, context: VerificationContext) -> VerificationResult:
        if self.authority not in context.http_allowlist:
            raise SandboxPolicyError(f"http_response target {self.authority} is not in the verifier allowlist")
        parts = urlsplit(self.url)
        connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parts.hostname, parts.port, timeout=self.timeout_seconds)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            connection.request(
                self.method, target, body=self.body.encode() if self.body is not None else None,
                headers={"User-Agent": "hermes-improve-verifier", "Connection": "close"},
            )
            response = connection.getresponse()
            payload = response.read(_MAX_INSPECT_BYTES + 1)
            status = response.status
        except (OSError, http.client.HTTPException) as exc:
            return VerificationResult(self.type_name, False, f"request failed: {exc}", self.label)
        finally:
            connection.close()
        problems = []
        if len(payload) > _MAX_INSPECT_BYTES:
            problems.append("response body exceeds 1 MiB")
        text = payload[:_MAX_INSPECT_BYTES].decode("utf-8", errors="replace")
        if status != self.expected_status:
            problems.append(f"status {status}, expected {self.expected_status}")
        if self.body_contains is not None and self.body_contains not in text:
            problems.append("expected body text absent")
        if self.body_regex is not None and re.search(self.body_regex, text) is None:
            problems.append("body did not match body_regex")
        return VerificationResult(self.type_name, not problems, "; ".join(problems) or f"status {status}", self.label)


@dataclass(frozen=True)
class MCPHealthVerifier:
    """Starts an MCP stdio server in the sandbox and checks initialize + tools/list."""

    type_name: ClassVar[str] = "mcp_health"
    argv: tuple[str, ...]
    cwd: str = "."
    timeout_seconds: float | None = None
    expected_tools: tuple[str, ...] = ()
    protocol_version: str = "2025-06-18"
    label: str = ""

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "MCPHealthVerifier":
        _check_keys(spec, cls.type_name, {"argv"}, {"cwd", "timeout_seconds", "expected_tools", "protocol_version"})
        if not isinstance(spec["argv"], list):
            raise ValueError("mcp_health argv must be an array")
        cwd = spec.get("cwd", ".")
        validate_relative_path(cwd, label="mcp_health cwd")
        tools = spec.get("expected_tools", [])
        if not isinstance(tools, list) or not all(isinstance(item, str) and item for item in tools):
            raise ValueError("mcp_health expected_tools must be an array of tool names")
        version = spec.get("protocol_version", "2025-06-18")
        if not isinstance(version, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", version):
            raise ValueError("mcp_health protocol_version must be a YYYY-MM-DD string")
        return cls(validate_argv(spec["argv"]), cwd, _timeout(spec), tuple(tools), version, _label(spec))

    def requests(self) -> bytes:
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": self.protocol_version, "capabilities": {},
                "clientInfo": {"name": "hermes-improve-verifier", "version": "1"},
            }},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        return "".join(json.dumps(message) + "\n" for message in messages).encode()

    def verify(self, context: VerificationContext) -> VerificationResult:
        executor = context.require_executor(self.type_name)
        workspace = context.require_workspace(self.type_name)
        result = executor.execute(
            self.argv, input_root=workspace, cwd=self.cwd,
            timeout=self.timeout_seconds or context.command_timeout, stdin=self.requests(),
        )
        responses: dict[int, Any] = {}
        for line in result.output.splitlines():
            ok, message = _parse_json(line.strip())
            if ok and isinstance(message, dict) and message.get("jsonrpc") == "2.0" and message.get("id") in {1, 2}:
                responses.setdefault(message["id"], message)
        problems = []
        initialize = responses.get(1, {})
        init_result = initialize.get("result") if isinstance(initialize, dict) else None
        if not isinstance(init_result, dict) or not isinstance(init_result.get("protocolVersion"), str) \
                or not isinstance(init_result.get("serverInfo"), dict):
            problems.append("initialize did not return protocolVersion and serverInfo")
        listing = responses.get(2, {})
        list_result = listing.get("result") if isinstance(listing, dict) else None
        tools = list_result.get("tools") if isinstance(list_result, dict) else None
        names: set[str] = set()
        if not isinstance(tools, list):
            problems.append("tools/list did not return a tools array")
        else:
            names = {tool.get("name") for tool in tools if isinstance(tool, dict) and isinstance(tool.get("name"), str)}
            missing = sorted(set(self.expected_tools) - names)
            if missing:
                problems.append("expected tools missing: " + ", ".join(missing))
        if result.timed_out:
            problems.append("MCP server did not exit after stdin closed")
        return VerificationResult(
            self.type_name, not problems, "; ".join(problems) or f"healthy, {len(names)} tools", self.label
        )


# --------------------------------------------------------------------------- registry


class VerifierRegistry:
    """Maps declarative verifier type names to validating factories."""

    def __init__(self, factories: Mapping[str, VerifierFactory] | None = None) -> None:
        self._factories: dict[str, VerifierFactory] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(self, type_name: str, factory: VerifierFactory) -> None:
        if not isinstance(type_name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", type_name):
            raise ValueError(f"verifier type name is invalid: {type_name!r}")
        if type_name in self._factories:
            raise ValueError(f"verifier type is already registered: {type_name!r}")
        if not callable(factory):
            raise TypeError("verifier factory must be callable")
        self._factories[type_name] = factory

    @property
    def types(self) -> frozenset[str]:
        return frozenset(self._factories)

    def create(self, spec: str | Mapping[str, Any]) -> Verifier:
        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, Mapping):
            raise ValueError("deterministic checks must be names or objects")
        type_name = spec.get("type")
        if not isinstance(type_name, str):
            raise ValueError("deterministic check objects require a string type")
        factory = self._factories.get(type_name)
        if factory is None:
            raise ValueError(f"unsupported deterministic verifier: {type_name!r}")
        return factory(spec)

    def build(self, specs: Sequence[str | Mapping[str, Any]]) -> tuple[Verifier, ...]:
        if not isinstance(specs, list) or len(specs) > MAX_CHECKS:
            raise ValueError(f"verification.deterministic must be an array of at most {MAX_CHECKS} checks")
        return tuple(self.create(spec) for spec in specs)


_CORE_VERIFIERS: tuple[type, ...] = (
    CommandVerifier, ExitCodeVerifier, UnitTestVerifier, IntegrationTestVerifier, BuildVerifier,
    LintVerifier, TypecheckVerifier, FileExistsVerifier, FileContentVerifier, UnchangedPathVerifier,
    JsonSchemaVerifier, RegexVerifier, GitDiffVerifier, HTTPResponseVerifier, MCPHealthVerifier,
)


def default_registry() -> VerifierRegistry:
    registry = VerifierRegistry()
    for verifier in _CORE_VERIFIERS:
        registry.register(verifier.type_name, verifier.from_spec)
    for kind in sorted(OutputCheckVerifier.KINDS):
        registry.register(kind, OutputCheckVerifier.factory(kind))
    return registry
