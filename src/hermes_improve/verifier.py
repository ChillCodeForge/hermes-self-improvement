from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .models import Benchmark, canonical_json


_MAX_CHECKS = 64


def verify_deterministic_output(
    benchmark: Benchmark,
    *,
    exit_code: int,
    output: str,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Run bounded, declarative checks against captured command output."""
    expected_exit = benchmark.ground_truth.get("expected_exit_code", 0)
    if isinstance(expected_exit, bool) or not isinstance(expected_exit, int):
        raise ValueError("ground_truth.expected_exit_code must be an integer")

    raw_checks = benchmark.verification.get("deterministic", [])
    if not isinstance(raw_checks, list) or len(raw_checks) > _MAX_CHECKS:
        raise ValueError(f"verification.deterministic must be an array of at most {_MAX_CHECKS} checks")

    checks: list[tuple[str, Any, Any]] = []
    has_exit_check = False
    for raw in raw_checks:
        if isinstance(raw, str):
            check_type, expected, path = raw, None, None
        elif isinstance(raw, Mapping):
            check_type = raw.get("type")
            expected = raw.get("expected")
            path = raw.get("path")
        else:
            raise ValueError("deterministic checks must be names or objects")
        if not isinstance(check_type, str):
            raise ValueError("deterministic check objects require a string type")
        if check_type == "exit_code":
            has_exit_check = True
            if expected is None:
                if isinstance(raw, Mapping) and "expected" in raw:
                    raise ValueError("exit_code verifier expected value must be an integer")
                expected = expected_exit
            if isinstance(expected, bool) or not isinstance(expected, int):
                raise ValueError("exit_code verifier expected value must be an integer")
            if expected != expected_exit:
                raise ValueError("exit_code verifier must match ground_truth.expected_exit_code")
        elif check_type in {"output_exact", "output_contains", "output_not_contains"}:
            if not isinstance(expected, str):
                raise ValueError(f"{check_type} verifier expected value must be a string")
        elif check_type == "json_output_equals":
            if not isinstance(raw, Mapping) or "expected" not in raw:
                raise ValueError("json_output_equals verifier requires an expected value")
        elif check_type == "json_path_equals":
            if not isinstance(path, list) or not path:
                raise ValueError("json_path_equals verifier requires a non-empty path array")
            if any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in path):
                raise ValueError("json_path_equals path elements must be strings or integers")
            if not isinstance(raw, Mapping) or "expected" not in raw:
                raise ValueError("json_path_equals verifier requires an expected value")
        else:
            raise ValueError(f"unsupported deterministic verifier: {check_type!r}")
        checks.append((check_type, expected, path))

    if not has_exit_check and len(checks) >= _MAX_CHECKS:
        raise ValueError(f"at most {_MAX_CHECKS} effective deterministic checks are supported")
    if not has_exit_check:
        checks.insert(0, ("exit_code", expected_exit, None))

    results: list[dict[str, Any]] = []
    for check_type, expected, path in checks:
        passed = _check(check_type, expected, path, exit_code, output)
        results.append({"type": check_type, "passed": passed})
    passed_count = sum(result["passed"] for result in results)
    return passed_count == len(results), 100.0 * passed_count / len(results), results


def _check(check_type: str, expected: Any, path: Any, exit_code: int, output: str) -> bool:
    if check_type == "exit_code":
        return exit_code == expected
    if check_type == "output_exact":
        return output == expected
    if check_type == "output_contains":
        return expected in output
    if check_type == "output_not_contains":
        return expected not in output

    try:
        parsed = json.loads(output, parse_constant=_reject_nonstandard_constant)
    except (ValueError, TypeError):
        return False
    if check_type == "json_output_equals":
        return canonical_json(parsed) == canonical_json(expected)
    if check_type == "json_path_equals":
        value = parsed
        for part in path:
            if isinstance(value, Mapping) and isinstance(part, str) and part in value:
                value = value[part]
            elif isinstance(value, list) and isinstance(part, int) and 0 <= part < len(value):
                value = value[part]
            else:
                return False
        return canonical_json(value) == canonical_json(expected)
    raise AssertionError(f"unhandled verifier {check_type!r}")


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")
