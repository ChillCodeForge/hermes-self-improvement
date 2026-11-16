from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

from hermes_improve.models import Benchmark
from hermes_improve.sandbox import LocalSandboxExecutor, SandboxLimits, SandboxPolicyError
from hermes_improve.verifier import run_deterministic_verification, verify_deterministic_output
from hermes_improve.verifiers import (
    VerificationContext,
    VerifierRegistry,
    default_registry,
)


FIXTURES = Path(__file__).parent / "fixtures"
PY = sys.executable


def executor(**limits):
    return LocalSandboxExecutor(SandboxLimits(**limits), acknowledge_no_isolation=True)


def workspace_pair(tmp_path, before, after):
    source, workspace = tmp_path / "source", tmp_path / "workspace"
    for root, files in ((source, before), (workspace, after)):
        root.mkdir(parents=True)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return source, workspace


def context(tmp_path, *, before=None, after=None, output="", exit_code=0, **kwargs):
    source, workspace = workspace_pair(tmp_path, before or {}, after or {})
    return VerificationContext(
        exit_code=exit_code, output=output, workspace=workspace, source=source,
        executor=kwargs.pop("executor", executor()), **kwargs,
    )


def check(spec, ctx):
    return default_registry().create(spec).verify(ctx)


def test_default_registry_provides_all_spec_verifiers():
    assert {
        "command", "exit_code", "unit_test", "integration_test", "build", "lint", "typecheck",
        "file_exists", "file_content", "unchanged_path", "json_schema", "regex", "git_diff",
        "http_response", "mcp_health",
    } <= default_registry().types


def test_registry_rejects_duplicates_unknown_types_and_unknown_fields():
    registry = VerifierRegistry()
    registry.register("custom", lambda spec: None)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", lambda spec: None)
    with pytest.raises(ValueError, match="unsupported deterministic verifier"):
        default_registry().create({"type": "made_up"})
    with pytest.raises(ValueError, match="unsupported fields"):
        default_registry().create({"type": "file_exists", "path": "a", "follow_symlinks": True})
    with pytest.raises(ValueError, match="at most 64"):
        default_registry().build([{"type": "exit_code"}] * 65)
    assert default_registry().create("exit_code").expected == 0


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../../b", "."])
def test_file_verifiers_reject_path_escape_at_spec_time(path):
    with pytest.raises(ValueError):
        default_registry().create({"type": "file_exists", "path": path})
    with pytest.raises(ValueError):
        default_registry().create({"type": "file_content", "path": path, "contains": "x"})


def test_file_exists_does_not_follow_symlinks_out_of_workspace(tmp_path):
    ctx = context(tmp_path, after={"real.txt": "x"})
    (ctx.workspace / "link.txt").symlink_to("/etc/hostname")
    assert check({"type": "file_exists", "path": "real.txt"}, ctx).passed
    assert not check({"type": "file_exists", "path": "link.txt"}, ctx).passed
    assert check({"type": "file_exists", "path": "missing.txt", "must_exist": False}, ctx).passed
    assert not check({"type": "file_content", "path": "link.txt", "contains": "a"}, ctx).passed


def test_file_content_modes(tmp_path):
    ctx = context(tmp_path, after={"settings.json": '{"enabled": true}'})
    digest = hashlib.sha256(b'{"enabled": true}').hexdigest()
    assert check({"type": "file_content", "path": "settings.json", "contains": "enabled"}, ctx).passed
    assert check({"type": "file_content", "path": "settings.json", "not_contains": "secret"}, ctx).passed
    assert check({"type": "file_content", "path": "settings.json", "equals": '{"enabled": true}'}, ctx).passed
    assert check({"type": "file_content", "path": "settings.json", "sha256": digest}, ctx).passed
    assert not check({"type": "file_content", "path": "settings.json", "contains": "disabled"}, ctx).passed
    with pytest.raises(ValueError, match="exactly one"):
        default_registry().create({"type": "file_content", "path": "a", "contains": "x", "equals": "y"})


def test_unchanged_path_and_git_diff_use_workspace_changes(tmp_path):
    ctx = context(
        tmp_path,
        before={"src/app.py": "v1", "tests/test_app.py": "t", "docs/readme.md": "d"},
        after={"src/app.py": "v2", "tests/test_app.py": "t", "src/new.py": "n"},
    )
    assert check({"type": "unchanged_path", "path": "tests"}, ctx).passed
    assert not check({"type": "unchanged_path", "path": "docs"}, ctx).passed
    assert check({"type": "git_diff", "allowed_paths": ["src/", "docs/*"], "required_paths": ["src/app.py"]}, ctx).passed
    assert not check({"type": "git_diff", "forbidden_paths": ["docs/"]}, ctx).passed
    assert not check({"type": "git_diff", "max_changed_files": 2}, ctx).passed
    result = check({"type": "git_diff", "allowed_paths": ["src/app.py"]}, ctx)
    assert not result.passed and "src/new.py" in result.detail
    with pytest.raises(ValueError, match="at least one constraint"):
        default_registry().create({"type": "git_diff"})


def test_json_schema_validates_output_and_files(tmp_path):
    schema = {
        "type": "object", "required": ["answer", "items"], "additionalProperties": False,
        "properties": {
            "answer": {"type": "integer", "minimum": 0},
            "items": {"type": "array", "minItems": 1, "items": {"type": "string", "pattern": "^[a-z]+$"}},
        },
    }
    ctx = context(tmp_path, output='{"answer": 3, "items": ["ok"]}', after={"bad.json": '{"answer": true, "items": []}'})
    assert check({"type": "json_schema", "schema": schema}, ctx).passed
    bad = check({"type": "json_schema", "schema": schema, "source": "file", "path": "bad.json"}, ctx)
    assert not bad.passed and "answer" in bad.detail
    assert not check({"type": "json_schema", "schema": schema}, context(tmp_path / "x", output="NaN")).passed
    with pytest.raises(ValueError, match="unsupported keywords"):
        default_registry().create({"type": "json_schema", "schema": {"$ref": "#/x"}})


def test_regex_verifier_on_output_and_file(tmp_path):
    ctx = context(tmp_path, output="release-4\n", after={"log.txt": "ERROR: boom"})
    assert check({"type": "regex", "pattern": r"^release-\d+$"}, ctx).passed
    assert not check({"type": "regex", "pattern": "ERROR", "source": "file", "path": "log.txt", "must_match": False}, ctx).passed
    with pytest.raises(ValueError, match="valid regex"):
        default_registry().create({"type": "regex", "pattern": "("})


def test_command_style_verifiers_run_in_sandbox_copy_of_workspace(tmp_path):
    ctx = context(tmp_path, after={"marker.txt": "ready"})
    script = "import pathlib, sys; sys.exit(0 if pathlib.Path('marker.txt').read_text() == 'ready' else 1)"
    for kind in ("command", "build", "lint", "typecheck"):
        assert check({"type": kind, "argv": [PY, "-c", script]}, ctx).passed, kind
    writer = {"type": "command", "argv": [PY, "-c", "open('marker.txt', 'w').write('mutated')"]}
    assert check(writer, ctx).passed
    assert (ctx.workspace / "marker.txt").read_text() == "ready", "verifier commands must not mutate the evidence"
    assert not check({"type": "command", "argv": [PY, "-c", "print('nope')"], "output_contains": "yes"}, ctx).passed
    assert check({"type": "command", "argv": [PY, "-c", "raise SystemExit(3)"], "expected_exit_code": 3}, ctx).passed


def test_unit_and_integration_tests_reject_vacuous_runs(tmp_path):
    ctx = context(tmp_path, after={"test_sample.py": "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"})
    assert check({"type": "unit_test", "argv": [PY, "-m", "unittest", "-q"]}, ctx).passed
    empty = context(tmp_path / "empty", after={"README.md": "no tests"})
    for kind in ("unit_test", "integration_test"):
        result = check({"type": kind, "argv": [PY, "-m", "unittest", "-q"]}, empty)
        assert not result.passed and "no tests" in result.detail


def test_command_verifier_never_runs_on_host_without_executor(tmp_path):
    ctx = context(tmp_path, after={"a": "b"}, executor=None)
    with pytest.raises(SandboxPolicyError, match="host execution is not allowed"):
        check({"type": "command", "argv": [PY, "-c", "pass"]}, ctx)


def test_local_executor_scrubs_environment_and_enforces_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCTION_API_TOKEN", "do-not-leak")
    source = tmp_path / "in"
    source.mkdir()
    sandbox = executor(timeout_seconds=5)
    result = sandbox.execute([PY, "-c", "import os, json; print(json.dumps(sorted(os.environ)))"], input_root=source)
    visible = set(json.loads(result.output))
    assert "PRODUCTION_API_TOKEN" not in visible
    assert visible <= {"PATH", "HOME", "TMPDIR", "LANG", "PYTHONDONTWRITEBYTECODE", "LC_CTYPE"}
    slow = sandbox.execute([PY, "-c", "import time; time.sleep(30)"], input_root=source, timeout=1)
    assert slow.timed_out and slow.exit_code == 124
    with pytest.raises(SandboxPolicyError, match="within the sandbox limit"):
        sandbox.execute([PY, "-c", "pass"], input_root=source, timeout=60)
    with pytest.raises(SandboxPolicyError, match="acknowledge_no_isolation"):
        LocalSandboxExecutor()


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.com/")
            self.end_headers()
            return
        body = b'{"status": "ok", "version": 3}' if self.path == "/health" else b"missing"
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return None


@pytest.fixture
def http_service():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_http_response_verifier_only_reaches_allowlisted_hosts(tmp_path, http_service):
    ctx = context(tmp_path, http_allowlist=frozenset({http_service}))
    url = f"http://{http_service}"
    assert check({"type": "http_response", "url": f"{url}/health", "body_contains": '"ok"'}, ctx).passed
    assert check({"type": "http_response", "url": f"{url}/health", "body_regex": r'"version": \d'}, ctx).passed
    assert not check({"type": "http_response", "url": f"{url}/nope"}, ctx).passed
    redirect = check({"type": "http_response", "url": f"{url}/redirect"}, ctx)
    assert not redirect.passed and "302" in redirect.detail
    closed = context(tmp_path / "closed")
    with pytest.raises(SandboxPolicyError, match="not in the verifier allowlist"):
        check({"type": "http_response", "url": f"{url}/health"}, closed)
    with pytest.raises(ValueError, match="without credentials"):
        default_registry().create({"type": "http_response", "url": "http://user:pw@127.0.0.1/"})


def test_mcp_health_verifier_speaks_json_rpc_over_stdio_in_sandbox(tmp_path):
    ctx = context(tmp_path, after={"README.md": "server"})
    shutil.copy(FIXTURES / "mcp" / "stdio_server.py", ctx.workspace / "server.py")
    healthy = check({"type": "mcp_health", "argv": [PY, "server.py"], "expected_tools": ["read_file", "search"]}, ctx)
    assert healthy.passed, healthy.detail
    missing = check({"type": "mcp_health", "argv": [PY, "server.py"], "expected_tools": ["write_file"]}, ctx)
    assert not missing.passed and "write_file" in missing.detail
    broken = check({"type": "mcp_health", "argv": [PY, "server.py", "--broken"]}, ctx)
    assert not broken.passed and "tools/list" in broken.detail
    dead = check({"type": "mcp_health", "argv": [PY, "-c", "print('not json-rpc')"]}, ctx)
    assert not dead.passed and "initialize" in dead.detail


def benchmark(checks, *, expected_exit=0):
    return Benchmark.from_dict({
        "schema_version": 1, "id": "verifier-task", "version": 1,
        "source": {"type": "manual", "provenance": "unit-test"},
        "task": {"title": "Verifier test", "prompt": "Return a structured result"},
        "taxonomy": {"categories": [], "difficulty": "easy", "required_capabilities": []},
        "environment": {"container_image": "example/image@sha256:" + "a" * 64, "network_policy": "disabled"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": checks, "subjective": []},
        "ground_truth": {"type": "deterministic_test", "confidence": 1.0, "expected_exit_code": expected_exit},
        "benchmark": {"split": "TRAIN", "suites": ["SMOKE"], "cluster_id": "verifier", "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    })


def test_deterministic_verification_adds_implicit_exit_check_and_scores(tmp_path):
    ctx = context(tmp_path, after={"out.txt": "done"}, exit_code=2)
    outcome = run_deterministic_verification(benchmark([{"type": "file_exists", "path": "out.txt"}]), ctx)
    assert not outcome.passed
    assert outcome.quality == 50.0
    assert outcome.summaries() == [
        {"type": "exit_code", "passed": False}, {"type": "file_exists", "passed": True},
    ]


def test_output_only_verification_rejects_workspace_checks():
    with pytest.raises(ValueError, match="post-run workspace"):
        verify_deterministic_output(
            benchmark([{"type": "file_exists", "path": "out.txt"}]), exit_code=0, output=""
        )


def test_local_executor_snapshot_contains_workspace(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    (source / "a.txt").write_text("a")
    out = tmp_path / "out"
    out.mkdir()
    result = executor().execute(
        [PY, "-c", "open('b.txt', 'w').write('b')"], input_root=source, snapshot_dir=out,
    )
    assert result.snapshot is not None and result.snapshot.is_file()
    assert not (source / "b.txt").exists()
    assert os.path.getsize(result.snapshot) > 0
