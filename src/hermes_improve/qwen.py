from __future__ import annotations

import hashlib
import http.client
import json
import ipaddress
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlparse

if TYPE_CHECKING:
    from .config import ModelsConfig


_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class QwenConfig:
    base_url: str
    model: str
    judge_model: str | None = None
    api_key: str | None = None
    timeout: float = 60
    max_concurrency: int = 2
    log_dir: str | None = None
    # Per logical role model, resolved from config/models.yaml role entries
    # (environment-variable names only; no model name is ever hardcoded).
    role_models: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_env(cls, models: "ModelsConfig | None" = None, environ: Mapping[str, str] | None = None) -> "QwenConfig":
        if models is None:
            from .config import config_from_environment

            models = config_from_environment().models
        environ = os.environ if environ is None else environ
        endpoint = models.endpoint_env
        base_url = environ.get(endpoint["base_url"], "").rstrip("/")
        model_env = shared_model_env(models)
        model = environ.get(model_env, "")
        if not base_url or not model:
            raise ValueError(f"{endpoint['base_url']} and {model_env} must be set for local model requests")
        timeout = float(environ.get(endpoint["timeout"], "60"))
        concurrency = int(environ.get(endpoint["max_concurrency"], "2"))
        if timeout <= 0 or concurrency < 1:
            raise ValueError(f"{endpoint['timeout']} and {endpoint['max_concurrency']} must be positive")
        config = cls(
            base_url=base_url,
            model=model,
            judge_model=(
                environ.get(models.roles["qwen_judge"].model_env) if "qwen_judge" in models.roles else None
            ),
            api_key=environ.get(endpoint["api_key"]),
            timeout=timeout,
            max_concurrency=concurrency,
            log_dir=environ.get("QWEN_LOG_DIR") or None,
            role_models=MappingProxyType(resolve_role_models(models, environ)),
        )
        config.validate_endpoint()
        return config

    def model_for(self, role: str) -> str:
        """Model name used for ``role``: its configured model, else the shared default."""
        if role not in QWEN_ROLES:
            raise ValueError(f"unknown Qwen logical role: {role}")
        if role in self.role_models:
            return self.role_models[role]
        if role == "qwen_judge" and self.judge_model:
            return self.judge_model
        return self.model

    def validate_endpoint(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Qwen base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Qwen base URL must not contain credentials, query, or fragment")
        hostname = parsed.hostname.lower()
        try:
            local = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            local = hostname == "localhost"
        if not local and os.environ.get("QWEN_ALLOW_REMOTE", "").lower() != "true":
            raise ValueError("Qwen endpoint must be localhost unless QWEN_ALLOW_REMOTE=true is explicitly set")
        if (
            not self.model.strip() or self.timeout <= 0 or not math.isfinite(self.timeout)
            or self.max_concurrency < 1
            or (self.judge_model is not None and not self.judge_model.strip())
        ):
            raise ValueError("model, positive timeout, and positive concurrency are required")
        for role, model in self.role_models.items():
            if role not in QWEN_ROLES or not isinstance(model, str) or not model.strip():
                raise ValueError(f"role model for {role!r} must be a non-empty model name of a known role")


QWEN_ROLES = frozenset({"qwen_solver", "qwen_optimizer", "qwen_dreamer", "qwen_judge"})


def shared_model_env(models: "ModelsConfig") -> str:
    """Environment variable naming the shared default model, taken from the role configuration.

    It is the variable all roles fall back to when they name a single common
    ``fallback_model_env``; otherwise the ``qwen_solver`` role's own
    ``model_env``. The variable name is not hardcoded here.
    """
    fallbacks = {entry.fallback_model_env for entry in models.roles.values() if entry.fallback_model_env}
    if len(fallbacks) == 1:
        return next(iter(fallbacks))
    solver = models.roles.get("qwen_solver")
    if solver is None:
        raise ValueError("model configuration must define the qwen_solver role")
    return solver.model_env


def resolve_role_models(models: "ModelsConfig", environ: Mapping[str, str]) -> dict[str, str]:
    """Resolve each logical role's model from its configured environment variables.

    A role whose ``model_env`` is unset uses ``fallback_model_env`` (default
    ``QWEN_MODEL``); a role with neither set is omitted and uses the shared model.
    """
    resolved = {}
    for role, entry in models.roles.items():
        if role not in QWEN_ROLES:
            raise ValueError(f"unknown Qwen logical role in model configuration: {role}")
        value = environ.get(entry.model_env) or environ.get(entry.fallback_model_env or "QWEN_MODEL")
        if value is not None and value.strip():
            resolved[role] = value
    return resolved


class _ResponseFormatRejected(Exception):
    """The endpoint answered HTTP 400/422 naming ``response_format`` (no JSON mode support)."""


class OpenAICompatibleQwen:
    """Small synchronous adapter for local OpenAI-compatible chat endpoints.

    Logical roles share the endpoint but keep separate models (``role_models``
    from config/models.yaml; ``qwen_judge`` falls back to ``judge_model``),
    statistics and, when ``log_dir`` is set, separate
    JSONL logs (``<log_dir>/<role>.jsonl``). Logs record digests, token usage,
    latency and status, never prompt or response text.
    """

    def __init__(self, config: QwenConfig) -> None:
        config.validate_endpoint()
        self.config = config
        self._semaphore = threading.BoundedSemaphore(config.max_concurrency)
        self._stats_lock = threading.Lock()
        self._stats: dict[str, dict[str, float | int]] = {}
        self._log_lock = threading.Lock()
        self._log_dir = Path(config.log_dir) if config.log_dir else None
        self._json_mode_supported = True
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    @property
    def stats(self) -> dict[str, dict[str, float | int]]:
        with self._stats_lock:
            return {role: dict(values) for role, values in self._stats.items()}

    def role_log_path(self, role: str) -> Path | None:
        if role not in QWEN_ROLES:
            raise ValueError(f"unknown Qwen logical role: {role}")
        return self._log_dir / f"{role}.jsonl" if self._log_dir is not None else None

    def complete(self, *, role: str, system_prompt: str, user_payload: str, json_object: bool = False) -> str:
        """One chat completion for a logical role.

        ``json_object=True`` (callers whose contract is a single JSON object:
        Dreaming REM/DEEP, optimizer, candidate content, judge) sends
        ``response_format: {"type": "json_object"}``. Small local models
        otherwise answer with a top-level array or prose and every strict
        parse fails (lab finding F6). An endpoint that rejects the field with
        HTTP 400/422 naming ``response_format`` is retried once without it, and
        JSON mode stays off for this client afterwards.
        """
        if role not in QWEN_ROLES:
            raise ValueError(f"unknown Qwen logical role: {role}")
        if not isinstance(system_prompt, str) or not isinstance(user_payload, str):
            raise TypeError("Qwen prompts and payloads must be strings")
        model = self.config.model_for(role)
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"[logical_role={role}]\n{system_prompt}"},
                {"role": "user", "content": user_payload},
            ],
            "temperature": 0,
        }
        if json_object is True and self._json_mode_supported:
            request["response_format"] = {"type": "json_object"}
        body = json.dumps(request).encode("utf-8")
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("Qwen request exceeds the 1 MiB payload limit")
        started = time.monotonic()
        try:
            try:
                content, prompt_tokens, completion_tokens = self._request(body)
            except _ResponseFormatRejected:
                self._json_mode_supported = False
                request.pop("response_format", None)
                body = json.dumps(request).encode("utf-8")
                content, prompt_tokens, completion_tokens = self._request(body)
        except (RuntimeError, ValueError) as exc:
            self._log(role, model, body, None, started, 0, 0, str(exc))
            raise
        elapsed_ms = self._log(role, model, body, content, started, prompt_tokens, completion_tokens, None)
        with self._stats_lock:
            metrics = self._stats.setdefault(role, {
                "requests": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0,
            })
            metrics["requests"] = int(metrics["requests"]) + 1
            metrics["input_tokens"] = int(metrics["input_tokens"]) + prompt_tokens
            metrics["output_tokens"] = int(metrics["output_tokens"]) + completion_tokens
            metrics["latency_ms"] = float(metrics["latency_ms"]) + elapsed_ms
        return content

    def _request(self, body: bytes) -> tuple[str, int, int]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            f"{self.config.base_url}/v1/chat/completions", data=body, headers=headers, method="POST"
        )
        with self._semaphore:
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
                    if len(raw) > _MAX_RESPONSE_BYTES:
                        raise ValueError("Qwen response exceeds the 2 MiB limit")
                    payload = json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096) if exc.fp is not None else b""
                if exc.code in {400, 422} and b"response_format" in detail:
                    raise _ResponseFormatRejected() from exc
                raise RuntimeError(f"local model request failed: {exc}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"local model request failed: {exc}") from exc
            except (OSError, http.client.HTTPException) as exc:
                # A dropped or reset connection (e.g. the model server restarting) is a
                # transport failure like a timeout: callers fail closed on RuntimeError.
                raise RuntimeError(f"local model request failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("local model response must be a JSON object")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("local model response has an invalid chat-completion shape") from exc
        if not isinstance(content, str):
            raise ValueError("local model response content must be a string")
        usage = payload.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        if (
            isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
            or isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int)
        ):
            prompt_tokens = completion_tokens = 0
        return content, prompt_tokens, completion_tokens

    def _log(
        self, role: str, model: str, body: bytes, content: str | None, started: float,
        prompt_tokens: int, completion_tokens: int, error: str | None,
    ) -> float:
        elapsed_ms = (time.monotonic() - started) * 1000
        path = self.role_log_path(role)
        if path is None:
            return elapsed_ms
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": model,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "response_sha256": hashlib.sha256(content.encode()).hexdigest() if content is not None else None,
            "status": "ok" if error is None else "error",
            "error": error[:500] if error else None,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "latency_ms": round(elapsed_ms, 3),
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._log_lock:
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(line)
        return elapsed_ms
