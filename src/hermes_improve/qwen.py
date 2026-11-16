from __future__ import annotations

import hashlib
import json
import ipaddress
import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


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

    @classmethod
    def from_env(cls) -> "QwenConfig":
        base_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
        model = os.environ.get("QWEN_MODEL", "")
        if not base_url or not model:
            raise ValueError("QWEN_BASE_URL and QWEN_MODEL must be set for local model requests")
        timeout = float(os.environ.get("QWEN_TIMEOUT", "60"))
        concurrency = int(os.environ.get("QWEN_MAX_CONCURRENCY", "2"))
        if timeout <= 0 or concurrency < 1:
            raise ValueError("QWEN_TIMEOUT and QWEN_MAX_CONCURRENCY must be positive")
        config = cls(
            base_url=base_url,
            model=model,
            judge_model=os.environ.get("QWEN_JUDGE_MODEL"),
            api_key=os.environ.get("QWEN_API_KEY"),
            timeout=timeout,
            max_concurrency=concurrency,
            log_dir=os.environ.get("QWEN_LOG_DIR") or None,
        )
        config.validate_endpoint()
        return config

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


QWEN_ROLES = frozenset({"qwen_solver", "qwen_optimizer", "qwen_dreamer", "qwen_judge"})


class OpenAICompatibleQwen:
    """Small synchronous adapter for local OpenAI-compatible chat endpoints.

    Logical roles share the endpoint but keep separate models (``qwen_judge``
    may use ``judge_model``), statistics and, when ``log_dir`` is set, separate
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

    def complete(self, *, role: str, system_prompt: str, user_payload: str) -> str:
        if role not in QWEN_ROLES:
            raise ValueError(f"unknown Qwen logical role: {role}")
        if not isinstance(system_prompt, str) or not isinstance(user_payload, str):
            raise TypeError("Qwen prompts and payloads must be strings")
        model = self.config.judge_model if role == "qwen_judge" and self.config.judge_model else self.config.model
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": f"[logical_role={role}]\n{system_prompt}"},
                {"role": "user", "content": user_payload},
            ],
            "temperature": 0,
        }).encode("utf-8")
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("Qwen request exceeds the 1 MiB payload limit")
        started = time.monotonic()
        try:
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"local model request failed: {exc}") from exc
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
