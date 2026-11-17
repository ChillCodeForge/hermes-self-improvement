"""Loopback proxy that meters token usage between Hermes and its model endpoint.

Hermes' ``batch_runner`` discards the token counters that ``run_conversation``
returns (see ``hermes_runner.py``), so the evaluation runner cannot read
tokens from Hermes itself. ``MeteringProxy`` sits between batch_runner and the
configured OpenAI-compatible model endpoint instead: batch_runner gets the
proxy's loopback URL as ``--base_url``, the proxy forwards every request
unchanged to the real endpoint and reads the ``usage`` object of each
generation response on the way back.

Measurement is all-or-nothing per run. ``UsageReport.complete`` is true only
when every successful (2xx) generation response carried a usage object and no
generation request was still in flight when the run ended. One response
without usage (for example a server that ignores ``stream_options``) marks the
whole run as unmeasured, so the policy sees "not measured" instead of an
undercount. Non-2xx responses are passed through and counted separately; the
provider returned no completion for them.

Usage shapes understood:

* OpenAI chat/completions, non-streaming: top-level ``usage`` with
  ``prompt_tokens``/``completion_tokens``;
* OpenAI streaming with ``stream_options.include_usage`` (Hermes v0.22.2 sets
  it, ``agent/chat_completion_helpers.py``): the final SSE chunk's ``usage``;
* OpenAI Responses API: ``usage`` or ``response.usage`` with
  ``input_tokens``/``output_tokens`` (streaming: ``response.completed``);
* Anthropic Messages: ``message.usage`` in ``message_start`` plus ``usage`` in
  ``message_delta``; per field the largest reported value is kept;
* Ollama native ``/api/chat`` and ``/api/generate``: ``prompt_eval_count`` and
  ``eval_count`` of the final (``done``) object, JSON or NDJSON stream.

Every path is forwarded to the upstream host, not only paths below the
configured base path. Hermes probes the server root to recognise the backend,
for example Ollama ``/api/tags`` and ``/api/show``. Its context-window floor
(64K in v0.22.2) and the Ollama ``num_ctx`` handling depend on that probe.
Before this change the proxy answered such probes with 404, Hermes did not
recognise Ollama behind the loopback URL, and it ran a model it would refuse
when called directly (lab finding F5). Only generation paths are metered.

The proxy never modifies request or response bodies and never logs headers
(the ``Authorization`` header carries the model API key). It listens on
127.0.0.1 only; Hermes tools run in containers with networking disabled and
cannot reach it.
"""

from __future__ import annotations

import http.client
import http.server
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping


_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding",
}
_GENERATION_SUFFIXES = ("/chat/completions", "/completions", "/responses", "/messages", "/api/chat", "/api/generate")
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_PARSED_RESPONSE_BYTES = 64 * 1024 * 1024
_READ_CHUNK = 64 * 1024


@dataclass(frozen=True)
class CallUsage:
    """Usage of one successful generation response."""

    path: str
    status: int
    streamed: bool
    input_tokens: int | None
    output_tokens: int | None

    @property
    def reported(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


@dataclass(frozen=True)
class UsageReport:
    calls: tuple[CallUsage, ...]
    failed_calls: int
    in_flight: int

    @property
    def generation_calls(self) -> int:
        return len(self.calls)

    @property
    def complete(self) -> bool:
        return self.in_flight == 0 and all(call.reported for call in self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens or 0 for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens or 0 for call in self.calls)

    @property
    def max_context_tokens(self) -> int:
        return max((call.input_tokens or 0 for call in self.calls), default=0)

    def as_record(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "generation_calls": self.generation_calls,
            "failed_calls": self.failed_calls,
            "in_flight": self.in_flight,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": [
                {"path": call.path, "status": call.status, "streamed": call.streamed,
                 "input_tokens": call.input_tokens, "output_tokens": call.output_tokens}
                for call in self.calls
            ],
        }


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage_from(container: object) -> tuple[int | None, int | None]:
    if not isinstance(container, Mapping):
        return None, None
    if "prompt_eval_count" in container or "eval_count" in container:
        # Ollama native API: counters at the top level of the final object.
        return _token_count(container.get("prompt_eval_count")), _token_count(container.get("eval_count"))
    usage = container.get("usage")
    if not isinstance(usage, Mapping):
        return None, None
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _token_count(usage.get("prompt_tokens")), _token_count(usage.get("completion_tokens"))
    return _token_count(usage.get("input_tokens")), _token_count(usage.get("output_tokens"))


def extract_usage(document: object) -> tuple[int | None, int | None]:
    """Return ``(input, output)`` tokens from one JSON response or SSE event."""
    found_in, found_out = None, None
    candidates = [document]
    if isinstance(document, Mapping):
        candidates += [document.get("response"), document.get("message")]
    for candidate in candidates:
        tokens_in, tokens_out = _usage_from(candidate)
        if tokens_in is not None:
            found_in = tokens_in if found_in is None else max(found_in, tokens_in)
        if tokens_out is not None:
            found_out = tokens_out if found_out is None else max(found_out, tokens_out)
    return found_in, found_out


def usage_from_body(body: bytes, streamed: bool) -> tuple[int | None, int | None]:
    """Parse a complete response body (JSON, or SSE when ``streamed``)."""
    if not streamed:
        try:
            return extract_usage(json.loads(body))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # not one JSON document: try NDJSON (Ollama native streaming)
    tokens_in = tokens_out = None
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if streamed:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
        else:
            payload = line
        if not payload or payload == b"[DONE]":
            continue
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        event_in, event_out = extract_usage(event)
        if event_in is not None:
            tokens_in = event_in if tokens_in is None else max(tokens_in, event_in)
        if event_out is not None:
            tokens_out = event_out if tokens_out is None else max(tokens_out, event_out)
    return tokens_in, tokens_out


@dataclass
class _Meter:
    lock: threading.Condition = field(default_factory=threading.Condition)
    calls: list[CallUsage] = field(default_factory=list)
    failed: int = 0
    in_flight: int = 0

    def snapshot(self, settle_seconds: float) -> UsageReport:
        # A client can hold the complete response before the handler thread has
        # booked it; wait (bounded) until every started request is booked.
        with self.lock:
            self.lock.wait_for(lambda: self.in_flight == 0, timeout=settle_seconds)
            return UsageReport(tuple(self.calls), self.failed, self.in_flight)


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream: urllib.parse.SplitResult, meter: _Meter, timeout: float) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.upstream = upstream
        self.base_path = upstream.path.rstrip("/")
        self.meter = meter
        self.upstream_timeout = timeout
        self.opener = urllib.request.build_opener(_NoRedirect)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - urllib hook
        return None


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return None

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._forward(b"")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if size < 0 or size > _MAX_REQUEST_BYTES:
            self.send_error(413, "request body too large for metering proxy")
            return
        self._forward(self.rfile.read(size))

    def _forward(self, body: bytes) -> None:
        split = urllib.parse.urlsplit(self.path)
        path = split.path
        if not path.startswith("/") or split.scheme or split.netloc:
            self.send_error(400, "origin-form request path required")
            return
        upstream = self.server.upstream
        target = urllib.parse.urlunsplit((upstream.scheme, upstream.netloc, path, split.query, ""))
        headers = {key: value for key, value in self.headers.items() if key.lower() not in _HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"
        generation = self.command == "POST" and path.rstrip("/").endswith(_GENERATION_SUFFIXES)
        meter = self.server.meter
        if generation:
            with meter.lock:
                meter.in_flight += 1
        try:
            self._relay(target, body, headers, path, generation)
        finally:
            if generation:
                with meter.lock:
                    meter.in_flight -= 1
                    meter.lock.notify_all()

    def _relay(self, target: str, body: bytes, headers: dict[str, str], path: str, generation: bool) -> None:
        request = urllib.request.Request(
            target, data=body if self.command == "POST" else None, headers=headers, method=self.command,
        )
        try:
            response = self.server.opener.open(request, timeout=self.server.upstream_timeout)
        except urllib.error.HTTPError as error:
            response = error
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            self._record_failure(generation)
            self.send_error(502, "model endpoint unreachable")
            return
        status = response.status if hasattr(response, "status") else response.code
        streamed = "text/event-stream" in (response.headers.get("Content-Type") or "")
        self.send_response_only(status)
        for key, value in response.headers.items():
            if key.lower() not in _HOP_BY_HOP or key.lower() == "content-length":
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        captured = bytearray()
        overflow = False
        cut = False
        try:
            while True:
                chunk = response.read1(_READ_CHUNK) if hasattr(response, "read1") else response.read(_READ_CHUNK)
                if not chunk:
                    break
                if generation and not overflow:
                    if len(captured) + len(chunk) > _MAX_PARSED_RESPONSE_BYTES:
                        overflow = True
                        captured.clear()
                    else:
                        captured += chunk
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            cut = True
        finally:
            response.close()
        if not generation:
            return
        if not 200 <= status < 300:
            self._record_failure(True)
            return
        tokens_in, tokens_out = (None, None) if overflow or cut else usage_from_body(bytes(captured), streamed)
        with self.server.meter.lock:
            self.server.meter.calls.append(CallUsage(path, status, streamed, tokens_in, tokens_out))

    def _record_failure(self, generation: bool) -> None:
        if generation:
            with self.server.meter.lock:
                self.server.meter.failed += 1


class MeteringProxy:
    """Context manager running one loopback proxy for one evaluation run.

    ``base_url`` is the URL to hand to Hermes; ``report()`` returns the usage
    seen so far. Leaving the context stops the server.
    """

    def __init__(self, upstream_base_url: str, *, timeout: float = 600.0) -> None:
        upstream = urllib.parse.urlsplit(upstream_base_url)
        if upstream.scheme not in {"http", "https"} or not upstream.netloc:
            raise ValueError("metering proxy upstream must be an http(s) URL")
        if upstream.query or upstream.fragment:
            raise ValueError("metering proxy upstream must not carry a query or fragment")
        self._upstream = upstream
        self._timeout = timeout
        self._meter = _Meter()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MeteringProxy":
        self._server = _Server(self._upstream, self._meter, self._timeout)
        self._thread = threading.Thread(target=self._server.serve_forever, name="metering-proxy", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("metering proxy is not running")
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}{self._server.base_path}"

    def report(self, settle_seconds: float = 5.0) -> UsageReport:
        """Usage booked so far; waits up to ``settle_seconds`` for open requests to finish.

        Requests still open after that stay in ``in_flight`` and make the
        report incomplete.
        """
        return self._meter.snapshot(settle_seconds)
