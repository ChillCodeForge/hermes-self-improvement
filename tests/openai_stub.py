"""Local OpenAI-compatible chat-completions and embeddings servers for tests (real HTTP on loopback)."""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Callable
from typing import Any


Reply = Callable[[dict[str, Any]], "str | tuple[int, str]"]


class OpenAIStub:
    """Serves ``POST /v1/chat/completions``; ``reply`` maps the request body to content.

    ``reply`` may return a string (sent as the assistant message content) or a
    ``(status, raw_body)`` tuple to simulate transport-level failures.
    """

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                stub.requests.append(body)
                stub.headers.append(dict(self.headers))
                if self.path != "/v1/chat/completions":
                    self._send(404, b"not found")
                    return
                answer = stub.reply(body)
                if isinstance(answer, tuple):
                    status, raw = answer
                    self._send(status, raw.encode())
                    return
                payload = json.dumps({
                    "id": f"chatcmpl-{len(stub.requests)}",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                }).encode()
                self._send(200, payload)

            def _send(self, status, payload):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "OpenAIStub":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def user_payloads(self) -> list[dict[str, Any]]:
        return [json.loads(request["messages"][1]["content"]) for request in self.requests]


class EmbeddingStub:
    """Serves ``POST /v1/embeddings`` on loopback; ``embed`` maps one input text to a vector.

    ``reverse`` returns the data rows in reverse index order (clients must sort
    by ``index``); ``failure`` makes every request answer with that raw body.
    """

    def __init__(self, embed: Callable[[str], list[float]], *, reverse: bool = False,
                 failure: tuple[int, str] | None = None) -> None:
        self.embed = embed
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                stub.requests.append(body)
                stub.headers.append(dict(self.headers))
                if self.path != "/v1/embeddings":
                    self._send(404, b"not found")
                    return
                if failure is not None:
                    self._send(failure[0], failure[1].encode())
                    return
                rows = [
                    {"object": "embedding", "index": index, "embedding": stub.embed(text)}
                    for index, text in enumerate(body["input"])
                ]
                if reverse:
                    rows.reverse()
                payload = json.dumps({
                    "object": "list", "model": body.get("model"), "data": rows,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }).encode()
                self._send(200, payload)

            def _send(self, status, payload):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> "EmbeddingStub":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def inputs(self) -> list[str]:
        return [text for request in self.requests for text in request["input"]]


ScriptedReply = Callable[[str, str, bytes], "tuple[int, str, bytes]"]


def chat_completion_json(prompt_tokens: int | None, completion_tokens: int | None) -> bytes:
    """Non-streaming chat.completion body; ``None`` token counts omit ``usage``."""
    body: dict[str, Any] = {
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    if prompt_tokens is not None:
        body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "total_tokens": prompt_tokens + (completion_tokens or 0)}
    return json.dumps(body).encode()


def chat_completion_sse(prompt_tokens: int | None, completion_tokens: int | None) -> bytes:
    """Streaming body as sent with ``stream_options.include_usage`` (usage in the last chunk)."""
    events = [
        {"id": "c", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "o"}}]},
        {"id": "c", "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"content": "k"}, "finish_reason": "stop"}]},
    ]
    if prompt_tokens is not None:
        events.append({"id": "c", "object": "chat.completion.chunk", "choices": [],
                       "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                 "total_tokens": prompt_tokens + (completion_tokens or 0)}})
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


class ScriptedModelStub:
    """Loopback model endpoint whose ``reply(method, path, body)`` returns ``(status, content_type, raw)``.

    SSE bodies (``text/event-stream``) are sent with chunked transfer encoding in
    several writes, like a real streaming server; others with Content-Length.
    Every request's method, path, headers and body are recorded.
    """

    def __init__(self, reply: ScriptedReply) -> None:
        self.reply = reply
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _handle(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                stub.requests.append((self.command, self.path, dict(self.headers), body))
                status, content_type, raw = stub.reply(self.command, self.path, body)
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                if content_type == "text/event-stream":
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for piece in raw.split(b"\n\n"):
                        if piece:
                            piece += b"\n\n"
                            self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                            self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

            do_GET = _handle
            do_POST = _handle

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> "ScriptedModelStub":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
