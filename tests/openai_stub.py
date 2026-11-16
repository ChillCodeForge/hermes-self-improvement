"""Local OpenAI-compatible chat-completions server for tests (real HTTP on loopback)."""

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
