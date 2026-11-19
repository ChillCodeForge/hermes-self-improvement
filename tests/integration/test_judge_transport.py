"""Judge transport failures fail closed instead of crashing the evaluation (lab finding).

A dropped connection to the local model (e.g. an Ollama restart while the
judge runs) raised ``http.client.RemoteDisconnected``/``ConnectionResetError``
straight through the judge. Judging happens after every paired run, so the
whole evaluation was discarded with nothing recorded.
"""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from hermes_improve.judge import PairwiseJudge
from hermes_improve.qwen import OpenAICompatibleQwen, QwenConfig


class BrokenServer:
    """Loopback HTTP endpoint that reads each request and then misbehaves.

    ``mode``: ``close`` closes without a response, ``reset`` aborts the
    connection with RST, ``garbage`` answers with a non-HTTP status line.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.connections = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self.base_url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"

    def __enter__(self) -> "BrokenServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            self.connections += 1
            with connection:
                connection.settimeout(2)
                data = b""
                try:
                    while b"\r\n\r\n" not in data:
                        chunk = connection.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                except OSError:
                    continue
                if self.mode == "reset":
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                elif self.mode == "garbage":
                    connection.sendall(b"NOT-HTTP garbage\r\n\r\n")


def client(server: BrokenServer) -> OpenAICompatibleQwen:
    return OpenAICompatibleQwen(QwenConfig(server.base_url, "solver-model", judge_model="judge-model", timeout=5))


@pytest.mark.parametrize("mode", ["close", "reset", "garbage"])
def test_transport_failures_are_wrapped_as_local_model_request_failures(mode):
    with BrokenServer(mode) as server:
        model = client(server)
        with pytest.raises(RuntimeError, match="local model request failed"):
            model.complete(role="qwen_judge", system_prompt="s", user_payload="{}", json_object=True)
    assert server.connections >= 1


@pytest.mark.parametrize("mode", ["close", "reset"])
def test_dropped_judge_connection_yields_unreliable_verdict(mode):
    with BrokenServer(mode) as server:
        verdict = PairwiseJudge(client(server)).compare(
            task_prompt="Summarize the incident", criteria=["mentions root cause"],
            baseline_output="baseline answer", candidate_output="candidate answer",
        )
    assert verdict.result == "unreliable"
    assert verdict.confidence is None and verdict.uncertainty is None
    errors = [item["error"] for item in verdict.presentations]
    assert all(error and "local model request failed" in error for error in errors)
