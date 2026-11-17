from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from hermes_improve.metering_proxy import MeteringProxy, extract_usage, usage_from_body
from openai_stub import ScriptedModelStub, chat_completion_json, chat_completion_sse


def post(url, body, headers=None):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.headers.get("Content-Type"), response.read()


def test_non_streaming_usage_is_metered_and_body_and_auth_pass_unchanged():
    raw = chat_completion_json(120, 30)
    with ScriptedModelStub(lambda method, path, body: (200, "application/json", raw)) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            assert proxy.base_url.startswith("http://127.0.0.1:") and proxy.base_url.endswith("/v1")
            assert proxy.base_url != upstream.base_url
            status, _type, body = post(proxy.base_url + "/chat/completions", {"model": "m"},
                                       {"Authorization": "Bearer secret-key"})
            report = proxy.report()
    assert status == 200 and body == raw
    method, path, headers, sent = upstream.requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert headers["Authorization"] == "Bearer secret-key"
    assert json.loads(sent) == {"model": "m"}
    assert report.complete and report.generation_calls == 1 and report.failed_calls == 0
    assert (report.input_tokens, report.output_tokens, report.max_context_tokens) == (120, 30, 120)


def test_streamed_usage_from_include_usage_chunk_and_stream_is_byte_identical():
    raw = chat_completion_sse(200, 15)
    with ScriptedModelStub(lambda method, path, body: (200, "text/event-stream", raw)) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            status, content_type, body = post(proxy.base_url + "/chat/completions", {"stream": True})
            report = proxy.report()
    assert status == 200 and content_type == "text/event-stream" and body == raw
    assert report.complete and report.calls[0].streamed
    assert (report.input_tokens, report.output_tokens) == (200, 15)


def test_one_generation_response_without_usage_makes_the_run_unmeasured():
    replies = iter([chat_completion_json(10, 5), chat_completion_sse(None, None)])

    def reply(method, path, body):
        raw = next(replies)
        return 200, "text/event-stream" if raw.startswith(b"data:") else "application/json", raw

    with ScriptedModelStub(reply) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            post(proxy.base_url + "/chat/completions", {})
            post(proxy.base_url + "/chat/completions", {"stream": True})
            report = proxy.report()
    assert report.generation_calls == 2 and not report.complete
    assert report.as_record()["calls"][1]["input_tokens"] is None


def test_error_responses_pass_through_and_count_as_failed_not_unmeasured():
    with ScriptedModelStub(lambda method, path, body: (429, "application/json", b'{"error":"rate"}')) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(proxy.base_url + "/chat/completions", {})
            report = proxy.report()
    assert caught.value.code == 429 and caught.value.read() == b'{"error":"rate"}'
    assert report.failed_calls == 1 and report.generation_calls == 0 and report.complete


def test_unreachable_upstream_is_502_and_not_metered_as_success():
    with ScriptedModelStub(lambda *args: (200, "application/json", b"{}")) as upstream:
        dead = upstream.base_url
    with MeteringProxy(dead, timeout=5) as proxy:
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(proxy.base_url + "/chat/completions", {})
        report = proxy.report()
    assert caught.value.code == 502 and report.failed_calls == 1 and report.generation_calls == 0


def test_non_generation_requests_are_forwarded_but_not_metered():
    with ScriptedModelStub(lambda method, path, body: (200, "application/json", b'{"data":[]}')) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            with urllib.request.urlopen(proxy.base_url + "/models?x=1", timeout=10) as response:
                assert response.read() == b'{"data":[]}'
            report = proxy.report()
    assert upstream.requests[0][:2] == ("GET", "/v1/models?x=1")
    assert report.generation_calls == 0 and report.complete


def test_paths_outside_the_endpoint_base_are_refused():
    with ScriptedModelStub(lambda *args: (200, "application/json", b"{}")) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            root = proxy.base_url.rsplit("/v1", 1)[0]
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(root + "/admin/chat/completions", {})
    assert caught.value.code == 404 and upstream.requests == []


def test_upstream_redirects_are_not_followed():
    with ScriptedModelStub(lambda *args: (307, "application/json", b"{}")) as upstream:
        with MeteringProxy(upstream.base_url) as proxy:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post(proxy.base_url + "/chat/completions", {})
            report = proxy.report()
    assert caught.value.code == 307 and len(upstream.requests) == 1 and report.failed_calls == 1


@pytest.mark.parametrize("url", ["ftp://host/v1", "http://", "http://host/v1?key=1"])
def test_invalid_upstream_is_rejected(url):
    with pytest.raises(ValueError):
        MeteringProxy(url)


def test_usage_shapes_of_responses_api_and_anthropic_messages():
    assert extract_usage({"usage": {"input_tokens": 7, "output_tokens": 3}}) == (7, 3)
    completed = {"type": "response.completed", "response": {"usage": {"input_tokens": 9, "output_tokens": 4}}}
    assert extract_usage(completed) == (9, 4)
    anthropic = (
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
        b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":40}}\n\n'
    )
    assert usage_from_body(anthropic, streamed=True) == (25, 40)


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": True, "completion_tokens": 1},
    {"prompt_tokens": -1, "completion_tokens": 1},
    {"prompt_tokens": "5", "completion_tokens": 1},
    {"prompt_tokens": 5},
])
def test_malformed_usage_counts_as_missing(usage):
    tokens_in, tokens_out = extract_usage({"usage": usage})
    assert tokens_in is None or tokens_out is None


def test_unparseable_body_counts_as_missing():
    assert usage_from_body(b"not json", streamed=False) == (None, None)
