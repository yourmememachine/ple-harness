"""Regression tests for the model-client fix round.

Covers: lazy stream walking with cancel/finalize/deadline checks
between chunks, cancellation raised by the usage emit propagating
instead of being swallowed, pre-stream HTTP errors surfacing as
hard failures with no usage recorded, and the cache-read count
precedence between the top level and prompt_tokens_details.

Run from the build directory:
    .venv/bin/python -m pytest tests/test_model_client_fix.py -v
"""

import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_client import (  # noqa: E402
    InMemoryUsageLedger,
    ModelCallError,
    ModelCallInterrupted,
    ModelClient,
    normalize_usage,
)

MESSAGES = [{"role": "user", "content": "hi"}]


def _chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def _http_error(code):
    return urllib.error.HTTPError("http://proxy.invalid", code, "err",
                                  hdrs=None, fp=None)


# ---------------------------------------------------------------------
# Streaming: walked lazily, stop signals checked between chunks
# ---------------------------------------------------------------------

def test_cancel_between_chunks_stops_stream_without_draining_it():
    cancel = threading.Event()
    pulled = []

    def transport(payload):
        def gen():
            for i in range(10):
                pulled.append(i)
                if i == 2:
                    cancel.set()
                yield _chunk(f"c{i}")
        return gen()

    ledger = InMemoryUsageLedger()
    client = ModelClient(ledger=ledger, transport=transport,
                         config={"cancel_flag": cancel, "retry_delays": ()})
    with pytest.raises(ModelCallInterrupted) as excinfo:
        client.complete(MESSAGES)
    # Chunks 0 and 1 were processed; chunk 2 was pulled, then the
    # stop was seen before it was processed; the rest never moved.
    assert excinfo.value.partial.content == "c0c1"
    assert pulled == [0, 1, 2]
    assert len(ledger.records) == 1


def test_never_ending_stream_yields_to_cancellation():
    cancel = threading.Event()

    def transport(payload):
        def gen():
            while True:
                yield _chunk("x")
        return gen()

    client = ModelClient(transport=transport,
                         config={"cancel_flag": cancel, "retry_delays": ()})
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        start = time.monotonic()
        with pytest.raises(ModelCallInterrupted):
            client.complete(MESSAGES)
        assert time.monotonic() - start < 5
    finally:
        timer.cancel()


def test_finalize_flag_stops_stream():
    finalize = threading.Event()
    finalize.set()

    def transport(payload):
        def gen():
            yield _chunk("x")
            yield _chunk("y")
        return gen()

    client = ModelClient(transport=transport,
                         config={"finalize_flag": finalize,
                                 "retry_delays": ()})
    with pytest.raises(ModelCallInterrupted) as excinfo:
        client.complete(MESSAGES)
    assert "finalize" in str(excinfo.value)


def test_deadline_via_time_left_stops_stream():
    def transport(payload):
        def gen():
            yield _chunk("x")
        return gen()

    client = ModelClient(transport=transport,
                         config={"time_left": lambda: 0.0,
                                 "retry_delays": ()})
    with pytest.raises(ModelCallInterrupted) as excinfo:
        client.complete(MESSAGES)
    assert "deadline" in str(excinfo.value)
    assert excinfo.value.partial.content == ""


# ---------------------------------------------------------------------
# Usage emit: cancellation propagates, plain failures stay defects
# ---------------------------------------------------------------------

class SessionCancelled(Exception):
    """Named to match the session layer's exception."""


def _full_response_transport(payload):
    return {"message": {"content": "done"},
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "finish_reason": "stop"}


def test_usage_emit_cancellation_propagates():
    def emit(frame):
        raise SessionCancelled("the session has ended")

    client = ModelClient(emit=emit, transport=_full_response_transport,
                         config={"retry_delays": ()})
    with pytest.raises(SessionCancelled):
        client.complete(MESSAGES)


def test_usage_emit_plain_failure_is_recorded_not_raised():
    def emit(frame):
        raise RuntimeError("boom")

    client = ModelClient(emit=emit, transport=_full_response_transport,
                         config={"retry_delays": ()})
    response = client.complete(MESSAGES)
    assert response.content == "done"
    assert any("usage emit failed" in d for d in client.defects)


# ---------------------------------------------------------------------
# Pre-stream HTTP errors: hard failures, no usage recorded
# ---------------------------------------------------------------------

def _client_with_failing_urlopen(monkeypatch, code, retry_delays=()):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(code)
        raise _http_error(code)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ledger = InMemoryUsageLedger()
    emitted = []
    client = ModelClient(ledger=ledger, emit=emitted.append,
                         base_url="http://proxy.invalid/v1",
                         api_key="test-key",
                         config={"retry_delays": retry_delays})
    return client, ledger, emitted, calls


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_prestream_http_error_is_a_hard_failure(monkeypatch, code):
    client, ledger, emitted, calls = _client_with_failing_urlopen(
        monkeypatch, code)
    with pytest.raises(ModelCallError) as excinfo:
        client.complete(MESSAGES)
    assert f"HTTP {code}" in str(excinfo.value)
    assert ledger.records == []  # no usage for a call never accepted
    assert emitted == []
    assert calls == [code]  # non-transient codes are not retried


def test_prestream_503_retries_then_fails_hard(monkeypatch):
    client, ledger, emitted, calls = _client_with_failing_urlopen(
        monkeypatch, 503, retry_delays=(0, 0))
    with pytest.raises(ModelCallError):
        client.complete(MESSAGES)
    assert calls == [503, 503, 503]  # initial attempt plus two retries
    assert ledger.records == []
    assert emitted == []


def test_midstream_http_error_is_interrupted_with_confirmed_usage():
    def transport(payload):
        def gen():
            yield {"choices": [{"delta": {"content": "part"}}],
                   "usage": {"input_tokens": 7, "output_tokens": 2}}
            raise _http_error(500)
        return gen()

    ledger = InMemoryUsageLedger()
    client = ModelClient(ledger=ledger, transport=transport,
                         config={"retry_delays": ()})
    with pytest.raises(ModelCallInterrupted) as excinfo:
        client.complete(MESSAGES)
    assert excinfo.value.partial.content == "part"
    assert len(ledger.records) == 1
    assert ledger.records[0]["input_tokens"] == 7
    assert ledger.records[0]["output_tokens"] == 2


# ---------------------------------------------------------------------
# Cache-read precedence
# ---------------------------------------------------------------------

def test_cache_read_top_level_wins_and_disagreement_is_a_defect():
    counts, defects = normalize_usage(
        {"cache_read_tokens": 5,
         "prompt_tokens_details": {"cached_tokens": 3}})
    assert counts["cache_read_tokens"] == 5
    assert any("disagrees" in d for d in defects)


def test_cache_read_details_fill_only_when_top_level_absent():
    counts, defects = normalize_usage(
        {"prompt_tokens_details": {"cached_tokens": 9}})
    assert counts["cache_read_tokens"] == 9
    assert not any("disagrees" in d for d in defects)


def test_cache_read_agreement_is_not_a_defect():
    counts, defects = normalize_usage(
        {"cache_read_tokens": 4,
         "prompt_tokens_details": {"cached_tokens": 4}})
    assert counts["cache_read_tokens"] == 4
    assert not any("disagrees" in d for d in defects)
