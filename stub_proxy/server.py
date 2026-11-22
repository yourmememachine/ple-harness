#!/usr/bin/env python3
"""A stub model server for local harness development.

It speaks the same OpenAI-compatible chat-completions API as the kit's
model proxy (kit/model_proxy, a LiteLLM proxy), so the harness and the
kit's own model probe can be exercised end to end with no real model
access and no cost:

    MODEL_PROXY_URL=http://127.0.0.1:4100
    MODEL_PROXY_KEY=stub-local-key

The harness is pointed at `<proxy>/v1` (MODEL_BASE_URL), so chat calls
arrive at `/v1/chat/completions`; the bare `/chat/completions` path is
served too. Every reply is canned text, but the response shape is the
real one — `choices[0].message`, `finish_reason`, and a full `usage`
block with integer token counts — because the harness's usage reporting
is part of the protocol under test.

Behaviour modes (per request: JSON field `stub_mode`, else the
`X-Stub-Mode` header, else the `?mode=` query parameter, else the
server default from --mode / STUB_MODE):

    ok         Reply at once with a JSON completion.
    slow       Stream the reply as server-sent events, one chunk every
               STUB_CHUNK_DELAY seconds (default 0.4). Without
               `stream: true`, wait STUB_SLOW_SECONDS (default 3) and
               then reply with JSON.
    interrupt  Start a streamed reply, send two chunks, then drop the
               connection mid-stream (no closing chunk, no [DONE]).
               Without `stream: true`, send a truncated JSON body and
               close. Simulates a model stream dying halfway.

Other endpoints:

    GET /health        {"status": "ok"}
    GET /v1/models     The eight official google-cloud model ids.
    GET /__requests    The requests received so far, as JSON (test aid).

Authentication: if a key is configured (--key / STUB_PROXY_KEY),
requests must present it as `Authorization: Bearer <key>`, as with the
real proxy's master key. With no key configured, any (or no)
Authorization header is accepted.

Standard library only; Python 3.12.

Run:
    python3 stub_proxy/server.py [--host 127.0.0.1] [--port 4100]
                                 [--key stub-local-key] [--mode ok]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

MODELS = (
    "gemini-4-argon",
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemma-4-26b",
    "claude-opus-5-5",
    "gpt-oss-120b",
    "glm-5.2",
    "llama-4-maverick",
)
DEFAULT_MODEL = "claude-opus-5-5"
MODES = ("ok", "slow", "interrupt")

_lock = threading.Lock()
_requests: list[dict] = []
_counter = 0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: about four characters per token, at least 1."""
    return max(1, (len(text) + 3) // 4)


def _message_text(content) -> str:
    """Extracts the text of a chat message content (string or parts list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _canned_reply(messages: list) -> str:
    """The text the stub answers with, as a function of the conversation."""
    last_user = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user = _message_text(message.get("content"))
            break
    if "word OK" in last_user:
        # The kit's model probe asks for exactly this; answer it exactly.
        return "OK"
    return (
        "Stub response: this reply was generated locally by the stub "
        "model server. No real model was called."
    )


def _usage(messages: list, reply: str) -> dict:
    prompt = sum(
        _estimate_tokens(_message_text(m.get("content")))
        for m in messages
        if isinstance(m, dict)
    )
    completion = _estimate_tokens(reply)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "StubProxy/1.0"

    # -- plumbing -----------------------------------------------------

    def log_message(self, fmt, *args):  # one short line per request
        if not getattr(self.server, "quiet", False):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, code: str = "stub_error") -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": "invalid_request_error",
                       "code": code}},
        )

    def _authorised(self) -> bool:
        key = getattr(self.server, "api_key", "")
        if not key:
            return True
        return self.headers.get("Authorization") == f"Bearer {key}"

    # -- routes -------------------------------------------------------

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/health"):
            self._send_json(200, {"status": "ok"})
        elif path in ("/models", "/v1/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "created": 0,
                     "owned_by": "stub"}
                    for name in MODELS
                ],
            })
        elif path == "/__requests":
            with _lock:
                snapshot = list(_requests)
            self._send_json(200, {"requests": snapshot})
        else:
            self._error(404, f"no such path: {path}", "not_found")

    def do_POST(self):
        split = urlsplit(self.path)
        if split.path not in ("/chat/completions", "/v1/chat/completions"):
            self._error(404, f"no such path: {split.path}", "not_found")
            return
        if not self._authorised():
            self._error(401, "invalid API key", "invalid_api_key")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            request = json.loads(raw or b"{}")
        except ValueError:
            self._error(400, "request body is not valid JSON")
            return
        if not isinstance(request, dict):
            self._error(400, "request body must be a JSON object")
            return

        mode = (
            request.get("stub_mode")
            or self.headers.get("X-Stub-Mode")
            or parse_qs(split.query).get("mode", [None])[0]
            or getattr(self.server, "default_mode", "ok")
        )
        if mode not in MODES:
            self._error(400, f"unknown stub mode: {mode!r} (use one of {MODES})")
            return

        model = request.get("model") or DEFAULT_MODEL
        messages = request.get("messages") or []
        stream = bool(request.get("stream"))
        reply = _canned_reply(messages)
        usage = _usage(messages, reply)

        global _counter
        with _lock:
            _counter += 1
            call_id = f"chatcmpl-stub-{_counter}"
            _requests.append({
                "id": call_id,
                "model": model,
                "mode": mode,
                "stream": stream,
                "messages": len(messages),
                "usage": usage,
            })

        if mode == "interrupt":
            self._interrupt(model, call_id, reply, stream)
        elif stream:
            delay = 0.4 if mode == "slow" else 0.0
            self._stream(model, call_id, reply, usage, request, delay)
        else:
            if mode == "slow":
                time.sleep(float(os.environ.get("STUB_SLOW_SECONDS", "3")))
            self._send_json(200, {
                "id": call_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": usage,
            })

    # -- streaming ----------------------------------------------------

    def _chunks(self, reply: str, size: int = 12) -> list[str]:
        return [reply[i:i + size] for i in range(0, len(reply), size)] or [""]

    def _sse(self, payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    def _chunk(self, call_id: str, model: str, delta: dict,
               finish: str | None = None) -> dict:
        return {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def _open_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _stream(self, model, call_id, reply, usage, request, delay):
        self._open_stream()
        chunks = self._chunks(reply)
        for i, piece in enumerate(chunks):
            delta = {"content": piece}
            if i == 0:
                delta = {"role": "assistant", "content": piece}
            self.wfile.write(self._sse(self._chunk(call_id, model, delta)))
            self.wfile.flush()
            if delay:
                time.sleep(delay)
        self.wfile.write(
            self._sse(self._chunk(call_id, model, {}, finish="stop")))
        self.wfile.flush()
        options = request.get("stream_options") or {}
        if options.get("include_usage"):
            final = self._chunk(call_id, model, {})
            final["choices"] = []
            final["usage"] = usage
            self.wfile.write(self._sse(final))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _interrupt(self, model, call_id, reply, stream):
        """Sends the start of a reply, then drops the connection."""
        if stream:
            self._open_stream()
            pieces = self._chunks(reply)
            for piece in pieces[:2]:
                self.wfile.write(
                    self._sse(self._chunk(call_id, model,
                                          {"content": piece})))
            self.wfile.flush()
        else:
            body = json.dumps({
                "id": call_id,
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            # Declare more than will arrive, then stop mid-body.
            self.send_header("Content-Length", str(len(body) + 512))
            self.end_headers()
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
        # Drop the connection without closing the stream or the body.
        try:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER,
                b"\x01\x00\x00\x00\x00\x00\x00\x00",  # linger on, timeout 0 -> RST
            )
        except OSError:
            pass
        self.close_connection = True
        self.connection.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("STUB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("STUB_PORT", "4100")))
    parser.add_argument("--key", default=os.environ.get("STUB_PROXY_KEY", ""))
    parser.add_argument("--mode", default=os.environ.get("STUB_MODE", "ok"),
                        choices=MODES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    setattr(server, "api_key", args.key)
    server.default_mode = args.mode
    server.quiet = args.quiet
    print(
        f"stub proxy listening on http://{args.host}:{args.port} "
        f"(mode={args.mode}, auth={'on' if args.key else 'off'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
