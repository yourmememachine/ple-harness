"""Smoke tests for the harness entry.

Two tests:

1. `test_stub_proxy_replies_with_usage` — the stub model server
   (stub_proxy/server.py) answers a chat-completions call in the shape
   the kit's model probe and the harness's model client expect,
   including integer usage counts.

2. `test_serve_done_happy_path` — starts serve.py with the model
   pointed at the stub proxy, connects over WebSocket, sends one task
   frame in the kit's protocol (kit/protocol.py `task_message`), plays
   the evaluator's side of the protocol for any request frames
   (exec / read_file / write_file / http answered against a temporary
   workdir), and asserts that exactly one terminal frame arrives and
   that it is `done`.

The second test skips while serve.py does not exist yet.

Run with the build venv:

    .venv/bin/python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
import websockets

BUILD_DIR = Path(__file__).resolve().parent.parent
SERVE_PY = BUILD_DIR / "serve.py"
STUB_PY = BUILD_DIR / "stub_proxy" / "server.py"
STUB_KEY = "stub-local-key"
MODEL = "claude-opus-5-5"
TASK_DEADLINE_SEC = 120


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Proc:
    """A subprocess whose output is captured for failure messages."""

    def __init__(self, argv, env=None, cwd=None):
        self.proc = subprocess.Popen(
            argv,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> str:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        out = ""
        if self.proc.stdout is not None:
            try:
                out = self.proc.stdout.read() or ""
            except ValueError:
                out = ""
        return out


def _start_stub(port: int) -> _Proc:
    proc = _Proc(
        [sys.executable, str(STUB_PY), "--port", str(port),
         "--key", STUB_KEY, "--quiet"]
    )
    deadline = time.time() + 15
    while time.time() < deadline:
        if not proc.alive():
            raise AssertionError(f"stub proxy exited early:\n{proc.stop()}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as resp:
                if resp.status == 200:
                    return proc
        except OSError:
            time.sleep(0.2)
    raise AssertionError("stub proxy did not become ready in 15 s")


def _chat(port: int, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {STUB_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def test_stub_proxy_replies_with_usage():
    port = _free_port()
    stub = _start_stub(port)
    try:
        reply = _chat(port, {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the word OK."}],
            "max_completion_tokens": 512,
        })
    finally:
        stub.stop()
    message = reply["choices"][0]["message"]
    assert message["content"] == "OK"
    assert reply["choices"][0]["finish_reason"] == "stop"
    usage = reply["usage"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert isinstance(usage[key], int) and usage[key] >= 0
    assert usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )


# ---------------------------------------------------------------------------
# The evaluator side of the wire protocol, against a real temp workdir.


def _answer_request(frame: dict, workdir: Path) -> dict:
    """Builds the `result` frame for one harness request frame."""
    rid = frame["id"]
    kind = frame["type"]
    if kind == "exec":
        timeout = min(int(frame.get("timeout_sec") or 120), 120)
        try:
            done = subprocess.run(
                ["bash", "-c", frame["command"]],
                cwd=workdir,
                timeout=timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = done.stdout.decode(errors="replace")
            return {"ok": True, "exit_code": done.returncode,
                    "output": output[:100_000],
                    "truncated": len(output) > 100_000, "timed_out": False}
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or b"").decode(errors="replace")
            return {"ok": True, "exit_code": 124, "output": output[:100_000],
                    "truncated": False, "timed_out": True}
    if kind == "read_file":
        path = Path(frame["path"])
        if not path.is_absolute():
            path = workdir / path
        if not path.is_file():
            return {"ok": False, "error": f"not a regular file: {frame['path']}"}
        data = path.read_bytes()
        return {"ok": True,
                "content_b64": base64.b64encode(data).decode(),
                "size": len(data), "truncated": False}
    if kind == "write_file":
        path = Path(frame["path"])
        if not path.is_absolute():
            path = workdir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(frame["content_b64"])
        with path.open("ab" if frame.get("append") else "wb") as fh:
            fh.write(data)
        return {"ok": True, "size": path.stat().st_size}
    if kind == "http":
        return {"ok": False,
                "error": "this test configures no MCP servers"}
    return {"ok": False, "error": f"unknown request type: {kind}"}


async def _run_one_task(ws_url: str, workdir: Path) -> list[dict]:
    """Sends one task and returns every frame received, in order."""
    task = {
        "type": "task",
        "protocol": 1,
        "session": "smoke-1",
        "instruction": (
            "Write the word OK into a file named answer.txt in the "
            "workdir, then finish with a short summary."
        ),
        "workdir": str(workdir),
        "limits": {
            "agent_timeout_sec": TASK_DEADLINE_SEC,
            "exec_timeout_sec": 600,
            "max_output_bytes": 100_000,
            "max_write_bytes": 10_000_000,
        },
        "mcp_servers": [],
    }
    frames: list[dict] = []
    async with websockets.connect(
        ws_url, ping_interval=None, max_size=32 << 20
    ) as ws:
        await ws.send(json.dumps(task))
        deadline = time.time() + TASK_DEADLINE_SEC
        terminal_seen = False
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise AssertionError(
                    f"no terminal frame within {TASK_DEADLINE_SEC} s; "
                    f"frames so far: {[f.get('type') for f in frames]}"
                )
            # After the terminal frame, keep reading briefly so a
            # duplicate terminal frame is caught by the assertions.
            wait = 1.0 if terminal_seen else left
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=wait)
            except asyncio.TimeoutError:
                if terminal_seen:
                    break
                raise
            except websockets.ConnectionClosed:
                # A clean close after the terminal frame is the normal
                # end of a task; before it, the assertions below report
                # the missing terminal frame.
                break
            frame = json.loads(raw)
            frames.append(frame)
            kind = frame.get("type")
            if kind in ("exec", "read_file", "write_file", "http"):
                result = _answer_request(frame, workdir)
                await ws.send(json.dumps(
                    {"type": "result", "id": frame["id"], **result}))
            elif kind in ("done", "error"):
                terminal_seen = True
    return frames


def test_serve_done_happy_path():
    if not SERVE_PY.is_file():
        pytest.skip(
            "serve.py has not been built yet; the smoke test runs once "
            "it exists"
        )
    if not (BUILD_DIR / "harness.py").is_file():
        pytest.skip(
            "harness.py has not been built yet; serve.py imports it "
            "when a task arrives, so the smoke test runs once it exists"
        )
    stub_port = _free_port()
    serve_port = _free_port()
    stub = _start_stub(stub_port)
    env = dict(
        os.environ,
        PORT=str(serve_port),
        MODEL_NAME=MODEL,
        MODEL_BASE_URL=f"http://127.0.0.1:{stub_port}/v1",
        OPENAI_BASE_URL=f"http://127.0.0.1:{stub_port}/v1",
        PYTHONUNBUFFERED="1",
    )
    env["MODEL_API_KEY"] = STUB_KEY
    env["OPENAI_API_KEY"] = STUB_KEY
    serve = _Proc([sys.executable, str(SERVE_PY)], env=env, cwd=BUILD_DIR)
    try:
        # Wait until the server accepts TCP connections (the evaluator
        # allows 120 s; a local smoke test should not need anywhere
        # near that).
        deadline = time.time() + 60
        while True:
            if not serve.alive():
                raise AssertionError(
                    f"serve.py exited before listening:\n{serve.stop()}")
            try:
                with socket.create_connection(("127.0.0.1", serve_port),
                                              timeout=2):
                    break
            except OSError:
                if time.time() > deadline:
                    raise AssertionError(
                        "serve.py did not listen within 60 s:\n"
                        f"{serve.stop()}")
                time.sleep(0.2)
        with tempfile.TemporaryDirectory() as tmp:
            frames = asyncio.run(
                _run_one_task(f"ws://127.0.0.1:{serve_port}", Path(tmp)))
    finally:
        serve_output = serve.stop()
        stub.stop()
    terminals = [f for f in frames if f.get("type") in ("done", "error")]
    assert len(terminals) == 1, (
        f"expected exactly one terminal frame, got "
        f"{[f.get('type') for f in frames]}; serve.py output:\n"
        f"{serve_output[-2000:]}"
    )
    assert terminals[0]["type"] == "done", (
        f"terminal frame is {terminals[0]['type']!r}, expected 'done': "
        f"{terminals[0]}; serve.py output:\n{serve_output[-2000:]}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
