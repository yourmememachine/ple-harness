"""Cancel-path cleanup tests for the session layer and the loop.

A cancel can land at any moment in a run. These tests pin what must
hold in each position:

- cancel while the loop is blocked waiting on an evaluator result:
  the loop thread unwinds and exactly one terminal frame (`done`)
  reaches the wire;
- cancel while the loop is in Finalize: the same single-`done`
  invariant, and the closing summary the loop leaves on disk is its
  own — parseable and complete;
- cancel after the loop has already finished: nothing changes;
- a run that started a background job through `exec` and is then
  cancelled leaves no background process behind, and a deliverable
  is left holding either the old or the new complete content, never
  a partial write.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve  # noqa: E402
from harness import SUMMARY_FIELDS, run_harness  # noqa: E402
from tools import ToolDispatcher  # noqa: E402
from websockets.exceptions import ConnectionClosedOK  # noqa: E402

CLOSE = object()

INSTRUCTION = ("Write the result to out/result.txt and make sure the "
               "result file exists before finishing.")

BRIEF = {
    "deliverables": ["out/result.txt"],
    "constraints": [{
        "id": "c1", "text": "result file must exist", "hard": True,
        "source_span": "Write the result to out/result.txt",
        "check": {"type": "file_exists", "path": "out/result.txt"},
    }],
    "work_items": [{"id": "w1", "title": "Write result",
                    "deliverable": "out/result.txt"}],
}


class FakeWS:
    """In-memory stand-in for a WebSocket connection."""

    def __init__(self):
        self.inbound = asyncio.Queue()
        self.sent: list[str] = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def recv(self):
        item = await self.inbound.get()
        if item is CLOSE:
            raise ConnectionClosedOK(None, None)
        return item

    async def close(self):
        self.closed = True

    def feed(self, obj):
        self.inbound.put_nowait(
            obj if isinstance(obj, str) else json.dumps(obj))

    def frames(self):
        return [json.loads(p) for p in self.sent]

    def terminal(self):
        return [f for f in self.frames()
                if f.get("type") in ("done", "error")]

    def requests(self):
        return [f for f in self.frames() if f.get("type") in serve.REQUESTS]


async def wait_for(predicate, timeout=10.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _task_frame(session: str, workdir: Path) -> dict:
    return {
        "type": "task",
        "protocol": 1,
        "session": session,
        "instruction": INSTRUCTION,
        "workdir": str(workdir),
        "limits": {"agent_timeout_sec": None, "exec_timeout_sec": 600},
        "mcp_servers": [],
    }


def _harness_runner(client, exited: threading.Event, extra_config=None,
                    transport: str = "local"):
    """A session runner that drives the real loop with a scripted
    model client.

    ``transport="local"`` runs tool calls against the workdir
    directly; ``transport="session"`` keeps the session's own wiring,
    so tool calls travel as request frames over the wire.
    """
    def runner(task, emit, workdir, config):
        cfg = {**config, "model_client": client,
               "token_cap": 100_000, "estimated_call_tokens": 10}
        if transport == "local":
            cfg.update({"emit_is_transport": False, "request": None})
        if extra_config:
            cfg.update(extra_config)
        try:
            return run_harness(task, emit, workdir, cfg)
        finally:
            exited.set()
    return runner


def _answer_request(frame: dict, workdir: Path) -> dict:
    """The evaluator's side of one request frame, against the real
    workdir (same semantics as the local transport)."""
    kind = frame["type"]
    if kind == "read_file":
        path = workdir / frame["path"]
        if not path.is_file():
            return {"ok": False, "error": "not a regular file"}
        data = path.read_bytes()
        import base64
        return {"ok": True, "size": len(data),
                "content_b64": base64.b64encode(data).decode(),
                "truncated": False}
    if kind == "write_file":
        import base64
        path = workdir / frame["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = base64.b64decode(frame["content_b64"])
        with path.open("ab" if frame.get("append") else "wb") as fh:
            fh.write(data)
        return {"ok": True, "size": path.stat().st_size}
    return {"ok": False, "error": f"unanswered in this test: {kind}"}


async def _pump(ws: FakeWS, workdir: Path, stop: asyncio.Event,
                skip_types=()) -> None:
    """Answers request frames as they appear, except skipped types,
    which are left pending on purpose."""
    answered = 0
    while not stop.is_set():
        requests = ws.requests()
        while answered < len(requests):
            frame = requests[answered]
            answered += 1
            if frame["type"] in skip_types:
                continue
            result = _answer_request(frame, workdir)
            ws.feed({"type": "result", "id": frame["id"], **result})
        await asyncio.sleep(0.01)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------
# Cancel during an emit wait
# ---------------------------------------------------------------------

async def _cancel_during_emit_wait(tmp_path: Path) -> None:
    def client(messages, tools=None, phase="execute"):
        if phase == "brief":
            return {"content": json.dumps(BRIEF),
                    "usage": {"input_tokens": 20, "output_tokens": 0}}
        if phase == "execute":
            # A tool call whose result never arrives: the loop thread
            # is blocked inside the emit bridge when the cancel lands.
            return {"tool_calls": [{
                "name": "exec",
                "arguments": {"command": "sleep 5"}}],
                "usage": {"input_tokens": 30, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    exited = threading.Event()
    ws = FakeWS()
    workdir = tmp_path / "work"
    session = serve.Session(
        ws, _harness_runner(client, exited, transport="session"),
        {"state_root": str(tmp_path / "state")})
    run_task = asyncio.ensure_future(session.run())
    stop = asyncio.Event()
    pump = asyncio.ensure_future(
        _pump(ws, workdir, stop, skip_types=("exec",)))
    ws.feed(_task_frame("cancel-emit", workdir))

    # The setup requests (candidate write, probes) are answered by
    # the pump; the exec request is left pending, so the loop thread
    # is blocked inside the emit when the cancel lands.
    assert await wait_for(
        lambda: any(r["type"] == "exec" for r in ws.requests()))
    exec_requests = [r for r in ws.requests() if r["type"] == "exec"]
    assert len(exec_requests) == 1
    ws.feed({"type": "cancel", "reason": "timeout"})
    await asyncio.wait_for(run_task, 20)
    stop.set()
    await pump
    await asyncio.sleep(0.1)

    terminal = ws.terminal()
    assert len(terminal) == 1, ws.frames()
    assert terminal[0]["type"] == "done"  # cancel never answers error
    # No new requests were issued after the cancel.
    assert [r for r in ws.requests() if r["type"] == "exec"] == exec_requests
    # The loop thread unwound instead of staying blocked on the emit.
    assert exited.wait(5)


def test_cancel_during_emit_wait_unwinds_with_one_done(tmp_path):
    asyncio.run(_cancel_during_emit_wait(tmp_path))


# ---------------------------------------------------------------------
# Cancel during Finalize
# ---------------------------------------------------------------------

async def _cancel_during_finalize(tmp_path: Path) -> None:
    finalize_entered = threading.Event()
    release_finalize = threading.Event()

    def client(messages, tools=None, phase="execute"):
        if phase == "brief":
            return {"content": json.dumps(BRIEF),
                    "usage": {"input_tokens": 20, "output_tokens": 0}}
        if phase == "execute":
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        if phase == "finalize":
            # The Finalize audit call blocks: the cancel lands while
            # the loop is inside Finalize, before it writes anything.
            finalize_entered.set()
            release_finalize.wait(timeout=60)
            return {"content": "{}",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    exited = threading.Event()
    ws = FakeWS()
    session = serve.Session(
        ws, _harness_runner(client, exited),
        {"state_root": str(tmp_path / "state")})
    run_task = asyncio.ensure_future(session.run())
    ws.feed(_task_frame("cancel-finalize", tmp_path / "work"))

    assert await wait_for(finalize_entered.is_set)
    ws.feed({"type": "cancel", "reason": "timeout"})
    # The supervisor answers inside its cancel window even though
    # the loop cannot return yet.
    assert await wait_for(lambda: len(ws.terminal()) == 1, timeout=20)
    terminal = ws.terminal()
    assert terminal[0]["type"] == "done"

    # Let the loop finish Finalize on its own; nothing more may be
    # sent, and the loop's own closing records land on disk.
    release_finalize.set()
    assert exited.wait(10)
    await asyncio.wait_for(run_task, 20)
    await asyncio.sleep(0.1)
    assert len(ws.terminal()) == 1

    state_dir = Path(session.state_dir)
    summary = json.loads(
        (state_dir / "summary.json").read_text("utf-8"))
    assert set(SUMMARY_FIELDS) <= set(summary)
    assert summary["task_id"] == "cancel-finalize"
    assert summary["phase"] == "finalize"
    closing = (state_dir / "closing_summary.md").read_text("utf-8")
    assert closing.startswith("Task cancel-finalize:")
    assert "phase finalize" in closing
    assert "All hard checks pass:" in closing


def test_cancel_during_finalize_keeps_one_done_and_loop_summary(tmp_path):
    asyncio.run(_cancel_during_finalize(tmp_path))


# ---------------------------------------------------------------------
# Cancel after completion
# ---------------------------------------------------------------------

async def _cancel_after_completion(tmp_path: Path) -> None:
    async def runner(task, emit, workdir, config):
        return {"status": "done", "final_message": "already finished"}

    ws = FakeWS()
    session = serve.Session(ws, runner,
                            {"state_root": str(tmp_path / "state")})
    run_task = asyncio.ensure_future(session.run())
    ws.feed(_task_frame("cancel-late", tmp_path / "work"))
    await asyncio.wait_for(run_task, 10)
    await asyncio.sleep(0.05)

    before = ws.frames()
    assert len(ws.terminal()) == 1
    assert ws.terminal()[0]["final_message"] == "already finished"

    ws.feed({"type": "cancel", "reason": "timeout"})
    await asyncio.sleep(0.3)
    # The late cancel changes nothing: no new frame of any kind, and
    # the original outcome stands.
    assert ws.frames() == before
    assert len(ws.terminal()) == 1
    assert ws.terminal()[0]["final_message"] == "already finished"


def test_cancel_after_completion_changes_nothing(tmp_path):
    asyncio.run(_cancel_after_completion(tmp_path))


# ---------------------------------------------------------------------
# Subprocess hygiene
# ---------------------------------------------------------------------

def test_dispatcher_stop_all_background_kills_the_group(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.exec("sleep 300", background=True)
    data = result["data"]
    pgid = int(data["process_group"])
    try:
        assert _group_alive(pgid)
        assert dispatcher.stop_all_background() == 1
        assert not _group_alive(pgid)
        assert dispatcher._background == {}
    finally:
        _kill_group(pgid)


async def _cancelled_run_hygiene(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    released = threading.Event()
    calls = {"execute": 0}
    pgid_box: dict[str, int] = {}

    def client(messages, tools=None, phase="execute"):
        if phase == "brief":
            return {"content": json.dumps(BRIEF),
                    "usage": {"input_tokens": 20, "output_tokens": 0}}
        if phase == "execute":
            calls["execute"] += 1
            if calls["execute"] == 1:
                return {"tool_calls": [
                    {"name": "write_file", "arguments": {
                        "path": "out/result.txt",
                        "content": "complete v1\n"}},
                    {"name": "exec", "arguments": {
                        "command": "echo $$ > bgpgid.txt; sleep 300",
                        "background": True}}],
                    "usage": {"input_tokens": 30, "output_tokens": 0}}
            if calls["execute"] == 2:
                # Blocked mid-run when the cancel lands; afterwards
                # the loop writes the new deliverable version whole.
                released.wait(timeout=60)
                return {"tool_calls": [{
                    "name": "write_file", "arguments": {
                        "path": "out/result.txt",
                        "content": "complete v2\n"}}],
                    "usage": {"input_tokens": 30, "output_tokens": 0}}
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    exited = threading.Event()
    ws = FakeWS()
    session = serve.Session(
        ws, _harness_runner(client, exited),
        {"state_root": str(tmp_path / "state")})
    run_task = asyncio.ensure_future(session.run())
    ws.feed(_task_frame("cancel-hygiene", workdir))

    marker = workdir / "bgpgid.txt"
    assert await wait_for(
        lambda: marker.exists() and marker.read_text().strip().isdigit())
    pgid = int(marker.read_text().strip())
    pgid_box["pgid"] = pgid
    try:
        assert _group_alive(pgid)

        ws.feed({"type": "cancel", "reason": "timeout"})
        assert await wait_for(lambda: len(ws.terminal()) == 1,
                              timeout=20)
        assert ws.terminal()[0]["type"] == "done"

        released.set()
        assert exited.wait(15)
        await asyncio.wait_for(run_task, 20)
        await asyncio.sleep(0.2)
        assert len(ws.terminal()) == 1

        # The deliverable holds one complete version or the other,
        # never a partial write.
        content = (workdir / "out" / "result.txt").read_bytes()
        assert content in (b"complete v1\n", b"complete v2\n")

        # No registered background process survives the cancelled run.
        assert not _group_alive(pgid), (
            f"background process group {pgid} is still alive after "
            "the cancelled run finished")
    finally:
        _kill_group(pgid)


def test_cancelled_run_leaves_no_background_process(tmp_path):
    asyncio.run(_cancelled_run_hygiene(tmp_path))
