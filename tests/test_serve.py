"""Tests for serve.py: the session layer and its validating serialiser.

Run from the build directory:
    .venv/bin/python -m unittest discover -s tests -v
"""

import asyncio
import base64
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serve  # noqa: E402
from websockets.exceptions import ConnectionClosedOK  # noqa: E402

CLOSE = object()

TASK_FRAME = {
    "type": "task",
    "protocol": 1,
    "session": "test-session",
    "instruction": "do the thing",
    "workdir": "/app",
    "limits": {"agent_timeout_sec": None, "exec_timeout_sec": 600},
    "mcp_servers": [],
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
            obj if isinstance(obj, str) else json.dumps(obj)
        )

    def frames(self):
        return [json.loads(p) for p in self.sent]

    def terminal(self):
        return [f for f in self.frames() if f.get("type") in ("done", "error")]

    def requests(self):
        return [f for f in self.frames() if f.get("type") in serve.REQUESTS]


async def wait_for(predicate, timeout=5.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class SerialiserTests(unittest.IsolatedAsyncioTestCase):
    def make(self, servers=()):
        sent = []

        async def raw(payload):
            sent.append(payload)

        return serve.Serialiser(raw, list(servers)), sent

    async def test_second_terminal_frame_refused(self):
        s, sent = self.make()
        self.assertTrue(await s.send_terminal("done", "finished"))
        self.assertFalse(await s.send_terminal("error", "too late"))
        self.assertFalse(await s.send_terminal("done", "also too late"))
        frames = [json.loads(p) for p in sent]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0], {"type": "done", "final_message": "finished"})

    async def test_terminal_via_send_rejected(self):
        s, sent = self.make()
        with self.assertRaises(serve.FrameError):
            await s.send({"type": "done", "final_message": "sneaky"})
        with self.assertRaises(serve.FrameError):
            await s.send({"type": "error", "message": "sneaky"})
        self.assertEqual(sent, [])

    async def test_usage_zero_fill_and_coercion(self):
        s, sent = self.make()
        await s.send({"type": "usage", "model": "m"})
        await s.send({"type": "usage"})
        await s.send(
            {
                "type": "usage",
                "model": "m",
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_tokens": None,
            }
        )
        await s.send(
            {
                "type": "usage",
                "model": "m",
                "input_tokens": True,
                "output_tokens": 1.5,
                "cache_read_tokens": "9",
            }
        )
        await s.send(
            {
                "type": "usage",
                "model": "m",
                "input_tokens": 7,
                "output_tokens": 12.0,
                "cache_read_tokens": 2**60,
            }
        )
        frames = [json.loads(p) for p in sent]
        self.assertEqual(len(frames), 5)
        for frame in frames[:4]:
            self.assertEqual(frame["input_tokens"], 0)
            self.assertEqual(frame["output_tokens"], 0)
            self.assertEqual(frame["cache_read_tokens"], 0)
        self.assertEqual(frames[0]["model"], "m")
        self.assertEqual(frames[1]["model"], "")
        self.assertEqual(frames[4]["input_tokens"], 7)
        self.assertEqual(frames[4]["output_tokens"], 12)
        self.assertEqual(frames[4]["cache_read_tokens"], 2**53 - 1)
        for frame in frames:  # every count on the wire is a true integer
            for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
                self.assertIs(type(frame[key]), int)

    async def test_usage_never_raises_on_garbage(self):
        s, sent = self.make()
        await s.send({"type": "usage", "model": 42, "input_tokens": object()})
        frame = json.loads(sent[0])
        self.assertEqual(frame["model"], "")
        self.assertEqual(frame["input_tokens"], 0)

    async def test_oversize_frame_refused_locally(self):
        s, sent = self.make()
        with self.assertRaises(serve.FrameTooLarge):
            await s.send(
                {"type": "exec", "id": "1", "command": "x" * (17 << 20)}
            )
        self.assertEqual(sent, [])

    async def test_write_payload_over_cap_refused(self):
        s, sent = self.make()
        big = base64.b64encode(b"x" * (serve.MAX_WRITE_BYTES + 1)).decode()
        with self.assertRaises(serve.FrameError):
            await s.send(
                {
                    "type": "write_file",
                    "id": "1",
                    "path": "/app/f",
                    "content_b64": big,
                }
            )
        self.assertEqual(sent, [])

    async def test_strings_sanitised(self):
        s, sent = self.make()
        await s.send({"type": "log", "text": "a\ud800b\x00c"})
        payload = sent[0]
        payload.encode("utf-8")  # must not raise
        text = json.loads(payload)["text"]
        self.assertNotIn("\x00", text)
        self.assertNotIn("\ud800", text)

    async def test_http_restricted_to_mcp_servers(self):
        servers = [{"name": "docs", "url": "http://mcp.local/sse"}]
        s, sent = self.make(servers)
        with self.assertRaises(serve.FrameError):
            await s.send(
                {"type": "http", "id": "1", "url": "http://evil.example/x"}
            )
        await s.send({"type": "http", "id": "2", "url": "http://mcp.local/sse"})
        self.assertEqual(len(sent), 1)

    async def test_request_validation(self):
        s, sent = self.make()
        with self.assertRaises(serve.FrameError):
            await s.send({"type": "exec", "id": "1", "command": ""})
        with self.assertRaises(serve.FrameError):
            await s.send({"type": "exec", "command": "ls"})  # no id
        with self.assertRaises(serve.FrameError):
            await s.send({"type": "exec", "id": "1", "command": "ls",
                          "timeout_sec": "soon"})
        await s.send(
            {"type": "exec", "id": "1", "command": "ls", "timeout_sec": 9999}
        )
        frame = json.loads(sent[0])
        self.assertEqual(frame["timeout_sec"], 600)  # clamped to the cap


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = {"state_root": self.tmp.name}

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, runner, frame=None):
        ws = FakeWS()
        session = serve.Session(ws, runner, self.config)
        run_task = asyncio.ensure_future(session.run())
        ws.feed(TASK_FRAME if frame is None else frame)
        return ws, session, run_task

    async def settle(self):
        await asyncio.sleep(0.05)

    async def test_success_sends_exactly_one_done(self):
        async def runner(task, emit, workdir, config):
            return {"status": "done", "final_message": "all delivered"}

        ws, session, run_task = self.start(runner)
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "done")
        self.assertEqual(terminal[0]["final_message"], "all delivered")

    async def test_error_outcome_and_exception(self):
        async def err_runner(task, emit, workdir, config):
            return {"status": "error", "message": "cannot proceed"}

        ws, session, run_task = self.start(err_runner)
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        self.assertEqual(
            ws.terminal(), [{"type": "error", "message": "cannot proceed"}]
        )

        async def boom_runner(task, emit, workdir, config):
            raise RuntimeError("boom")

        ws, session, run_task = self.start(boom_runner)
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "error")
        self.assertIn("boom", terminal[0]["message"])

    async def test_result_routing_by_id(self):
        async def runner(task, emit, workdir, config):
            result = await emit({"type": "exec", "command": "ls"})
            return {"status": "done", "final_message": result["output"]}

        ws, session, run_task = self.start(runner)
        self.assertTrue(await wait_for(lambda: len(ws.requests()) == 1))
        rid = ws.requests()[0]["id"]
        ws.feed({"type": "result", "id": "bogus", "ok": True, "output": "no"})
        ws.feed({"type": "result", "id": rid, "ok": True, "output": "yes"})
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        self.assertEqual(ws.terminal()[0]["final_message"], "yes")

    async def test_cancel_done_no_new_requests(self):
        calls = []

        async def runner(task, emit, workdir, config):
            calls.append("start")
            await emit({"type": "exec", "command": "sleep 999"})
            calls.append("resumed")  # never reached: no result ever comes
            return {"status": "done", "final_message": "unreachable"}

        ws, session, run_task = self.start(runner)
        self.assertTrue(await wait_for(lambda: len(ws.requests()) == 1))
        start = time.monotonic()
        ws.feed({"type": "cancel", "reason": "timeout"})
        await asyncio.wait_for(run_task, 5)
        elapsed = time.monotonic() - start
        await self.settle()
        self.assertLess(elapsed, 5)  # the window is 10 s; answer far sooner
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "done")  # cancel never errors
        self.assertEqual(len(ws.requests()), 1)  # no new requests after cancel
        self.assertEqual(calls, ["start"])

    async def test_cancel_uses_loop_outcome_from_grace(self):
        async def runner(task, emit, workdir, config):
            try:
                await emit({"type": "exec", "command": "sleep 999"})
            except serve.SessionCancelled:
                return {"status": "done", "final_message": "loop summary"}
            return {"status": "done", "final_message": "unreachable"}

        ws, session, run_task = self.start(runner)
        self.assertTrue(await wait_for(lambda: len(ws.requests()) == 1))
        ws.feed({"type": "cancel", "reason": "timeout"})
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["final_message"], "loop summary")

    async def test_cancel_falls_back_to_disk_summary(self):
        async def runner(task, emit, workdir, config):
            Path(config["state_dir"], serve.SUMMARY_FILENAME).write_text(
                "disk summary", encoding="utf-8"
            )
            await emit({"type": "exec", "command": "sleep 999"})
            return {"status": "done", "final_message": "unreachable"}

        ws, session, run_task = self.start(runner)
        self.assertTrue(await wait_for(lambda: len(ws.requests()) == 1))
        ws.feed({"type": "cancel", "reason": "timeout"})
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "done")
        self.assertEqual(terminal[0]["final_message"], "disk summary")

    async def test_terminal_exactly_once_under_races(self):
        for i in range(25):
            delay = 0.004 * (i % 5)

            async def runner(task, emit, workdir, config, delay=delay):
                await asyncio.sleep(delay)
                return {"status": "done", "final_message": "raced"}

            ws, session, run_task = self.start(runner)

            async def feed_cancel(delay=delay):
                await asyncio.sleep(delay)
                ws.feed({"type": "cancel", "reason": "timeout"})

            cancel_task = asyncio.ensure_future(feed_cancel())
            await asyncio.wait_for(run_task, 5)
            await cancel_task
            await self.settle()
            self.assertEqual(
                len(ws.terminal()), 1, f"iteration {i}: {ws.frames()}"
            )
            self.assertEqual(ws.terminal()[0]["type"], "done")

    async def test_watchdog_finalize_and_self_done(self):
        async def runner(task, emit, workdir, config):
            await asyncio.sleep(3600)  # a wedged loop
            return {"status": "done", "final_message": "unreachable"}

        frame = {**TASK_FRAME, "limits": {"agent_timeout_sec": 3}}
        ws, session, run_task = self.start(runner, frame)
        start = time.monotonic()
        await asyncio.wait_for(run_task, 10)
        elapsed = time.monotonic() - start
        await self.settle()
        # force-Finalize at 3 - 0.75 = 2.25 s; self-done at 3 - 0.15 = 2.85 s
        self.assertTrue(session.finalize_event.is_set())
        self.assertGreaterEqual(elapsed, 2.5)
        self.assertLess(elapsed, 8)
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "done")
        self.assertEqual(terminal[0]["final_message"], serve.STOPPED_MESSAGE)

    async def test_first_frame_not_a_task(self):
        async def runner(task, emit, workdir, config):
            return {"status": "done", "final_message": "unreachable"}

        ws = FakeWS()
        session = serve.Session(ws, runner, self.config)
        run_task = asyncio.ensure_future(session.run())
        ws.feed({"type": "hello"})
        await asyncio.wait_for(run_task, 5)
        await self.settle()
        terminal = ws.terminal()
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["type"], "error")
        self.assertIn("expected a task message first", terminal[0]["message"])


if __name__ == "__main__":
    unittest.main()
