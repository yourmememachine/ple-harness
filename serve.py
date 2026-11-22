"""WebSocket session layer for the harness entry.

serve.py owns the socket and the session lifecycle. It accepts one
evaluator connection per task, reads frames from the moment the
connection opens, routes results to the requests waiting on them, runs
the agent loop in harness.py as a task, and guarantees that exactly
one terminal frame (`done` or `error`) is ever sent for a session.

The loop never touches the socket. The production loop
(harness.run_harness) is a plain synchronous function:

    outcome = run_harness(task, emit, workdir, config)

It runs in a worker thread and is handed a synchronous bridge emit:
`emit(frame)` blocks the loop thread until the frame is sent and, for
request frames, until the evaluator's result arrives. A runner that
is itself a coroutine function is instead awaited on the event loop
and handed the async emit (`await emit(frame)`); both forms share
the semantics below.

- task: the evaluator's `task` frame, defensively normalised (missing
  or null fields replaced by logged defaults).
- emit: sends one non-terminal frame through the session's validating
  serialiser. For request frames (`exec`, `read_file`, `write_file`,
  `http`) it returns the evaluator's `result` frame for that request;
  for one-way frames (`log`, `usage`) it returns None. Request ids
  are assigned here, monotonically per session, and never reused; any
  `id` in the passed frame is replaced. emit raises FrameError for a
  locally invalid frame (nothing is sent) and SessionCancelled when
  the session is ending (cancel received, terminal frame sent, or
  socket closed). The synchronous bridge raises SessionCancelled in
  the loop thread when the session is cancelled, so a blocked loop
  unwinds promptly without any new model calls.
- workdir: the task's working directory inside the task container.
- config: a dict with `session`, `state_dir`, `limits`,
  `mcp_servers`, `deadline` (a monotonic timestamp or None),
  `cancel_event` and `finalize_event` (asyncio events the loop may
  poll: cancel means stop now, finalize means wrap up),
  `cancel_flag` and `finalize_flag` (threading events mirroring
  them, safe to poll from the loop's worker thread), `time_left()`
  returning seconds remaining or None, and `emit_is_transport` set
  true: the loop's tool dispatcher takes its request transport from
  the emit callable itself, so tool calls flow through this session
  as request frames instead of running locally.

run_harness returns an outcome dict, translated by the supervisor
into the session's single terminal frame:

    {"status": "done", "final_message": "..."}  -> one `done` frame
    {"status": "error", "message": "..."}       -> one `error` frame

The harness's own outcome vocabulary is accepted as well:
"finalized" closes as `done` with its final_message, and
"error_outcome" closes as `error` with its error text.

Any other return value, or an exception escaping the loop, becomes an
`error` frame. On evaluator cancel the supervisor always answers
`done` (never `error`): it uses the loop's own outcome if the loop
returns one inside the grace window, otherwise the closing summary
held in disk state (state_dir / closing_summary.md), otherwise a
fixed stopped message. The cancel path makes no model calls and
issues no new requests; the terminal frame is on the wire well
inside the evaluator's 10 s window.

Protocol constants follow serve protocol v1 (kit/protocol.py).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import contextlib
import itertools
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
from typing import Any

try:  # the one non-stdlib dependency; imported here for its exception type
    from websockets.exceptions import ConnectionClosed
except Exception:  # pragma: no cover - websockets missing

    class ConnectionClosed(Exception):  # type: ignore[no-redef]
        """Stand-in so the module still imports without websockets."""


# --- serve protocol v1 limits (kit/protocol.py) ---
VERSION = 1
DEFAULT_PORT = 8765
START_SEC = 120  # accept + first frame budget at the evaluator
MAX_FRAME_BYTES = 16 << 20
MAX_IN_FLIGHT = 8
MAX_ID_CHARS = 128
MAX_LOG_CHARS = 20_000
MAX_MODEL_CHARS = 200
MAX_COUNT = 2**53 - 1  # largest usage count the protocol carries
MAX_WRITE_BYTES = 10_000_000
MAX_HTTP_BYTES = 10_000_000
DEFAULT_EXEC_TIMEOUT_SEC = 120
EXEC_TIMEOUT_SEC = 600

REQUESTS = ("exec", "read_file", "write_file", "http")
NOTES = ("log", "usage")
ENDS = ("done", "error")
HTTP_METHODS = ("GET", "POST", "DELETE", "PUT", "PATCH", "HEAD", "OPTIONS")

# --- this layer's own limits ---
FINAL_MESSAGE_CAP = 1_000_000  # chars; the protocol's final-message ceiling
ERROR_MESSAGE_CAP = 2_000  # chars
SUMMARY_FILENAME = "closing_summary.md"  # disk closing summary, in state_dir
STOPPED_MESSAGE = "(stopped: time is up)"
# Cancel endgame: the loop gets CANCEL_LOOP_GRACE_SEC to return its own
# outcome after the cancel lands, then CANCEL_UNWIND_SEC to finish
# unwinding once cancelled. 2 + 4 s keeps the terminal frame on the
# wire by +8 s at the latest, inside the evaluator's 10 s window.
CANCEL_LOOP_GRACE_SEC = 2.0
CANCEL_UNWIND_SEC = 4.0
# Watchdog milestones, as fractions of the task's stated limit with
# absolute caps: force-Finalize at limit - min(20 min, 25%),
# self-done at limit - min(5 min, 5%).
FORCE_FINALIZE_CAP_SEC = 1200.0
FORCE_FINALIZE_FRACTION = 0.25
SELF_DONE_CAP_SEC = 300.0
SELF_DONE_FRACTION = 0.05


class FrameError(ValueError):
    """A frame failed local validation; nothing was sent."""


class FrameTooLarge(FrameError):
    """The serialised frame exceeds the protocol's 16 MiB limit."""


class SessionCancelled(Exception):
    """The session is ending; raised into pending emits and loop awaits."""


# ---------------------------------------------------------------------------
# Sanitising and field helpers
# ---------------------------------------------------------------------------


def _clean_text(value: str) -> str:
    """Returns `value` safe to put on the wire: no NULs, valid UTF-8.

    Unpaired surrogates (which model-authored text can carry) are
    replaced, so the result always encodes as UTF-8.
    """
    if "\x00" in value:
        value = value.replace("\x00", "")
    return value.encode("utf-8", "replace").decode("utf-8")


def _sanitise(obj: Any) -> Any:
    """Recursively cleans every string in a frame (keys and values)."""
    if isinstance(obj, str):
        return _clean_text(obj)
    if isinstance(obj, dict):
        return {
            (_clean_text(k) if isinstance(k, str) else k): _sanitise(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and (
        isinstance(value, int)
        or isinstance(value, float)
        and math.isfinite(value)
    )


def _req_text(frame: dict[str, Any], key: str) -> str:
    value = frame.get(key)
    if not isinstance(value, str) or not value:
        raise FrameError(f"{key}: a non-empty string")
    return value


def _clamp_timeout(frame: dict[str, Any]) -> None:
    """Validates `timeout_sec` in place, clamping it to 1..600."""
    value = frame.get("timeout_sec")
    if value is None:
        frame.pop("timeout_sec", None)
        return
    if not _is_number(value):
        raise FrameError("timeout_sec: a number")
    frame["timeout_sec"] = min(max(int(value), 1), EXEC_TIMEOUT_SEC)


def _check_b64(frame: dict[str, Any], key: str, cap: int, required: bool) -> None:
    value = frame.get(key)
    if value is None:
        if required:
            raise FrameError(f'{key}: required, base64 text ("" for empty)')
        frame.pop(key, None)
        return
    if not isinstance(value, str) or len(value) > (cap + 2) // 3 * 4 + 4:
        raise FrameError(f"{key}: base64 text of at most {cap} bytes")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise FrameError(f"{key}: not valid base64") from None
    if len(data) > cap:
        raise FrameError(f"{key}: more than {cap} bytes")


def mcp_allows(url: str, servers: list[dict[str, Any]]) -> bool:
    """Whether `url` is one of the task's MCP server URLs or under one.

    Mirrors the evaluator's rule: an exact match, a query string added
    to a query-less server URL, or a sub-path. Paths that could
    traverse upward after decoding are rejected.
    """
    try:
        path = urllib.parse.urlsplit(url).path
    except ValueError:
        return False
    decoded = urllib.parse.unquote(path)
    if (
        re.search(r"%2f|%5c|\\", path, re.IGNORECASE)
        or ";" in decoded
        or {".", ".."} & set(decoded.split("/"))
    ):
        return False
    for server in servers:
        base = str(server.get("url") or "")
        if base and (
            url == base
            or ("?" not in base and url.startswith(base + "?"))
            or url.startswith(base.rstrip("/") + "/")
        ):
            return True
    return False


def _usage_count(value: Any) -> tuple[int, bool]:
    """Coerces one usage count to a true integer in 0..MAX_COUNT.

    Returns (count, ok). Absent/null counts are a clean 0 (ok). Whole
    numbers in range pass through. Anything else — bool, fractional
    float, non-number, out of range — is coerced (ok=False) so the
    emitted frame always carries true integers and the usage path
    never has to raise.
    """
    if value is None:
        return 0, True
    if isinstance(value, bool):
        return 0, False
    if isinstance(value, int):
        if 0 <= value <= MAX_COUNT:
            return value, True
        return min(max(value, 0), MAX_COUNT), False
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return _usage_count(int(value))
    return 0, False


# ---------------------------------------------------------------------------
# The serialiser: the one validating gate every outbound frame passes
# ---------------------------------------------------------------------------


class Serialiser:
    """Validates, sanitises, and sends every outbound frame.

    One instance per session. Non-terminal frames go through `send`;
    terminal frames only through `send_terminal`, which claims the
    session's single terminal slot first — a second terminal frame is
    refused unconditionally, whoever asks for it.
    """

    def __init__(self, send_raw, mcp_servers: list[dict[str, Any]]):
        # send_raw: async callable(str) writing one frame to the socket
        self._send_raw = send_raw
        self._mcp_servers = mcp_servers
        self._lock = asyncio.Lock()  # one frame on the wire at a time
        self._terminal_claimed = False
        self.defects: list[str] = []  # local validation defects, bounded

    @property
    def terminal_claimed(self) -> bool:
        return self._terminal_claimed

    def claim_terminal(self) -> bool:
        """Atomically claims the session's one terminal frame.

        Synchronous check-and-set: within the event loop no other
        path can interleave between the check and the claim.
        """
        if self._terminal_claimed:
            return False
        self._terminal_claimed = True
        return True

    def note_defect(self, message: str) -> None:
        self.defects.append(message)
        del self.defects[:-100]
        print(f"serve: frame defect: {message}", file=sys.stderr, flush=True)

    # -- validation ---------------------------------------------------

    def _prepare(self, frame: Any) -> dict[str, Any]:
        """Returns the wire-ready frame, or raises FrameError."""
        if not isinstance(frame, dict):
            raise FrameError("a frame is one JSON object")
        if frame.get("type") in REQUESTS:
            # A NUL in a field the evaluator acts on (command, path,
            # url) is refused, not stripped: deleting the byte would
            # silently join the two halves into a request the caller
            # never literally wrote. Stripping still applies to log,
            # final, and error text, where it cannot change what is
            # executed or read. Checked on the raw frame, before
            # sanitising removes the evidence.
            for key in ("command", "path", "url"):
                value = frame.get(key)
                if isinstance(value, str) and "\x00" in value:
                    raise FrameError(f"{key}: contains a NUL byte")
        frame = _sanitise(frame)
        kind = frame.get("type")
        if kind in ENDS:
            raise FrameError("terminal frames are sent by the supervisor only")
        if kind in REQUESTS:
            self._prepare_request(frame, kind)
        elif kind == "log":
            text = frame.get("text")
            if not isinstance(text, str):
                raise FrameError("log text: a string")
            frame["text"] = text[:MAX_LOG_CHARS]
        elif kind == "usage":
            return self._prepare_usage(frame)
        else:
            raise FrameError(f"unknown type {kind!r:.100}")
        return frame

    def _prepare_request(self, frame: dict[str, Any], kind: str) -> None:
        rid = frame.get("id")
        if (
            not isinstance(rid, str)
            or not rid
            or len(rid) > MAX_ID_CHARS
        ):
            raise FrameError(
                f"id: a non-empty string of at most {MAX_ID_CHARS} characters"
            )
        if kind == "exec":
            _req_text(frame, "command")
            _clamp_timeout(frame)
        elif kind == "read_file":
            _req_text(frame, "path")
        elif kind == "write_file":
            _req_text(frame, "path")
            _check_b64(frame, "content_b64", MAX_WRITE_BYTES, required=True)
            append = frame.get("append")
            if append is None:
                frame.pop("append", None)
            elif not isinstance(append, bool):
                raise FrameError("append: true or false")
        else:  # http
            url = _req_text(frame, "url")
            if not mcp_allows(url, self._mcp_servers):
                raise FrameError(
                    "http: the url is not one of the task's MCP servers"
                )
            method = frame.get("method")
            if method is not None:
                if not isinstance(method, str) or method.upper() not in HTTP_METHODS:
                    raise FrameError(f"method: one of {', '.join(HTTP_METHODS)}")
                frame["method"] = method.upper()
            headers = frame.get("headers")
            if headers is not None and (
                not isinstance(headers, dict)
                or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in headers.items()
                )
            ):
                raise FrameError("headers: an object of string values")
            _check_b64(frame, "body_b64", MAX_HTTP_BYTES, required=False)
            _clamp_timeout(frame)

    def _prepare_usage(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Normalises a usage frame. Never raises.

        The frame is rebuilt with exactly the protocol's fields:
        absent or null counts become 0, every count leaves as a true
        integer in range, and `model` leaves as a string.
        """
        model = frame.get("model")
        if model is not None and not isinstance(model, str):
            self.note_defect(f"usage model not a string: {type(model).__name__}")
            model = None
        out: dict[str, Any] = {
            "type": "usage",
            "model": (model or "")[:MAX_MODEL_CHARS],
        }
        for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
            count, ok = _usage_count(frame.get(key))
            if not ok:
                self.note_defect(f"usage {key} coerced from {frame.get(key)!r:.60}")
            out[key] = count
        return out

    # -- sending ------------------------------------------------------

    async def send(self, frame: Any) -> None:
        """Validates and sends one non-terminal frame.

        Raises FrameError (or FrameTooLarge) for a locally invalid
        frame; nothing is sent in that case.
        """
        clean = self._prepare(frame)
        payload = json.dumps(clean)
        size = len(payload.encode("utf-8"))
        if size > MAX_FRAME_BYTES:
            raise FrameTooLarge(
                f"{clean.get('type')}: the frame is {size} bytes, over"
                f" the protocol's {MAX_FRAME_BYTES >> 20} MiB"
            )
        async with self._lock:
            await self._send_raw(payload)

    async def send_terminal(self, kind: str, text: Any) -> bool:
        """Claims the terminal slot and sends `done`/`error`.

        Returns True only for the one call that actually sends; every
        later call returns False and sends nothing.
        """
        if kind not in ENDS:
            raise ValueError(f"not a terminal kind: {kind!r}")
        if not self.claim_terminal():
            return False
        field = "final_message" if kind == "done" else "message"
        cap = FINAL_MESSAGE_CAP if kind == "done" else ERROR_MESSAGE_CAP
        body = _clean_text(text if isinstance(text, str) else str(text))[:cap]
        payload = json.dumps({"type": kind, field: body})
        while len(payload.encode("utf-8")) > MAX_FRAME_BYTES:
            # unreachable under the caps
            body = body[: len(body) // 2]
            payload = json.dumps({"type": kind, field: body})
        async with self._lock:
            try:
                await self._send_raw(payload)
            except Exception:
                # One fallback attempt: a failed send must not
                # silently consume the session's only terminal
                # frame. If the fallback fails too, the claim is
                # released so a later closing path can retry, and
                # the failure propagates to the caller.
                try:
                    await self._send_raw(payload)
                except Exception:
                    self._terminal_claimed = False
                    self.note_defect("terminal send failed twice; claim released")
                    raise
        return True


# ---------------------------------------------------------------------------
# Task frame parsing (defensive: logged defaults, never an exception)
# ---------------------------------------------------------------------------


def parse_task(raw: Any) -> dict[str, Any] | None:
    """Parses and normalises the evaluator's first frame.

    Returns None when the frame is not a task message at all. Missing
    or null fields degrade to defaults, printed to stderr, so a
    differently shaped task message still starts a session.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(msg, dict) or msg.get("type") != "task":
        return None

    def defaulted(what: str) -> None:
        print(f"serve: task field defaulted: {what}", file=sys.stderr, flush=True)

    instruction = msg.get("instruction")
    if not isinstance(instruction, str):
        defaulted("instruction -> ''")
        instruction = ""
    workdir = msg.get("workdir")
    if not isinstance(workdir, str) or not workdir:
        defaulted("workdir -> '/'")
        workdir = "/"
    limits = msg.get("limits")
    if not isinstance(limits, dict):
        defaulted("limits -> {}")
        limits = {}
    timeout = limits.get("agent_timeout_sec")
    if not _is_number(timeout) or timeout <= 0:
        if timeout is not None:
            defaulted("agent_timeout_sec -> None (no deadline)")
        timeout = None
    servers = msg.get("mcp_servers")
    if not isinstance(servers, list):
        if servers is not None:
            defaulted("mcp_servers -> []")
        servers = []
    servers = [s for s in servers if isinstance(s, dict)]
    session = msg.get("session")
    if not isinstance(session, str) or not session:
        session = f"task-{os.getpid()}-{next(_session_counter)}"
    return {
        **msg,
        "type": "task",
        "protocol": msg.get("protocol", VERSION),
        "session": session,
        "instruction": instruction,
        "workdir": workdir,
        "limits": {**limits, "agent_timeout_sec": timeout},
        "mcp_servers": servers,
    }


_session_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# One session: reader, loop, watchdog, and the supervising arbiter paths
# ---------------------------------------------------------------------------


class Session:
    """One task session on one evaluator connection."""

    def __init__(self, ws: Any, runner, config: dict[str, Any] | None = None):
        # runner: run_harness(task, emit, workdir, config); a coroutine
        # function is awaited with the async emit, a plain function is
        # run in a worker thread with the synchronous bridge emit.
        self.ws = ws
        self._runner = runner
        self._loop: asyncio.AbstractEventLoop | None = None
        self._base_config = config or {}
        self.serialiser: Serialiser | None = None
        self.task: dict[str, Any] | None = None
        self.state_dir: str | None = None
        self.cancel_event = asyncio.Event()
        self.finalize_event = asyncio.Event()
        # Threading mirrors of the two signals: the loop and the
        # model client run in a worker thread, where polling an
        # asyncio event is not safe; these are set alongside them.
        self._cancel_flag = threading.Event()
        self._finalize_flag = threading.Event()
        self._self_done = asyncio.Event()
        self._ended = asyncio.Event()
        self._pending: dict[str, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._inflight = asyncio.Semaphore(MAX_IN_FLIGHT)
        self._receipt: float | None = None
        self._limit_sec: float | None = None

    # -- time ---------------------------------------------------------

    @property
    def deadline(self) -> float | None:
        if self._receipt is None or self._limit_sec is None:
            return None
        return self._receipt + self._limit_sec

    def time_left(self) -> float | None:
        end = self.deadline
        return None if end is None else end - time.monotonic()

    # -- emit: the loop's only way out --------------------------------

    async def emit(self, frame: Any) -> dict[str, Any] | None:
        """Sends one non-terminal frame; see the module docstring."""
        if self._ended.is_set() or (
            self.serialiser is not None and self.serialiser.terminal_claimed
        ):
            raise SessionCancelled("the session has ended")
        kind = frame.get("type") if isinstance(frame, dict) else None
        if kind in REQUESTS:
            if self.cancel_event.is_set():
                raise SessionCancelled("cancelled: no new requests")
            async with self._inflight:
                # Re-check after acquiring the semaphore: a request
                # that waited for a slot while a cancel landed must
                # not be sent once the slot frees.
                if self.cancel_event.is_set() or self._ended.is_set():
                    raise SessionCancelled("cancelled: no new requests")
                return await self._request(frame)
        await self.serialiser.send(frame)
        return None

    async def _request(self, frame: dict[str, Any]) -> dict[str, Any]:
        rid = str(next(self._ids))
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future
        try:
            await self.serialiser.send({**frame, "id": rid})
        except ConnectionClosed:
            self._pending.pop(rid, None)
            raise SessionCancelled("the connection closed") from None
        except BaseException:
            self._pending.pop(rid, None)
            raise
        try:
            return await future
        finally:
            self._pending.pop(rid, None)
            if not future.done():
                future.cancel()

    def _bridge_emit(self, frame: Any) -> dict[str, Any] | None:
        """The synchronous emit handed to a loop running in a thread.

        Schedules the async emit on the session's event loop and
        blocks the loop thread until it completes, returning the
        evaluator's result dict for request frames and None for
        one-way frames. When the session ends or is cancelled, the
        pending emit is abandoned and SessionCancelled is raised in
        the loop thread, so the loop unwinds promptly; exceptions the
        async emit itself raises (FrameError, SessionCancelled)
        propagate unchanged.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            raise SessionCancelled("the session has ended")
        future = asyncio.run_coroutine_threadsafe(self.emit(frame), loop)
        kind = frame.get("type") if isinstance(frame, dict) else None
        try:
            while True:
                try:
                    return future.result(timeout=0.05)
                except TimeoutError:
                    if self._ended.is_set() or (
                        self.serialiser is not None
                        and self.serialiser.terminal_claimed
                    ):
                        future.cancel()
                        raise SessionCancelled(
                            "the session has ended"
                        ) from None
                    if kind in REQUESTS and self.cancel_event.is_set():
                        # a request blocked somewhere the pending-
                        # request failure cannot reach (for example
                        # the in-flight semaphore): abandon it here
                        future.cancel()
                        raise SessionCancelled(
                            "cancelled: no new requests"
                        ) from None
        except concurrent.futures.CancelledError:
            raise SessionCancelled("the session has ended") from None

    # -- inbound --------------------------------------------------------

    async def _raw_send(self, payload: str) -> None:
        await self.ws.send(payload)

    def _fail_pending(self) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    SessionCancelled("the session is ending")
                )
        self._pending.clear()

    def _on_cancel(self) -> None:
        if not self.cancel_event.is_set():
            self.cancel_event.set()
            self._cancel_flag.set()
            self._fail_pending()

    async def _reader(self) -> None:
        """Routes inbound frames from the moment the session starts.

        Results go strictly to the pending request with their id; a
        result for an unknown or abandoned id is dropped. A `cancel`
        frame pre-empts the loop: pending awaits fail at once.
        """
        try:
            while True:
                raw = await self.ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    raw = bytes(raw).decode("utf-8", "replace")
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "result":
                    rid = msg.get("id")
                    future = (
                        self._pending.pop(rid, None)
                        if isinstance(rid, str)
                        else None
                    )
                    if future is not None and not future.done():
                        future.set_result(msg)
                elif kind == "cancel":
                    self._on_cancel()
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as exc:  # the reader must never take the session down
            print(
                f"serve: reader stopped: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            self._ended.set()
            self._fail_pending()

    # -- watchdog -------------------------------------------------------

    async def _watchdog(self) -> None:
        """Sets the finalize signal, then the self-done signal.

        Milestones are fractions of the task's stated limit with
        absolute caps, so a short-limit task is still worked, not
        finalised in its first minute.
        """
        limit = self._limit_sec
        receipt = self._receipt
        if limit is None or receipt is None:
            return
        force_at = receipt + limit - min(
            FORCE_FINALIZE_CAP_SEC, FORCE_FINALIZE_FRACTION * limit
        )
        done_at = receipt + limit - min(
            SELF_DONE_CAP_SEC, SELF_DONE_FRACTION * limit
        )
        delay = force_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self.finalize_event.set()
        self._finalize_flag.set()
        delay = done_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        self._self_done.set()

    # -- the loop -------------------------------------------------------

    def _loop_config(self) -> dict[str, Any]:
        return {
            "session": self.task["session"],
            "state_dir": self.state_dir,
            "limits": self.task["limits"],
            "mcp_servers": self.task["mcp_servers"],
            "deadline": self.deadline,
            "cancel_event": self.cancel_event,
            "finalize_event": self.finalize_event,
            "cancel_flag": self._cancel_flag,
            "finalize_flag": self._finalize_flag,
            "time_left": self.time_left,
            # The production tool transport: the dispatcher resolves
            # its request callable from the emit it is handed (the
            # harness reads `request`/`transport` first, then this
            # flag), so exec/read_file/write_file go out as request
            # frames through the serialiser and mcp_call rides the
            # http request type — nothing runs locally.
            "emit_is_transport": True,
        }

    async def _run_loop(self) -> Any:
        if asyncio.iscoroutinefunction(self._runner):
            return await self._runner(
                self.task, self.emit, self.task["workdir"],
                self._loop_config(),
            )
        # A synchronous runner (the production harness) runs in a
        # worker thread with the synchronous bridge emit, so the event
        # loop stays free to read, route results, and supervise.
        self._loop = asyncio.get_running_loop()
        return await asyncio.to_thread(
            self._runner,
            self.task,
            self._bridge_emit,
            self.task["workdir"],
            self._loop_config(),
        )

    # -- closing --------------------------------------------------------

    def _fallback_summary(self) -> str:
        """The closing summary from disk state, or the stopped message."""
        if self.state_dir:
            path = os.path.join(self.state_dir, SUMMARY_FILENAME)
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read(FINAL_MESSAGE_CAP)
                if text.strip():
                    return text
            except OSError:
                pass
        return STOPPED_MESSAGE

    @staticmethod
    def _outcome_frame(outcome: Any) -> tuple[str, str]:
        if isinstance(outcome, dict):
            status = outcome.get("status")
            if status in ("done", "finalized"):
                return "done", str(outcome.get("final_message") or "")
            if status == "error":
                return "error", str(outcome.get("message") or "harness error")
            if status == "error_outcome":
                return "error", str(
                    outcome.get("error")
                    or outcome.get("message")
                    or "harness error"
                )
        return "error", "the harness returned an invalid outcome"

    async def _finish_from_loop(self, loop_task: asyncio.Task) -> None:
        if loop_task.cancelled():
            kind, text = "error", "the harness was stopped before it finished"
            with contextlib.suppress(ConnectionClosed):
                await self.serialiser.send_terminal(kind, text)
            return
        exc = loop_task.exception()
        if exc is not None:
            print(
                f"serve: harness error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            kind, text = "error", f"{type(exc).__name__}: {exc}"
        else:
            kind, text = self._outcome_frame(loop_task.result())
        with contextlib.suppress(ConnectionClosed):
            await self.serialiser.send_terminal(kind, text)

    async def _finish_cancel(self, loop_task: asyncio.Task) -> None:
        """The cancel path: `done` inside the window, no new requests.

        The loop first gets a short grace to return its own outcome
        (it can catch SessionCancelled from a pending emit and close
        with its summary); after that it is cancelled and given a
        bounded unwind before the disk summary is used instead.
        """
        outcome = None
        if not loop_task.done():
            await asyncio.wait({loop_task}, timeout=CANCEL_LOOP_GRACE_SEC)
        if loop_task.done() and not loop_task.cancelled():
            if loop_task.exception() is None:
                outcome = loop_task.result()
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.wait({loop_task}, timeout=CANCEL_UNWIND_SEC)
            if (
                loop_task.done()
                and not loop_task.cancelled()
                and loop_task.exception() is None
            ):
                outcome = loop_task.result()
        if isinstance(outcome, dict) and outcome.get("status") in (
            "done",
            "finalized",
        ):
            text = str(outcome.get("final_message") or "")
        else:
            text = self._fallback_summary()
        with contextlib.suppress(ConnectionClosed):
            await self.serialiser.send_terminal("done", text)

    async def _finish_watchdog(self, loop_task: asyncio.Task) -> None:
        """Self-done at the hard wall: send `done`, then stop the loop."""
        with contextlib.suppress(ConnectionClosed):
            await self.serialiser.send_terminal("done", self._fallback_summary())
        if not loop_task.done():
            loop_task.cancel()
            await asyncio.wait({loop_task}, timeout=CANCEL_UNWIND_SEC)

    async def _supervise(self, loop_task: asyncio.Task) -> None:
        """Races the loop against cancel, self-done, and socket loss.

        Whichever path runs, the serialiser's terminal claim decides
        what actually reaches the wire, so concurrent paths cannot
        produce two terminal frames.
        """
        waiters = {
            loop_task,
            asyncio.ensure_future(self.cancel_event.wait()),
            asyncio.ensure_future(self._ended.wait()),
        }
        if self._limit_sec is not None:
            waiters.add(asyncio.ensure_future(self._self_done.wait()))
        try:
            while True:
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                if self.cancel_event.is_set():
                    await self._finish_cancel(loop_task)
                    return
                if loop_task in done:
                    await self._finish_from_loop(loop_task)
                    return
                if self._self_done.is_set():
                    await self._finish_watchdog(loop_task)
                    return
                if self._ended.is_set():
                    # the socket is gone: no terminal frame can be sent
                    if not loop_task.done():
                        loop_task.cancel()
                        await asyncio.wait(
                            {loop_task}, timeout=CANCEL_UNWIND_SEC
                        )
                    return
                # a waiter finished without a decided path: keep waiting
        finally:
            for waiter in waiters:
                if waiter is not loop_task:
                    waiter.cancel()

    # -- session entry ----------------------------------------------------

    @staticmethod
    def _safe_dir_name(session_id: str) -> str:
        """A session id reduced to one safe path component.

        Anything outside [A-Za-z0-9_-] collapses to an underscore, so
        separators and dot runs from an arbitrary id cannot traverse
        out of the state root.
        """
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("._")
        return name or "session"

    def _open_state_dir(self, session_id: str) -> None:
        root = self._base_config.get("state_root") or os.path.join(
            os.environ.get("HOME") or ".", "state"
        )
        path = os.path.join(root, self._safe_dir_name(session_id))
        try:
            if os.path.commonpath(
                [os.path.abspath(path), os.path.abspath(root)]
            ) != os.path.abspath(root):
                self.state_dir = None
                return
            os.makedirs(path, exist_ok=True)
            self.state_dir = path
        except (OSError, ValueError):
            self.state_dir = None

    async def run(self) -> None:
        """Runs the session from the first frame to the terminal frame."""
        try:
            raw = await asyncio.wait_for(self.ws.recv(), START_SEC)
        except (asyncio.TimeoutError, ConnectionClosed):
            return
        self.task = parse_task(raw)
        if self.task is None:
            self.serialiser = Serialiser(self._raw_send, [])
            with contextlib.suppress(ConnectionClosed):
                await self.serialiser.send_terminal(
                    "error", "expected a task message first"
                )
            with contextlib.suppress(Exception):
                await self.ws.close()
            return
        self._receipt = time.monotonic()
        timeout = self.task["limits"].get("agent_timeout_sec")
        self._limit_sec = float(timeout) if timeout is not None else None
        self.serialiser = Serialiser(self._raw_send, self.task["mcp_servers"])
        self._open_state_dir(self.task["session"])

        reader = asyncio.ensure_future(self._reader())
        watchdog = (
            asyncio.ensure_future(self._watchdog())
            if self._limit_sec is not None
            else None
        )
        loop_task = asyncio.ensure_future(self._run_loop())
        try:
            await self._supervise(loop_task)
        finally:
            self._fail_pending()
            for task_ in (reader, watchdog, loop_task):
                if task_ is not None and not task_.done():
                    task_.cancel()
            with contextlib.suppress(Exception):
                await self.ws.close()


# ---------------------------------------------------------------------------
# Server entry: lazy bind on 0.0.0.0:$PORT, harness imported on first task
# ---------------------------------------------------------------------------

_runner_cache: dict[str, Any] = {}


def _load_runner():
    """Imports harness.py lazily, on the first connection.

    A failed import does not stop the listener: the session for that
    task ends with an `error` frame naming the failure, and the
    import is retried on the next connection.
    """
    if "runner" not in _runner_cache:
        import harness

        _runner_cache["runner"] = harness.run_harness
    return _runner_cache["runner"]


async def handle_connection(ws: Any, config: dict[str, Any] | None = None) -> None:
    try:
        runner = _load_runner()
    except Exception as exc:  # configuration failure: fail this task loudly

        async def runner(task, emit, workdir, cfg, _exc=exc):
            raise _exc

    await Session(ws, runner, config).run()


async def serve_forever(port: int | None = None) -> None:
    from websockets.asyncio.server import serve as _ws_serve

    port = int(port or os.environ.get("PORT", DEFAULT_PORT))
    async with _ws_serve(
        handle_connection,
        "0.0.0.0",
        port,
        max_size=MAX_FRAME_BYTES,
        ping_interval=None,
    ):
        print(f"listening on {port}", flush=True)
        await asyncio.Future()


def run(port: int | None = None) -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_forever(port))


if __name__ == "__main__":
    run()
