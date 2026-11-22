"""Regression tests for the tools.py fix round.

Covers: workdir path confinement, raise-only mutating classification,
cancellation propagation through dispatch, MCP recovery on the
evaluator-path failure shapes, write-integrity probe rules, opt-in
line-number stripping, background exec lifecycle, exec/read size
caps, and the exec deadline clamp.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import (  # noqa: E402
    IntegrityError,
    MCPAmbiguousResult,
    MCPLayer,
    StubMCPTransport,
    ToolDispatcher,
    ToolError,
)


# ---------------------------------------------------------------------
# Fix 1: path confinement
# ---------------------------------------------------------------------

def test_absolute_path_rejected(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch("read_file", {"path": "/etc/hostname"})
    assert result["ok"] is False
    result = dispatcher.dispatch(
        "write_file", {"path": "/tmp/ple-escape-abs.txt", "content": "x"})
    assert result["ok"] is False
    assert not Path("/tmp/ple-escape-abs.txt").exists()


def test_dotdot_escape_rejected(tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    dispatcher = ToolDispatcher(workdir)
    result = dispatcher.dispatch(
        "write_file", {"path": "../escape.txt", "content": "x"})
    assert result["ok"] is False
    assert not (tmp_path / "escape.txt").exists()
    result = dispatcher.dispatch("read_file", {"path": "../escape.txt"})
    assert result["ok"] is False


def test_symlinked_dir_escape_rejected(tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / "link").symlink_to(outside, target_is_directory=True)
    dispatcher = ToolDispatcher(workdir)
    result = dispatcher.dispatch(
        "write_file", {"path": "link/esc.txt", "content": "x"})
    assert result["ok"] is False
    assert not (outside / "esc.txt").exists()


def test_symlinked_deliverable_path_rejected(tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("ORIGINAL")
    (workdir / "deliverable.txt").symlink_to(target)
    dispatcher = ToolDispatcher(workdir)
    result = dispatcher.dispatch(
        "write_file", {"path": "deliverable.txt", "content": "OVERWRITTEN"})
    assert result["ok"] is False
    assert target.read_text() == "ORIGINAL"


def test_symlink_staying_inside_workdir_allowed(tmp_path):
    workdir = tmp_path / "wd"
    (workdir / "real").mkdir(parents=True)
    (workdir / "alias").symlink_to(workdir / "real", target_is_directory=True)
    dispatcher = ToolDispatcher(workdir)
    result = dispatcher.dispatch(
        "write_file", {"path": "alias/a.txt", "content": "inside"})
    assert result["ok"] is True
    assert (workdir / "real" / "a.txt").read_text() == "inside"


def test_confinement_applies_on_evaluator_transport(tmp_path):
    seen = []

    def fake(frame):
        seen.append(frame)
        if frame["type"] == "read_file":
            return {"ok": False, "error": "not found"}
        return {"ok": True, "size": 1}

    dispatcher = ToolDispatcher(tmp_path, request=fake)
    result = dispatcher.dispatch(
        "write_file", {"path": "../escape.txt", "content": "x"})
    assert result["ok"] is False
    assert seen == []  # rejected before any transport call


# ---------------------------------------------------------------------
# Fix 2: mutating classification is raise-only
# ---------------------------------------------------------------------

def test_mutating_flag_cannot_lower_classification():
    layer = MCPLayer({}, transport=StubMCPTransport())
    assert layer.is_mutating("delete_record", explicit=False) is True
    assert layer.is_mutating("archive_report", explicit=False) is True
    assert layer.is_mutating("rename_file", explicit=False) is True
    assert layer.is_mutating("transfer_funds", explicit=False) is True
    assert layer.is_mutating("patch_item", explicit=False) is True
    assert layer.is_mutating("get_record", explicit=True) is True
    assert layer.is_mutating("get_record", explicit=False) is False
    assert layer.is_mutating("get_record") is False


def test_mutating_config_floor_applies():
    layer = MCPLayer({}, transport=StubMCPTransport(),
                     config={"mutating_tools": {"frobnicate"}})
    assert layer.is_mutating("frobnicate", explicit=False) is True


class DropAfterApplyTransport(StubMCPTransport):
    """Applies tools/call effects, then drops the first response."""

    def __init__(self):
        super().__init__()
        self.applied = 0
        self._dropped = False

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/call":
            self.applied += 1
            if not self._dropped:
                self._dropped = True
                raise ConnectionError("response lost")
        return super().__call__(url, payload, headers)


def test_model_mutating_false_cannot_defeat_double_apply_protection():
    transport = DropAfterApplyTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    with pytest.raises(MCPAmbiguousResult):
        layer.call("s", "delete_record", {}, mutating=False)
    assert transport.applied == 1


# ---------------------------------------------------------------------
# Fix 3: dispatch re-raises cancellation
# ---------------------------------------------------------------------

class SessionCancelled(Exception):
    """Same class name as serve.py's cancellation exception."""


def test_dispatch_reraises_session_cancelled(tmp_path):
    def cancelling(frame):
        raise SessionCancelled("cancelled")

    dispatcher = ToolDispatcher(tmp_path, request=cancelling)
    with pytest.raises(SessionCancelled):
        dispatcher.dispatch("read_file", {"path": "a.txt"})


def test_dispatch_still_swallows_ordinary_errors(tmp_path):
    def broken(frame):
        raise RuntimeError("boom")

    dispatcher = ToolDispatcher(tmp_path, request=broken)
    result = dispatcher.dispatch("read_file", {"path": "a.txt"})
    assert result["ok"] is False
    assert result["error_class"] == "internal"


# ---------------------------------------------------------------------
# Fix 4: MCP recovery on evaluator-path failure shapes
# ---------------------------------------------------------------------

class EvaluatorHTTP:
    """Fake evaluator http callable: frames in, result dicts out.

    ``fail_calls`` counts how many tools/call requests report a
    transport failure (ok: False) before succeeding.
    """

    def __init__(self, fail_calls: int = 0):
        self.fail_calls = fail_calls
        self.calls = 0
        self.inits = 0

    def __call__(self, frame):
        payload = json.loads(base64.b64decode(frame["body_b64"]))
        method = payload.get("method")
        if method == "initialize":
            self.inits += 1
            body = {"jsonrpc": "2.0", "id": payload.get("id"),
                    "result": {"protocolVersion": "2025-06-18"}}
            return {"ok": True, "status": 200,
                    "headers": {"Mcp-Session-Id": "eval-session"},
                    "body_b64": base64.b64encode(
                        json.dumps(body).encode()).decode()}
        if method == "notifications/initialized":
            return {"ok": True, "status": 202, "headers": {},
                    "body_b64": ""}
        if method == "tools/call":
            self.calls += 1
            if self.calls <= self.fail_calls:
                return {"ok": False, "error": "connection reset by peer"}
            body = {"jsonrpc": "2.0", "id": payload.get("id"),
                    "result": {"done": True}}
            return {"ok": True, "status": 200, "headers": {},
                    "body_b64": base64.b64encode(
                        json.dumps(body).encode()).decode()}
        return {"ok": True, "status": 200, "headers": {}, "body_b64": ""}


def test_evaluator_transport_loss_recovers_non_mutating():
    http = EvaluatorHTTP(fail_calls=1)
    layer = MCPLayer({"s": "http://x"}, http_request=http)
    result = layer.call("s", "get_record", {})
    assert result == {"done": True}
    assert http.calls == 2
    assert http.inits == 2  # session was re-initialised before retry


def test_evaluator_transport_loss_mutating_is_ambiguous():
    http = EvaluatorHTTP(fail_calls=1)
    layer = MCPLayer({"s": "http://x"}, http_request=http)
    with pytest.raises(MCPAmbiguousResult):
        layer.call("s", "create_record", {})
    assert http.calls == 1  # never blindly retried


class SessionExpiryTransport(StubMCPTransport):
    """First tools/call answers HTTP 404 (session gone), then works."""

    def __init__(self):
        super().__init__()
        self.expired_once = False

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/call" and not self.expired_once:
            self.expired_once = True
            return {"status": 404, "headers": {},
                    "body": {"error": "no such session"}}
        return super().__call__(url, payload, headers)


def test_http_404_session_expiry_recovers():
    layer = MCPLayer({"s": "http://x"}, transport=SessionExpiryTransport())
    result = layer.call("s", "get_record", {})
    assert "content" in result


# ---------------------------------------------------------------------
# Fix 5: write integrity probe rules
# ---------------------------------------------------------------------

class NoSizeRequest:
    """Transport that writes for real but never reports a size and
    answers reads without one either (probe returns None)."""

    def __init__(self, root: Path):
        self.root = root

    def __call__(self, frame):
        kind = frame.get("type")
        if kind == "read_file":
            path = self.root / frame["path"]
            if not path.exists():
                return {"ok": False, "error": "not found"}
            return {"ok": True, "content_b64": base64.b64encode(
                path.read_bytes()).decode(), "truncated": False}
        if kind == "write_file":
            path = self.root / frame["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            data = base64.b64decode(frame["content_b64"])
            with open(path, "ab" if frame.get("append") else "wb") as fh:
                fh.write(data)
            return {"ok": True}
        raise AssertionError(kind)


def test_none_probe_does_not_waive_final_verification(tmp_path):
    dispatcher = ToolDispatcher(tmp_path, request=NoSizeRequest(tmp_path))
    with pytest.raises(IntegrityError):
        dispatcher.write_bytes("a.bin", b"hello")


def test_append_with_none_probe_refused(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    dispatcher = ToolDispatcher(tmp_path, request=NoSizeRequest(tmp_path))
    with pytest.raises(IntegrityError):
        dispatcher.write_bytes("a.bin", b"more", append=True)


class DropBothRequest:
    """Writes land, but the first write response AND the resend
    response are both lost."""

    def __init__(self, root: Path):
        self.root = root
        self.writes = 0

    def __call__(self, frame):
        kind = frame.get("type")
        if kind == "read_file":
            path = self.root / frame["path"]
            if not path.exists():
                return {"ok": False, "error": "not found", "size": 0}
            data = path.read_bytes()
            return {"ok": True, "size": len(data),
                    "content_b64": base64.b64encode(data).decode(),
                    "truncated": False}
        if kind == "write_file":
            self.writes += 1
            path = self.root / frame["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            data = base64.b64decode(frame["content_b64"])
            if self.writes == 1:
                raise ConnectionError("first response lost, not landed")
            with open(path, "ab" if frame.get("append") else "wb") as fh:
                fh.write(data)
            if self.writes == 2:
                raise ConnectionError("resend response lost after landing")
            return {"ok": True, "size": path.stat().st_size}
        raise AssertionError(kind)


def test_resend_lost_response_is_probed_not_raw(tmp_path):
    fake = DropBothRequest(tmp_path)
    dispatcher = ToolDispatcher(tmp_path, request=fake)
    # First send lost before landing; resend lands but its response is
    # lost too. The probe after the resend must see the landed chunk
    # and the write completes instead of leaking a ConnectionError.
    dispatcher.write_bytes("a.bin", b"payload")
    assert (tmp_path / "a.bin").read_bytes() == b"payload"


class DropResendUnlandedRequest(DropBothRequest):
    """Resend response is lost and the resend never landed."""

    def __call__(self, frame):
        if frame.get("type") == "write_file" and self.writes == 1:
            self.writes += 1
            raise ConnectionError("resend lost, not landed")
        return super().__call__(frame)


def test_resend_unlanded_reports_integrity_not_raw_error(tmp_path):
    fake = DropResendUnlandedRequest(tmp_path)
    dispatcher = ToolDispatcher(tmp_path, request=fake)
    result = dispatcher.dispatch("write_file",
                                 {"path": "a.bin", "content": "payload"})
    assert result["ok"] is False
    assert result["error_class"] == "integrity"


# ---------------------------------------------------------------------
# Fix 6: write_file never silently alters content
# ---------------------------------------------------------------------

def test_numbered_tsv_round_trips_byte_identical(tmp_path):
    rows = "".join(f"{i}\tvalue-{i}\n" for i in range(1, 11))
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch(
        "write_file", {"path": "data.tsv", "content": rows})
    assert result["ok"] is True
    assert (tmp_path / "data.tsv").read_text() == rows


def test_strip_line_numbers_is_opt_in(tmp_path):
    rows = "".join(f"{i}\tvalue-{i}\n" for i in range(1, 11))
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch(
        "write_file", {"path": "data.tsv", "content": rows,
                       "strip_line_numbers": True})
    assert result["ok"] is True
    written = (tmp_path / "data.tsv").read_text()
    assert written != rows
    assert written.startswith("value-1\n")


# ---------------------------------------------------------------------
# Fix 8: background exec lifecycle
# ---------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def test_background_jobs_reaped(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch("exec", {"command": "true",
                                          "background": True})
    assert result["ok"] is True
    pid = result["data"]["pid"]
    deadline = time.time() + 5
    reaped: list[int] = []
    while time.time() < deadline and pid not in reaped:
        reaped = dispatcher.reap_background()
        time.sleep(0.05)
    assert pid in reaped
    assert pid not in dispatcher._background


def test_stop_all_background_kills_group(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch(
        "exec", {"command": "sleep 60 & wait", "background": True})
    pid = result["data"]["pid"]
    assert _pid_alive(pid)
    stopped = dispatcher.stop_all_background()
    assert stopped == 1
    deadline = time.time() + 5
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.05)
    assert not _pid_alive(pid)


def test_stop_background_unknown_pid_signals_nothing(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    assert dispatcher.stop_background(999999) is False


def test_foreground_exec_kills_group_on_return(tmp_path):
    dispatcher = ToolDispatcher(tmp_path)
    result = dispatcher.dispatch(
        "exec", {"command": "sleep 60 & echo $! > child.pid; echo done"})
    assert result["ok"] is True
    child = int((tmp_path / "child.pid").read_text().strip())
    deadline = time.time() + 5
    while time.time() < deadline and _pid_alive(child):
        time.sleep(0.05)
    assert not _pid_alive(child)


# ---------------------------------------------------------------------
# Fix 9: size caps
# ---------------------------------------------------------------------

def test_exec_output_capped_head_and_tail(tmp_path):
    dispatcher = ToolDispatcher(
        tmp_path, config={"max_exec_output_chars": 1000})
    result = dispatcher.dispatch(
        "exec", {"command": "printf 'a%.0s' $(seq 1 5000)"})
    assert result["ok"] is True
    output = result["data"]["output"]
    assert result["truncated"] is True
    assert len(output) <= 1000
    assert output.startswith("aaa")
    assert output.endswith("aaa")


def test_read_file_bounded(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 5000)
    dispatcher = ToolDispatcher(tmp_path, config={"max_read_bytes": 100})
    result = dispatcher.dispatch("read_file", {"path": "big.txt"})
    assert result["ok"] is False
    with pytest.raises(ToolError):
        dispatcher.read_bytes("big.txt")


# ---------------------------------------------------------------------
# Fix 10: exec deadline clamp
# ---------------------------------------------------------------------

def test_exec_clamped_to_remaining_task_time(tmp_path):
    dispatcher = ToolDispatcher(
        tmp_path, config={"deadline": time.time() + 4})
    start = time.time()
    result = dispatcher.dispatch(
        "exec", {"command": "sleep 30", "timeout_s": 600})
    elapsed = time.time() - start
    assert result["ok"] is True
    assert result["data"]["timed_out"] is True
    assert elapsed < 10  # clamped to ~2 s of remaining time, not 30/600


def test_exec_with_expired_deadline_refused(tmp_path):
    dispatcher = ToolDispatcher(
        tmp_path, config={"deadline": time.time() - 1})
    result = dispatcher.dispatch("exec", {"command": "true"})
    assert result["ok"] is False
