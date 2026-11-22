"""Regression tests for the shared monotonic clock contract,
annotation-aware mutating classification, background-job tracking
on the evaluator transport, schema-compile honesty, the late-return
ambiguity path, and discovery accounting (duplicates, partial
fetches, the per-server catalogue bound)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import (MCPAmbiguousResult, MCPError, MCPLayer,  # noqa: E402
                   StubMCPTransport, ToolDispatcher)


def serve_style_config(span: float = 3600.0) -> dict:
    """The loop config exactly as serve's session builds it: a
    monotonic deadline plus the ``time_left`` callable."""
    deadline = time.monotonic() + span
    return {"deadline": deadline,
            "time_left": lambda: deadline - time.monotonic()}


def call_frames(transport) -> list[dict]:
    return [c for c in transport.calls if c.get("method") == "tools/call"]


# ---------------------------------------------------------------------
# Clock contract: numeric deadlines are monotonic, time_left preferred
# ---------------------------------------------------------------------

def test_serve_style_config_exec_runs_and_mcp_budget_sane(tmp_path):
    dispatcher = ToolDispatcher(tmp_path, config=serve_style_config())
    result = dispatcher.dispatch("exec", {"command": "echo hello"})
    assert result["ok"] is True
    assert "hello" in result["data"]["output"]

    layer = MCPLayer({"s": "http://x"}, transport=StubMCPTransport(),
                     config=serve_style_config())
    budget = layer._remaining_call_timeout()
    assert 0 < budget <= 300
    # And a call under that config is actually sent, not refused.
    assert "content" in layer.call("s", "get_record", {})


def test_monotonic_deadline_in_the_past_refuses(tmp_path):
    transport = StubMCPTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport,
                     config={"deadline": time.monotonic() - 5})
    with pytest.raises(MCPError, match="no task time remaining"):
        layer.call("s", "get_record", {})
    assert call_frames(transport) == []

    dispatcher = ToolDispatcher(
        tmp_path, config={"deadline": time.monotonic() - 5})
    result = dispatcher.dispatch("exec", {"command": "true"})
    assert result["ok"] is False


# ---------------------------------------------------------------------
# Mutating classification: widened name floor and server annotations
# ---------------------------------------------------------------------

class ApplyThenLoseTransport(StubMCPTransport):
    """Applies each tools/call, then loses the first response."""

    def __init__(self, tools=None):
        super().__init__(tools=tools)
        self.applied = 0
        self._dropped = False

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/call":
            self.applied += 1
            if not self._dropped:
                self._dropped = True
                raise ConnectionError("response lost")
        return super().__call__(url, payload, headers)


def test_widened_name_floor_classifies_realistic_mutating_names():
    layer = MCPLayer({}, transport=StubMCPTransport())
    for name in ("place_order", "commit_changes", "reserve_seat",
                 "reject_invoice", "sign_document"):
        assert layer.is_mutating(name, explicit=False) is True, name
    assert layer.is_mutating("get_record", explicit=False) is False


def test_place_order_lost_response_applied_once_and_ambiguous():
    transport = ApplyThenLoseTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    with pytest.raises(MCPAmbiguousResult):
        layer.call("s", "place_order", {"item": "book"}, mutating=False)
    assert transport.applied == 1


UPDATE_PREFS = {
    "name": "update_preferences",
    "description": "Update stored preferences",
    "inputSchema": {"type": "object",
                    "properties": {"theme": {"type": "string"}}},
    "annotations": {"readOnlyHint": True},
}
FETCH_DATA = {
    "name": "fetch_data",
    "description": "Fetch a data extract",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"readOnlyHint": False, "destructiveHint": True},
}
PLAIN_GET = {
    "name": "get_record",
    "description": "Fetch one record",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {"destructiveHint": False},
}


def test_annotations_carried_into_catalogue_and_classification():
    transport = StubMCPTransport(
        tools=[UPDATE_PREFS, FETCH_DATA, PLAIN_GET])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    layer.discover_all()
    record = layer.catalogue["s"]["tools"]["update_preferences"]
    assert record["annotations"] == {"readOnlyHint": True}
    # The name floor alone would call update_preferences mutating;
    # the server's read-only declaration settles it the other way.
    assert layer.is_mutating("update_preferences", explicit=False,
                             server="s") is False
    # A destructive hint classifies a read-shaped name as mutating.
    assert layer.is_mutating("fetch_data", explicit=False,
                             server="s") is True
    # Annotations without a read-only declaration: a clearly
    # read-only name stays non-mutating.
    assert layer.is_mutating("get_record", explicit=False,
                             server="s") is False


def test_readonly_hint_tool_is_not_held_back_by_lost_response():
    transport = ApplyThenLoseTransport(tools=[UPDATE_PREFS])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    layer.discover_all()
    result = layer.call("s", "update_preferences", {"theme": "dark"})
    assert "content" in result
    # Non-mutating: the lost response was retried after
    # re-initialisation instead of being reported ambiguous.
    assert transport.applied == 2


# ---------------------------------------------------------------------
# Background jobs on the evaluator transport
# ---------------------------------------------------------------------

class FakeExecTransport:
    """Answers dispatcher exec frames the way the evaluator would:
    the background start wrapper's pid on stdout, and a kill frame
    that reports the group dead — or fails outright."""

    def __init__(self, kill_works: bool = True):
        self.frames: list[dict] = []
        self.kill_works = kill_works

    def __call__(self, frame):
        self.frames.append(frame)
        command = frame.get("command", "")
        if "setsid" in command:
            return {"ok": True, "exit_code": 0, "output": "4321\n"}
        if command.startswith("kill -TERM"):
            if not self.kill_works:
                raise ConnectionError("transport down")
            return {"ok": True, "exit_code": 0, "output": "dead\n"}
        return {"ok": True, "exit_code": 0, "output": ""}


def test_transport_background_job_registered_and_stopped(tmp_path):
    transport = FakeExecTransport()
    dispatcher = ToolDispatcher(tmp_path, request=transport)
    result = dispatcher.dispatch(
        "exec", {"command": "sleep 60", "background": True})
    assert result["ok"] is True
    assert result["data"]["pid"] == 4321
    assert 4321 in dispatcher._background_remote
    stopped = dispatcher.stop_all_background()
    assert stopped == 1
    kills = [f for f in transport.frames
             if str(f.get("command", "")).startswith("kill -TERM")]
    assert len(kills) == 1
    assert dispatcher.unreconciled_background == []
    assert 4321 not in dispatcher._background_remote


def test_transport_background_stop_unreconciled_when_kill_fails(tmp_path):
    transport = FakeExecTransport(kill_works=False)
    dispatcher = ToolDispatcher(tmp_path, request=transport)
    dispatcher.dispatch("exec", {"command": "sleep 60",
                                 "background": True})
    stopped = dispatcher.stop_all_background()
    assert stopped == 1  # attempted, and reported as such
    assert dispatcher.unreconciled_background == [4321]
    # Still registered, so a later stop_all retries the kill.
    assert 4321 in dispatcher._background_remote


# ---------------------------------------------------------------------
# Schema compilation: unenforced keywords quarantine the tool
# ---------------------------------------------------------------------

def _tool_with(prop_schema, name: str = "t_tool") -> dict:
    return {"name": name, "description": "schema case",
            "inputSchema": {"type": "object",
                            "properties": {"x": prop_schema}}}


SCHEMA_CASES = [
    ({"type": "string", "pattern": "^[a-z]+$"}, "pattern"),
    ({"type": "string", "format": "email"}, "format"),
    ({"type": "integer", "minimum": 10}, "minimum"),
    ({"type": "number", "maximum": 5}, "maximum"),
    ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "oneOf"),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "anyOf"),
    ({"type": "string", "enum": "abc"}, "enum"),
    ({"type": "object", "additionalProperties": False},
     "additionalProperties"),
    ({"type": "array",
      "items": {"type": "object",
                "properties": {"y": {"type": "string"}},
                "required": ["y"]}}, "items"),
]


@pytest.mark.parametrize("prop_schema,keyword", SCHEMA_CASES)
def test_unenforced_keyword_quarantines_the_tool(prop_schema, keyword):
    transport = StubMCPTransport(tools=[_tool_with(prop_schema)])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    entry = layer.discover_all()["servers"]["s"]
    assert entry["callable"] == 0
    assert entry["quarantined_tools"] == ["t_tool"]
    reason = layer.catalogue["s"]["quarantined"][0]["reason"]
    assert keyword in reason
    with pytest.raises(MCPError, match="quarantined"):
        layer.call("s", "t_tool", {"x": 1})
    assert call_frames(transport) == []


def test_covered_subset_still_compiles():
    tool = {"name": "ok_tool", "description": "covered subset only",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["a", "b"]},
                    "tags": {"type": "array",
                             "items": {"type": "string"}},
                    "nested": {"type": "object",
                               "properties": {"n": {"type": "integer"}},
                               "required": ["n"]}},
                "required": ["id"]}}
    transport = StubMCPTransport(tools=[tool])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    entry = layer.discover_all()["servers"]["s"]
    assert entry["callable"] == 1
    assert entry["quarantined_tools"] == []


# ---------------------------------------------------------------------
# Late returns: mutating calls take the ambiguity path
# ---------------------------------------------------------------------

class SlowApplyTransport(StubMCPTransport):
    """tools/call is applied, but the answer takes its time."""

    def __init__(self):
        super().__init__()
        self.applied = 0

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/call":
            self.applied += 1
            time.sleep(0.3)
        return super().__call__(url, payload, headers)


def test_mutating_slow_success_is_ambiguous_and_applied_once():
    transport = SlowApplyTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport,
                     config={"mcp_call_timeout_sec": 0.05})
    with pytest.raises(MCPAmbiguousResult):
        layer.call("s", "delete_record", {})
    assert transport.applied == 1
    assert len(call_frames(transport)) == 1


def test_mutating_slow_success_with_confirming_probe():
    transport = SlowApplyTransport()
    layer = MCPLayer(
        {"s": "http://x"}, transport=transport,
        config={"mcp_call_timeout_sec": 0.05,
                "effect_probes": {"delete_record":
                                  lambda server, tool, args: True}})
    result = layer.call("s", "delete_record", {})
    assert result["effect_confirmed"] is True
    assert transport.applied == 1


def test_non_mutating_slow_success_stays_a_plain_timeout():
    transport = SlowApplyTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport,
                     config={"mcp_call_timeout_sec": 0.05})
    with pytest.raises(MCPError, match="per-call timeout") as excinfo:
        layer.call("s", "get_record", {})
    assert not isinstance(excinfo.value, MCPAmbiguousResult)


# ---------------------------------------------------------------------
# Discovery accounting: duplicates, partial fetches, the size bound
# ---------------------------------------------------------------------

STRICT_DUP = {"name": "dup", "description": "strict version",
              "inputSchema": {"type": "object",
                              "properties": {"a": {"type": "string"}},
                              "required": ["a"]}}
LAX_DUP = {"name": "dup", "description": "lax version",
           "inputSchema": {"type": "object", "properties": {}}}
BROKEN_DUP = {"name": "dup", "description": "broken version",
              "inputSchema": {"type": "object",
                              "properties": {"a": {"type": "funky"}}}}
GOOD_DUP = {"name": "dup", "description": "good version",
            "inputSchema": {"type": "object", "properties": {}}}
TOOL_ONE = {"name": "one", "description": "first",
            "inputSchema": {"type": "object", "properties": {}}}
TOOL_TWO = {"name": "two", "description": "second",
            "inputSchema": {"type": "object", "properties": {}}}


class TwoPageTransport(StubMCPTransport):
    """Serves page_one, then page_two under a cursor."""

    def __init__(self, page_one, page_two):
        super().__init__()
        self._pages = [page_one, page_two]

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/list":
            self.calls.append(dict(payload))
            cursor = (payload.get("params") or {}).get("cursor")
            if cursor is None:
                return {"status": 200, "headers": {},
                        "body": {"jsonrpc": "2.0",
                                 "id": payload.get("id"),
                                 "result": {"tools": self._pages[0],
                                            "nextCursor": "page-2"}}}
            return {"status": 200, "headers": {},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"tools": self._pages[1]}}}
        return super().__call__(url, payload, headers)


def test_duplicate_name_first_wins_and_is_rejected_with_reason():
    transport = TwoPageTransport([STRICT_DUP], [LAX_DUP])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    entry = layer.discover_all()["servers"]["s"]
    assert entry["advertised"] == 2
    assert entry["indexed"] == 1
    assert entry["callable"] == 1
    assert entry["advertised"] == \
        entry["indexed"] + len(entry["rejected"])
    assert entry["rejected"] == [{"name": "dup",
                                  "reason": "duplicate name"}]
    assert entry["quarantined_tools"] == []
    # The first schema governs: its required argument is enforced.
    with pytest.raises(MCPError, match="missing required argument"):
        layer.call("s", "dup", {})
    assert "content" in layer.call("s", "dup", {"a": "x"})


def test_duplicate_name_first_broken_quarantines_the_name():
    transport = TwoPageTransport([BROKEN_DUP], [GOOD_DUP])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    entry = layer.discover_all()["servers"]["s"]
    assert entry["callable"] == 0
    assert entry["quarantined_tools"] == ["dup"]
    assert entry["rejected"] == [{"name": "dup",
                                  "reason": "duplicate name"}]
    assert "dup" not in layer.catalogue["s"]["tools"]


class SecondPageFailsTransport(StubMCPTransport):
    """Page 1 answers with two tools and a cursor; page 2 fails."""

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/list":
            self.calls.append(dict(payload))
            cursor = (payload.get("params") or {}).get("cursor")
            if cursor is None:
                return {"status": 200, "headers": {},
                        "body": {"jsonrpc": "2.0",
                                 "id": payload.get("id"),
                                 "result": {"tools": [TOOL_ONE, TOOL_TWO],
                                            "nextCursor": "page-2"}}}
            return {"status": 500, "headers": {},
                    "body": {"error": "internal"}}
        return super().__call__(url, payload, headers)


def test_mid_pagination_failure_keeps_the_seen_count():
    transport = SecondPageFailsTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    entry = layer.discover_all()["servers"]["s"]
    assert entry["status"] == "quarantined"
    assert entry["advertised"] == 2
    assert entry["callable"] == 0


def test_catalogue_past_the_per_server_bound_is_quarantined():
    tools = [{"name": f"tool_{i}", "description": "bulk",
              "inputSchema": {"type": "object", "properties": {}}}
             for i in range(5)]
    transport = StubMCPTransport(tools=tools)
    layer = MCPLayer({"s": "http://x"}, transport=transport,
                     config={"mcp_max_tools_per_server": 3})
    entry = layer.discover_all()["servers"]["s"]
    assert entry["status"] == "quarantined"
    assert "bound" in entry["reason"]
    assert entry["callable"] == 0
    assert entry["advertised"] == 5
    assert layer.find_tools("tool") == []
