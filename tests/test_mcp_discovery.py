"""Tests for MCP discovery: full pagination, the on-disk catalogue
index, local schema validation, quarantine, recall counts, and the
per-call timeout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import run_harness  # noqa: E402
from tools import (MCPError, MCPLayer, StubMCPTransport,  # noqa: E402
                   ToolDispatcher)

GET_RECORD = {
    "name": "get_record",
    "description": "Fetch one record by id",
    "inputSchema": {
        "type": "object",
        "properties": {"id": {"type": "string"},
                       "count": {"type": "integer"}},
        "required": ["id"],
    },
}
LIST_RECORDS = {
    "name": "list_records",
    "description": "List all records in the store",
    "inputSchema": {"type": "object", "properties": {}},
}
BROKEN_TOOL = {
    "name": "broken_tool",
    "description": "Advertises a schema that cannot compile",
    "inputSchema": {"type": "object",
                    "properties": {"x": {"type": "funky"}}},
}


def list_calls(transport) -> list[dict]:
    return [c for c in transport.calls if c.get("method") == "tools/list"]


def call_calls(transport) -> list[dict]:
    return [c for c in transport.calls if c.get("method") == "tools/call"]


# ---------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------

class PagedTransport(StubMCPTransport):
    """Serves the catalogue in two pages joined by a cursor."""

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/list":
            self.calls.append(dict(payload))
            cursor = (payload.get("params") or {}).get("cursor")
            if cursor is None:
                return {"status": 200, "headers": {},
                        "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                                 "result": {"tools": [GET_RECORD],
                                            "nextCursor": "page-2"}}}
            return {"status": 200, "headers": {},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"tools": [LIST_RECORDS]}}}
        return super().__call__(url, payload, headers)


def test_discovery_pages_fully():
    transport = PagedTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    recall = layer.discover_all()
    entry = recall["servers"]["s"]
    assert entry["status"] == "ok"
    assert entry["advertised"] == 2
    assert entry["indexed"] == 2
    assert entry["callable"] == 2
    assert len(list_calls(transport)) == 2  # both pages were fetched
    assert recall["totals"] == {"advertised": 2, "indexed": 2,
                                "callable": 2}


class EndlessPagesTransport(StubMCPTransport):
    """Always offers another page: pagination never terminates."""

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/list":
            self.calls.append(dict(payload))
            return {"status": 200, "headers": {},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"tools": [GET_RECORD],
                                        "nextCursor": "more"}}}
        return super().__call__(url, payload, headers)


def test_pagination_beyond_bound_is_a_defect_not_truncation():
    transport = EndlessPagesTransport()
    layer = MCPLayer({"s": "http://x"}, transport=transport,
                     config={"mcp_max_list_pages": 3})
    recall = layer.discover_all()
    entry = recall["servers"]["s"]
    # The defect is recorded (quarantine with a pagination reason),
    # not a silently truncated catalogue presented as complete.
    assert entry["status"] == "quarantined"
    assert "pagination" in entry["reason"]
    assert entry["callable"] == 0
    assert len(list_calls(transport)) == 3


# ---------------------------------------------------------------------
# Schema validation before sending
# ---------------------------------------------------------------------

def discovered_layer():
    transport = StubMCPTransport(tools=[GET_RECORD, LIST_RECORDS])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    layer.discover_all()
    return layer, transport


def test_unknown_tool_refused_locally():
    layer, transport = discovered_layer()
    with pytest.raises(MCPError, match="unknown tool"):
        layer.call("s", "no_such_tool", {})
    assert call_calls(transport) == []  # never sent


def test_missing_required_argument_refused_locally():
    layer, transport = discovered_layer()
    with pytest.raises(MCPError, match="missing required argument"):
        layer.call("s", "get_record", {"count": 1})
    assert call_calls(transport) == []


def test_wrong_typed_argument_refused_locally():
    layer, transport = discovered_layer()
    with pytest.raises(MCPError, match="expected type"):
        layer.call("s", "get_record", {"id": 42})
    with pytest.raises(MCPError, match="expected type"):
        layer.call("s", "get_record", {"id": "r1", "count": "three"})
    with pytest.raises(MCPError, match="expected type"):
        # bool is not an integer for validation purposes
        layer.call("s", "get_record", {"id": "r1", "count": True})
    assert call_calls(transport) == []


def test_valid_call_passes_validation_and_is_sent():
    layer, transport = discovered_layer()
    result = layer.call("s", "get_record", {"id": "r1", "count": 2})
    assert "content" in result
    assert len(call_calls(transport)) == 1


def test_uncompilable_schema_quarantines_only_that_tool():
    transport = StubMCPTransport(tools=[GET_RECORD, BROKEN_TOOL])
    layer = MCPLayer({"s": "http://x"}, transport=transport)
    recall = layer.discover_all()
    entry = recall["servers"]["s"]
    assert entry["status"] == "ok"  # the server itself is fine
    assert entry["advertised"] == 2
    assert entry["indexed"] == 2
    assert entry["callable"] == 1
    assert entry["quarantined_tools"] == ["broken_tool"]
    with pytest.raises(MCPError, match="quarantined"):
        layer.call("s", "broken_tool", {"x": 1})
    assert call_calls(transport) == []
    # The healthy tool on the same server still works.
    assert "content" in layer.call("s", "get_record", {"id": "r1"})


# ---------------------------------------------------------------------
# Quarantine on a dead server
# ---------------------------------------------------------------------

class DeadTransport:
    """Every request fails at the connection level."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, payload, headers):
        self.calls.append(dict(payload))
        raise ConnectionError("connection refused")


def test_dead_server_is_quarantined_and_discovery_continues():
    transport = DeadTransport()
    layer = MCPLayer({"dead": "http://x", "also-dead": "http://y"},
                     transport=transport)
    recall = layer.discover_all()
    for name in ("dead", "also-dead"):
        entry = recall["servers"][name]
        assert entry["status"] == "quarantined"
        assert "ConnectionError" in entry["reason"]
        assert entry["callable"] == 0
    assert layer.find_tools("record") == []  # nothing is offered
    with pytest.raises(MCPError, match="quarantined"):
        layer.call("dead", "get_record", {"id": "r1"})


def test_protocol_error_server_is_quarantined():
    class GarbageTransport(StubMCPTransport):
        def __call__(self, url, payload, headers):
            if payload.get("method") == "tools/list":
                return {"status": 500, "headers": {},
                        "body": {"error": "internal"}}
            return super().__call__(url, payload, headers)

    layer = MCPLayer({"s": "http://x"}, transport=GarbageTransport())
    entry = layer.discover_all()["servers"]["s"]
    assert entry["status"] == "quarantined"
    assert entry["reason"]


# ---------------------------------------------------------------------
# find_tools
# ---------------------------------------------------------------------

def test_find_tools_ranks_and_returns_schemas():
    layer, _ = discovered_layer()
    found = layer.find_tools("fetch record")
    assert found[0]["name"] == "get_record"
    assert found[0]["input_schema"]["required"] == ["id"]
    found = layer.find_tools("list")
    assert [f["name"] for f in found] == ["list_records"]
    assert layer.find_tools("nonexistentvocabulary") == []


# ---------------------------------------------------------------------
# Per-call timeout
# ---------------------------------------------------------------------

class SlowTransport(StubMCPTransport):
    """tools/call answers, but slowly."""

    def __call__(self, url, payload, headers):
        if payload.get("method") == "tools/call":
            time.sleep(0.3)
        return super().__call__(url, payload, headers)


def test_call_that_overruns_its_timeout_is_reported():
    layer = MCPLayer({"s": "http://x"}, transport=SlowTransport(),
                     config={"mcp_call_timeout_sec": 0.05})
    with pytest.raises(MCPError, match="per-call timeout"):
        layer.call("s", "get_record", {})


def test_call_refused_when_no_task_time_remains():
    layer = MCPLayer({"s": "http://x"}, transport=StubMCPTransport(),
                     config={"deadline": time.time() - 5})
    with pytest.raises(MCPError, match="no task time remaining"):
        layer.call("s", "get_record", {})


# ---------------------------------------------------------------------
# The index on disk and the run outcome
# ---------------------------------------------------------------------

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


def scripted_client(seen):
    def client(messages, tools=None, phase="execute"):
        if phase == "brief":
            return {"content": json.dumps(BRIEF),
                    "usage": {"input_tokens": 20, "output_tokens": 0}}
        if phase == "execute":
            if not seen.get("looked"):
                seen["looked"] = True
                seen["tools_offered"] = [t["function"]["name"]
                                         for t in (tools or [])]
                return {"tool_calls": [{
                    "name": "find_tools",
                    "arguments": {"query": "fetch record"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 0}}
            if not seen.get("called"):
                seen["called"] = True
                return {"tool_calls": [{
                    "name": "mcp_call",
                    "arguments": {"server": "docs", "tool": "get_record",
                                  "args": {"id": "r1"}}}],
                    "usage": {"input_tokens": 10, "output_tokens": 0}}
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}
    return client


def test_run_indexes_catalogue_and_reports_recall(tmp_path):
    transport = StubMCPTransport(tools=[GET_RECORD, BROKEN_TOOL])
    seen: dict = {}
    outcome = run_harness(
        {"instruction": INSTRUCTION, "id": "t-mcp",
         "mcp_servers": [{"name": "docs", "url": "http://mcp.local/"}]},
        lambda frame: None, tmp_path,
        {"model_client": scripted_client(seen),
         "mcp_transport": transport,
         "token_cap": 10_000, "estimated_call_tokens": 10})
    assert outcome["status"] == "finalized"

    recall = outcome["discovery_recall"]
    assert recall["totals"] == {"advertised": 2, "indexed": 2,
                                "callable": 1}
    assert recall["servers"]["docs"]["quarantined_tools"] == ["broken_tool"]

    # The index is on disk in the state dir, complete and re-readable.
    state_dir = Path(outcome["state_dir"])
    index = json.loads((state_dir / "mcp_catalogue.json").read_text())
    assert index["totals"] == recall["totals"]
    tools = index["servers"]["docs"]["tools"]
    assert [t["name"] for t in tools] == ["get_record"]
    assert tools[0]["description"] == "Fetch one record by id"
    assert tools[0]["input_schema"]["required"] == ["id"]
    assert index["servers"]["docs"]["quarantined_tools"][0]["name"] == \
        "broken_tool"
    # Atomic write: no temporary siblings are left behind.
    assert list(state_dir.glob("*.tmp")) == []

    # Recall landed in the ledger as its own event kind, separate
    # from the call outcomes recorded as actions.
    events = [json.loads(line) for line in
              (state_dir / "ledger.jsonl").read_text().splitlines()]
    discovery_events = [e for e in events if e["kind"] == "mcp_discovery"]
    assert len(discovery_events) == 1
    assert discovery_events[0]["advertised"] == 2
    assert discovery_events[0]["callable"] == 1

    # find_tools was offered to the model and its result came back.
    assert "find_tools" in seen["tools_offered"]


def test_run_with_dead_mcp_server_still_completes(tmp_path):
    def client(messages, tools=None, phase="execute"):
        if phase == "brief":
            return {"content": json.dumps(BRIEF),
                    "usage": {"input_tokens": 20, "output_tokens": 0}}
        if phase == "execute":
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    outcome = run_harness(
        {"instruction": INSTRUCTION, "id": "t-mcp-dead",
         "mcp_servers": [{"name": "dead", "url": "http://mcp.local/"}]},
        lambda frame: None, tmp_path,
        {"model_client": client, "mcp_transport": DeadTransport(),
         "token_cap": 10_000, "estimated_call_tokens": 10})
    # Discovery failure is not fatal: the run finalizes, with the
    # server recorded as quarantined in the outcome.
    assert outcome["status"] == "finalized"
    entry = outcome["discovery_recall"]["servers"]["dead"]
    assert entry["status"] == "quarantined"
    assert entry["callable"] == 0


def test_dispatcher_mcp_call_uses_discovery_validation(tmp_path):
    transport = StubMCPTransport(tools=[GET_RECORD])
    dispatcher = ToolDispatcher(
        tmp_path, config={"mcp_servers": [{"name": "s",
                                           "url": "http://x"}],
                          "mcp_transport": transport})
    dispatcher.mcp.discover_all()
    refused = dispatcher.dispatch(
        "mcp_call", {"server": "s", "tool": "get_record",
                     "args": {"count": 1}})
    assert refused["ok"] is False
    assert "missing required argument" in refused["summary"]
    assert call_calls(transport) == []
    sent = dispatcher.dispatch(
        "mcp_call", {"server": "s", "tool": "get_record",
                     "args": {"id": "r1"}})
    assert sent["ok"] is True
    assert len(call_calls(transport)) == 1
