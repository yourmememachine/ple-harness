"""The model-facing tool surface: exactly four tools.

``read_file``, ``write_file``, ``exec`` and ``mcp_call`` are dispatched
to evaluator requests through a transport callable (a function taking
a protocol frame dict and returning its result dict). When no
transport is supplied, a local transport with the same semantics runs
against ``workdir`` directly, which is how the loop is exercised in
tests.

Contracts worth knowing:

- ``write_file`` writes in chunks of at most ~5 MB using the native
  ``append`` flag. Append is not idempotent, so the write carries an
  integrity contract: the expected size is tracked chunk by chunk,
  every chunk result is checked against it, a chunk whose response is
  lost is probed by file size (never blindly re-sent), and the final
  size and hash are read back and compared before the write counts as
  done. A size probe that cannot answer never waives verification:
  an unverifiable write is an integrity failure, not a pass. File
  content is written exactly as supplied; the line-number prefixes
  added by ``read_file`` are stripped only when the caller explicitly
  passes ``strip_line_numbers=True``.
- All tool paths are confined to the workdir: paths are normalised
  against it, absolute paths and ``..`` escapes are rejected, and
  symlinks are resolved so a target outside the workdir is rejected
  on both the local and the evaluator transport.
- ``exec`` runs each command in a fresh ``bash`` subshell in the
  workdir: files persist, ``cd`` and exports do not. The timeout is
  capped at 600 s and, when a task deadline is supplied, at the
  remaining task time. A foreground command runs in its own process
  group, and the whole group is killed when the call returns or times
  out. Background execution is explicit: a background job starts in
  its own new process group (``setsid``), its output goes to a log
  file, finished jobs are reaped, and jobs are stopped by signalling
  their process group (SIGTERM, then SIGKILL), never by guessing at
  individual process ids.
- ``mcp_call`` goes through the MCP layer in this module: sessions are
  initialised lazily, a dropped session is re-initialised, and a
  mutating call whose response is lost is probed for its effect
  before any retry — and is not retried at all when no probe exists.
  Servers are discovered before the loop starts (see
  :meth:`MCPLayer.discover_all`): the advertised catalogue is paged
  in full, indexed on disk, and every call is validated locally
  against the advertised schema before it is sent — an unknown tool,
  a missing required argument, or a wrongly typed argument is
  refused locally and never reaches the server. A server that fails
  discovery is quarantined with its reason and its tools are not
  offered; discovery failure is never fatal to the run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

MAX_WRITE_BYTES = 10_000_000
DEFAULT_CHUNK_BYTES = 5_000_000
MAX_EXEC_TIMEOUT_SEC = 600
DEFAULT_EXEC_TIMEOUT_SEC = 120
RESULT_SUMMARY_CHARS = 2_000
MAX_EXEC_OUTPUT_CHARS = 100_000
MAX_READ_BYTES = 10_000_000

TOOL_NAMES = ("read_file", "write_file", "exec", "mcp_call")

_MUTATING_HINTS = ("create", "update", "delete", "write", "send", "set",
                   "add", "remove", "move", "assign", "post", "put",
                   "patch", "insert", "submit", "approve", "publish",
                   "archive", "rename", "transfer", "apply", "pay",
                   "close", "execute", "duplicate", "destroy", "drop",
                   "edit", "modify", "upload", "import", "replace",
                   "merge", "cancel", "refund", "charge", "book")


class ToolError(Exception):
    """A tool call failed in a way the model should hear about."""


class IntegrityError(ToolError):
    """A write did not land as intended; the file state is uncertain."""


class MCPError(ToolError):
    """The MCP layer could not complete a call."""


class MCPAmbiguousResult(MCPError):
    """A mutating call's outcome is unknown and no probe exists."""


class MCPTransportLost(MCPError):
    """The transport reported the request or its response as lost."""


class MCPSessionExpired(MCPError):
    """The server says the MCP session is gone and must be reopened."""


# ---------------------------------------------------------------------
# MCP catalogue schemas: compilation and argument validation
# ---------------------------------------------------------------------

MAX_LIST_PAGES = 100
DEFAULT_MCP_CALL_TIMEOUT_SEC = 300

_SCHEMA_TYPES = ("string", "integer", "number", "boolean", "object",
                 "array", "null")


def _schema_problems(schema: Any, where: str = "schema") -> list[str]:
    """Structural problems that keep a JSON schema from compiling.

    Only the subset the local validator enforces is checked: known
    ``type`` values, ``properties`` as a mapping of schemas,
    ``required`` as a list of names, and ``items`` as a schema.
    Anything outside that subset is a compile failure, so a tool
    whose schema uses it is quarantined rather than offered with
    validation silently weaker than advertised.
    """
    if not isinstance(schema, Mapping):
        return [f"{where}: not an object"]
    problems: list[str] = []
    declared = schema.get("type")
    if declared is not None:
        types = declared if isinstance(declared, list) else [declared]
        for one in types:
            if one not in _SCHEMA_TYPES:
                problems.append(f"{where}: unknown type {one!r}")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            problems.append(f"{where}.properties: not an object")
        else:
            for name, sub in properties.items():
                problems.extend(
                    _schema_problems(sub, f"{where}.properties.{name}"))
    required = schema.get("required")
    if required is not None:
        if not isinstance(required, (list, tuple)) or \
                not all(isinstance(r, str) for r in required):
            problems.append(f"{where}.required: not a list of names")
    items = schema.get("items")
    if items is not None:
        problems.extend(_schema_problems(items, f"{where}.items"))
    return problems


def compile_tool_schema(entry: Mapping[str, Any]) -> tuple[bool, str]:
    """Whether one advertised tool entry can be offered for calls.

    Returns ``(True, "")`` for a compilable entry, else
    ``(False, reason)``. A tool that fails here is quarantined —
    excluded from the callable set — without affecting the rest of
    its server's tools.
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return False, "advertised tool has no usable name"
    schema = entry.get("inputSchema", entry.get("input_schema"))
    if schema is None:
        schema = {}
    if not isinstance(schema, Mapping):
        return False, f"tool {name}: input schema is not an object"
    declared = schema.get("type")
    if declared is not None and declared != "object":
        return False, f"tool {name}: input schema type is {declared!r}, " \
                      "not \"object\""
    problems = _schema_problems(schema)
    if problems:
        return False, f"tool {name}: " + "; ".join(problems)
    return True, ""


def _type_matches(declared: Any, value: Any) -> bool:
    types = declared if isinstance(declared, list) else [declared]
    for one in types:
        if one == "string" and isinstance(value, str):
            return True
        if one == "integer" and isinstance(value, int) \
                and not isinstance(value, bool):
            return True
        if one == "number" and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            return True
        if one == "boolean" and isinstance(value, bool):
            return True
        if one == "object" and isinstance(value, Mapping):
            return True
        if one == "array" and isinstance(value, (list, tuple)):
            return True
        if one == "null" and value is None:
            return True
    return False


def validate_arguments(schema: Mapping[str, Any],
                       args: Mapping[str, Any],
                       where: str = "args") -> Optional[str]:
    """Validate ``args`` against an advertised input schema.

    Returns a refusal string, or None when the arguments conform.
    Required arguments must be present and every declared property
    that is present must match its declared type (recursing into
    nested objects and array items) and its ``enum`` when one is
    advertised. Undeclared extra arguments are passed through, as
    servers routinely accept them.
    """
    for name in schema.get("required") or []:
        if name not in args:
            return f"{where}: missing required argument {name!r}"
    properties = schema.get("properties") or {}
    for name, value in args.items():
        sub = properties.get(name)
        if not isinstance(sub, Mapping):
            continue
        declared = sub.get("type")
        if declared is not None and not _type_matches(declared, value):
            return (f"{where}.{name}: expected type {declared!r}, "
                    f"got {type(value).__name__}")
        enum = sub.get("enum")
        if isinstance(enum, (list, tuple)) and value not in enum:
            return f"{where}.{name}: {value!r} is not one of the " \
                   "advertised values"
        if isinstance(value, Mapping) and (
                sub.get("properties") or sub.get("required")):
            problem = validate_arguments(sub, value, f"{where}.{name}")
            if problem:
                return problem
        if isinstance(value, (list, tuple)) and \
                isinstance(sub.get("items"), Mapping):
            for index, item in enumerate(value):
                item_type = sub["items"].get("type")
                if item_type is not None and \
                        not _type_matches(item_type, item):
                    return (f"{where}.{name}[{index}]: expected type "
                            f"{item_type!r}, got {type(item).__name__}")
    return None


def find_tools_schema() -> dict[str, Any]:
    """OpenAI-style function schema for the catalogue search tool."""
    return {
        "type": "function",
        "function": {
            "name": "find_tools",
            "description": "Search the discovered MCP tool catalogue "
                           "and return matching tools with their "
                           "schemas. Only tools returned here can be "
                           "called with mcp_call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "server": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    }


def _result(ok: bool, summary: str, data: Any = None,
            error_class: Optional[str] = None,
            truncated: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": ok, "summary": summary[:RESULT_SUMMARY_CHARS]}
    if data is not None:
        out["data"] = data
    if truncated:
        out["truncated"] = True
    if error_class is not None:
        out["error_class"] = error_class
    return out


_LINE_NUMBER_PREFIX = re.compile(r"^\d+\t")


def strip_line_number_prefixes(text: str) -> str:
    """Remove this module's own read_file line-number prefixes.

    Only applies when nearly every non-empty line carries the
    ``<number><TAB>`` prefix, so ordinary numbered content is left
    alone.
    """
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return text
    marked = sum(1 for ln in nonempty if _LINE_NUMBER_PREFIX.match(ln))
    if marked < max(1, int(0.8 * len(nonempty))):
        return text
    return "\n".join(_LINE_NUMBER_PREFIX.sub("", ln) for ln in lines)


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas for the four tools."""
    def fn(name: str, description: str, properties: dict[str, Any],
           required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    return [
        fn("read_file",
           "Read a text file from the workdir. The result carries line "
           "numbers. Binary files return metadata only.",
           {"path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"}},
           ["path"]),
        fn("write_file",
           "Write a file in the workdir. Default is a full-file "
           "replacement; append=true adds a chunk at the end. Large "
           "deliverables should be generated by an exec script, not "
           "authored chunk by chunk.",
           {"path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean"},
            "strip_line_numbers": {"type": "boolean"}},
           ["path", "content"]),
        fn("exec",
           "Run a bash command in the workdir. Each call is a fresh "
           "subshell: files persist, cd and exports do not. Default "
           "timeout 120 s, maximum 600 s. With background=true the "
           "command starts detached in its own new process group, "
           "writes its output to a log file, and the result carries "
           "the process-group id; stop it by signalling that group.",
           {"command": {"type": "string"},
            "timeout_s": {"type": "integer"},
            "background": {"type": "boolean"}},
           ["command"]),
        fn("mcp_call",
           "Call a tool on one of the task's MCP servers, through the "
           "harness's MCP session layer.",
           {"server": {"type": "string"},
            "tool": {"type": "string"},
            "args": {"type": "object"},
            "mutating": {"type": "boolean"}},
           ["server", "tool"]),
    ]


# ---------------------------------------------------------------------
# MCP layer
# ---------------------------------------------------------------------

class StubMCPTransport:
    """Deterministic stand-in MCP transport for local runs and tests.

    Answers ``initialize`` and ``tools/list`` and echoes ``tools/call``
    arguments back as the result. Records every payload it receives.
    """

    def __init__(self, tools: Optional[list[dict[str, Any]]] = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._tools = tools or []

    def __call__(self, url: str, payload: Mapping[str, Any],
                 headers: Mapping[str, str]) -> dict[str, Any]:
        self.calls.append(dict(payload))
        method = payload.get("method")
        if method == "initialize":
            return {"status": 200,
                    "headers": {"Mcp-Session-Id": "stub-session"},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"protocolVersion": "2025-06-18",
                                        "capabilities": {},
                                        "serverInfo": {"name": "stub",
                                                       "version": "0"}}}}
        if method == "notifications/initialized":
            return {"status": 202, "headers": {}, "body": {}}
        if method == "tools/list":
            return {"status": 200, "headers": {},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"tools": list(self._tools)}}}
        if method == "tools/call":
            params = payload.get("params") or {}
            return {"status": 200, "headers": {},
                    "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                             "result": {"content": [{
                                 "type": "text",
                                 "text": json.dumps({
                                     "tool": params.get("name"),
                                     "arguments": params.get("arguments", {}),
                                 })}],
                                 "stub": True}}}
        return {"status": 200, "headers": {},
                "body": {"jsonrpc": "2.0", "id": payload.get("id"),
                         "result": {}}}


class MCPLayer:
    """Streamable-HTTP MCP sessions over a transport callable.

    The transport takes ``(url, payload, headers)`` and returns a dict
    with ``status``, ``headers`` and ``body`` (a parsed JSON value or
    text). When the layer is built on an evaluator ``http`` request
    callable instead, frames are constructed here and the result's
    body is decoded.
    """

    VERSION = "2025-06-18"

    def __init__(self, servers: Any = None,
                 transport: Optional[Callable[..., Any]] = None,
                 http_request: Optional[Callable[[dict[str, Any]], Any]] = None,
                 config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.servers: dict[str, str] = {}
        if isinstance(servers, Mapping):
            for name, url in servers.items():
                self.servers[str(name)] = str(url)
        elif isinstance(servers, (list, tuple)):
            for entry in servers:
                if isinstance(entry, Mapping) and entry.get("name"):
                    self.servers[str(entry["name"])] = str(entry.get("url") or "")
        self.transport = transport or cfg.get("mcp_transport")
        self.http_request = http_request
        self.effect_probes: dict[str, Callable[..., Any]] = dict(
            cfg.get("effect_probes") or {})
        if callable(cfg.get("effect_probe")):
            self.effect_probes["*"] = cfg["effect_probe"]
        self.mutating_tools = set(cfg.get("mutating_tools") or ())
        self.sessions: dict[str, dict[str, Any]] = {}
        self._ids = 0
        self._cfg = cfg
        self.max_list_pages = max(1, int(cfg.get("mcp_max_list_pages",
                                                 MAX_LIST_PAGES)))
        try:
            self.call_timeout_sec = float(cfg.get(
                "mcp_call_timeout_sec", DEFAULT_MCP_CALL_TIMEOUT_SEC))
        except (TypeError, ValueError):
            self.call_timeout_sec = float(DEFAULT_MCP_CALL_TIMEOUT_SEC)
        # Discovery state: server -> {"status", "tools", "quarantined"}
        # for servers discovery has visited. Empty until discover_all
        # (or discover) runs; call validation applies only to servers
        # present here, so undiscovered use keeps its prior behaviour.
        self.catalogue: dict[str, dict[str, Any]] = {}
        self.discovery_recall: dict[str, Any] = {
            "servers": {},
            "totals": {"advertised": 0, "indexed": 0, "callable": 0}},

    # -- call timeout ----------------------------------------------------

    def _remaining_call_timeout(
            self, explicit: Optional[float] = None) -> float:
        """Seconds one MCP call may still take.

        The bound is the configured per-call cap, tightened by the
        remaining task time under the same deadline machinery the
        dispatcher uses for ``exec`` (config ``deadline`` epoch or a
        ``time_left`` callable, minus a 2 s safety margin). No call
        may hang past the remaining task time.
        """
        budget = self.call_timeout_sec
        if explicit is not None:
            try:
                budget = min(budget, float(explicit))
            except (TypeError, ValueError):
                pass
        remaining: Optional[float] = None
        deadline = self._cfg.get("deadline")
        if isinstance(deadline, (int, float)):
            remaining = float(deadline) - time.time()
        else:
            time_left = self._cfg.get("time_left")
            if callable(time_left):
                try:
                    value = time_left()
                except Exception:
                    value = None
                if isinstance(value, (int, float)):
                    remaining = float(value)
        if remaining is not None:
            budget = min(budget, remaining - 2)
        return budget

    # -- plumbing ------------------------------------------------------

    def _next_id(self) -> int:
        self._ids += 1
        return self._ids

    def _send(self, server: str, payload: Mapping[str, Any],
              notify: bool = False) -> tuple[int, dict[str, str], Any]:
        url = self.servers.get(server)
        if not url:
            raise MCPError(f"unknown MCP server: {server}")
        session = self.sessions.get(server) or {}
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if session.get("session_id"):
            headers["Mcp-Session-Id"] = str(session["session_id"])
        if session.get("protocol_version"):
            headers["MCP-Protocol-Version"] = str(session["protocol_version"])
        if self.transport is not None:
            reply = self.transport(url, dict(payload), headers)
            if not isinstance(reply, Mapping):
                raise MCPError("MCP transport returned no reply")
            status = int(reply.get("status", 200))
            out_headers = {str(k): str(v)
                           for k, v in (reply.get("headers") or {}).items()}
            body = reply.get("body", reply.get("result", {}))
            return status, out_headers, body
        if self.http_request is None:
            raise MCPError("no MCP transport is configured")
        frame = {
            "type": "http", "method": "POST", "url": url, "headers": headers,
            "body_b64": base64.b64encode(
                json.dumps(payload).encode("utf-8")).decode(),
            "timeout_sec": max(1, int(self._remaining_call_timeout())),
        }
        result = self.http_request(frame)
        if not isinstance(result, Mapping) or not result.get("ok", True):
            error = result.get("error") if isinstance(result, Mapping) else None
            # On the evaluator path a not-ok result is how a lost
            # request/response is reported; classify it structurally
            # so callers can apply the recovery rules to it.
            raise MCPTransportLost(
                f"MCP request failed: {error or 'no result'}")
        raw = base64.b64decode(result.get("body_b64") or "")
        out_headers = {str(k): str(v)
                       for k, v in (result.get("headers") or {}).items()}
        return int(result.get("status", 200)), out_headers, raw

    @staticmethod
    def _messages(body: Any) -> list[dict[str, Any]]:
        if isinstance(body, Mapping):
            return [dict(body)]
        if isinstance(body, (bytes, bytearray)):
            body = bytes(body).decode("utf-8", "replace")
        if not isinstance(body, str):
            return []
        text = body.strip()
        chunks: list[str] = []
        if text.startswith(("{", "[")):
            chunks = [text]
        else:
            for block in re.split(r"\r?\n\r?\n", text):
                data = "\n".join(
                    ln[5:].lstrip() for ln in block.splitlines()
                    if ln.startswith("data:"))
                if data:
                    chunks.append(data)
        found: list[dict[str, Any]] = []
        for chunk in chunks:
            try:
                data = json.loads(chunk)
            except ValueError:
                continue
            for item in (data if isinstance(data, list) else [data]):
                if isinstance(item, Mapping):
                    found.append(dict(item))
        return found

    def _rpc(self, server: str, method: str,
             params: Optional[Mapping[str, Any]] = None,
             notify: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = dict(params)
        rid = None
        if not notify:
            rid = self._next_id()
            payload["id"] = rid
        status, headers, body = self._send(server, payload, notify)
        session = self.sessions.setdefault(server, {})
        for key, value in headers.items():
            if key.lower() == "mcp-session-id":
                session["session_id"] = value
        if notify:
            return {}
        if status >= 400:
            detail = f"{method}: HTTP {status}: {str(body)[:200]}"
            if status in (404, 410) or self._session_expired_text(body):
                raise MCPSessionExpired(detail)
            raise MCPError(detail)
        for message in self._messages(body):
            if message.get("id") == rid:
                if "error" in message:
                    detail = f"{method}: {message['error']}"
                    if self._session_expired_text(message["error"]):
                        raise MCPSessionExpired(detail)
                    raise MCPError(detail)
                result = message.get("result")
                return result if isinstance(result, dict) else {}
        raise MCPError(f"{method}: no answer for id {rid}")

    # -- session lifecycle ----------------------------------------------

    def initialize(self, server: str) -> dict[str, Any]:
        """Open (or reopen) the session for ``server``."""
        self.sessions[server] = {}
        result = self._rpc(server, "initialize", {
            "protocolVersion": self.VERSION,
            "capabilities": {},
            "clientInfo": {"name": "harness", "version": "1.0"},
        })
        version = result.get("protocolVersion")
        session = self.sessions.setdefault(server, {})
        session["protocol_version"] = (
            version if isinstance(version, str) and version else self.VERSION)
        session["initialized"] = True
        self._rpc(server, "notifications/initialized", notify=True)
        return result

    def reset(self, server: str) -> None:
        self.sessions.pop(server, None)

    def ensure_session(self, server: str) -> None:
        session = self.sessions.get(server) or {}
        if not session.get("initialized"):
            self.initialize(server)

    def list_tools(self, server: str) -> list[dict[str, Any]]:
        """The server's full advertised catalogue, paged completely.

        Pagination is followed to the end; it is never silently
        truncated. If the server still offers a next page after
        ``max_list_pages`` pages, that is recorded as a defect — an
        :class:`MCPError` the discovery path quarantines the server
        with — rather than a partial catalogue presented as whole.
        """
        self.ensure_session(server)
        tools: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(self.max_list_pages):
            page = self._rpc(server, "tools/list",
                             {"cursor": cursor} if cursor else {})
            found = page.get("tools") if isinstance(page.get("tools"), list) else []
            tools.extend(t for t in found if isinstance(t, Mapping))
            cursor = page.get("nextCursor")
            if not cursor:
                return tools
        raise MCPError(
            f"tools/list for {server}: pagination exceeded "
            f"{self.max_list_pages} pages; the catalogue is incomplete "
            "and is recorded as a defect, not truncated silently")

    # -- discovery ---------------------------------------------------------

    def discover(self, server: str) -> dict[str, Any]:
        """Fetch and index one server's catalogue.

        Returns the server's recall entry. A server that cannot be
        reached or answers with a protocol error is quarantined: the
        entry records the reason, no tools are offered from it, and
        discovery of the remaining servers continues — the failure is
        not fatal to the run.
        """
        entry: dict[str, Any] = {
            "server": server, "status": "ok", "reason": None,
            "advertised": 0, "indexed": 0, "callable": 0,
            "quarantined_tools": [], "rejected": []}
        try:
            advertised = self.list_tools(server)
        except Exception as exc:
            if type(exc).__name__ == "SessionCancelled":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            entry["status"] = "quarantined"
            entry["reason"] = reason
            self.catalogue[server] = {
                "status": "quarantined", "reason": reason,
                "tools": {}, "quarantined": []}
            return entry
        entry["advertised"] = len(advertised)
        tools: dict[str, dict[str, Any]] = {}
        quarantined: list[dict[str, Any]] = []
        for raw in advertised:
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                entry["rejected"].append("<unnamed>")
                continue
            schema = raw.get("inputSchema", raw.get("input_schema"))
            record = {
                "name": name,
                "description": str(raw.get("description") or ""),
                "input_schema": dict(schema) if isinstance(schema, Mapping)
                else schema,
            }
            ok, why = compile_tool_schema(raw)
            if ok:
                tools[name] = record
            else:
                quarantined.append({"name": name, "reason": why})
                entry["quarantined_tools"].append(name)
        entry["indexed"] = len(tools) + len(quarantined)
        entry["callable"] = len(tools)
        self.catalogue[server] = {
            "status": "ok", "reason": None,
            "tools": tools, "quarantined": quarantined}
        return entry

    def discover_all(self, index_path: Any = None) -> dict[str, Any]:
        """Discover every configured server and index the catalogue.

        The index is written atomically to ``index_path`` when one is
        given (a temporary sibling file, fsynced and renamed over the
        target), so a reader never sees a partial catalogue. Returns
        the discovery-recall block: per-server counts of advertised
        vs indexed vs callable tools, plus totals. These counts
        measure discovery and transport only; whether the discovered
        tools then succeed at the task is recorded separately, in the
        ledger's action entries.
        """
        servers: dict[str, Any] = {}
        totals = {"advertised": 0, "indexed": 0, "callable": 0}
        for server in list(self.servers):
            entry = self.discover(server)
            servers[server] = entry
            for key in totals:
                totals[key] += int(entry.get(key, 0))
        recall = {"servers": servers, "totals": totals}
        self.discovery_recall = recall
        if index_path is not None:
            index: dict[str, Any] = {"servers": {}, "totals": totals}
            for server, entry in servers.items():
                cat = self.catalogue.get(server) or {}
                index["servers"][server] = {
                    "status": entry["status"],
                    "reason": entry["reason"],
                    "tools": [cat["tools"][name]
                              for name in sorted(cat.get("tools") or {})],
                    "quarantined_tools": cat.get("quarantined") or [],
                    "rejected": entry["rejected"],
                }
            path = Path(index_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
            with open(tmp, "wb") as handle:
                handle.write(json.dumps(index, indent=1, sort_keys=True,
                                        ensure_ascii=False).encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        return recall

    def find_tools(self, query: str, server: Optional[str] = None,
                   limit: int = 10) -> list[dict[str, Any]]:
        """Ranked candidate tools from the discovered catalogue.

        Matches query tokens against tool names (weighted double) and
        descriptions; only callable tools are returned — quarantined
        servers and quarantined tools are never offered. An empty
        query returns the callable catalogue in name order, bounded
        by ``limit``.
        """
        tokens = [t for t in re.split(r"[^a-z0-9]+", str(query).lower()) if t]
        scored: list[tuple[int, str, str, dict[str, Any]]] = []
        for name, cat in self.catalogue.items():
            if server is not None and name != server:
                continue
            if cat.get("status") != "ok":
                continue
            for tool_name, record in (cat.get("tools") or {}).items():
                if not tokens:
                    score = 1
                else:
                    hay_name = tool_name.lower()
                    hay_desc = str(record.get("description") or "").lower()
                    score = sum(2 for t in tokens if t in hay_name) \
                        + sum(1 for t in tokens if t in hay_desc)
                if score > 0:
                    scored.append((score, name, tool_name, {
                        "server": name, "name": tool_name,
                        "description": record.get("description", ""),
                        "input_schema": record.get("input_schema") or {},
                    }))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in scored[:max(0, int(limit))]]

    def catalogue_context_lines(self) -> list[str]:
        """Names-and-one-line-descriptions index for loop context."""
        lines: list[str] = []
        for server in sorted(self.catalogue):
            cat = self.catalogue[server]
            if cat.get("status") != "ok":
                lines.append(
                    f"{server}: unavailable (quarantined: "
                    f"{cat.get('reason')})")
                continue
            for name in sorted(cat.get("tools") or {}):
                record = cat["tools"][name]
                desc = str(record.get("description") or "").splitlines()
                lines.append(f"{server}/{name}: "
                             f"{desc[0] if desc else ''}".rstrip())
            for item in cat.get("quarantined") or []:
                lines.append(f"{server}/{item['name']}: unavailable "
                             f"(quarantined: {item['reason']})")
        return lines

    def validate_call(self, server: str, tool: str,
                      args: Mapping[str, Any]) -> Optional[str]:
        """Local refusal reason for a proposed call, or None to send.

        Applies only once the server has been discovered: an
        undiscovered server keeps the layer's prior behaviour. A
        discovered server refuses, before anything is sent, a call
        to a quarantined server, an unknown or quarantined tool, or
        arguments that miss a required field or mistype a declared
        one.
        """
        cat = self.catalogue.get(server)
        if cat is None:
            return None
        if cat.get("status") != "ok":
            return (f"server {server} is quarantined "
                    f"({cat.get('reason')}); its tools are not offered")
        tools = cat.get("tools") or {}
        if tool not in tools:
            for item in cat.get("quarantined") or []:
                if item.get("name") == tool:
                    return (f"tool {tool} on {server} is quarantined "
                            f"({item['reason']}); it is not offered")
            return (f"unknown tool {tool!r} on server {server}: not in "
                    "the discovered catalogue")
        schema = tools[tool].get("input_schema")
        if isinstance(schema, Mapping):
            return validate_arguments(schema, args,
                                      where=f"{server}/{tool} args")
        return None

    # -- calls -----------------------------------------------------------

    def is_mutating(self, tool: str,
                    explicit: Optional[bool] = None) -> bool:
        """Classify a tool as mutating. The classification is a floor:
        the configured set and the name heuristic always apply, and an
        explicit caller-supplied flag may only raise the caution
        (``True``), never lower it."""
        if explicit:
            return True
        if tool in self.mutating_tools:
            return True
        lowered = tool.lower()
        return any(hint in lowered for hint in _MUTATING_HINTS)

    @staticmethod
    def _session_expired_text(value: Any) -> bool:
        text = str(value).lower()
        return "session" in text and any(
            word in text for word in ("expired", "invalid", "not found",
                                      "missing", "unknown", "404", "gone",
                                      "no longer"))

    @classmethod
    def _session_expired(cls, exc: Exception) -> bool:
        return isinstance(exc, MCPSessionExpired) or \
            cls._session_expired_text(exc)

    def _probe(self, server: str, tool: str,
               args: Mapping[str, Any]) -> Optional[bool]:
        probe = self.effect_probes.get(tool) or self.effect_probes.get("*")
        if probe is None:
            return None
        try:
            return bool(probe(server, tool, dict(args)))
        except Exception:
            return None

    def call(self, server: str, tool: str,
             args: Optional[Mapping[str, Any]] = None,
             mutating: Optional[bool] = None,
             timeout_sec: Optional[float] = None) -> dict[str, Any]:
        """Call one MCP tool, validated locally and time-bounded.

        Once the server has been discovered, the call is validated
        against the advertised schema first: an unknown tool, a
        missing required argument, or a wrongly typed argument is
        refused locally and never sent. The call is bounded by the
        per-call timeout (the configured cap tightened by the
        remaining task time): with no time remaining the call is
        refused before sending, and a call that returns after its
        bound is reported as a timeout rather than trusted.
        """
        arguments = dict(args or {})
        refusal = self.validate_call(server, tool, arguments)
        if refusal is not None:
            raise MCPError(f"mcp_call refused locally: {refusal}")
        budget = self._remaining_call_timeout(timeout_sec)
        if budget <= 0:
            raise MCPError(
                f"tools/call {tool}: no task time remaining; not sent")
        started = time.monotonic()
        result = self._call_recovering(server, tool, arguments,
                                       mutating=mutating)
        if time.monotonic() - started > budget:
            raise MCPError(
                f"tools/call {tool}: exceeded the per-call timeout "
                f"of {budget:.1f}s")
        return result

    def _call_recovering(self, server: str, tool: str,
                         args: Optional[Mapping[str, Any]] = None,
                         mutating: Optional[bool] = None) -> dict[str, Any]:
        """Call one MCP tool, with session recovery and safe retries.

        A lost response to a mutating call is probed before any
        retry; without a probe the call is reported ambiguous rather
        than repeated into graded state.
        """
        arguments = dict(args or {})
        is_mut = self.is_mutating(tool, mutating)
        self.ensure_session(server)
        try:
            return self._rpc(server, "tools/call",
                             {"name": tool, "arguments": arguments})
        except Exception as exc:
            # Recovery is driven by the failure's shape, not only its
            # type: the evaluator path reports lost responses and dead
            # sessions as MCPTransportLost / MCPSessionExpired raised
            # at the _send/_rpc boundary, alongside the raw connection
            # errors a direct transport produces.
            lost = isinstance(exc, (ConnectionError, TimeoutError, OSError,
                                    MCPTransportLost))
            expired = self._session_expired(exc)
            if not (lost or expired):
                raise MCPError(f"tools/call {tool}: {exc}") from exc
            if is_mut:
                probed = self._probe(server, tool, arguments)
                if probed is True:
                    return {"probed": True, "effect_confirmed": True,
                            "tool": tool}
                if probed is None:
                    raise MCPAmbiguousResult(
                        f"tools/call {tool}: response lost and no effect "
                        "probe is available; not retried") from exc
            self.reset(server)
            self.initialize(server)
            try:
                return self._rpc(server, "tools/call",
                                 {"name": tool, "arguments": arguments})
            except Exception as retry_exc:
                retry_lost = isinstance(
                    retry_exc, (ConnectionError, TimeoutError, OSError,
                                MCPTransportLost)) or \
                    self._session_expired(retry_exc)
                if retry_lost and is_mut:
                    raise MCPAmbiguousResult(
                        f"tools/call {tool}: retry response lost; outcome "
                        "unknown and not retried again") from retry_exc
                if isinstance(retry_exc, MCPError):
                    raise
                raise MCPError(
                    f"tools/call {tool}: {retry_exc}") from retry_exc


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

class ToolDispatcher:
    """Validates and runs the four model-facing tools."""

    def __init__(self, workdir: Any,
                 request: Optional[Callable[[dict[str, Any]], Any]] = None,
                 config: Optional[Mapping[str, Any]] = None,
                 mcp_layer: Optional[MCPLayer] = None) -> None:
        cfg = dict(config or {})
        self.workdir = Path(workdir)
        self.request = request or cfg.get("request") or cfg.get("transport")
        self.config = cfg
        self.chunk_bytes = int(cfg.get("write_chunk_bytes", DEFAULT_CHUNK_BYTES))
        self.chunk_bytes = max(1, min(self.chunk_bytes, MAX_WRITE_BYTES))
        self.readback_limit = int(cfg.get("integrity_readback_limit",
                                          MAX_WRITE_BYTES))
        self.max_read_bytes = int(cfg.get("max_read_bytes", MAX_READ_BYTES))
        self.max_exec_output_chars = int(cfg.get(
            "max_exec_output_chars", MAX_EXEC_OUTPUT_CHARS))
        self.bg_dir = Path(cfg.get("background_dir",
                                   self.workdir / ".ple-background"))
        self._background: dict[int, subprocess.Popen] = {}
        if mcp_layer is not None:
            self.mcp = mcp_layer
        else:
            servers = cfg.get("mcp_servers") or []
            transport = cfg.get("mcp_transport")
            if transport is None and self.request is None and cfg.get("mcp_stub"):
                transport = StubMCPTransport()
            self.mcp = MCPLayer(
                servers, transport=transport,
                http_request=self._request if self.request is not None else None,
                config=cfg)

    # -- transport -------------------------------------------------------

    def _confined_rel(self, path: str) -> str:
        """Normalise ``path`` against the workdir and return the
        workdir-relative path, rejecting anything that escapes.

        Absolute paths are rejected outright; the remainder is fully
        resolved (``..`` segments and symlinks included) and must land
        strictly inside the workdir."""
        raw = Path(str(path))
        if raw.is_absolute():
            raise ToolError(f"path escapes the workdir: {path}")
        if not str(path).strip() or str(path).strip() == ".":
            raise ToolError("path must name a file inside the workdir")
        root = Path(self.workdir).resolve()
        resolved = (root / raw).resolve()
        if resolved == root or root not in resolved.parents:
            raise ToolError(f"path escapes the workdir: {path}")
        return resolved.relative_to(root).as_posix()

    def _resolve(self, path: str) -> Path:
        return Path(self.workdir).resolve() / self._confined_rel(path)

    def _truncate_output(self, text: str) -> tuple[str, bool]:
        """Cap combined exec output, keeping head and tail."""
        cap = self.max_exec_output_chars
        if cap <= 0 or len(text) <= cap:
            return text, False
        marker = "\n...[output truncated]...\n"
        keep = max(0, cap - len(marker))
        head = keep // 2
        tail = keep - head
        return text[:head] + marker + (text[-tail:] if tail else ""), True

    def _request(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self.request is None:
            return self._local_request(frame)
        result = self.request(frame)
        if not isinstance(result, Mapping):
            raise ToolError(f"{frame.get('type')}: transport returned no result")
        return dict(result)

    def _local_request(self, frame: dict[str, Any]) -> dict[str, Any]:
        kind = frame.get("type")
        if kind == "read_file":
            path = self._resolve(str(frame.get("path", "")))
            if not path.is_file():
                return {"ok": False, "error": f"not a regular file: {path}"}
            size = path.stat().st_size
            if size > self.max_read_bytes:
                return {"ok": True, "size": size, "content_b64": "",
                        "truncated": True}
            data = path.read_bytes()
            return {"ok": True, "size": len(data),
                    "content_b64": base64.b64encode(data).decode(),
                    "truncated": False}
        if kind == "write_file":
            path = self._resolve(str(frame.get("path", "")))
            data = base64.b64decode(frame.get("content_b64") or "")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab" if frame.get("append") else "wb") as handle:
                handle.write(data)
            return {"ok": True, "size": path.stat().st_size}
        if kind == "exec":
            return self._local_exec(str(frame.get("command", "")),
                                    int(frame.get("timeout_sec",
                                                   DEFAULT_EXEC_TIMEOUT_SEC)),
                                    bool(frame.get("_background")))
        return {"ok": False, "error": f"unsupported local request: {kind}"}

    def _local_exec(self, command: str, timeout_sec: int,
                    background: bool) -> dict[str, Any]:
        if background:
            self.bg_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{int(time.time() * 1000)}-{os.getpid()}"
            log_path = self.bg_dir / f"{stamp}.log"
            handle = open(log_path, "ab")
            proc = subprocess.Popen(
                ["bash", "-c", command], cwd=str(self.workdir),
                stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            self._background[proc.pid] = proc
            return {"ok": True, "background": True, "pid": proc.pid,
                    "process_group": os.getpgid(proc.pid),
                    "log_path": str(log_path), "exit_code": None,
                    "output": "", "timed_out": False}
        proc = subprocess.Popen(
            ["bash", "-c", command], cwd=str(self.workdir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True)
        pgid = os.getpgid(proc.pid)
        try:
            out, _ = proc.communicate(timeout=timeout_sec)
            timed_out = False
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            out, _ = b"", None
        finally:
            # The command ran in its own process group; kill the whole
            # group on timeout and on return alike, so children the
            # command started (and did not itself detach with setsid)
            # cannot outlive the call.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if timed_out:
                out, _ = proc.communicate()
        text, truncated = self._truncate_output(
            out.decode("utf-8", "replace"))
        return {"ok": True, "exit_code": exit_code, "output": text,
                "timed_out": timed_out, "truncated": truncated}

    # -- read --------------------------------------------------------------

    def read_bytes(self, path: str) -> tuple[bytes, dict[str, Any]]:
        rel = self._confined_rel(path)
        result = self._request({"type": "read_file", "path": rel})
        if not result.get("ok"):
            raise ToolError(f"read_file {path}: {result.get('error') or 'failed'}")
        if result.get("truncated"):
            raise ToolError(f"read_file {path}: file exceeds one read")
        size = result.get("size")
        if isinstance(size, int) and size > self.max_read_bytes:
            raise ToolError(
                f"read_file {path}: file exceeds the "
                f"{self.max_read_bytes}-byte read bound")
        data = base64.b64decode(result.get("content_b64") or "")
        if len(data) > self.max_read_bytes:
            raise ToolError(
                f"read_file {path}: file exceeds the "
                f"{self.max_read_bytes}-byte read bound")
        return data, {"size": int(result.get("size", len(data)))}

    def probe_size(self, path: str) -> Optional[int]:
        """Best-effort current size of ``path``; None when unknown."""
        try:
            rel = self._confined_rel(path)
        except ToolError:
            return None
        return self._probe_size_rel(rel)

    def _probe_size_rel(self, rel: str) -> Optional[int]:
        try:
            result = self._request({"type": "read_file", "path": rel})
        except Exception:
            return None
        if isinstance(result.get("size"), int):
            return int(result["size"])
        if not result.get("ok"):
            return 0 if self.request is None else None
        return None

    def read_file(self, path: str, offset: int = 0,
                  limit: Optional[int] = None) -> dict[str, Any]:
        data, meta = self.read_bytes(path)
        digest = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode("utf-8")
            if "\x00" in text:
                raise UnicodeDecodeError("utf-8", data, 0, 1, "NUL byte")
        except UnicodeDecodeError:
            return _result(True, f"binary file {path}: {len(data)} bytes",
                           {"path": path, "binary": True, "size": len(data),
                            "sha256": digest})
        lines = text.split("\n")
        start = max(0, int(offset))
        end = len(lines) if limit is None else start + max(0, int(limit))
        shown = lines[start:end]
        numbered = "\n".join(f"{start + i + 1}\t{ln}"
                             for i, ln in enumerate(shown))
        return _result(
            True, f"read {len(shown)} of {len(lines)} lines from {path}",
            {"path": path, "content": numbered, "size": len(data),
             "sha256": digest, "line_count": len(lines), "offset": start},
            truncated=end < len(lines))

    # -- write ---------------------------------------------------------------

    def write_bytes(self, path: str, data: bytes,
                    append: bool = False) -> dict[str, Any]:
        """Write ``data`` in chunks under the integrity contract."""
        if not isinstance(data, (bytes, bytearray)):
            raise ToolError("write_file: data must be bytes")
        payload = bytes(data)
        rel = self._confined_rel(path)
        current = 0
        if append:
            probed = self._probe_size_rel(rel)
            if probed is None:
                raise IntegrityError(
                    f"write_file {path}: cannot verify the current size "
                    "before appending; refusing to write unverified")
            current = probed
        chunks = [payload[i:i + self.chunk_bytes]
                  for i in range(0, len(payload), self.chunk_bytes)] or [b""]
        sent = 0
        for index, chunk in enumerate(chunks):
            use_append = append or index > 0
            expected_after = (current if index == 0 and not append
                              else current + sent) + len(chunk) \
                if append or index > 0 else len(chunk)
            if not append and index == 0:
                expected_after = len(chunk)
            before_chunk = expected_after - len(chunk)
            frame = {"type": "write_file", "path": rel, "append": use_append,
                     "content_b64": base64.b64encode(chunk).decode()}
            try:
                result = self._request(frame)
                if not result.get("ok"):
                    raise ToolError(
                        f"write_file {path}: {result.get('error') or 'failed'}")
                reported = result.get("size")
                if isinstance(reported, int) and reported != expected_after:
                    raise IntegrityError(
                        f"write_file {path}: chunk {index} reported size "
                        f"{reported}, expected {expected_after}")
            except ToolError:
                raise
            except Exception as exc:
                # The response may be lost after the write landed. Probe
                # by size: landed -> continue; not landed -> resend once;
                # anything else -> the file state is uncertain.
                probed = self._probe_size_rel(rel)
                if probed == expected_after:
                    pass
                elif probed == before_chunk:
                    # The resend gets the same probe protection as the
                    # first send: if its response is lost too, probe
                    # again and report the outcome as an integrity
                    # result, never as a raw transport error.
                    try:
                        result = self._request(frame)
                    except Exception as resend_exc:
                        reprobed = self._probe_size_rel(rel)
                        if reprobed == expected_after:
                            result = None
                        else:
                            raise IntegrityError(
                                f"write_file {path}: resent chunk {index} "
                                f"outcome unknown (probed size {reprobed}, "
                                f"expected {expected_after})") from resend_exc
                    if result is not None:
                        if not result.get("ok"):
                            raise ToolError(
                                f"write_file {path}: "
                                f"{result.get('error') or 'failed'}"
                            ) from exc
                        reported = result.get("size")
                        if isinstance(reported, int) and \
                                reported != expected_after:
                            raise IntegrityError(
                                f"write_file {path}: resent chunk reported "
                                f"size {reported}, expected {expected_after}"
                            ) from exc
                else:
                    raise IntegrityError(
                        f"write_file {path}: chunk {index} outcome unknown "
                        f"(probed size {probed}, expected {expected_after})"
                    ) from exc
            sent += len(chunk)
        final_size = (current + len(payload)) if append else len(payload)
        probed = self._probe_size_rel(rel)
        if probed is None:
            raise IntegrityError(
                f"write_file {path}: final size could not be verified "
                f"(expected {final_size}); the write is not confirmed")
        if probed != final_size:
            raise IntegrityError(
                f"write_file {path}: final size {probed}, expected {final_size}")
        digest = hashlib.sha256(payload).hexdigest()
        if append:
            digest = None  # the hash of the whole file is checked by callers
        elif final_size <= self.readback_limit:
            back, _ = self.read_bytes(path)
            if hashlib.sha256(back).hexdigest() != digest:
                raise IntegrityError(
                    f"write_file {path}: read-back hash does not match intent")
        return {"path": path, "size": final_size, "sha256": digest,
                "chunks": len(chunks), "append": append}

    def write_file(self, path: str, content: Any,
                   append: bool = False,
                   strip_line_numbers: bool = False) -> dict[str, Any]:
        if isinstance(content, str):
            # Content is written exactly as supplied. Stripping the
            # read_file line-number prefixes is strictly opt-in: by
            # default a numbered document (a TSV whose first column is
            # numbers, say) round-trips byte-identical.
            if strip_line_numbers:
                content = strip_line_number_prefixes(content)
            data = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray)):
            data = bytes(content)
        else:
            raise ToolError("write_file: content must be text or bytes")
        info = self.write_bytes(path, data, append=append)
        return _result(
            True,
            f"wrote {info['size']} bytes to {path} "
            f"({info['chunks']} chunk{'s' if info['chunks'] != 1 else ''})",
            info)

    # -- exec ------------------------------------------------------------------

    def _remaining_timeout(self, deadline: Optional[float]) -> Optional[int]:
        """Seconds of task time left (minus a safety margin), or None
        when no deadline is known."""
        remaining: Optional[float] = None
        if deadline is not None:
            remaining = float(deadline) - time.time()
        else:
            cfg_deadline = self.config.get("deadline")
            if isinstance(cfg_deadline, (int, float)):
                remaining = float(cfg_deadline) - time.time()
            else:
                time_left = self.config.get("time_left")
                if callable(time_left):
                    try:
                        value = time_left()
                    except Exception:
                        value = None
                    if isinstance(value, (int, float)):
                        remaining = float(value)
        if remaining is None:
            return None
        return int(remaining - 2)

    def exec(self, command: str, timeout_s: int = DEFAULT_EXEC_TIMEOUT_SEC,
             background: bool = False,
             deadline: Optional[float] = None) -> dict[str, Any]:
        if not isinstance(command, str) or not command:
            raise ToolError("exec: command must be non-empty text")
        timeout = max(1, min(int(timeout_s), MAX_EXEC_TIMEOUT_SEC))
        remaining = self._remaining_timeout(deadline)
        if remaining is not None:
            if remaining <= 0:
                raise ToolError("exec: no task time remaining")
            timeout = max(1, min(timeout, remaining))
        self.reap_background()
        if background and self.request is None:
            result = self._local_exec(command, timeout, True)
            return _result(True, f"background job started (pid {result['pid']}, "
                                 f"process group {result['process_group']})",
                           result)
        if background:
            job = f"ple{int(time.time() * 1000)}"
            log_rel = f".ple-background/{job}.log"
            wrapped = (
                f"mkdir -p .ple-background && setsid bash -c "
                f"{shlex.quote(command)} > {shlex.quote(log_rel)} 2>&1 & "
                f"echo $!")
            result = self._request({"type": "exec", "command": wrapped,
                                    "timeout_sec": timeout})
            if not result.get("ok", True):
                raise ToolError(f"exec: {result.get('error') or 'failed'}")
            pid_text = str(result.get("output") or "").strip().splitlines()
            pid = int(pid_text[-1]) if pid_text and pid_text[-1].isdigit() else None
            return _result(True, f"background job started (pid {pid})",
                           {"background": True, "pid": pid,
                            "process_group": pid, "log_path": log_rel,
                            "exit_code": None, "output": ""})
        result = self._request({"type": "exec", "command": command,
                                "timeout_sec": timeout})
        if not result.get("ok", True):
            raise ToolError(f"exec: {result.get('error') or 'failed'}")
        exit_code = result.get("exit_code")
        output, capped = self._truncate_output(str(result.get("output", "")))
        truncated = bool(result.get("truncated")) or capped
        # A command that ran is a successful call whatever its exit
        # code; the code itself is part of the result data.
        return _result(
            True,
            f"exit code {exit_code}" + (" (timed out)"
                                        if result.get("timed_out") else ""),
            {"exit_code": exit_code, "output": output,
             "timed_out": bool(result.get("timed_out")),
             "truncated": truncated},
            truncated=truncated)

    # -- background job lifecycle -----------------------------------------

    def reap_background(self) -> list[int]:
        """Reap tracked background jobs that have finished, so exited
        jobs do not linger as zombies. Returns their pids."""
        finished: list[int] = []
        for pid, proc in list(self._background.items()):
            if proc.poll() is not None:
                finished.append(pid)
                del self._background[pid]
        return finished

    def stop_background(self, pid: int) -> bool:
        """Stop a registered local background job: SIGTERM its process
        group, then SIGKILL if it has not exited. Only jobs this
        dispatcher started are ever signalled. Returns True only when
        the job is actually finished afterwards."""
        proc = self._background.get(int(pid))
        if proc is None:
            return False
        if proc.poll() is not None:
            del self._background[int(pid)]
            return True
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        done = proc.poll() is not None
        if done:
            del self._background[int(pid)]
        return done

    def stop_all_background(self) -> int:
        """Stop every registered background job. The harness calls
        this before finishing so no started job outlives the task.
        Returns how many jobs were stopped."""
        self.reap_background()
        stopped = 0
        for pid in list(self._background):
            if self.stop_background(pid):
                stopped += 1
        return stopped

    # -- mcp -------------------------------------------------------------------

    def mcp_call(self, server: str, tool: str,
                 args: Optional[Mapping[str, Any]] = None,
                 mutating: Optional[bool] = None) -> dict[str, Any]:
        result = self.mcp.call(server, tool, args, mutating=mutating)
        return _result(True, f"mcp {server}/{tool} completed", result)

    # -- dispatch ----------------------------------------------------------------

    def dispatch(self, name: str,
                 arguments: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Validate and run one model-proposed call. Never raises,
        except that a session cancellation always propagates."""
        args = dict(arguments or {})
        try:
            if name == "read_file":
                return self.read_file(str(args["path"]),
                                      int(args.get("offset") or 0),
                                      args.get("limit"))
            if name == "write_file":
                return self.write_file(
                    str(args["path"]), args.get("content"),
                    bool(args.get("append", False)),
                    strip_line_numbers=bool(
                        args.get("strip_line_numbers", False)))
            if name == "exec":
                deadline = args.get("deadline")
                return self.exec(
                    str(args["command"]),
                    int(args.get("timeout_s")
                        or args.get("timeout_sec")
                        or DEFAULT_EXEC_TIMEOUT_SEC),
                    bool(args.get("background", False)),
                    deadline=float(deadline)
                    if isinstance(deadline, (int, float)) else None)
            if name == "mcp_call":
                return self.mcp_call(str(args["server"]), str(args["tool"]),
                                     args.get("args") or {},
                                     args.get("mutating"))
            return _result(False, f"unknown tool: {name}",
                           error_class="unknown_tool")
        except KeyError as exc:
            return _result(False, f"{name}: missing argument {exc}",
                           error_class="invalid_arguments")
        except IntegrityError as exc:
            return _result(False, str(exc), error_class="integrity")
        except MCPAmbiguousResult as exc:
            return _result(False, str(exc), error_class="mcp_ambiguous")
        except (ToolError, MCPError) as exc:
            return _result(False, str(exc), error_class="tool_error")
        except Exception as exc:  # surfaced to the model, never raised
            # Cancellation is not an error observation: a cancelled
            # session must unwind, not hand the loop a result to act
            # on. serve.py's SessionCancelled is recognised by class
            # name so this module needs no import of the server.
            if type(exc).__name__ == "SessionCancelled":
                raise
            return _result(False, f"{name}: {type(exc).__name__}: {exc}",
                           error_class="internal")
