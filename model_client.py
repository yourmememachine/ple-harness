"""Model-call layer for the harness.

One OpenAI-compatible client talks to the model proxy named by the
environment (``MODEL_PROXY_URL`` / ``MODEL_PROXY_KEY``, with the
evaluator's ``MODEL_BASE_URL`` / ``MODEL_API_KEY`` as fallback names).
Every completed call produces exactly one usage record in the ledger
and, when an ``emit`` callable is supplied, one protocol ``usage``
frame built from that record. Counts are taken only from what the
gateway confirmed: a missing or null count is recorded as 0, a count
of the wrong type is recorded as 0 and noted as a defect, and nothing
is estimated or invented. If a stream is interrupted, the record is
written from the counts confirmed before the interruption and the
partial text travels with the raised ``ModelCallInterrupted``.

The module also owns action fingerprinting: the canonicalised identity
of a tool call (tool name plus canonicalised arguments, whitespace-
insensitive, and comment-insensitive for shell commands), used by the
loop's repeat guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

MAX_COUNT = 2**53 - 1
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens")
USAGE_FRAME_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens")


def _is_true_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def normalize_usage(raw: Any) -> tuple[dict[str, int], list[str]]:
    """Return (counts, defects) for a gateway usage object.

    Missing or null counts become 0. A present count that is not a
    true integer in ``0 .. 2**53 - 1`` becomes 0 and is listed as a
    defect. Cache-write counts are read from the common field names.
    Never raises.
    """
    counts = {name: 0 for name in TOKEN_FIELDS}
    defects: list[str] = []
    if raw is None:
        return counts, defects
    if not isinstance(raw, Mapping):
        return counts, ["usage was not an object"]
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "cache_read_tokens": ("cache_read_tokens", "cached_tokens"),
        "cache_write_tokens": ("cache_write_tokens",
                               "cache_creation_input_tokens",
                               "cache_write_input_tokens"),
    }
    details = raw.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        if counts["cache_read_tokens"] == 0 and "cached_tokens" in details:
            raw = {**raw, "cache_read_tokens": details.get("cached_tokens")}
    for field_name, names in aliases.items():
        value = None
        present = False
        for name in names:
            if name in raw:
                value = raw.get(name)
                present = True
                break
        if not present or value is None:
            counts[field_name] = 0
        elif _is_true_int(value) and 0 <= value <= MAX_COUNT:
            counts[field_name] = int(value)
        else:
            counts[field_name] = 0
            defects.append(f"{field_name} was not a valid count")
    return counts, defects


def total_tokens(counts: Mapping[str, int]) -> int:
    """The budget unit: input + output + cache-read + cache-write."""
    return sum(int(counts.get(name, 0)) for name in TOKEN_FIELDS)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def canonicalise(value: Any) -> Any:
    """Canonical form of a JSON-like value for fingerprinting.

    Object keys are sorted, strings have their whitespace collapsed,
    and numbers are normalised so that ``1`` and ``1.0`` agree.
    """
    if isinstance(value, Mapping):
        return {str(k): canonicalise(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonicalise(v) for v in value]
    if isinstance(value, str):
        return _collapse_ws(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_exec_command(command: str) -> str:
    """Canonical form of a shell command for fingerprinting.

    Whitespace runs outside quotes collapse to one space and ``#``
    comments outside quotes are dropped, so a cosmetically varied
    repeat of the same command has the same identity. Quoted text is
    preserved exactly.
    """
    out: list[str] = []
    quote: Optional[str] = None
    escaped = False
    pending_space = False
    i = 0
    while i < len(command):
        ch = command[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and quote != "'":
            out.append(ch)
            escaped = True
            i += 1
            continue
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "#":
            while i < len(command) and command[i] != "\n":
                i += 1
            continue
        if ch.isspace():
            pending_space = True
            i += 1
            continue
        if pending_space and out:
            out.append(" ")
        pending_space = False
        out.append(ch)
        i += 1
    return "".join(out).strip()


def fingerprint_call(tool_name: str, arguments: Any) -> str:
    """Stable fingerprint of one proposed tool call."""
    args = arguments if isinstance(arguments, Mapping) else {}
    canon_args = canonicalise(dict(args))
    if tool_name == "exec" and isinstance(canon_args.get("command"), str):
        canon_args = dict(canon_args)
        canon_args["command"] = canonical_exec_command(str(args.get("command", "")))
    payload = {"tool": tool_name, "arguments": canon_args}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Short alias, used by call sites that prefer the noun form.
action_fingerprint = fingerprint_call


class InMemoryUsageLedger:
    """Small ledger used when no disk ledger is supplied."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append_usage(self, record: Mapping[str, Any]) -> dict[str, Any]:
        stored = dict(record)
        self.records.append(stored)
        return stored

    def totals(self) -> dict[str, int]:
        totals = {name: 0 for name in TOKEN_FIELDS}
        for record in self.records:
            for name in TOKEN_FIELDS:
                totals[name] += int(record.get(name, 0))
        return totals


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {n: 0 for n in TOKEN_FIELDS})
    model: str = ""
    finish_reason: Optional[str] = None
    raw: Any = None
    usage_recorded: bool = False
    defects: list[str] = field(default_factory=list)


class ModelCallInterrupted(Exception):
    """A streaming call stopped early; carries the confirmed usage."""

    def __init__(self, message: str, partial: ModelResponse):
        super().__init__(message)
        self.partial = partial


class ModelCallError(Exception):
    """A model call failed before any usable response arrived."""


def _parse_tool_call(raw: Mapping[str, Any]) -> dict[str, Any]:
    fn = raw.get("function") if isinstance(raw.get("function"), Mapping) else raw
    name = fn.get("name") or raw.get("name") or ""
    args = fn.get("arguments", raw.get("arguments", {}))
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
            args = parsed if isinstance(parsed, Mapping) else {"value": parsed}
        except ValueError:
            args = {"_raw": args}
    return {"id": raw.get("id"), "name": str(name),
            "arguments": dict(args) if isinstance(args, Mapping) else {}}


class ModelClient:
    """OpenAI-compatible streaming client with a usage ledger."""

    def __init__(
        self,
        ledger: Any = None,
        emit: Optional[Callable[[dict[str, Any]], Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
        transport: Optional[Callable[[dict[str, Any]], Any]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        cfg = dict(config or {})
        self.ledger = ledger if ledger is not None else InMemoryUsageLedger()
        self.emit = emit
        self.transport = transport or cfg.get("model_transport")
        self.base_url = (
            base_url
            or cfg.get("model_proxy_url")
            or os.environ.get("MODEL_PROXY_URL")
            or os.environ.get("MODEL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        self.api_key = (
            api_key
            or cfg.get("model_proxy_key")
            or os.environ.get("MODEL_PROXY_KEY")
            or os.environ.get("MODEL_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        chosen = (
            model
            or cfg.get("model_name")
            or os.environ.get("MODEL_NAME")
            or cfg.get("model_api")
            or os.environ.get("MODEL_API")
            or "google-cloud/claude-opus-5-5"
        )
        self.model = chosen.split("/", 1)[1] if "/" in chosen else chosen
        self.model_api = chosen if "/" in chosen else f"google-cloud/{chosen}"
        self.timeout = float(cfg.get("model_timeout_sec", 1800))
        self.retry_delays = tuple(cfg.get("retry_delays", (1.0, 4.0)))
        self.effort = cfg.get("reasoning_effort")
        self.max_output_tokens = cfg.get("max_output_tokens")
        self.defects: list[str] = []
        self._seq = 0

    # -- ledger ------------------------------------------------------

    def _record_usage(self, phase: str, counts: Mapping[str, int],
                      defects: Iterable[str]) -> dict[str, Any]:
        self._seq += 1
        record = {
            "seq": self._seq,
            "kind": "usage",
            "phase": phase,
            "model": self.model,
            **{name: int(counts.get(name, 0)) for name in TOKEN_FIELDS},
            "total_tokens": total_tokens(counts),
            "defects": list(defects),
        }
        try:
            append = getattr(self.ledger, "append_usage", None)
            if callable(append):
                append(record)
            elif callable(self.ledger):
                self.ledger(record)
        except Exception as exc:  # the ledger never raises into a session
            self.defects.append(f"ledger append failed: {exc}")
        if self.emit is not None:
            frame = {"type": "usage", "model": self.model}
            for name in USAGE_FRAME_FIELDS:
                frame[name] = int(counts.get(name, 0))
            try:
                self.emit(frame)
            except Exception as exc:
                self.defects.append(f"usage emit failed: {exc}")
        return record

    # -- transport ---------------------------------------------------

    def _endpoint(self) -> str:
        if not self.base_url:
            raise ModelCallError("no model proxy URL is configured")
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _http_chunks(self, payload: Mapping[str, Any]) -> Iterable[Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(), data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                **({"Authorization": f"Bearer {self.api_key}"}
                   if self.api_key else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            ctype = resp.headers.get_content_type()
            if ctype == "application/json":
                yield json.loads(resp.read().decode("utf-8"))
                return
            for line in resp:
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    yield json.loads(text)
                except ValueError:
                    continue

    def _chunks(self, payload: Mapping[str, Any]) -> Iterable[Any]:
        if self.transport is not None:
            produced = self.transport(dict(payload))
            if produced is None:
                return []
            if isinstance(produced, Mapping):
                return [produced]
            return list(produced)
        return self._http_chunks(payload)

    # -- calls ---------------------------------------------------------

    def build_payload(self, messages: list[Mapping[str, Any]],
                      tools: Optional[list[Mapping[str, Any]]] = None,
                      **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if self.effort:
            payload["reasoning_effort"] = self.effort
        if self.max_output_tokens:
            payload["max_completion_tokens"] = int(self.max_output_tokens)
        payload.update(overrides)
        return payload

    def complete(self, messages: list[Mapping[str, Any]],
                 tools: Optional[list[Mapping[str, Any]]] = None,
                 phase: str = "execute", **overrides: Any) -> ModelResponse:
        """Run one model call and return its normalised response.

        Retries apply only before the first streamed chunk arrives
        (429/5xx and connection failures). Once chunks have arrived, an
        interruption raises ``ModelCallInterrupted`` with the partial
        response and the usage confirmed so far already recorded.
        """
        payload = self.build_payload(messages, tools, **overrides)
        attempt = 0
        while True:
            content_parts: list[str] = []
            tool_parts: dict[int, dict[str, Any]] = {}
            usage_raw: Any = None
            finish_reason: Optional[str] = None
            started = False
            try:
                for chunk in self._chunks(payload):
                    started = True
                    if not isinstance(chunk, Mapping):
                        continue
                    if chunk.get("usage") is not None:
                        usage_raw = chunk.get("usage")
                    choices = chunk.get("choices") or []
                    if not choices:
                        # A non-streaming full response object.
                        message = chunk.get("message")
                        if isinstance(message, Mapping):
                            content_parts.append(str(message.get("content") or ""))
                            for i, tc in enumerate(message.get("tool_calls") or []):
                                tool_parts[i] = _parse_tool_call(tc)
                            finish_reason = chunk.get("finish_reason")
                        continue
                    choice = choices[0] if isinstance(choices[0], Mapping) else {}
                    if choice.get("finish_reason"):
                        finish_reason = choice.get("finish_reason")
                    delta = choice.get("delta") or choice.get("message") or {}
                    if isinstance(delta, Mapping):
                        if delta.get("content"):
                            content_parts.append(str(delta["content"]))
                        for tc in delta.get("tool_calls") or []:
                            idx = int(tc.get("index", len(tool_parts)))
                            slot = tool_parts.setdefault(
                                idx, {"id": tc.get("id"), "name": "", "arguments": {}})
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            piece = fn.get("arguments")
                            if isinstance(piece, str):
                                slot["_args_text"] = slot.get("_args_text", "") + piece
                            elif isinstance(piece, Mapping):
                                slot["arguments"] = dict(piece)
                break
            except urllib.error.HTTPError as exc:
                if not started and exc.code in (408, 409, 429) or (
                        not started and exc.code >= 500):
                    if attempt < len(self.retry_delays):
                        time.sleep(self.retry_delays[attempt])
                        attempt += 1
                        continue
                counts, defects = normalize_usage(usage_raw)
                partial = self._finish(content_parts, tool_parts, counts,
                                       finish_reason, None)
                self._record_usage(phase, counts, defects)
                partial.usage_recorded = True
                raise ModelCallInterrupted(
                    f"model call interrupted: HTTP {exc.code}", partial) from exc
            except Exception as exc:
                if not started and attempt < len(self.retry_delays) and isinstance(
                        exc, (ConnectionError, TimeoutError, OSError,
                              urllib.error.URLError)):
                    time.sleep(self.retry_delays[attempt])
                    attempt += 1
                    continue
                counts, defects = normalize_usage(usage_raw)
                partial = self._finish(content_parts, tool_parts, counts,
                                       finish_reason, None)
                self._record_usage(phase, counts, defects)
                partial.usage_recorded = True
                if started:
                    raise ModelCallInterrupted(
                        f"model call interrupted: {exc}", partial) from exc
                raise ModelCallError(f"model call failed: {exc}") from exc
        counts, defects = normalize_usage(usage_raw)
        response = self._finish(content_parts, tool_parts, counts,
                                finish_reason, None)
        self._record_usage(phase, counts, defects)
        response.usage_recorded = True
        response.defects = defects
        return response

    @staticmethod
    def _finish(content_parts: list[str], tool_parts: dict[int, dict[str, Any]],
                counts: Mapping[str, int], finish_reason: Optional[str],
                raw: Any) -> ModelResponse:
        tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_parts):
            slot = tool_parts[idx]
            args = slot.get("arguments") or {}
            if "_args_text" in slot:
                text = slot.pop("_args_text")
                try:
                    parsed = json.loads(text) if text.strip() else {}
                    args = parsed if isinstance(parsed, Mapping) else {"value": parsed}
                except ValueError:
                    args = {"_raw": text}
            tool_calls.append({"id": slot.get("id"), "name": slot.get("name", ""),
                               "arguments": dict(args)})
        return ModelResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            usage={name: int(counts.get(name, 0)) for name in TOKEN_FIELDS},
            finish_reason=finish_reason,
            raw=raw,
        )
