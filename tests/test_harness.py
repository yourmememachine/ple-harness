"""Tests for the harness loop, its state, budget and tools."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness as H  # noqa: E402
from harness import (BudgetGovernor, DeliverableStore, LedgerState,  # noqa: E402
                     all_hard_checks_pass, run_harness)
from model_client import fingerprint_call, normalize_usage  # noqa: E402
from tools import IntegrityError, ToolDispatcher  # noqa: E402

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


def make_client(script):
    """A scripted stand-in for the model client (callable form)."""
    def client(messages, tools=None, phase="execute"):
        return script(phase, messages)
    return client


def brief_reply(usage=20, brief=None):
    return {"content": json.dumps(brief or BRIEF),
            "usage": {"input_tokens": usage, "output_tokens": 0}}


# ---------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------

def test_deliverable_exists_early(tmp_path):
    seen = {}

    def script(phase, messages):
        if phase == "brief":
            return brief_reply()
        if phase == "execute":
            if "exists" not in seen:
                seen["exists"] = (tmp_path / "out/result.txt").exists()
                return {"tool_calls": [{
                    "name": "write_file",
                    "arguments": {"path": "out/result.txt",
                                  "content": "final answer\n"}}],
                    "usage": {"input_tokens": 30, "output_tokens": 0}}
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    frames = []
    outcome = run_harness(
        {"instruction": INSTRUCTION, "id": "t-early"}, frames.append,
        tmp_path,
        {"model_client": make_client(script), "token_cap": 10_000,
         "estimated_call_tokens": 10})
    assert seen["exists"] is True
    assert (tmp_path / "out/result.txt").read_text() == "final answer\n"
    assert outcome["first_deliverable_budget_fraction"] <= 0.20
    assert outcome["early_deliverable_within_target"] is True
    assert outcome["status"] == "finalized"
    assert outcome["deliverables"]["out/result.txt"]["status"] == "verified"
    assert outcome["all_hard_checks_pass"] is True
    assert outcome["terminal_frame_sent"] is False
    assert all(f.get("type") not in ("done", "error") for f in frames)


def test_execute_cannot_consume_the_reserve(tmp_path):
    def script(phase, messages):
        if phase == "brief":
            return brief_reply(usage=20)
        if phase == "execute":
            return {"tool_calls": [{
                "name": "write_file",
                "arguments": {"path": "out/result.txt",
                              "content": "answer\n"}}],
                "usage": {"input_tokens": 580, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 10, "output_tokens": 0}}

    outcome = run_harness(
        {"instruction": INSTRUCTION, "id": "t-reserve"}, lambda f: None,
        tmp_path,
        {"model_client": make_client(script), "token_cap": 1_000,
         "reserve_tokens": 400, "estimated_call_tokens": 10})
    budget = outcome["budget"]
    assert budget["spent"]["execute"] <= 600
    assert budget["spent_total"] <= 1_000
    # Finalize still ran on the reserve after Execute hit its ceiling.
    assert outcome["status"] == "finalized"
    assert budget["calls"]["finalize"] >= 1


def test_governor_reserve_unit():
    governor = BudgetGovernor(total_tokens=1_000, reserve_tokens=400)
    assert governor.can_spend("execute", 600)
    assert not governor.can_spend("execute", 601)
    governor.spend("execute", 600)
    assert not governor.can_spend("execute", 1)
    assert governor.can_spend("verify", 1)
    governor.spend("finalize", 400)
    assert governor.spent_total == 1_000
    assert not governor.can_spend("finalize", 1)


def test_unresolved_blocks_all_pass_claim(tmp_path):
    brief = json.loads(json.dumps(BRIEF))
    brief["constraints"] = [{
        "id": "c9", "text": "an uncheckable requirement", "hard": True,
        "source_span": "Write the result to out/result.txt",
    }]  # no check -> UNRESOLVED

    def script(phase, messages):
        if phase == "brief":
            return brief_reply(brief=brief)
        if phase == "execute":
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}",
                "usage": {"input_tokens": 5, "output_tokens": 0}}

    outcome = run_harness(
        {"instruction": INSTRUCTION, "id": "t-unresolved"}, lambda f: None,
        tmp_path,
        {"model_client": make_client(script), "token_cap": 10_000,
         "estimated_call_tokens": 10})
    assert outcome["all_hard_checks_pass"] is False
    assert outcome["checks"]["constraints"]["c9"]["status"] == "UNRESOLVED"
    assert outcome["deliverables"]["out/result.txt"]["status"] != "verified"


def test_all_hard_checks_pass_helper():
    assert all_hard_checks_pass({"constraints": {
        "a": {"hard": True, "status": "PASS"}}}) is True
    assert all_hard_checks_pass({"constraints": {
        "a": {"hard": False, "status": "UNRESOLVED"}}}) is False
    assert all_hard_checks_pass({"constraints": {
        "a": {"hard": True, "status": "FAIL"}}}) is False


# ---------------------------------------------------------------------
# Deliverable lifecycle
# ---------------------------------------------------------------------

def test_atomic_promotion_preserves_previous_verified(tmp_path):
    state = LedgerState(tmp_path / "state", "t-promo")
    dispatcher = ToolDispatcher(tmp_path)
    store = DeliverableStore(state, dispatcher)
    path = "out/result.txt"

    store.write_candidate(path, b"version one\n")
    assert store.promote(path) is True
    assert store.verified_copy(path) == b"version one\n"
    assert store.previous_verified(path) is None

    store.write_candidate(path, b"version two\n")
    # The graded path now holds v2, but v1 is still the verified copy.
    assert store.verified_copy(path) == b"version one\n"
    assert store.promote(path) is True
    assert store.verified_copy(path) == b"version two\n"
    assert store.previous_verified(path) == b"version one\n"

    # Corrupt the graded path; Finalize restores the verified copy.
    (tmp_path / "out/result.txt").write_bytes(b"truncated")
    assert store.restore_verified(path) is True
    assert (tmp_path / "out/result.txt").read_bytes() == b"version two\n"
    assert store.status(path) == "verified"


# ---------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------

def test_compaction_respects_64kib_cap(tmp_path):
    state = LedgerState(tmp_path / "state", "t-compact")
    for i in range(600):
        state.append_event("action", tool="exec",
                           target=f"command number {i}",
                           result_class="ok",
                           detail="x" * 480)
    compact = state.compact_summary()
    blob = json.dumps(compact, ensure_ascii=False,
                      sort_keys=True).encode("utf-8")
    assert len(blob) <= H.SUMMARY_CAP_BYTES == 64 * 1024
    assert compact["dropped_actions"] > 0
    assert len(state.closing_summary_text().encode("utf-8")) <= 64 * 1024


def test_summary_rebuilds_from_ledger(tmp_path):
    state = LedgerState(tmp_path / "state", "t-rebuild")
    state.append_usage({"phase": "execute", "model": "m",
                        "input_tokens": 7, "output_tokens": 3,
                        "cache_read_tokens": 1, "cache_write_tokens": 0})
    state.append_event("decision", text="chose approach A")
    rebuilt = state.rebuild_summary()
    assert rebuilt["usage"] == state.summary["usage"]
    assert rebuilt["model_calls"] == 1
    assert rebuilt["decisions"] == ["chose approach A"]


# ---------------------------------------------------------------------
# Chunked-append integrity
# ---------------------------------------------------------------------

class FakeRequest:
    """Evaluator-style request transport over a local directory."""

    def __init__(self, root: Path, drop_on_write: int | None = None,
                 corrupt_on_write: int | None = None):
        self.root = root
        self.writes = 0
        self.drop_on_write = drop_on_write
        self.corrupt_on_write = corrupt_on_write

    def __call__(self, frame):
        kind = frame.get("type")
        if kind == "read_file":
            path = self.root / frame["path"]
            if not path.exists():
                return {"ok": False, "error": "not found"}
            data = path.read_bytes()
            return {"ok": True, "size": len(data),
                    "content_b64": base64.b64encode(data).decode(),
                    "truncated": False}
        if kind == "write_file":
            self.writes += 1
            path = self.root / frame["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            data = base64.b64decode(frame["content_b64"])
            if self.corrupt_on_write == self.writes:
                data = bytes(b ^ 0xFF for b in data)
            mode = "ab" if frame.get("append") else "wb"
            with open(path, mode) as handle:
                handle.write(data)
            if self.drop_on_write == self.writes:
                raise ConnectionError("response lost after write")
            return {"ok": True, "size": path.stat().st_size}
        if kind == "exec":
            return {"ok": True, "stdout": "", "stderr": "", "exit_code": 0}
        raise AssertionError(f"unexpected frame {kind}")


def test_chunked_append_integrity(tmp_path):
    dispatcher = ToolDispatcher(tmp_path, request=FakeRequest(tmp_path),
                                config={"write_chunk_bytes": 4})
    dispatcher.write_bytes("data.bin", b"abc")
    result = dispatcher.write_bytes("data.bin", b"0123456789", append=True)
    assert result["size"] == 13
    assert (tmp_path / "data.bin").read_bytes() == b"abc0123456789"


def test_chunked_append_recovers_from_lost_response(tmp_path):
    fake = FakeRequest(tmp_path, drop_on_write=2)
    dispatcher = ToolDispatcher(tmp_path, request=fake,
                                config={"write_chunk_bytes": 4})
    dispatcher.write_bytes("data.bin", b"abc")
    fake.writes = 0
    dispatcher.write_bytes("data.bin", b"0123456789", append=True)
    assert (tmp_path / "data.bin").read_bytes() == b"abc0123456789"


def test_chunked_append_detects_corruption(tmp_path):
    fake = FakeRequest(tmp_path, corrupt_on_write=1)
    dispatcher = ToolDispatcher(tmp_path, request=fake,
                                config={"write_chunk_bytes": 4})
    with pytest.raises(IntegrityError):
        dispatcher.write_bytes("data.bin", b"0123456789")
    fake.corrupt_on_write = fake.writes + 1
    result = dispatcher.dispatch("write_file", {
        "path": "data.bin", "content": "0123456789"})
    assert result["ok"] is False
    assert result["error_class"] == "integrity"


# ---------------------------------------------------------------------
# Model client helpers
# ---------------------------------------------------------------------

def test_fingerprint_is_whitespace_and_comment_insensitive():
    a = fingerprint_call("exec", {"command": "ls   -la   /tmp"})
    b = fingerprint_call("exec", {"command": "ls -la /tmp # list it"})
    assert a == b
    c = fingerprint_call("exec", {"command": "ls -la /var"})
    assert a != c


def test_usage_normalisation_never_fabricates():
    counts, defects = normalize_usage({"input_tokens": 5,
                                       "output_tokens": None})
    assert counts["input_tokens"] == 5
    assert counts["output_tokens"] == 0
    counts, defects = normalize_usage({"input_tokens": "many"})
    assert counts["input_tokens"] == 0
    assert any("input_tokens" in defect for defect in defects)
    counts, _ = normalize_usage(
        {"prompt_tokens_details": {"cached_tokens": 9}})
    assert counts["cache_read_tokens"] == 9
