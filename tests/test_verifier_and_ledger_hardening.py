"""Regression tests for the verifier and ledger hardening round.

One test per finding:

- duplicate consistency-item ids must not collapse in the verifier
  (a FAIL hidden behind a later PASS under one shared id);
- a deliverable deleted during Execute is restored at Finalize from
  its verified copy, and the outcome reports the post-restore truth;
- ledger sequence numbers must not collide after a crash-and-resume
  (appends that landed after the last summary persist);
- a parseable but semantically invalid ledger line is counted torn
  and not folded at all (no phantom counts);
- usage defects are bounded, so a persisted summary stays under its
  64 KiB cap however long the defect strings are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness as H  # noqa: E402
from harness import (DeliverableStore, LedgerState, run_harness)  # noqa: E402
from tools import ToolDispatcher  # noqa: E402

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


def brief_reply(brief=None):
    return {"content": json.dumps(brief or BRIEF),
            "usage": {"input_tokens": 20, "output_tokens": 0}}


def _usage(state: LedgerState, input_tokens=100, output_tokens=10,
           defects=None):
    record = {"phase": "execute", "model": "m",
              "input_tokens": input_tokens,
              "output_tokens": output_tokens,
              "cache_read_tokens": 0, "cache_write_tokens": 0}
    if defects is not None:
        record["defects"] = defects
    state.append_usage(record)


# ---------------------------------------------------------------------
# B1: duplicate consistency-item ids
# ---------------------------------------------------------------------

def test_duplicate_consistency_ids_cannot_hide_a_failure(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out/result.txt").write_text("answer\n")
    state = LedgerState(tmp_path / "state", "t-consdup")
    dispatcher = ToolDispatcher(tmp_path)
    store = DeliverableStore(state, dispatcher)
    verifier = H.Verifier(state, dispatcher, store, tmp_path)
    brief = {
        "deliverables": [{"path": "out/result.txt"}],
        "constraints": [{
            "id": "c1", "hard": True,
            "check": {"type": "file_exists", "path": "out/result.txt"}}],
        "consistency_items": [
            {"id": "x1", "check": {"type": "file_exists",
                                   "path": "out/missing.dat"}},
            {"id": "x1", "check": {"type": "file_exists",
                                   "path": "out/result.txt"}},
        ],
    }
    checks = verifier.verify(brief)
    # The first item's FAIL keeps its own entry; the duplicate is
    # recorded separately as UNRESOLVED instead of overwriting it.
    assert checks["consistency"]["x1"]["status"] == "FAIL"
    assert checks["consistency"]["x1#dup1"]["status"] == "UNRESOLVED"
    assert checks["all_hard_checks_pass"] is False
    assert H.deliverable_checks_pass(checks, brief,
                                     "out/result.txt") is False


# ---------------------------------------------------------------------
# B2: deliverable deleted during Execute is restored at Finalize
# ---------------------------------------------------------------------

def test_deleted_deliverable_is_restored_at_finalize(tmp_path):
    cfg = {"token_cap": 100_000, "estimated_call_tokens": 10}
    task = {"instruction": INSTRUCTION, "id": "t-deleted"}

    wrote = {"done": False}

    def write_script(phase, messages):
        if phase == "brief":
            return brief_reply()
        if phase == "execute":
            if not wrote["done"]:
                wrote["done"] = True
                return {"tool_calls": [{
                    "name": "write_file",
                    "arguments": {"path": "out/result.txt",
                                  "content": "version one\n"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}", "usage": {"input_tokens": 5,
                                           "output_tokens": 0}}

    first = run_harness(task, lambda f: None, tmp_path,
                        {**cfg, "model_client": make_client(write_script)})
    assert first["deliverables"]["out/result.txt"]["status"] == "verified"
    assert (tmp_path / "out/result.txt").read_bytes() == b"version one\n"

    state = {"deleted": False}

    def delete_script(phase, messages):
        if phase == "brief":
            return brief_reply()
        if phase == "execute":
            if not state["deleted"]:
                state["deleted"] = True
                return {"tool_calls": [{
                    "name": "exec",
                    "arguments": {"command": "rm out/result.txt"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
            return {"content": "done",
                    "usage": {"input_tokens": 5, "output_tokens": 0}}
        return {"content": "{}", "usage": {"input_tokens": 5,
                                           "output_tokens": 0}}

    outcome = run_harness(task, lambda f: None, tmp_path,
                          {**cfg, "model_client": make_client(delete_script)})
    # The graded tree ends the run holding the verified bytes again,
    # and the outcome's reporting reflects the post-restore truth.
    assert (tmp_path / "out/result.txt").read_bytes() == b"version one\n"
    assert outcome["deliverables"]["out/result.txt"]["status"] == "verified"
    assert outcome["checks"]["deliverables"]["out/result.txt"]["exists"] \
        is True
    ledger = Path(outcome["state_dir"]) / "ledger.jsonl"
    decisions = [json.loads(line) for line in
                 ledger.read_text("utf-8").splitlines() if line.strip()]
    assert any(e.get("kind") == "decision"
               and "restored out/result.txt" in str(e.get("text", ""))
               and "missing" in str(e.get("text", ""))
               for e in decisions)


# ---------------------------------------------------------------------
# B3: ledger sequence collision after crash-and-resume
# ---------------------------------------------------------------------

def test_resume_after_crash_never_reissues_a_sequence(tmp_path):
    first = LedgerState(tmp_path / "state", "t-resume")
    first.append_event("decision", text="e1")
    _usage(first)
    first.append_event("decision", text="e3")
    first.persist()
    assert first.seq == 3

    resumed = LedgerState(tmp_path / "state", "t-resume")
    resumed.append_event("decision", text="pre-crash A")
    resumed.append_event("decision", text="pre-crash B")
    assert resumed.seq == 5
    # Crash: the summary is never persisted with seq 4-5 in it.

    recovered = LedgerState(tmp_path / "state", "t-resume")
    # The un-persisted tail is folded in at recovery, and the next
    # sequence continues from the highest one in the ledger.
    assert recovered.seq == 5
    assert recovered.summary["decisions"] == [
        "e1", "e3", "pre-crash A", "pre-crash B"]
    event = recovered.append_event("decision", text="post-resume C")
    assert event["seq"] == 6

    rebuilt = recovered.rebuild_summary()
    assert rebuilt == recovered.compact_summary()
    assert rebuilt["duplicate_events"] == 0
    assert rebuilt["decisions"] == [
        "e1", "e3", "pre-crash A", "pre-crash B", "post-resume C"]
    assert rebuilt["usage"]["input_tokens"] == 100


# ---------------------------------------------------------------------
# B4: semantically invalid ledger lines are torn, never folded
# ---------------------------------------------------------------------

def test_semantically_invalid_line_is_torn_and_not_folded(tmp_path):
    state = LedgerState(tmp_path / "state", "t-semantic")
    _usage(state)
    state.append_event("decision", text="kept")
    live = state.compact_summary()

    bad_usage = {"seq": state.seq + 1, "kind": "usage",
                 "phase": "execute",
                 "usage": {"input_tokens": 5, "output_tokens": "x",
                           "cache_read_tokens": 0,
                           "cache_write_tokens": 0},
                 "defects": []}
    bad_kind = {"seq": state.seq + 2, "kind": "bogus_kind"}
    with open(state.ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad_usage) + "\n")
        handle.write(json.dumps(bad_kind) + "\n")

    rebuilt = state.rebuild_summary()
    assert state.rebuild_defects["torn_lines"] == 2
    # No phantom +5 input tokens from the half-valid usage line, and
    # the invalid lines' sequences are not adopted either.
    assert rebuilt == live
    assert rebuilt["usage"]["input_tokens"] == 100
    assert rebuilt["model_calls"] == 1


# ---------------------------------------------------------------------
# B5: usage defects are bounded in the persisted summary
# ---------------------------------------------------------------------

def test_usage_defects_are_bounded_in_the_persisted_summary(tmp_path):
    state = LedgerState(tmp_path / "state", "t-defect-cap")
    for _ in range(60):
        _usage(state, input_tokens=1, output_tokens=0,
               defects=["x" * 5_000])
    defects = state.summary["usage_defects"]
    assert len(defects) == H.MAX_USAGE_DEFECTS
    assert all(len(d) <= H.FOLD_TEXT_CHARS
               + len(H.DEFECT_TRUNCATED_SUFFIX) for d in defects)
    assert all(d.endswith(H.DEFECT_TRUNCATED_SUFFIX) for d in defects)
    assert state.summary["dropped_entries"] >= 10

    persisted = state.persist()
    blob = json.dumps(persisted, ensure_ascii=False,
                      sort_keys=True).encode("utf-8")
    assert len(blob) <= H.SUMMARY_CAP_BYTES

    # The single-huge-defect shape: one 200,000-char defect string.
    huge = LedgerState(tmp_path / "state2", "t-defect-huge")
    _usage(huge, input_tokens=1, output_tokens=0,
           defects=["y" * 200_000])
    blob = json.dumps(huge.persist(), ensure_ascii=False,
                      sort_keys=True).encode("utf-8")
    assert len(blob) <= H.SUMMARY_CAP_BYTES
    assert huge.summary["usage_defects"][0].endswith(
        H.DEFECT_TRUNCATED_SUFFIX)
