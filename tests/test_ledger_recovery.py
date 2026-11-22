"""Crash-recovery tests for the ledger and its generated summary.

The ledger is append-only JSON lines. A crash can interrupt a run at
any point, including halfway through appending one line, and a later
process must be able to rebuild the summary from whatever is on disk.
These tests pin three properties of that rebuild:

- a torn final line (truncated JSON, the expected artifact of a
  crash mid-append) is tolerated: the rebuild does not raise and the
  summary still reflects every complete event;
- a partial ledger (a run interrupted before any closing event)
  rebuilds to exactly the summary the live fold held at that point,
  deterministically, including from a fresh state object that has
  only the ledger file;
- replaying a segment of the ledger does not double-count usage,
  model calls, or records in the rebuilt summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import LedgerState  # noqa: E402


def _populate(state: LedgerState) -> None:
    """A partial run: usage, an action, a candidate, a decision.

    No checks event and no closing event — the shape a ledger has
    when a run is interrupted mid-Execute.
    """
    state.append_event("phase", phase="execute")
    state.append_usage({"phase": "execute", "model": "m",
                        "input_tokens": 100, "output_tokens": 10,
                        "cache_read_tokens": 0, "cache_write_tokens": 0})
    state.append_event("action", tool="exec", target="ls",
                       result_class="ok", detail="listed the workdir")
    state.append_event("deliverable", path="out/result.txt",
                       status="candidate", sha256="ab" * 32, size=12)
    state.append_event("decision", text="chose approach A")
    state.append_usage({"phase": "execute", "model": "m",
                        "input_tokens": 50, "output_tokens": 5,
                        "cache_read_tokens": 0, "cache_write_tokens": 0})


def _lines(state: LedgerState) -> list[str]:
    return [line for line in
            state.ledger_path.read_text("utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------
# Torn final line
# ---------------------------------------------------------------------

def test_torn_final_line_is_tolerated_by_rebuild(tmp_path):
    state = LedgerState(tmp_path / "state", "t-torn")
    _populate(state)
    live = state.compact_summary()
    good_seq = state.seq

    # A crash mid-append leaves a truncated JSON fragment as the
    # last line: no closing braces, no trailing newline.
    with open(state.ledger_path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": %d, "kind": "usage", "usage": '
                     '{"input_tokens": 5' % (good_seq + 1))

    rebuilt = state.rebuild_summary()  # must not raise
    assert rebuilt == live
    assert rebuilt["updated_seq"] == good_seq
    assert rebuilt["usage"]["input_tokens"] == 150

    # A fresh state object over the same directory (a new process
    # recovering the run) rebuilds the same summary.
    fresh = LedgerState(tmp_path / "state", "t-torn")
    assert fresh.rebuild_summary() == live


def test_blank_lines_and_missing_ledger_are_tolerated(tmp_path):
    state = LedgerState(tmp_path / "state", "t-blank")
    _populate(state)
    live = state.compact_summary()
    with open(state.ledger_path, "a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert state.rebuild_summary() == live

    empty = LedgerState(tmp_path / "nostate", "t-empty")
    rebuilt = empty.rebuild_summary()  # no ledger file at all
    assert rebuilt["task_id"] == "t-empty"
    assert rebuilt["model_calls"] == 0
    assert rebuilt["usage"]["input_tokens"] == 0


# ---------------------------------------------------------------------
# Interrupted run: deterministic replay of a partial ledger
# ---------------------------------------------------------------------

def test_partial_ledger_rebuilds_the_live_summary(tmp_path):
    state = LedgerState(tmp_path / "state", "t-partial")
    _populate(state)
    # Nothing is persisted and nothing is closed: the run stops here.
    assert not state.summary_path.exists()
    live = state.compact_summary()

    rebuilt = state.rebuild_summary()
    assert rebuilt == live
    assert rebuilt["phase"] == "execute"
    assert rebuilt["model_calls"] == 2
    assert rebuilt["usage"] == {"input_tokens": 150, "output_tokens": 15,
                                "cache_read_tokens": 0,
                                "cache_write_tokens": 0}
    assert rebuilt["deliverables"]["out/result.txt"]["status"] == "candidate"
    assert rebuilt["all_hard_checks_pass"] is False

    # Deterministic replay from the ledger alone, in a new directory
    # with a fresh state object, gives the identical summary.
    other = tmp_path / "recovered"
    other.mkdir()
    (other / "ledger.jsonl").write_bytes(state.ledger_path.read_bytes())
    fresh = LedgerState(other, "t-partial")
    assert fresh.rebuild_summary() == live
    # And rebuilding twice gives the same result both times.
    assert fresh.rebuild_summary() == fresh.rebuild_summary()


# ---------------------------------------------------------------------
# Duplicate protection
# ---------------------------------------------------------------------

def test_duplicated_segment_does_not_double_count(tmp_path):
    state = LedgerState(tmp_path / "state", "t-dup")
    _populate(state)
    before = state.rebuild_summary()

    # Replay a segment: the first usage event and the decision event
    # are appended to the ledger a second time, byte for byte (the
    # shape of an at-least-once rewrite of part of the ledger).
    lines = _lines(state)
    usage_line = next(line for line in lines
                      if json.loads(line)["kind"] == "usage")
    decision_line = next(line for line in lines
                         if json.loads(line)["kind"] == "decision")
    with open(state.ledger_path, "a", encoding="utf-8") as handle:
        handle.write(usage_line + "\n")
        handle.write(decision_line + "\n")

    after = state.rebuild_summary()
    assert after["usage"] == before["usage"]
    assert after["model_calls"] == before["model_calls"]
    assert after["decisions"] == before["decisions"]
    assert after["recent_actions"] == before["recent_actions"]
