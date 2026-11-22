"""The agent loop: Brief -> Execute -> Finalize, socket-free.

``run_harness(task, emit, workdir, config)`` runs one task and returns
an outcome dict. The loop never sends a terminal frame: ``emit`` is
used only for non-terminal frames (``usage`` frames are emitted by the
model client from its ledger), and the supervisor that owns the
session turns the returned outcome into the closing frame.

Behaviour, in order:

- A deterministic inventory of the workdir is taken before any model
  call. The Brief call then produces deliverable paths, a work plan,
  and a hard/soft constraint checklist in which every hard constraint
  carries a source-span pointer to where it was read. A constraint
  that cannot be classified or given a check is UNRESOLVED: it counts
  as hard for every gating decision and blocks any "all hard checks
  pass" claim.
- Execute writes a first complete, submittable candidate at every
  declared deliverable path before spending on improvements (target:
  inside the first 15-20% of the token budget), then works the plan
  one model call at a time. A budget governor holds a protected
  reserve for verification and Finalize that Execute cannot consume,
  on top of per-phase allocations.
- Finalize re-reads the instruction in a fresh audit call, re-runs
  the full deterministic check set unconditionally, promotes passing
  deliverables from candidate to verified (atomically, preserving
  the previous verified copy), and restores the last verified copy
  over any graded path that no longer passes.

State lives on disk: an append-only ledger, from which the closing
summary is generated deterministically. The summary has defined
fields and a 64 KiB hard cap; when it grows past the cap the oldest
action detail is dropped first.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_client import (ModelClient, ModelCallInterrupted,  # noqa: E402
                          fingerprint_call, normalize_usage, total_tokens)
from tools import ToolDispatcher, find_tools_schema, tool_schemas  # noqa: E402

SUMMARY_CAP_BYTES = 64 * 1024
EARLY_DELIVERABLE_FRACTION = 0.20
MAX_TOOL_RESULT_CHARS = 8_000

PHASES = ("brief", "execute", "verify", "finalize")


class BudgetExceeded(Exception):
    """A phase tried to spend past its allocation or into the reserve."""


def _is_session_cancelled(exc: BaseException) -> bool:
    """True for the session layer's cancellation exception.

    Matched by class name so this module never imports the session
    layer at load time (which would be circular); a cancellation
    raised anywhere below the loop must propagate, never be folded
    into an ordinary result or outcome.
    """
    return any(cls.__name__ == "SessionCancelled"
               for cls in type(exc).__mro__)


class BudgetGovernor:
    """Absolute token/call budgets with a protected reserve.

    Non-reserve phases (brief, execute, discovery) may spend only up
    to ``total - reserve``. Verification and Finalize may spend the
    reserve as well, up to the total cap. Per-phase allocations cap
    each phase's own share on top of that.
    """

    RESERVE_PHASES = ("verify", "finalize")

    def __init__(self, total_tokens: int = 1_000_000,
                 reserve_tokens: Optional[int] = None,
                 reserve_fraction: float = 0.25,
                 allocations: Optional[Mapping[str, float]] = None,
                 total_calls: int = 200, reserve_calls: int = 4) -> None:
        self.total_tokens = max(0, int(total_tokens))
        if reserve_tokens is None:
            reserve_tokens = int(self.total_tokens * reserve_fraction)
        self.reserve_tokens = max(0, min(int(reserve_tokens), self.total_tokens))
        shares = {"brief": 0.10, "execute": 0.65, "verify": 0.15,
                  "finalize": 0.10}
        if allocations:
            shares.update({k: float(v) for k, v in allocations.items()})
        self.allocations = shares
        self.total_calls = max(0, int(total_calls))
        self.reserve_calls = max(0, int(reserve_calls))
        self.spent: dict[str, int] = {p: 0 for p in PHASES}
        self.calls: dict[str, int] = {p: 0 for p in PHASES}
        self.overrun_tokens = 0

    @property
    def spent_total(self) -> int:
        return sum(self.spent.values())

    @property
    def calls_total(self) -> int:
        return sum(self.calls.values())

    @property
    def non_reserve_ceiling(self) -> int:
        return self.total_tokens - self.reserve_tokens

    def phase_ceiling(self, phase: str) -> int:
        share = self.allocations.get(phase, 0.0)
        return int(self.total_tokens * share)

    def can_spend(self, phase: str, amount: int) -> bool:
        if amount < 0:
            return False
        if self.spent_total + amount > self.total_tokens:
            return False
        if phase not in self.RESERVE_PHASES:
            if self.spent_total + amount > self.non_reserve_ceiling:
                return False
        if phase in self.allocations:
            # Per-phase allocations bind every phase, reserve phases
            # included: verify and finalize each hold only their own
            # share of the budget.
            if self.spent.get(phase, 0) + amount > self.phase_ceiling(phase):
                return False
        return True

    def spend(self, phase: str, amount: int) -> None:
        if not self.can_spend(phase, amount):
            raise BudgetExceeded(
                f"{phase}: spending {amount} would exceed its allocation "
                f"or the protected reserve (spent {self.spent_total} of "
                f"{self.total_tokens}, reserve {self.reserve_tokens})")
        self.spent[phase] = self.spent.get(phase, 0) + int(amount)

    def record_spend(self, phase: str, amount: int) -> bool:
        """Charge an actual call cost; False when it broke the budget.

        The charge is split so the protected reserve stays inviolable:
        a non-reserve phase is charged only up to the non-reserve
        ceiling (and its own phase ceiling), a reserve phase only up
        to the total cap (and its own phase ceiling). Any excess is
        recorded as an overrun defect instead of being spent out of
        another phase's allocation, and False tells the caller to
        stop the phase.
        """
        amount = int(amount)
        if phase in self.RESERVE_PHASES:
            headroom = self.total_tokens - self.spent_total
        else:
            headroom = self.non_reserve_ceiling - self.spent_total
        if phase in self.allocations:
            headroom = min(headroom, self.phase_ceiling(phase)
                           - self.spent.get(phase, 0))
        charge = max(0, min(amount, headroom))
        self.spent[phase] = self.spent.get(phase, 0) + charge
        excess = amount - charge
        if excess > 0:
            self.overrun_tokens += excess
            return False
        return True

    def can_call(self, phase: str) -> bool:
        if self.calls_total + 1 > self.total_calls:
            return False
        if phase not in self.RESERVE_PHASES:
            reserve_phase_calls = sum(self.calls[p] for p in self.RESERVE_PHASES)
            if self.calls_total + 1 > self.total_calls - (
                    self.reserve_calls - reserve_phase_calls):
                return False
        return True

    def record_call(self, phase: str) -> None:
        self.calls[phase] = self.calls.get(phase, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "reserve_tokens": self.reserve_tokens,
            "spent": dict(self.spent),
            "spent_total": self.spent_total,
            "calls": dict(self.calls),
            "calls_total": self.calls_total,
            "allocations": dict(self.allocations),
            "overrun_tokens": self.overrun_tokens,
        }


# ---------------------------------------------------------------------
# Disk state and the ledger-generated summary
# ---------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=1, sort_keys=True,
                                   ensure_ascii=False).encode("utf-8"))


SUMMARY_FIELDS = (
    "task_id", "phase", "turn_count", "model_calls", "usage",
    "usage_defects", "deliverables", "constraints", "work_items",
    "checks", "decisions", "dead_ends", "recent_actions",
    "dropped_actions", "dropped_entries", "all_hard_checks_pass",
    "duplicate_events", "background_stopped",
    "updated_seq",
)

# Fold-time bounds: long model-authored strings are truncated in the
# summary (the ledger event keeps the full text) so no single field
# can push the summary past its cap on its own.
FOLD_TEXT_CHARS = 240

# Usage defects are bounded the same way: at most this many entries
# are kept in the summary, each truncated to FOLD_TEXT_CHARS with an
# explicit marker; anything past the bound is counted in
# ``dropped_entries`` instead of being recorded.
MAX_USAGE_DEFECTS = 50
DEFECT_TRUNCATED_SUFFIX = "…[truncated]"


def _empty_summary(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "phase": "brief",
        "turn_count": 0,
        "model_calls": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_tokens": 0, "cache_write_tokens": 0},
        "usage_defects": [],
        "deliverables": {},
        "constraints": {},
        "work_items": {},
        "checks": {},
        "decisions": [],
        "dead_ends": [],
        "recent_actions": [],
        "dropped_actions": 0,
        "dropped_entries": 0,
        "all_hard_checks_pass": False,
        "duplicate_events": 0,
        "background_stopped": 0,
        "updated_seq": 0,
    }


def _truncate_span(span: Any) -> Any:
    if isinstance(span, dict):
        out = dict(span)
        if isinstance(out.get("quote"), str):
            out["quote"] = out["quote"][:FOLD_TEXT_CHARS]
        return out
    if isinstance(span, str):
        return span[:FOLD_TEXT_CHARS]
    return span


def _compact_summary_dict(summary: dict[str, Any],
                          cap_bytes: int = SUMMARY_CAP_BYTES
                          ) -> dict[str, Any]:
    """Return ``summary`` degraded to fit ``cap_bytes``; never raises.

    Degradation order: oldest action detail first (counted), then
    action detail text, constraint text and work-item titles, check
    evidence, then per-entry detail (status-only entries), then whole
    map entries (counted). Decisions and dead ends are the protected
    record: they are trimmed in place, never evicted, ahead of any
    map entry being dropped.
    """
    summary = copy.deepcopy(summary)
    summary["dropped_actions"] = int(summary.get("dropped_actions", 0))
    summary["dropped_entries"] = int(summary.get("dropped_entries", 0))

    # Usage defects are bounded before anything else is measured:
    # each entry truncated, the list capped, the excess counted.
    # The fold applies the same bounds, so this step only bites on a
    # summary that was built or loaded without them.
    defects = summary.get("usage_defects")
    if isinstance(defects, list) and defects:
        bounded = []
        for defect in defects[:MAX_USAGE_DEFECTS]:
            text = str(defect)
            if len(text) > FOLD_TEXT_CHARS:
                text = text[:FOLD_TEXT_CHARS] + DEFECT_TRUNCATED_SUFFIX
            bounded.append(text)
        summary["dropped_entries"] += len(defects) - len(bounded)
        summary["usage_defects"] = bounded

    def size() -> int:
        return len(json.dumps(summary, ensure_ascii=False,
                              sort_keys=True).encode("utf-8"))

    while size() > cap_bytes and summary["recent_actions"]:
        summary["recent_actions"].pop(0)
        summary["dropped_actions"] += 1
    if size() > cap_bytes:
        for entry in summary["recent_actions"]:
            entry["detail"] = str(entry.get("detail", ""))[:80]
    if size() > cap_bytes:
        for entry in summary["constraints"].values():
            entry["text"] = str(entry.get("text", ""))[:120]
            if "source_span" in entry:
                entry["source_span"] = _truncate_span(entry["source_span"])
        for entry in summary["work_items"].values():
            entry["title"] = str(entry.get("title", ""))[:120]
    if size() > cap_bytes:
        checks = summary.get("checks") or {}
        for section in ("constraints", "consistency"):
            for entry in (checks.get(section) or {}).values():
                if isinstance(entry, dict) and "evidence" in entry:
                    entry["evidence"] = str(entry["evidence"])[:80]
    if size() > cap_bytes:
        for key in ("decisions", "dead_ends"):
            summary[key] = [str(text)[:120] for text in summary[key]]
    if size() > cap_bytes:
        summary["deliverables"] = {
            path: {"status": entry.get("status")}
            for path, entry in summary["deliverables"].items()}
        summary["constraints"] = {
            cid: {"status": entry.get("status"), "hard": entry.get("hard")}
            for cid, entry in summary["constraints"].items()}
        summary["work_items"] = {
            wid: {"status": entry.get("status")}
            for wid, entry in summary["work_items"].items()}
        summary["checks"] = {}
    for key in ("deliverables", "constraints", "work_items"):
        while size() > cap_bytes and summary[key]:
            summary[key].pop(next(iter(summary[key])))
            summary["dropped_entries"] += 1
    if size() > cap_bytes:
        for key in ("decisions", "dead_ends"):
            summary[key] = [str(text)[:32] for text in summary[key]]
    while size() > cap_bytes and (summary["decisions"]
                                  or summary["dead_ends"]):
        # Absolute last resort, reachable only with tens of thousands
        # of protected entries: trim the oldest protected entries and
        # count them, rather than raising out of the loop.
        if summary["dead_ends"]:
            summary["dead_ends"].pop(0)
        else:
            summary["decisions"].pop(0)
        summary["dropped_entries"] += 1
    return summary


_LEDGER_KINDS = frozenset({
    "usage", "turn", "phase", "action", "deliverable", "constraint",
    "work_item", "checks", "decision", "dead_end", "background_stop",
    "mcp_discovery",
})

_USAGE_COUNT_KEYS = ("input_tokens", "output_tokens",
                     "cache_read_tokens", "cache_write_tokens")


def _int_coercible(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _ledger_event_error(event: Any) -> Optional[str]:
    """Why a ledger line cannot be folded, or None when it can.

    Rebuild and crash recovery validate before folding: a line that
    parses as JSON but is semantically wrong — an unknown kind, a
    wrong-typed numeric field — must be rejected whole. Folding it
    until the bad field raises would leave a half-applied event in
    the summary (the valid fields already added, the event counted
    as torn).
    """
    if not isinstance(event, dict):
        return "ledger line is not a JSON object"
    kind = event.get("kind")
    if kind not in _LEDGER_KINDS:
        return f"unknown ledger event kind: {kind!r}"
    seq = event.get("seq")
    if seq is not None and (not isinstance(seq, int)
                            or isinstance(seq, bool)):
        return "ledger event seq is not an integer"
    if kind == "usage":
        usage = event.get("usage")
        if usage is not None:
            if not isinstance(usage, Mapping):
                return "usage event carries no usage object"
            for key in _USAGE_COUNT_KEYS:
                if key in usage and not _int_coercible(usage[key]):
                    return f"usage count {key} is not a number"
        defects = event.get("defects")
        if defects is not None and not isinstance(defects, (list, tuple)):
            return "usage defects are not a list"
    elif kind == "turn":
        if "turn" in event and not _int_coercible(event["turn"]):
            return "turn count is not a number"
    elif kind == "background_stop":
        if "stopped" in event and not _int_coercible(event["stopped"]):
            return "background stop count is not a number"
    return None


class LedgerState:
    """Append-only ledger plus the summary generated from it.

    The summary is maintained incrementally as events are appended
    and can be rebuilt deterministically from the ledger alone
    (:meth:`rebuild_summary`). Compaction enforces the 64 KiB cap by
    dropping the oldest action detail first.
    """

    def __init__(self, state_dir: Any, task_id: str) -> None:
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.task_id = task_id
        self.ledger_path = self.dir / "ledger.jsonl"
        self.summary_path = self.dir / "summary.json"
        self.seq = 0
        self.summary: dict[str, Any] = _empty_summary(task_id)
        # Sequence numbers this fold has already applied (the
        # duplicate guard) and the defect record of the most recent
        # rebuild: how many ledger lines were torn and how many
        # events were skipped as duplicates.
        self._applied_seqs: set[int] = set()
        self.rebuild_defects: dict[str, int] = {
            "torn_lines": 0, "duplicate_events": 0}
        summary_loaded = False
        foreign_summary = False
        if self.summary_path.exists():
            try:
                loaded = json.loads(self.summary_path.read_text("utf-8"))
                if isinstance(loaded, dict) and \
                        loaded.get("task_id") == task_id:
                    loaded_seq = int(loaded.get("updated_seq", 0))
                    self.summary = loaded
                    self.seq = loaded_seq
                    for key in ("duplicate_events", "background_stopped"):
                        self.summary.setdefault(key, 0)
                    summary_loaded = True
                elif isinstance(loaded, dict):
                    # The summary on disk belongs to another task:
                    # this state dir is not this task's to recover.
                    foreign_summary = True
            except (ValueError, OSError):
                pass
        if not foreign_summary:
            self._recover_ledger_tail(summary_loaded)

    def _recover_ledger_tail(self, summary_loaded: bool) -> None:
        """Fold ledger events the persisted summary never saw.

        Appends land in the ledger immediately; the summary persists
        later. A crash inside that window leaves events in the
        ledger that the resumed summary does not reflect — and if
        the next sequence were seeded from the stale summary alone,
        the resumed state would re-issue sequence numbers already
        present in the ledger, and a later rebuild would drop the
        new events as duplicates. Recovery scans the ledger
        (tolerating torn and invalid lines), folds every well-formed
        event past the persisted summary's position — or the whole
        ledger when no usable summary exists — and seeds the next
        sequence from the highest sequence actually present.
        """
        if not self.ledger_path.exists():
            return
        try:
            text = self.ledger_path.read_bytes().decode(
                "utf-8", errors="replace")
        except OSError:
            return
        base_seq = self.seq if summary_loaded else 0
        seen: set[int] = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if _ledger_event_error(event) is not None:
                continue
            seq = event.get("seq")
            seq = seq if isinstance(seq, int) and \
                not isinstance(seq, bool) else None
            if seq is not None:
                seen.add(seq)
            if summary_loaded:
                if seq is not None and seq > base_seq:
                    self._fold(event)
            else:
                self._fold(event)
        if seen:
            self._applied_seqs |= seen
            self.seq = max(self.seq, max(seen))
            self.summary["updated_seq"] = self.seq

    # -- ledger ------------------------------------------------------

    def append_event(self, kind: str, **fields: Any) -> dict[str, Any]:
        self.seq += 1
        event = {"seq": self.seq, "kind": kind, **fields}
        with open(self.ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False,
                                    default=str) + "\n")
        self._fold(event)
        self.summary["updated_seq"] = self.seq
        return event

    def append_usage(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Ledger entry point used by the model client."""
        return self.append_event(
            "usage", phase=record.get("phase", "execute"),
            model=record.get("model"),
            usage={k: int(record.get(k, 0)) for k in self.summary["usage"]},
            defects=[str(d) for d in (record.get("defects") or [])])

    def _fold(self, event: Mapping[str, Any]) -> bool:
        """Fold one event into the summary; False when it is skipped.

        An event carrying a sequence number is applied at most once:
        a replayed sequence is counted in ``duplicate_events`` and
        skipped, so replaying part of the ledger cannot double-count
        the record. Events without a sequence number (written before
        appends assigned sequences) always apply, in order.
        """
        summary = self.summary
        seq = event.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            if seq in self._applied_seqs:
                summary["duplicate_events"] = int(
                    summary.get("duplicate_events", 0)) + 1
                return False
            self._applied_seqs.add(seq)
        kind = event.get("kind")
        if kind == "usage":
            usage = event.get("usage") or {}
            for key in summary["usage"]:
                summary["usage"][key] += int(usage.get(key, 0))
            summary["model_calls"] += 1
            defects = summary.setdefault("usage_defects", [])
            for defect in event.get("defects") or []:
                text = str(defect)
                if len(text) > FOLD_TEXT_CHARS:
                    text = text[:FOLD_TEXT_CHARS] + DEFECT_TRUNCATED_SUFFIX
                if len(defects) < MAX_USAGE_DEFECTS:
                    defects.append(text)
                else:
                    summary["dropped_entries"] = int(
                        summary.get("dropped_entries", 0)) + 1
        elif kind == "turn":
            summary["turn_count"] = int(event.get("turn", summary["turn_count"]))
        elif kind == "phase":
            summary["phase"] = str(event.get("phase", summary["phase"]))
        elif kind == "action":
            summary["recent_actions"].append({
                "seq": event.get("seq"),
                "tool": event.get("tool"),
                "target": event.get("target"),
                "result_class": event.get("result_class"),
                "detail": str(event.get("detail", ""))[:500],
            })
        elif kind == "deliverable":
            entry = summary["deliverables"].setdefault(
                str(event.get("path")), {})
            for key in ("status", "sha256", "size"):
                if event.get(key) is not None:
                    entry[key] = event[key]
        elif kind == "constraint":
            entry = summary["constraints"].setdefault(
                str(event.get("id")), {})
            for key in ("text", "hard", "status", "source_span"):
                if event.get(key) is not None:
                    value = event[key]
                    if key == "text":
                        value = str(value)[:FOLD_TEXT_CHARS]
                    elif key == "source_span":
                        value = _truncate_span(value)
                    entry[key] = value
        elif kind == "work_item":
            entry = summary["work_items"].setdefault(
                str(event.get("id")), {})
            for key in ("title", "status", "reason", "acceptance_status"):
                if event.get(key) is not None:
                    value = event[key]
                    if key == "title":
                        value = str(value)[:FOLD_TEXT_CHARS]
                    entry[key] = value
        elif kind == "checks":
            summary["checks"] = event.get("checks") or {}
            summary["all_hard_checks_pass"] = bool(
                event.get("all_hard_checks_pass"))
        elif kind == "decision":
            summary["decisions"].append(str(event.get("text", ""))[:500])
        elif kind == "dead_end":
            summary["dead_ends"].append(str(event.get("text", ""))[:500])
        elif kind == "background_stop":
            summary["background_stopped"] = int(
                summary.get("background_stopped", 0)) + int(
                    event.get("stopped") or 0)
        return True

    def rebuild_summary(self) -> dict[str, Any]:
        """Regenerate the summary from the ledger alone.

        The fold starts from a genuinely empty summary — nothing is
        seeded from the in-memory summary — and the result passes
        through the same compaction as :meth:`persist`, so a rebuild
        of an uncompacted fold equals the persisted summary exactly.

        Recovery never raises on file content. Each line is parsed
        on its own: a malformed line — above all the torn final
        line a crash mid-append leaves behind — is skipped, the
        well-formed lines before and after it still fold, and the
        skip is recorded in :attr:`rebuild_defects` (the count of
        torn lines, alongside the duplicate count the fold itself
        records in the summary). A line that parses but fails field
        validation (see :func:`_ledger_event_error`) is counted
        torn and not folded at all, so a half-valid event cannot
        leave phantom counts in the rebuilt summary. Replayed
        events are applied at most once, by sequence number (see
        :meth:`_fold`).
        """
        fresh = LedgerState.__new__(LedgerState)
        fresh.dir, fresh.task_id = self.dir, self.task_id
        fresh.ledger_path, fresh.summary_path = self.ledger_path, self.summary_path
        fresh.seq = 0
        fresh.summary = _empty_summary(self.task_id)
        fresh._applied_seqs = set()
        fresh.rebuild_defects = {"torn_lines": 0, "duplicate_events": 0}
        torn = 0
        if self.ledger_path.exists():
            try:
                text = self.ledger_path.read_bytes().decode(
                    "utf-8", errors="replace")
            except OSError:
                text = ""
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    torn += 1
                    continue
                if _ledger_event_error(event) is not None:
                    torn += 1
                    continue
                seq = event.get("seq")
                fresh._fold(event)
                if isinstance(seq, int) and not isinstance(seq, bool):
                    fresh.seq = max(fresh.seq, seq)
                fresh.summary["updated_seq"] = fresh.seq
        self.rebuild_defects = {
            "torn_lines": torn,
            "duplicate_events": int(
                fresh.summary.get("duplicate_events", 0))}
        return _compact_summary_dict(fresh.summary)

    # -- compaction ------------------------------------------------------

    def compact_summary(self, cap_bytes: int = SUMMARY_CAP_BYTES) -> dict[str, Any]:
        """The closing summary, capped; degrades, never raises."""
        return _compact_summary_dict(self.summary, cap_bytes)

    def persist(self) -> dict[str, Any]:
        compact = self.compact_summary()
        _atomic_write_json(self.summary_path, compact)
        return compact

    def closing_summary_text(self) -> str:
        """Defined-structure closing summary, within the 64 KiB cap."""
        try:
            return self._closing_summary_text()
        except Exception:
            return (f"Task {self.task_id}: closing summary unavailable; "
                    "see the run record.")

    def _closing_summary_text(self) -> str:
        summary = self.compact_summary()
        lines = [f"Task {summary['task_id']}: phase {summary['phase']}, "
                 f"{summary['turn_count']} turns, "
                 f"{summary['model_calls']} model calls."]
        lines.append("Deliverables:")
        for path, entry in sorted(summary["deliverables"].items()):
            lines.append(f"- {path}: {entry.get('status', 'unknown')}"
                         f" ({entry.get('size', '?')} bytes)")
        if summary["constraints"]:
            lines.append("Hard-constraint outcomes:")
            for cid, entry in sorted(summary["constraints"].items()):
                lines.append(f"- {cid}: {entry.get('status', 'unknown')}"
                             f" — {entry.get('text', '')}")
        lines.append("All hard checks pass: "
                     + ("yes" if summary["all_hard_checks_pass"] else "no"))
        for text in summary["decisions"][-10:]:
            lines.append(f"Decision: {text}")
        text = "\n".join(lines)
        encoded = text.encode("utf-8")
        if len(encoded) > SUMMARY_CAP_BYTES:
            text = encoded[:SUMMARY_CAP_BYTES].decode("utf-8", "ignore")
        return text


# ---------------------------------------------------------------------
# Deliverable lifecycle
# ---------------------------------------------------------------------

def _slug(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


class DeliverableStore:
    """candidate -> complete -> verified, with atomic promotion.

    Candidate content is staged in state. Promotion writes the new
    verified copy into place by atomic rename only after the previous
    verified copy has itself been preserved, so a promotion can never
    destroy the last passing version. Finalize restores the verified
    copy over a graded path that no longer passes.
    """

    def __init__(self, state: LedgerState, tools: ToolDispatcher) -> None:
        self.state = state
        self.tools = tools
        self.root = state.dir / "deliverables"
        self.root.mkdir(parents=True, exist_ok=True)

    def _slot(self, path: str, kind: str) -> Path:
        return self.root / f"{_slug(path)}.{kind}"

    def status(self, path: str) -> str:
        entry = self.state.summary["deliverables"].get(path) or {}
        return str(entry.get("status", "unknown"))

    def stage_candidate(self, path: str, data: bytes) -> dict[str, Any]:
        _atomic_write(self._slot(path, "candidate"), data)
        info = {"status": "candidate",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data)}
        self.state.append_event("deliverable", path=path, **info)
        return info

    def write_candidate(self, path: str, data: bytes) -> dict[str, Any]:
        """Stage ``data`` and write it to the graded path as candidate."""
        info = self.stage_candidate(path, data)
        self.tools.write_bytes(path, data, append=False)
        return info

    def read_graded(self, path: str) -> Optional[bytes]:
        try:
            data, _ = self.tools.read_bytes(path)
            return data
        except Exception as exc:
            if _is_session_cancelled(exc):
                raise
            return None

    def sync_from_graded(self, path: str) -> Optional[dict[str, Any]]:
        data = self.read_graded(path)
        if data is None:
            return None
        if self.status(path) == "verified":
            verified = self._slot(path, "verified")
            if verified.exists() and \
                    hashlib.sha256(data).hexdigest() == \
                    hashlib.sha256(verified.read_bytes()).hexdigest():
                return {"status": "verified",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data)}
        return self.stage_candidate(path, data)

    def mark_complete(self, path: str) -> None:
        if self.status(path) == "candidate":
            self.state.append_event("deliverable", path=path,
                                    status="complete")

    def promote(self, path: str) -> bool:
        """Promote the current graded content to verified, atomically.

        The previous verified copy is preserved first; the state
        pointer (the summary) is updated only after both files are in
        place.
        """
        data = self.read_graded(path)
        if data is None:
            return False
        verified = self._slot(path, "verified")
        previous = self._slot(path, "previous")
        if verified.exists():
            _atomic_write(previous, verified.read_bytes())
        _atomic_write(verified, data)
        self.state.append_event(
            "deliverable", path=path, status="verified",
            sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        self.state.persist()
        return True

    def previous_verified(self, path: str) -> Optional[bytes]:
        previous = self._slot(path, "previous")
        return previous.read_bytes() if previous.exists() else None

    def verified_copy(self, path: str) -> Optional[bytes]:
        verified = self._slot(path, "verified")
        return verified.read_bytes() if verified.exists() else None

    def restore_verified(self, path: str) -> bool:
        data = self.verified_copy(path)
        if data is None:
            return False
        self.tools.write_bytes(path, data, append=False)
        self.state.append_event(
            "deliverable", path=path, status="verified",
            sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        return True


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def all_hard_checks_pass(checks: Mapping[str, Any]) -> bool:
    """True only when every constraint is resolved and passing.

    An UNRESOLVED constraint blocks the claim whatever its declared
    severity, as does any hard constraint not in PASS. The claim is
    never vacuous: an empty constraint set is not a pass, and the
    consistency section must be clean as well.
    """
    constraints = checks.get("constraints") or {}
    if not constraints:
        return False
    for entry in constraints.values():
        status = entry.get("status")
        if status == "UNRESOLVED":
            return False
        if entry.get("hard") and status != "PASS":
            return False
    for entry in (checks.get("consistency") or {}).values():
        if entry.get("status") != "PASS":
            return False
    return True


def deliverable_checks_pass(checks: Mapping[str, Any],
                            brief: Mapping[str, Any], path: str) -> bool:
    """The per-deliverable promotion gate.

    True only when at least one constraint was genuinely checked and
    every constraint applicable to ``path`` is PASS, and every
    consistency item is PASS. Constraints that name a deliverable
    apply to that deliverable; when none name it, the whole list
    applies. An empty constraint set can never promote.
    """
    constraints = brief.get("constraints") or []
    if not constraints:
        return False
    applicable = [c for c in constraints if c.get("deliverable") == path]
    if not applicable:
        applicable = list(constraints)
    results = checks.get("constraints") or {}
    for constraint in applicable:
        entry = results.get(str(constraint.get("id")))
        if entry is None or entry.get("status") != "PASS":
            return False
    for entry in (checks.get("consistency") or {}).values():
        if entry.get("status") != "PASS":
            return False
    return True


class Verifier:
    """Deterministic, constraint-first checks. No model calls."""

    def __init__(self, state: LedgerState, tools: ToolDispatcher,
                 store: DeliverableStore, workdir: Path) -> None:
        self.state = state
        self.tools = tools
        self.store = store
        self.workdir = workdir

    def deliverable_exists(self, path: str) -> bool:
        local = Path(path) if os.path.isabs(path) else self.workdir / path
        if self.tools.request is None:
            return local.is_file() and not local.is_symlink()
        return self.store.read_graded(path) is not None

    def run_check(self, spec: Mapping[str, Any]) -> tuple[str, str]:
        """Run one check spec; returns (status, evidence)."""
        kind = spec.get("type")
        try:
            if kind == "file_exists":
                ok = self.deliverable_exists(str(spec["path"]))
                return ("PASS" if ok else "FAIL",
                        f"file_exists {spec['path']}: {ok}")
            if kind == "file_contains":
                data = self.store.read_graded(str(spec["path"]))
                ok = data is not None and \
                    str(spec["text"]).encode() in data
                return ("PASS" if ok else "FAIL",
                        f"file_contains {spec['path']}")
            if kind == "file_min_size":
                data = self.store.read_graded(str(spec["path"]))
                ok = data is not None and len(data) >= int(spec["min_bytes"])
                return ("PASS" if ok else "FAIL",
                        f"file_min_size {spec['path']}")
            if kind == "file_sha256":
                data = self.store.read_graded(str(spec["path"]))
                ok = data is not None and \
                    hashlib.sha256(data).hexdigest() == spec["sha256"]
                return ("PASS" if ok else "FAIL", f"file_sha256 {spec['path']}")
            if kind == "command":
                result = self.tools.exec(str(spec["command"]),
                                         int(spec.get("timeout_s", 120)))
                expected = int(spec.get("expect_exit_code", 0))
                data = result.get("data") or {}
                ok = data.get("exit_code") == expected
                return ("PASS" if ok else "FAIL",
                        f"command exit {data.get('exit_code')}")
            return ("UNRESOLVED", f"unknown check type: {kind}")
        except Exception as exc:
            return ("FAIL", f"{kind}: {type(exc).__name__}: {exc}")

    def verify(self, brief: Mapping[str, Any]) -> dict[str, Any]:
        checks: dict[str, Any] = {"deliverables": {}, "constraints": {},
                                  "consistency": {}}
        for item in brief.get("deliverables") or []:
            path = item["path"] if isinstance(item, Mapping) else str(item)
            checks["deliverables"][path] = {
                "exists": self.deliverable_exists(path)}
        seen_ids: dict[str, int] = {}
        for constraint in brief.get("constraints") or []:
            cid = str(constraint.get("id"))
            if cid in seen_ids:
                # A duplicate id is rejected, never collapsed into the
                # first entry: it is recorded under its own key as
                # UNRESOLVED so it blocks any all-pass claim.
                seen_ids[cid] += 1
                dup_key = f"{cid}#dup{seen_ids[cid]}"
                checks["constraints"][dup_key] = {
                    "status": "UNRESOLVED",
                    "hard": bool(constraint.get("hard")),
                    "evidence": "duplicate constraint id rejected",
                    "source_span": constraint.get("source_span"),
                }
                self.state.append_event(
                    "constraint", id=dup_key, status="UNRESOLVED",
                    hard=bool(constraint.get("hard")),
                    source_span=constraint.get("source_span"))
                continue
            seen_ids[cid] = 0
            spec = constraint.get("check")
            if not isinstance(spec, Mapping):
                status, evidence = "UNRESOLVED", "no check defined"
            else:
                status, evidence = self.run_check(spec)
            if constraint.get("status") == "UNRESOLVED" and status == "PASS":
                status = "UNRESOLVED"
                evidence += "; constraint itself is UNRESOLVED"
            checks["constraints"][cid] = {
                "status": status,
                "hard": bool(constraint.get("hard")),
                "evidence": evidence,
                "source_span": constraint.get("source_span"),
            }
            self.state.append_event(
                "constraint", id=cid, status=status,
                hard=bool(constraint.get("hard")),
                source_span=constraint.get("source_span"))
        seen_item_ids: dict[str, int] = {}
        for item in brief.get("consistency_items") or []:
            iid = str(item.get("id")) if isinstance(item, Mapping) \
                else str(item)
            if iid in seen_item_ids:
                # The same duplicate treatment as constraints above:
                # a repeated id is recorded under its own key as
                # UNRESOLVED, never collapsed into the first entry —
                # collapsing would let a later PASS hide an earlier
                # FAIL behind the one shared id, and both all-pass
                # gating functions read this section.
                seen_item_ids[iid] += 1
                checks["consistency"][f"{iid}#dup{seen_item_ids[iid]}"] = {
                    "status": "UNRESOLVED",
                    "evidence": "duplicate consistency item id rejected"}
                continue
            seen_item_ids[iid] = 0
            spec = item.get("check") if isinstance(item, Mapping) else None
            if isinstance(spec, Mapping):
                status, evidence = self.run_check(spec)
            else:
                status, evidence = "UNRESOLVED", "no check defined"
            checks["consistency"][iid] = {
                "status": status, "evidence": evidence}
        checks["hard_failure_count"] = sum(
            1 for entry in checks["constraints"].values()
            if entry.get("hard") and entry.get("status") == "FAIL")
        checks["hard_unresolved_count"] = sum(
            1 for entry in checks["constraints"].values()
            if entry.get("hard") and entry.get("status") == "UNRESOLVED")
        checks["all_hard_checks_pass"] = all_hard_checks_pass(checks)
        self.state.append_event(
            "checks", checks=checks,
            all_hard_checks_pass=checks["all_hard_checks_pass"])
        return checks


def _run_acceptance(verifier: Verifier,
                    item: Mapping[str, Any]) -> tuple[bool, bool, str]:
    """Run a work item's acceptance check(s).

    Returns (ran, passed, evidence). An item with no acceptance spec
    has nothing to run; a spec (or list of specs) passes only when
    every check is PASS.
    """
    spec = item.get("acceptance")
    specs: list[Mapping[str, Any]] = []
    if isinstance(spec, Mapping):
        specs = [spec]
    elif isinstance(spec, (list, tuple)):
        specs = [s for s in spec if isinstance(s, Mapping)]
    if not specs:
        return (False, True, "no acceptance check defined")
    evidence: list[str] = []
    for one in specs:
        status, why = verifier.run_check(one)
        evidence.append(why)
        if status != "PASS":
            return (True, False, "; ".join(evidence))
    return (True, True, "; ".join(evidence))


# ---------------------------------------------------------------------
# Brief handling
# ---------------------------------------------------------------------

def _find_span(instruction: str, quote: str) -> Optional[dict[str, Any]]:
    if not quote:
        return None
    index = instruction.find(quote)
    if index < 0:
        return None
    return {"source": "instruction", "start": index,
            "end": index + len(quote), "quote": quote}


def _validate_span(span: Any, instruction: str) -> Optional[dict[str, Any]]:
    """Dereference a claimed source span against the instruction.

    Returns a verified span dict, or None when the claim cannot be
    found in the instruction text — an unverifiable span is no
    provenance at all, and the constraint lands in UNRESOLVED.
    """
    if isinstance(span, str):
        return _find_span(instruction, span)
    if isinstance(span, Mapping):
        quote = span.get("quote")
        if isinstance(quote, str) and quote:
            return _find_span(instruction, quote)
        start, end = span.get("start"), span.get("end")
        if (isinstance(start, int) and not isinstance(start, bool)
                and isinstance(end, int) and not isinstance(end, bool)
                and 0 <= start < end <= len(instruction)):
            return {"source": "instruction", "start": start, "end": end,
                    "quote": instruction[start:end]}
        source = span.get("source")
        if isinstance(source, str) and source and source != "instruction":
            return _find_span(instruction, source)
    return None


def normalise_constraint(raw: Any, instruction: str,
                         index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        raw = {"text": raw}
    raw = dict(raw or {})
    text = str(raw.get("text") or raw.get("description") or "")
    cid = str(raw.get("id") or f"c{index + 1}")
    severity = raw.get("severity") or raw.get("kind")
    if raw.get("hard") is not None:
        hard = bool(raw.get("hard"))
        classified = True
    elif isinstance(severity, str) and severity.lower() in ("hard", "soft"):
        hard = severity.lower() == "hard"
        classified = True
    else:
        hard = True  # unclassified counts as hard
        classified = False
    span = raw.get("source_span") or raw.get("provenance") or raw.get("source")
    if span is not None:
        span = _validate_span(span, instruction)
    if span is None:
        span = _find_span(instruction, text)
    check = raw.get("check") if isinstance(raw.get("check"), Mapping) else None
    status = "PENDING"
    if not classified or check is None or span is None:
        status = "UNRESOLVED"
    return {"id": cid, "text": text, "hard": hard, "classified": classified,
            "source_span": span, "check": check, "status": status,
            "deliverable": raw.get("deliverable") or raw.get("path")}


def normalise_brief(raw: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    instruction = str(task.get("instruction") or "")
    deliverables: list[dict[str, Any]] = []
    declared = (task.get("deliverables") or task.get("declared_deliverables")
                or task.get("deliverable_paths") or raw.get("deliverables")
                or [])
    for item in declared:
        if isinstance(item, Mapping):
            deliverables.append({"path": str(item.get("path") or item.get("name")),
                                 "format": item.get("format")})
        elif item:
            deliverables.append({"path": str(item), "format": None})
    constraints_raw = (task.get("constraints") or raw.get("constraints") or [])
    constraints = [normalise_constraint(c, instruction, i)
                   for i, c in enumerate(constraints_raw)]
    seen_ids: dict[str, int] = {}
    for constraint in constraints:
        cid = constraint["id"]
        if cid in seen_ids:
            # Duplicate ids are rejected, never collapsed: the later
            # entry keeps its own distinct key and is UNRESOLVED.
            seen_ids[cid] += 1
            constraint["id"] = f"{cid}#dup{seen_ids[cid]}"
            constraint["status"] = "UNRESOLVED"
        else:
            seen_ids[cid] = 0
    work_items: list[dict[str, Any]] = []
    items_raw = (task.get("work_items") or raw.get("work_items")
                 or raw.get("plan") or [])
    for i, item in enumerate(items_raw):
        if isinstance(item, Mapping):
            work_items.append({
                "id": str(item.get("id") or f"w{i + 1}"),
                "title": str(item.get("title") or item.get("text") or ""),
                "status": "pending",
                "acceptance": item.get("acceptance") or item.get("check"),
                "deliverable": item.get("deliverable") or item.get("path")})
        elif item:
            work_items.append({"id": f"w{i + 1}", "title": str(item),
                               "status": "pending", "acceptance": None,
                               "deliverable": None})
    if not work_items:
        for i, entry in enumerate(deliverables):
            work_items.append({"id": f"w{i + 1}",
                               "title": f"Produce {entry['path']}",
                               "status": "pending", "acceptance": None,
                               "deliverable": entry["path"]})
    consistency = (task.get("consistency_items")
                   or raw.get("consistency_items") or [])
    return {"deliverables": deliverables, "constraints": constraints,
            "work_items": work_items,
            "consistency_items": [dict(c) if isinstance(c, Mapping)
                                   else {"id": f"x{i + 1}", "check": None}
                                   for i, c in enumerate(consistency)],
            "unknowns": raw.get("unknowns") or []}


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except ValueError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(stripped[start:end + 1])
                return value if isinstance(value, dict) else None
            except ValueError:
                return None
    return None


def _placeholder_for(path: str) -> bytes:
    lowered = path.lower()
    if lowered.endswith(".json"):
        return b'{"status": "candidate"}\n'
    if lowered.endswith(".csv"):
        return b"item,value\n"
    if lowered.endswith((".md", ".txt")):
        return ("# Candidate deliverable\n\n"
                "This file is a complete, submittable first version. "
                "It is refined as the work proceeds.\n").encode()
    if lowered.endswith((".yaml", ".yml")):
        return b"status: candidate\n"
    if lowered.endswith(".html"):
        return (b"<!DOCTYPE html>\n<html><body>"
                b"<p>Candidate deliverable.</p></body></html>\n")
    return b"Candidate deliverable.\n"


# ---------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert engineer working on a task in a Linux container "
    "with no internet access. Your tools act inside it. Read the task "
    "and its files before starting, compute with code rather than by "
    "hand, write each deliverable exactly where and in the format the "
    "task asks, and check what you wrote. A first complete version of "
    "every deliverable is written early; later turns improve it."
)


def _write_closing_summary(state: LedgerState, state_dir: Path) -> None:
    """Writes the closing summary where the session layer reads it.

    serve.py's cancel fallback reads <state_dir>/closing_summary.md;
    the loop keeps that file current with the same text it returns
    as the outcome's final_message.
    """
    try:
        path = state_dir / "closing_summary.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(state.closing_summary_text(), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def run_harness(task: Any, emit: Optional[Callable[[dict[str, Any]], Any]],
                workdir: Any, config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Run one task and return its outcome dict. Sends no terminal frame."""
    cfg = dict(config or {})
    emit_fn = emit or (lambda frame: None)
    if isinstance(task, str):
        task = {"instruction": task}
    task = dict(task or {})
    instruction = str(task.get("instruction") or "")
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    task_id = str(task.get("session") or task.get("id")
                  or hashlib.sha256(instruction.encode()).hexdigest()[:16])
    state_dir = Path(cfg.get("state_dir")
                     or root / ".ple-state" / task_id)
    state = LedgerState(state_dir, task_id)

    governor = BudgetGovernor(
        total_tokens=int(cfg.get("token_cap", 1_000_000)),
        reserve_tokens=cfg.get("reserve_tokens"),
        reserve_fraction=float(cfg.get("reserve_fraction", 0.25)),
        allocations=cfg.get("allocations"),
        total_calls=int(cfg.get("max_model_calls", 200)),
        reserve_calls=int(cfg.get("reserve_calls", 4)))
    estimated_call = int(cfg.get("estimated_call_tokens", 1_000))

    request = cfg.get("request") or cfg.get("transport")
    if request is None and cfg.get("emit_is_transport"):
        request = emit_fn
    tools = ToolDispatcher(root, request=request, config={
        **cfg,
        "mcp_servers": task.get("mcp_servers") or cfg.get("mcp_servers") or [],
    })
    store = DeliverableStore(state, tools)

    client = cfg.get("model_client") or cfg.get("model")
    usage_recorded_by_client = client is not None and not callable(client)
    if client is None:
        client = ModelClient(ledger=state, emit=emit_fn, config=cfg)
        usage_recorded_by_client = True
    elif callable(client):
        call_fn = client
        client = _CallableClient(call_fn)

    admission = int(cfg.get("worst_case_call_tokens") or estimated_call)

    def call_model(phase: str, messages: list[dict[str, Any]],
                   tool_defs: Optional[list] = None) -> Any:
        if not governor.can_call(phase):
            raise BudgetExceeded(f"{phase}: model-call reserve reached")
        if not governor.can_spend(phase, admission):
            raise BudgetExceeded(f"{phase}: token allocation reached")
        governor.record_call(phase)
        try:
            response = client.complete(messages, tools=tool_defs, phase=phase)
        except ModelCallInterrupted as exc:
            response = exc.partial
            state.append_event("dead_end", text=f"interrupted call: {exc}")
        usage = getattr(response, "usage", None) or {}
        counts, defects = normalize_usage(usage)
        for defect in getattr(response, "defects", None) or []:
            if defect not in defects:
                defects.append(defect)
        if not getattr(response, "usage_recorded", usage_recorded_by_client):
            state.append_usage({"phase": phase,
                                "model": getattr(response, "model", None),
                                "defects": defects,
                                **counts})
        if not governor.record_spend(phase, total_tokens(counts)):
            # The call's actuals broke the budget: the governor has
            # charged only up to the protected ceilings and recorded
            # the excess as an overrun. Honour the signal — the phase
            # stops here instead of spending the reserve.
            state.append_event(
                "dead_end",
                text=f"{phase}: call actuals exceeded the admission "
                     "bound; overrun recorded, phase stopped")
            raise BudgetExceeded(
                f"{phase}: actual spend exceeded its allocation or "
                "the protected reserve")
        return response

    def finish_background() -> int:
        """Stop every background job this run started; never raises.

        Called on every exit path out of the loop, before the
        closing summary is finalised, so no job outlives the run
        and the summary records how many were stopped. A failure
        inside the stop is swallowed: it must never mask the run's
        own outcome.
        """
        try:
            stopped = int(tools.stop_all_background())
        except Exception:
            return 0
        if stopped:
            try:
                state.append_event("background_stop", stopped=stopped)
            except Exception:
                pass
        return stopped

    outcome: dict[str, Any] = {"task_id": task_id, "status": "started",
                               "terminal_frame_sent": False}
    try:
        # ---- inventory (no model call) ----
        inventory = cfg.get("inventory")
        if inventory is None:
            inventory = _inventory(root, tools)
        state.append_event("phase", phase="brief")

        # ---- MCP discovery (Brief time, no model call) ----
        # Each named server's catalogue is paged in full and indexed
        # on disk in the task state dir; a server that fails
        # discovery is quarantined with its reason and its tools are
        # not offered. Recall counts (advertised vs indexed vs
        # callable) are recorded in the ledger and the outcome,
        # separately from whether the tools then succeed at calls.
        discovery_recall: dict[str, Any] = {
            "servers": {},
            "totals": {"advertised": 0, "indexed": 0, "callable": 0}}
        catalogue_lines: list[str] = []
        if tools.mcp.servers:
            try:
                discovery_recall = tools.mcp.discover_all(
                    state_dir / "mcp_catalogue.json")
                catalogue_lines = tools.mcp.catalogue_context_lines()
            except Exception as exc:
                if _is_session_cancelled(exc):
                    raise
                state.append_event(
                    "dead_end", text=f"MCP discovery failed: {exc}")
            for server, entry in (discovery_recall.get("servers")
                                  or {}).items():
                state.append_event(
                    "mcp_discovery", server=server,
                    status=entry.get("status"),
                    reason=entry.get("reason"),
                    advertised=entry.get("advertised"),
                    indexed=entry.get("indexed"),
                    callable=entry.get("callable"),
                    quarantined_tools=entry.get("quarantined_tools"))
        outcome["discovery_recall"] = discovery_recall

        # ---- Brief ----
        brief_raw: dict[str, Any] = {}
        brief_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
             "Write the brief for this task as one JSON object with keys "
             "deliverables (list of paths), constraints (list of objects "
             "with text, hard true/false, a source_span quoting the "
             "instruction span each came from, and a check), work_items, "
             "and unknowns.\n\nInstruction:\n" + instruction
             + "\n\nInventory:\n" + json.dumps(inventory, default=str)[:6000]
             + ("\n\nMCP tool catalogue (names and descriptions; use "
                "find_tools during Execute to get a tool's schema "
                "before calling it):\n" + "\n".join(catalogue_lines)
                if catalogue_lines else "")},
        ]
        try:
            reply = call_model("brief", brief_messages)
            brief_raw = _extract_json(getattr(reply, "content", "") or "") or {}
            if not brief_raw and not task.get("deliverables"):
                retry = call_model("brief", brief_messages + [
                    {"role": "user", "content":
                     "That reply was not a JSON brief. Reply with only the "
                     "JSON object. The instruction was:\n" + instruction}])
                brief_raw = _extract_json(getattr(retry, "content", "") or "") or {}
        except BudgetExceeded:
            brief_raw = {}
        brief = normalise_brief(brief_raw, task)
        brief["mcp_catalogue"] = catalogue_lines
        _atomic_write_json(state_dir / "brief.json", brief)
        for constraint in brief["constraints"]:
            state.append_event(
                "constraint", id=constraint["id"], text=constraint["text"],
                hard=constraint["hard"], status=constraint["status"],
                source_span=constraint["source_span"])
        for item in brief["work_items"]:
            state.append_event("work_item", id=item["id"],
                               title=item["title"], status=item["status"])
        state.persist()

        # ---- early candidate deliverables ----
        spent_before = governor.spent_total
        for entry in brief["deliverables"]:
            path = entry["path"]
            if store.read_graded(path) is None:
                store.write_candidate(path, _placeholder_for(path))
            else:
                store.sync_from_graded(path)
        outcome["first_deliverable_budget_fraction"] = (
            governor.spent_total / governor.total_tokens
            if governor.total_tokens else 0.0)
        outcome["early_deliverable_within_target"] = (
            outcome["first_deliverable_budget_fraction"]
            <= EARLY_DELIVERABLE_FRACTION + 1e-9
            or spent_before <= governor.total_tokens * EARLY_DELIVERABLE_FRACTION)

        # ---- Execute ----
        state.append_event("phase", phase="execute")
        verifier = Verifier(state, tools, store, root)
        observations: list[str] = []
        fingerprints: dict[str, dict[str, Any]] = {}
        seen_results: set[str] = set()
        no_progress = 0
        replanned = False
        max_turns = int(cfg.get("max_turns", 50))
        turn = 0
        finish_requested = False
        # Soft wall-clock transition: the task's time budget is read
        # once; at the soft limit no new work starts and the run
        # moves to verification and Finalize.
        clock = cfg.get("clock") or time.monotonic
        started = clock()
        time_budget: Optional[float] = None
        for key in ("agent_timeout_sec", "time_budget_sec", "timeout_sec"):
            raw_budget = task.get(key) or cfg.get(key)
            if raw_budget:
                time_budget = float(raw_budget)
                break
        if cfg.get("soft_time_limit_sec"):
            soft_limit: Optional[float] = float(cfg["soft_time_limit_sec"])
        elif time_budget:
            soft_limit = 0.8 * time_budget
        else:
            soft_limit = None
        time_left_fn = cfg.get("time_left") if callable(
            cfg.get("time_left")) else None
        soft_transition = False
        while turn < max_turns and not finish_requested:
            turn += 1
            state.append_event("turn", turn=turn)
            if soft_limit is not None:
                elapsed_now = clock() - started
                remaining = None
                if time_left_fn is not None:
                    remaining = time_left_fn()
                elif time_budget:
                    remaining = time_budget - elapsed_now
                if elapsed_now >= soft_limit or (
                        remaining is not None and time_budget
                        and remaining <= time_budget - soft_limit):
                    soft_transition = True
                    state.append_event(
                        "decision",
                        text="soft wall-clock reached: no new work "
                             "items start; moving to verification "
                             "and Finalize")
                    break
            hashes_before = _deliverable_hashes(store, brief)
            messages = _execute_messages(state, brief, observations,
                                         governor)
            tool_defs = tool_schemas()
            if catalogue_lines:
                tool_defs = tool_defs + [find_tools_schema()]
            try:
                reply = call_model("execute", messages, tool_defs)
            except BudgetExceeded:
                break
            content = getattr(reply, "content", "") or ""
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                if content:
                    state.append_event("decision",
                                       text=f"model proposed finishing: "
                                            f"{content[:300]}")
                finish_requested = True
                break
            new_info = False
            for call in calls[:4]:
                name = str(call.get("name") or "")
                args = call.get("arguments") or {}
                fp = fingerprint_call(name, args)
                record = fingerprints.setdefault(
                    fp, {"count": 0, "result_class": None})
                if record["count"] >= 3:
                    # Repeat guard, checked BEFORE dispatch: the
                    # predicted result class is the last recorded
                    # class for this fingerprint, so an identical
                    # repeat is blocked without executing — no side
                    # effects, no spend on a call already made 3
                    # times with the same outcome.
                    observations.append(
                        f"{name}: blocked — this exact call has already "
                        f"been made 3 times with the same result "
                        f"({record['result_class']}). It was not run.")
                    state.append_event(
                        "action", tool=name,
                        target=str(args.get("path") or args.get("command")
                                   or args.get("tool") or "")[:200],
                        result_class="blocked",
                        detail="repeat guard: blocked before dispatch")
                    continue
                if name == "find_tools":
                    found = tools.mcp.find_tools(
                        str(args.get("query") or ""),
                        server=args.get("server"))
                    result = {
                        "ok": True,
                        "summary": f"find_tools: {len(found)} candidate "
                                   "tool(s)",
                        "data": found}
                else:
                    result = tools.dispatch(name, args)
                result_class = ("ok" if result.get("ok") else
                                str(result.get("error_class") or "error"))
                blob = json.dumps(result, default=str, sort_keys=True)
                if hashlib.sha256(blob.encode()).hexdigest() not in seen_results:
                    seen_results.add(hashlib.sha256(blob.encode()).hexdigest())
                    new_info = True
                record["count"] += 1
                record["result_class"] = result_class
                state.append_event(
                    "action", tool=name,
                    target=str(args.get("path") or args.get("command")
                               or args.get("tool") or "")[:200],
                    result_class=result_class,
                    detail=str(result.get("summary", ""))[:300])
                observations.append(
                    f"{name}: {json.dumps(result, default=str)[:MAX_TOOL_RESULT_CHARS]}")
                if name == "write_file" and result.get("ok"):
                    written = str(args.get("path") or "")
                    if any(d["path"] == written for d in brief["deliverables"]):
                        store.sync_from_graded(written)
                        for item in brief["work_items"]:
                            if item.get("deliverable") != written:
                                continue
                            if item["status"] == "complete":
                                continue
                            ran, passed, evidence = _run_acceptance(
                                verifier, item)
                            if ran and not passed:
                                # Acceptance runs as soon as the item
                                # is marked done, not at the end: a
                                # failure reopens the item while
                                # Execute budget remains to repair it.
                                item["status"] = "pending"
                                state.append_event(
                                    "work_item", id=item["id"],
                                    status="pending",
                                    reason=f"acceptance failed: {evidence}",
                                    acceptance_status="FAIL")
                                observations.append(
                                    f"work item {item['id']}: acceptance "
                                    f"check failed ({evidence}); the "
                                    "item is reopened.")
                            else:
                                item["status"] = "complete"
                                store.mark_complete(written)
                                if ran:
                                    state.append_event(
                                        "work_item", id=item["id"],
                                        status="complete",
                                        acceptance_status="PASS")
            hashes_after = _deliverable_hashes(store, brief)
            if hashes_after == hashes_before and not new_info:
                no_progress += 1
            else:
                no_progress = 0
            if no_progress >= 3:
                if not replanned:
                    replanned = True
                    no_progress = 0
                    observations.append(
                        "No progress for 3 turns. Re-plan: pick a "
                        "different work item or state a new hypothesis; "
                        "do not repeat the same calls.")
                    state.append_event("decision",
                                       text="forced re-plan after 3 "
                                            "no-progress turns")
                else:
                    break
            if governor.spent.get("execute", 0) >= \
                    governor.phase_ceiling("execute"):
                break

        # ---- Finalize ----
        state.append_event("phase", phase="finalize")
        if cfg.get("finalize_audit", True):
            try:
                audit = call_model("finalize", [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":
                     "Re-read this instruction and compare it with the "
                     "constraint list. Reply with one JSON object: "
                     "{\"unresolved\": [ids that cannot be classified or "
                     "checked], \"missing\": [constraint texts that are "
                     "missing]}.\n\nInstruction:\n" + instruction
                     + "\n\nConstraints:\n"
                     + json.dumps(brief["constraints"], default=str)[:6000]},
                ])
                parsed = _extract_json(getattr(audit, "content", "") or "") or {}
                unresolved_ids = {str(x) for x in parsed.get("unresolved") or []}
                for constraint in brief["constraints"]:
                    if constraint["id"] in unresolved_ids:
                        constraint["status"] = "UNRESOLVED"
                        state.append_event(
                            "constraint", id=constraint["id"],
                            status="UNRESOLVED")
                for missing in parsed.get("missing") or []:
                    extra = normalise_constraint(
                        {"text": str(missing)}, instruction,
                        len(brief["constraints"]))
                    brief["constraints"].append(extra)
                    state.append_event(
                        "constraint", id=extra["id"], text=extra["text"],
                        hard=extra["hard"], status=extra["status"],
                        source_span=extra["source_span"])
            except BudgetExceeded:
                pass
            except Exception as exc:
                if _is_session_cancelled(exc):
                    raise
                state.append_event("dead_end", text=f"audit call failed: {exc}")
        checks = verifier.verify(brief)
        for entry in brief["deliverables"]:
            path = entry["path"]
            exists = (checks["deliverables"].get(path) or {}).get("exists")
            if exists and deliverable_checks_pass(checks, brief, path):
                store.promote(path)
            elif exists and store.verified_copy(path) is not None:
                # Per-path restore: only a deliverable whose own
                # checks fail is restored, and the discarded bytes
                # are noted in the record.
                current = store.read_graded(path)
                verified = store.verified_copy(path)
                if current != verified:
                    state.append_event(
                        "decision",
                        text=f"restored {path} from the last verified "
                             "copy; discarded bytes that failed this "
                             "deliverable's own checks (sha256 "
                             f"{hashlib.sha256(current or b'').hexdigest()[:12]})")
                    store.restore_verified(path)
            elif not exists:
                verified = store.verified_copy(path)
                if verified is not None:
                    # The graded file is gone — deleted during
                    # Execute — but a verified copy survives: restore
                    # it so the graded tree ends the run holding the
                    # verified bytes, then re-check the path. If the
                    # restore does not land, the deliverable is
                    # recorded as not verified, whatever the record
                    # said before.
                    state.append_event(
                        "decision",
                        text=f"restored {path} from the last verified "
                             "copy; the graded file was missing at "
                             "Finalize")
                    restored = False
                    try:
                        restored = store.restore_verified(path)
                    except Exception as exc:
                        if _is_session_cancelled(exc):
                            raise
                        restored = False
                    after = store.read_graded(path) if restored else None
                    if after is not None and hashlib.sha256(
                            after).hexdigest() == hashlib.sha256(
                            verified).hexdigest():
                        checks["deliverables"][path] = {"exists": True}
                    else:
                        state.append_event(
                            "deliverable", path=path,
                            status="missing" if after is None
                            else "unverified")
        finish_background()
        state.persist()
        outcome.update({
            "status": "finalized",
            "checks": checks,
            "all_hard_checks_pass": checks["all_hard_checks_pass"],
            "hard_failure_count": checks.get("hard_failure_count", 0),
            "hard_unresolved_count": checks.get("hard_unresolved_count", 0),
            "wall_time_sec": clock() - started,
            "soft_time_transition": soft_transition,
            "deliverables": copy.deepcopy(state.summary["deliverables"]),
            "usage": dict(state.summary["usage"]),
            "budget": governor.snapshot(),
            "state_dir": str(state_dir),
            "final_message": state.closing_summary_text(),
            "turns": turn,
        })
        _write_closing_summary(state, state_dir)
        return outcome
    except Exception as exc:  # the supervisor owns the closing frame
        if _is_session_cancelled(exc):
            # The run is being torn down by the session layer, but
            # its background jobs are still stopped and the stop is
            # recorded in the persisted summary before the
            # cancellation propagates; none of this may mask it.
            finish_background()
            try:
                state.persist()
            except Exception:
                pass
            _write_closing_summary(state, state_dir)
            raise
        finish_background()
        state.append_event("dead_end",
                           text=f"loop failure: {type(exc).__name__}: {exc}")
        try:
            state.persist()
        except Exception:
            pass
        _write_closing_summary(state, state_dir)
        try:
            final_message = state.closing_summary_text()
        except Exception:
            final_message = (f"Task {task_id}: closing summary "
                             "unavailable; see the run record.")
        outcome.update({
            "status": "error_outcome",
            "error": f"{type(exc).__name__}: {exc}",
            "state_dir": str(state_dir),
            "final_message": final_message,
        })
        return outcome


class _CallableClient:
    """Adapts a plain callable from config to the client interface."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def complete(self, messages: list[dict[str, Any]], tools: Any = None,
                 phase: str = "execute") -> Any:
        result = self._fn(messages, tools=tools, phase=phase)
        if isinstance(result, Mapping):
            from model_client import ModelResponse
            usage, defects = normalize_usage(result.get("usage"))
            return ModelResponse(
                content=str(result.get("content") or ""),
                tool_calls=list(result.get("tool_calls") or []),
                usage=usage, model=str(result.get("model") or ""),
                usage_recorded=bool(result.get("usage_recorded", False)),
                defects=defects)
        return result


def _inventory(root: Path, tools: ToolDispatcher) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if tools.request is None:
        for path in sorted(root.rglob("*")):
            if ".ple-state" in path.parts or ".ple-background" in path.parts:
                continue
            if path.is_file():
                entries.append({"path": str(path.relative_to(root)),
                                "size": path.stat().st_size})
            if len(entries) >= 500:
                break
    return entries


def _deliverable_hashes(store: DeliverableStore,
                        brief: Mapping[str, Any]) -> dict[str, Optional[str]]:
    hashes: dict[str, Optional[str]] = {}
    for entry in brief.get("deliverables") or []:
        data = store.read_graded(entry["path"])
        hashes[entry["path"]] = (
            hashlib.sha256(data).hexdigest() if data is not None else None)
    return hashes


def _execute_messages(state: LedgerState, brief: Mapping[str, Any],
                      observations: list[str],
                      governor: BudgetGovernor) -> list[dict[str, Any]]:
    summary = state.compact_summary()
    context = {
        "brief": {"deliverables": brief["deliverables"],
                  "constraints": brief["constraints"],
                  "work_items": brief["work_items"],
                  "mcp_catalogue": brief.get("mcp_catalogue") or []},
        "summary": {k: summary[k] for k in
                    ("phase", "turn_count", "deliverables", "checks",
                     "all_hard_checks_pass")},
        "recent_observations": observations[-6:],
        "budget": {"spent_total": governor.spent_total,
                   "total_tokens": governor.total_tokens,
                   "reserve_tokens": governor.reserve_tokens},
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
         "Continue the task. Work the next incomplete item, or reply "
         "with no tool calls and a summary when everything is done.\n"
         + json.dumps(context, default=str)[:24000]},
    ]
