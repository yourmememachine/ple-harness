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
from tools import ToolDispatcher, tool_schemas  # noqa: E402

SUMMARY_CAP_BYTES = 64 * 1024
EARLY_DELIVERABLE_FRACTION = 0.20
MAX_TOOL_RESULT_CHARS = 8_000

PHASES = ("brief", "execute", "verify", "finalize")


class BudgetExceeded(Exception):
    """A phase tried to spend past its allocation or into the reserve."""


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
            if self.spent.get(phase, 0) + amount > max(
                    self.phase_ceiling(phase),
                    self.reserve_tokens if phase in self.RESERVE_PHASES else 0):
                # A reserve phase may always use at least the reserve.
                if not (phase in self.RESERVE_PHASES
                        and self.spent_total + amount <= self.total_tokens):
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
        """Charge an actual call cost; False when it broke the budget."""
        try:
            self.spend(phase, amount)
            return True
        except BudgetExceeded:
            self.spent[phase] = self.spent.get(phase, 0) + int(amount)
            return False

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
        }


# ---------------------------------------------------------------------
# Disk state and the ledger-generated summary
# ---------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
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
    "deliverables", "constraints", "work_items", "checks",
    "decisions", "dead_ends", "recent_actions", "dropped_actions",
    "all_hard_checks_pass", "updated_seq",
)


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
        self.summary: dict[str, Any] = {
            "task_id": task_id,
            "phase": "brief",
            "turn_count": 0,
            "model_calls": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0},
            "deliverables": {},
            "constraints": {},
            "work_items": {},
            "checks": {},
            "decisions": [],
            "dead_ends": [],
            "recent_actions": [],
            "dropped_actions": 0,
            "all_hard_checks_pass": False,
            "updated_seq": 0,
        }
        if self.summary_path.exists():
            try:
                loaded = json.loads(self.summary_path.read_text("utf-8"))
                if isinstance(loaded, dict) and loaded.get("task_id") == task_id:
                    self.summary = loaded
                    self.seq = int(loaded.get("updated_seq", 0))
            except (ValueError, OSError):
                pass

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
            usage={k: int(record.get(k, 0)) for k in self.summary["usage"]})

    def _fold(self, event: Mapping[str, Any]) -> None:
        kind = event.get("kind")
        summary = self.summary
        if kind == "usage":
            usage = event.get("usage") or {}
            for key in summary["usage"]:
                summary["usage"][key] += int(usage.get(key, 0))
            summary["model_calls"] += 1
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
                    entry[key] = event[key]
        elif kind == "work_item":
            entry = summary["work_items"].setdefault(
                str(event.get("id")), {})
            for key in ("title", "status", "reason"):
                if event.get(key) is not None:
                    entry[key] = event[key]
        elif kind == "checks":
            summary["checks"] = event.get("checks") or {}
            summary["all_hard_checks_pass"] = bool(
                event.get("all_hard_checks_pass"))
        elif kind == "decision":
            summary["decisions"].append(str(event.get("text", ""))[:500])
        elif kind == "dead_end":
            summary["dead_ends"].append(str(event.get("text", ""))[:500])

    def rebuild_summary(self) -> dict[str, Any]:
        """Regenerate the summary from the ledger alone."""
        fresh = LedgerState.__new__(LedgerState)
        fresh.dir, fresh.task_id = self.dir, self.task_id
        fresh.ledger_path, fresh.summary_path = self.ledger_path, self.summary_path
        fresh.seq = 0
        fresh.summary = copy.deepcopy(self.summary)
        for key in ("turn_count", "model_calls", "dropped_actions"):
            fresh.summary[key] = 0
        fresh.summary["usage"] = {k: 0 for k in self.summary["usage"]}
        for key in ("recent_actions", "decisions", "dead_ends"):
            fresh.summary[key] = []
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                fresh.seq = int(event.get("seq", fresh.seq))
                fresh._fold(event)
                fresh.summary["updated_seq"] = fresh.seq
        return fresh.summary

    # -- compaction ------------------------------------------------------

    def compact_summary(self, cap_bytes: int = SUMMARY_CAP_BYTES) -> dict[str, Any]:
        """The closing summary, capped: oldest action detail goes first."""
        summary = copy.deepcopy(self.summary)
        summary["dropped_actions"] = int(summary.get("dropped_actions", 0))

        def size() -> int:
            return len(json.dumps(summary, ensure_ascii=False,
                                  sort_keys=True).encode("utf-8"))

        while size() > cap_bytes and summary["recent_actions"]:
            summary["recent_actions"].pop(0)
            summary["dropped_actions"] += 1
        while size() > cap_bytes and summary["dead_ends"]:
            summary["dead_ends"].pop(0)
        while size() > cap_bytes and summary["decisions"]:
            summary["decisions"].pop(0)
        if size() > cap_bytes:
            for entry in summary["recent_actions"]:
                entry["detail"] = entry.get("detail", "")[:80]
        if size() > cap_bytes:
            blob = json.dumps(summary, ensure_ascii=False,
                              sort_keys=True).encode("utf-8")
            raise ValueError(
                f"summary is {len(blob)} bytes, over the {cap_bytes} cap, "
                "with no droppable detail left")
        return summary

    def persist(self) -> dict[str, Any]:
        compact = self.compact_summary()
        _atomic_write_json(self.summary_path, compact)
        return compact

    def closing_summary_text(self) -> str:
        """Defined-structure closing summary, within the 64 KiB cap."""
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
        except Exception:
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
    severity, as does any hard constraint not in PASS.
    """
    constraints = checks.get("constraints") or {}
    for entry in constraints.values():
        status = entry.get("status")
        if status == "UNRESOLVED":
            return False
        if entry.get("hard") and status != "PASS":
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
        for constraint in brief.get("constraints") or []:
            cid = str(constraint.get("id"))
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
        for item in brief.get("consistency_items") or []:
            spec = item.get("check") if isinstance(item, Mapping) else None
            if isinstance(spec, Mapping):
                status, evidence = self.run_check(spec)
            else:
                status, evidence = "UNRESOLVED", "no check defined"
            checks["consistency"][str(item.get("id"))] = {
                "status": status, "evidence": evidence}
        checks["all_hard_checks_pass"] = all_hard_checks_pass(checks)
        self.state.append_event(
            "checks", checks=checks,
            all_hard_checks_pass=checks["all_hard_checks_pass"])
        return checks


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
    if isinstance(span, str):
        span = _find_span(instruction, span) or {"source": span}
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

    def call_model(phase: str, messages: list[dict[str, Any]],
                   tool_defs: Optional[list] = None) -> Any:
        if not governor.can_call(phase):
            raise BudgetExceeded(f"{phase}: model-call reserve reached")
        if not governor.can_spend(phase, estimated_call):
            raise BudgetExceeded(f"{phase}: token allocation reached")
        governor.record_call(phase)
        try:
            response = client.complete(messages, tools=tool_defs, phase=phase)
        except ModelCallInterrupted as exc:
            response = exc.partial
            state.append_event("dead_end", text=f"interrupted call: {exc}")
        usage = getattr(response, "usage", None) or {}
        counts, _ = normalize_usage(usage)
        if not getattr(response, "usage_recorded", usage_recorded_by_client):
            state.append_usage({"phase": phase,
                                "model": getattr(response, "model", None),
                                **counts})
        governor.record_spend(phase, total_tokens(counts))
        return response

    outcome: dict[str, Any] = {"task_id": task_id, "status": "started",
                               "terminal_frame_sent": False}
    try:
        # ---- inventory (no model call) ----
        inventory = cfg.get("inventory")
        if inventory is None:
            inventory = _inventory(root, tools)
        state.append_event("phase", phase="brief")

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
             + "\n\nInventory:\n" + json.dumps(inventory, default=str)[:6000]},
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
        while turn < max_turns and not finish_requested:
            turn += 1
            state.append_event("turn", turn=turn)
            hashes_before = _deliverable_hashes(store, brief)
            messages = _execute_messages(state, brief, observations,
                                         governor)
            try:
                reply = call_model("execute", messages, tool_schemas())
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
                result = tools.dispatch(name, args)
                result_class = ("ok" if result.get("ok") else
                                str(result.get("error_class") or "error"))
                blob = json.dumps(result, default=str, sort_keys=True)
                if hashlib.sha256(blob.encode()).hexdigest() not in seen_results:
                    seen_results.add(hashlib.sha256(blob.encode()).hexdigest())
                    new_info = True
                if record["count"] >= 3 and record["result_class"] == result_class:
                    observations.append(
                        f"{name}: blocked — this exact call has already "
                        f"been made 3 times with the same result.")
                    state.append_event(
                        "action", tool=name, target=str(args)[:200],
                        result_class="blocked", detail="repeat guard")
                    continue
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
                            if item.get("deliverable") == written:
                                item["status"] = "complete"
                                store.mark_complete(written)
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
                state.append_event("dead_end", text=f"audit call failed: {exc}")
        checks = verifier.verify(brief)
        for entry in brief["deliverables"]:
            path = entry["path"]
            exists = (checks["deliverables"].get(path) or {}).get("exists")
            if exists and checks["all_hard_checks_pass"]:
                store.promote(path)
            elif exists and store.verified_copy(path) is not None:
                current = store.read_graded(path)
                verified = store.verified_copy(path)
                if current != verified:
                    store.restore_verified(path)
        state.persist()
        outcome.update({
            "status": "finalized",
            "checks": checks,
            "all_hard_checks_pass": checks["all_hard_checks_pass"],
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
        state.append_event("dead_end",
                           text=f"loop failure: {type(exc).__name__}: {exc}")
        try:
            state.persist()
        except Exception:
            pass
        _write_closing_summary(state, state_dir)
        outcome.update({
            "status": "error_outcome",
            "error": f"{type(exc).__name__}: {exc}",
            "state_dir": str(state_dir),
            "final_message": state.closing_summary_text(),
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
            usage, _ = normalize_usage(result.get("usage"))
            return ModelResponse(
                content=str(result.get("content") or ""),
                tool_calls=list(result.get("tool_calls") or []),
                usage=usage, model=str(result.get("model") or ""),
                usage_recorded=bool(result.get("usage_recorded", False)))
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
                  "work_items": brief["work_items"]},
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
