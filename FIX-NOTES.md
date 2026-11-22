# Integration fix notes

## Problem

serve.py treated the harness entry point as a coroutine function:
it awaited `run_harness(...)` and handed the loop its async
`Session.emit`. `harness.run_harness` is a plain synchronous
function, so the await raised `TypeError: object dict can't be used
in 'await' expression` and the session closed with an `error` frame.
The model client also calls `emit` synchronously, which left the
async emit's coroutines un-awaited (`RuntimeWarning: coroutine
'Session.emit' was never awaited`).

## The unified contract

`run_harness(task, emit, workdir, config)` is synchronous, and the
`emit` it receives is synchronous:

- `emit(frame_dict)` blocks the calling thread until the frame is
  sent and, for request frames, until the evaluator's result
  arrives; it returns the result dict for request frames and None
  for `log`/`usage` frames.
- `emit` raises `serve.SessionCancelled` in the loop thread when the
  session is cancelled or has ended, and `serve.FrameError` for a
  locally invalid frame.

## Changes

serve.py:

- `Session._run_loop` now branches on the runner: a coroutine
  function is awaited on the event loop with the async emit (the
  behaviour the session tests exercise); a plain function runs in a
  worker thread via `asyncio.to_thread` and receives the new
  synchronous bridge, `Session._bridge_emit`.
- `_bridge_emit` schedules `Session.emit` on the session's event
  loop with `asyncio.run_coroutine_threadsafe` and blocks on the
  resulting future. It polls the future so that session end, a
  claimed terminal frame, or cancellation of a request blocked
  outside the pending-request map (for example on the in-flight
  semaphore) abandons the emit and raises `SessionCancelled` in the
  loop thread. Exceptions raised by the async emit itself propagate
  unchanged. The supervisor's existing cancel path is untouched: the
  loop unwinds promptly, no new model calls are made, and `done` is
  still sent inside the window.
- `_outcome_frame` and the cancel path now accept the harness's own
  outcome vocabulary alongside the documented one: `finalized`
  closes as `done` with the outcome's `final_message`, and
  `error_outcome` closes as `error` with its `error` text.
  Previously both fell through to "the harness returned an invalid
  outcome".

harness.py:

- `run_harness` now writes its closing summary to
  `<state_dir>/closing_summary.md` (via `_write_closing_summary`,
  atomic tmp-file replace) on both the finalized and the failure
  path. That is the exact path serve.py's cancel fallback reads;
  before, the loop only wrote `summary.json` and returned the text
  in the outcome, so a cancelled session would have fallen back to
  the fixed stopped message instead of the loop's own summary.

model_client.py: no change needed. Its one emit call site was
already synchronous; it now receives the bridge emit, so usage
frames reach the wire and no coroutine is left un-awaited.

requirements.txt (new): the install step in submission.yaml runs
`uv pip install -r requirements.txt`, but the file did not exist.
It now declares the one non-stdlib runtime dependency, `websockets`
(imported by serve.py); harness.py, model_client.py and tools.py
use only the standard library.

## Verification

`.venv/bin/python -m pytest tests/ -q` — 33 passed (was 32 passed,
1 failed). The suite also passes with `RuntimeWarning` promoted to
an error, confirming no un-awaited coroutines remain.
