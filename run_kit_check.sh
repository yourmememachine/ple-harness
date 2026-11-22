#!/usr/bin/env bash
# Runs the kit's automated check against this entry, with the model
# stages pointed at the local stub proxy (stub_proxy/server.py) so the
# whole run needs no real model access.
#
#   ./run_kit_check.sh [--cancel-test] [extra kit check flags...]
#
# Environment overrides:
#   STUB_PORT   port for the stub proxy        (default 4100)
#   STUB_KEY    key the check presents to it   (default stub-local-key)
#
# The full output is kept in kit-check-last.log next to this script.
# A per-stage PASS/FAIL summary is printed at the end. Exit status is
# 0 only when the check prints "ready to submit"; a run can pass
# several stages and still not be ready (for example, the entry must
# sit at the root of its own git repo with everything committed).
#
# After the summary, check_secrets.sh runs as an advisory stage: it
# scans the tracked files and the git history for credential-shaped
# strings and prints PASS/FAIL, but its result never changes the
# kit verdict or this script's exit status.
#
# Note: the build/listen/smoke stages run containers, so on a machine
# without Docker the run stops at the build stage. The stages before
# it (submission.yaml, model, files) still run and are reported.

set -u
export PATH="$HOME/.local/bin:$PATH"

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$BUILD_DIR/../stage0/ple-repo"
STUB_PORT="${STUB_PORT:-4100}"
STUB_KEY="${STUB_KEY:-stub-local-key}"
LOG="$BUILD_DIR/kit-check-last.log"
PYTHON="$BUILD_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

if [ ! -d "$KIT_DIR/kit" ]; then
  echo "kit not found at $KIT_DIR (expected the google-research/ple clone)" >&2
  exit 2
fi

# --- start the stub proxy ---------------------------------------------
"$PYTHON" "$BUILD_DIR/stub_proxy/server.py" \
  --port "$STUB_PORT" --key "$STUB_KEY" --quiet &
STUB_PID=$!
trap 'kill "$STUB_PID" 2>/dev/null' EXIT

for _ in $(seq 1 50); do
  if curl -s "http://127.0.0.1:$STUB_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done
if ! curl -s "http://127.0.0.1:$STUB_PORT/health" >/dev/null 2>&1; then
  echo "stub proxy did not come up on port $STUB_PORT" >&2
  exit 2
fi
echo "stub proxy: http://127.0.0.1:$STUB_PORT (pid $STUB_PID)"

# --- run the check ------------------------------------------------------
export MODEL_PROXY_URL="http://127.0.0.1:$STUB_PORT"
export MODEL_PROXY_KEY="$STUB_KEY"

cd "$KIT_DIR"
uv run python -m kit check "$BUILD_DIR" "$@" 2>&1 | tee "$LOG"
CHECK_RC=${PIPESTATUS[0]}

# --- per-stage summary --------------------------------------------------
echo
echo "=== stage results (from $LOG) ==="
grep -E '^\[(PASS|FAIL|WARN|SKIP)\]' "$LOG" || echo "(no stage lines found)"

# --- advisory secret scan (never changes the verdict) -------------------
echo
echo "=== secret scan (advisory) ==="
if [ -x "$BUILD_DIR/check_secrets.sh" ]; then
  if "$BUILD_DIR/check_secrets.sh"; then
    echo "[PASS] secrets: no credential-shaped strings in tracked files or git history"
  else
    echo "[FAIL] secrets: credential-shaped strings found (advisory only; kit verdict unchanged)"
  fi
else
  echo "[SKIP] secrets: check_secrets.sh not found or not executable"
fi
echo
if grep -q "ready to submit" "$LOG"; then
  echo "VERDICT: ready to submit"
  exit 0
fi
echo "VERDICT: not ready (kit check exit $CHECK_RC; see the FAIL lines above)"
exit 1
