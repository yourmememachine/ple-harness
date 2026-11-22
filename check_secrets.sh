#!/usr/bin/env bash
# Secret scan: searches the tracked files and the full git history
# of this repository for credential-shaped strings.
#
#   ./check_secrets.sh
#
# Exit status 0 (PASS) when nothing matches, 1 (FAIL) when any
# pattern matches. The scan covers:
#   - private-key blocks (PEM headers)
#   - assignments of a quoted literal to a credential-shaped name
#     (api_key / access_token / auth_token / client_secret /
#     password / passwd)
#   - well-known key shapes (AWS access key ids, GitHub tokens,
#     OpenAI-style keys, Slack tokens)
#
# This script is excluded from its own scan: it contains the pattern
# text by necessity. Variable references and environment lookups are
# not matches — only literal values are.
#
# run_kit_check.sh runs this scan as an advisory stage; there it
# reports PASS/FAIL without changing the kit verdict.

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "FAIL: not a git repository: $REPO_DIR" >&2
  exit 1
fi

SELF=':(exclude)check_secrets.sh'

PATTERNS=(
  -e '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  -e '(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{8,}["'"'"']'
  -e '(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})'
)

FOUND=0

# --- tracked files (working tree) --------------------------------------
HITS="$(git grep -I -n -E "${PATTERNS[@]}" -- . "$SELF" 2>/dev/null || true)"
if [ -n "$HITS" ]; then
  echo "tracked files:"
  echo "$HITS"
  FOUND=1
fi

# --- git history (every reachable commit) -------------------------------
for REV in $(git rev-list --all); do
  HITS="$(git grep -I -n -E "${PATTERNS[@]}" "$REV" -- . "$SELF" 2>/dev/null || true)"
  if [ -n "$HITS" ]; then
    echo "history at $REV:"
    echo "$HITS"
    FOUND=1
  fi
done

if [ "$FOUND" -ne 0 ]; then
  echo "FAIL: credential-shaped strings found (see above)"
  exit 1
fi
echo "PASS: no credential-shaped strings in tracked files or git history"
exit 0
