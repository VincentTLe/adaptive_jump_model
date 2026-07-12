#!/usr/bin/env bash
# Tool-free handoff helper — works for ANY agent (Codex, Cursor, Claude) without a skill runtime.
#
#   handoff.sh                 print git context + the JSONL entry schema (agent fills it in)
#   handoff.sh '<json-entry>'  append the given one-line JSON entry to .agent/session-log.jsonl,
#                              then regenerate .agent/session-log.html
#
# The JSONL file is the source of truth (next agent reads it). The HTML is the human view.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$ROOT/.agent"
LOG="$AGENT_DIR/session-log.jsonl"
mkdir -p "$AGENT_DIR"

if [ "${1:-}" = "" ]; then
  echo "== git status (short) =="; git -C "$ROOT" status --short 2>/dev/null || echo "(not a git repo)"
  echo; echo "== diff --stat =="; git -C "$ROOT" diff --stat 2>/dev/null || true
  echo; echo "== last 5 commits =="; git -C "$ROOT" log --oneline -5 2>/dev/null || true
  echo
  echo "Pass ONE canonical JSON entry to this helper; it validates before append."
  echo "Schema (fill honestly; never fabricate a commit sha or a verification that did not happen):"
  echo '{"ts":"<UTC ISO8601, e.g. 2026-05-25T21:30Z>","agent":"claude|codex|cursor","model":"<model>","goal":"...","files":["path (+/- tag)"],"verification":["cmd + result, or: not verified this session"],"commit":"<sha|uncommitted>","next":"...","notes":"<gotchas, optional>"}'
  echo
  echo "Run: bash \"$HERE/handoff.sh\" '<one-line-json-entry>'"
  exit 0
fi

python3 "$HERE/render_log.py" "$AGENT_DIR" --check-log
python3 "$HERE/render_log.py" "$AGENT_DIR" --validate-entry "$1"
printf '%s\n' "$1" >> "$LOG"
python3 "$HERE/render_log.py" "$AGENT_DIR"
echo "Appended entry + regenerated $AGENT_DIR/session-log.html"
