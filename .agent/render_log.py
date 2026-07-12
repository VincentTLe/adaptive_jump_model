#!/usr/bin/env python3
"""Validate and render the append-only agent handoff log."""

import argparse
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

STRING_FIELDS = ("ts", "agent", "model", "goal", "commit", "next", "notes")
LIST_FIELDS = ("files", "verification")
CANONICAL_FIELDS = frozenset((*STRING_FIELDS, *LIST_FIELDS))
LEGACY_FIELDS = CANONICAL_FIELDS | {"next_step"}


def esc(value: object) -> str:
    return html.escape(str(value)) if value not in (None, "") else ""


def _require_string_list(entry: dict[str, object], field: str) -> None:
    value = entry[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")


def validate_canonical_entry(entry: object) -> None:
    """Validate one new entry against the canonical handoff schema."""
    if not isinstance(entry, dict):
        raise ValueError("entry must be a JSON object")
    missing = sorted(CANONICAL_FIELDS - entry.keys())
    if missing:
        label = "field" if len(missing) == 1 else "fields"
        raise ValueError(f"missing required {label}: {', '.join(missing)}")
    unexpected = sorted(entry.keys() - CANONICAL_FIELDS)
    if unexpected:
        label = "field" if len(unexpected) == 1 else "fields"
        raise ValueError(f"unexpected {label}: {', '.join(unexpected)}")
    for field in STRING_FIELDS:
        if not isinstance(entry[field], str):
            raise ValueError(f"{field} must be a string")
    for field in LIST_FIELDS:
        _require_string_list(entry, field)


def parse_canonical_entry(raw: str) -> dict[str, object]:
    if "\n" in raw or "\r" in raw:
        raise ValueError("entry must be one physical line")
    try:
        entry = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid entry JSON: {exc.msg}") from exc
    validate_canonical_entry(entry)
    return entry


def validate_stored_entry(entry: object) -> None:
    """Accept the canonical schema plus documented historical variations."""
    if not isinstance(entry, dict):
        raise ValueError("entry must be a JSON object")
    unexpected = sorted(entry.keys() - LEGACY_FIELDS)
    if unexpected:
        raise ValueError(f"unexpected fields: {', '.join(unexpected)}")
    required = CANONICAL_FIELDS - {"model", "next"}
    missing = sorted(required - entry.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    has_next = "next" in entry
    has_next_step = "next_step" in entry
    if has_next == has_next_step:
        raise ValueError("entry must contain exactly one of next or next_step")
    for field in ("ts", "agent", "goal", "commit", "notes"):
        if not isinstance(entry[field], str):
            raise ValueError(f"{field} must be a string")
    for field in ("model", "next", "next_step"):
        if field in entry and not isinstance(entry[field], str):
            raise ValueError(f"{field} must be a string")
    _require_string_list(entry, "files")
    verification = entry["verification"]
    if not isinstance(verification, str):
        _require_string_list(entry, "verification")


def load_entries(jsonl: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if not jsonl.exists():
        return entries
    for line_number, raw in enumerate(
        jsonl.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            raise ValueError(f"{jsonl}:{line_number}: blank line")
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{jsonl}:{line_number}: invalid JSON: {exc.msg}") from exc
        try:
            validate_stored_entry(entry)
        except ValueError as exc:
            raise ValueError(f"{jsonl}:{line_number}: {exc}") from exc
        entries.append(entry)
    return entries


def files_html(files: object) -> str:
    if not files:
        return '<span class="dim">—</span>'
    return "<ul>" + "".join(f"<li><code>{esc(f)}</code></li>" for f in files) + "</ul>"


def verification_html(verification: object) -> str:
    if not verification:
        return '<span class="dim">—</span>'
    if isinstance(verification, str):
        return f"<p>{esc(verification)}</p>"
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in verification) + "</ul>"


def card(entry: dict[str, object]) -> str:
    model = esc(entry.get("model", ""))
    who = esc(entry.get("agent", "?")) + (f" · {model}" if model else "")
    notes = entry.get("notes")
    next_step = entry.get("next", entry.get("next_step", ""))
    timestamp = esc(entry.get("ts", ""))
    goal = esc(entry.get("goal", ""))
    files = files_html(entry.get("files"))
    verification = verification_html(entry.get("verification"))
    commit = esc(entry.get("commit", "")) or "—"
    next_html = esc(next_step) or '<span class="dim">—</span>'
    notes_html = (
        f'<p class="notes"><span class="lbl">Notes</span>{esc(notes)}</p>'
        if notes
        else ""
    )
    return f"""    <article class="card">
      <div class="head">
        <span class="ts">{timestamp}</span><span class="who">{who}</span>
      </div>
      <p class="goal">{goal}</p>
      <div class="grid">
        <div><span class="lbl">Files touched</span>{files}</div>
        <div><span class="lbl">Verification</span>{verification}</div>
        <div><span class="lbl">Commit</span><p><code>{commit}</code></p></div>
        <div><span class="lbl">Next step</span><p>{next_html}</p></div>
      </div>
      {notes_html}
    </article>"""


def render(agent_dir: Path, entries: list[dict[str, object]]) -> None:
    output = agent_dir / "session-log.html"
    entries_rev = list(reversed(entries))  # newest first for display
    body = (
        "\n".join(card(e) for e in entries_rev)
        if entries_rev
        else '<p class="empty">No sessions logged yet.</p>'
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session log — {esc(agent_dir.resolve().parent.name)}</title>
<style>
  :root{{--bg:#0f1117;--bg2:#161922;--bg3:#1f2330;--fg:#e7ecef;
    --dim:#9aa5b1;--accent:#7bdcb5;--cyan:#7bf0ff;--border:#2a3040;}}
  *{{box-sizing:border-box;}} body{{margin:0;background:var(--bg);color:var(--fg);
    font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;
    line-height:1.6;font-size:15px;}}
  code{{font-family:'JetBrains Mono',ui-monospace,monospace;
    background:var(--bg3);color:var(--cyan);
    padding:.06em .35em;border-radius:4px;font-size:.86em;}}
  header{{max-width:880px;margin:0 auto;padding:42px 28px 8px;}}
  header h1{{margin:0;font-size:26px;font-weight:800;color:var(--accent);
    letter-spacing:-.02em;}}
  header .sub{{color:var(--dim);font-size:13px;margin:.3em 0 0;}}
  main{{max-width:880px;margin:0 auto;padding:18px 28px 60px;display:flex;
    flex-direction:column;gap:14px;}}
  .card{{background:var(--bg2);border:1px solid var(--border);
    border-radius:12px;padding:16px 18px;
    border-left:3px solid var(--accent);}}
  .head{{display:flex;justify-content:space-between;align-items:baseline;
    gap:12px;flex-wrap:wrap;}}
  .ts{{font-family:'JetBrains Mono',monospace;color:var(--cyan);font-size:13px;}}
  .who{{color:var(--dim);font-size:12.5px;text-transform:uppercase;letter-spacing:.5px;}}
  .goal{{font-size:16px;font-weight:600;margin:.5em 0 .8em;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;}}
  @media(max-width:620px){{.grid{{grid-template-columns:1fr;}}}}
  .lbl{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--accent);
    font-weight:700;margin-bottom:2px;}}
  .grid ul{{margin:.2em 0;padding-left:1.1em;}} .grid li{{margin:.15em 0;}}
  .grid p{{margin:.2em 0;}}
  .notes{{margin:.9em 0 0;padding-top:.7em;
    border-top:1px dashed var(--border);font-size:14px;}}
  .notes .lbl{{display:inline;margin-right:.4em;}}
  .dim{{color:var(--dim);}} .empty{{color:var(--dim);text-align:center;padding:40px;}}
</style></head>
<body>
<header>
  <h1>Session log</h1>
  <p class="sub">{len(entries)} session(s) · newest first · {generated}</p>
</header>
<main>
{body}
</main>
</body></html>"""
    agent_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")
    print(f"Wrote {output} ({len(entries)} entries)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_dir", nargs="?", type=Path, default=Path(".agent"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-log", action="store_true")
    mode.add_argument("--validate-entry")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_entry is not None:
            parse_canonical_entry(args.validate_entry)
            print("Entry is valid")
            return 0
        entries = load_entries(args.agent_dir / "session-log.jsonl")
        if args.check_log:
            print(f"Validated {len(entries)} existing entries")
            return 0
        render(args.agent_dir, entries)
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
