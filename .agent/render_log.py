#!/usr/bin/env python3
"""Render .agent/session-log.jsonl into a dark visual timeline at .agent/session-log.html.

Usage:  render_log.py [AGENT_DIR]      (AGENT_DIR defaults to ./.agent)

Stdlib only. Malformed JSONL lines are skipped, never fatal. The JSONL is the
source of truth (what the next agent reads); this HTML is the human-friendly view.
"""
import sys
import json
import html
from pathlib import Path
from datetime import datetime, timezone

AGENT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".agent")
JSONL = AGENT_DIR / "session-log.jsonl"
OUT = AGENT_DIR / "session-log.html"


def esc(x):
    return html.escape(str(x)) if x not in (None, "") else ""


def load_entries():
    out = []
    if not JSONL.exists():
        return out
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip a bad line rather than crash the render
    return out


def files_html(files):
    if not files:
        return '<span class="dim">—</span>'
    if isinstance(files, str):
        files = [files]
    return "<ul>" + "".join(f"<li><code>{esc(f)}</code></li>" for f in files) + "</ul>"


def card(e):
    model = esc(e.get("model", ""))
    who = esc(e.get("agent", "?")) + (f" · {model}" if model else "")
    notes = e.get("notes")
    notes_html = (
        f'<p class="notes"><span class="lbl">Notes</span>{esc(notes)}</p>' if notes else ""
    )
    return f"""    <article class="card">
      <div class="head"><span class="ts">{esc(e.get('ts',''))}</span><span class="who">{who}</span></div>
      <p class="goal">{esc(e.get('goal',''))}</p>
      <div class="grid">
        <div><span class="lbl">Files touched</span>{files_html(e.get('files'))}</div>
        <div><span class="lbl">Verification</span><p>{esc(e.get('verification','')) or '<span class="dim">—</span>'}</p></div>
        <div><span class="lbl">Commit</span><p><code>{esc(e.get('commit','')) or '—'}</code></p></div>
        <div><span class="lbl">Next step</span><p>{esc(e.get('next','')) or '<span class="dim">—</span>'}</p></div>
      </div>
      {notes_html}
    </article>"""


def main():
    entries = load_entries()
    entries_rev = list(reversed(entries))  # newest first for display
    body = "\n".join(card(e) for e in entries_rev) if entries_rev else '<p class="empty">No sessions logged yet.</p>'
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Session log — {esc(AGENT_DIR.resolve().parent.name)}</title>
<style>
  :root{{--bg:#0f1117;--bg2:#161922;--bg3:#1f2330;--fg:#e7ecef;--dim:#9aa5b1;--accent:#7bdcb5;--cyan:#7bf0ff;--border:#2a3040;}}
  *{{box-sizing:border-box;}} body{{margin:0;background:var(--bg);color:var(--fg);
    font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;line-height:1.6;font-size:15px;}}
  code{{font-family:'JetBrains Mono',ui-monospace,monospace;background:var(--bg3);color:var(--cyan);
    padding:.06em .35em;border-radius:4px;font-size:.86em;}}
  header{{max-width:880px;margin:0 auto;padding:42px 28px 8px;}}
  header h1{{margin:0;font-size:26px;font-weight:800;color:var(--accent);letter-spacing:-.02em;}}
  header .sub{{color:var(--dim);font-size:13px;margin:.3em 0 0;}}
  main{{max-width:880px;margin:0 auto;padding:18px 28px 60px;display:flex;flex-direction:column;gap:14px;}}
  .card{{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:16px 18px;
    border-left:3px solid var(--accent);}}
  .head{{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;}}
  .ts{{font-family:'JetBrains Mono',monospace;color:var(--cyan);font-size:13px;}}
  .who{{color:var(--dim);font-size:12.5px;text-transform:uppercase;letter-spacing:.5px;}}
  .goal{{font-size:16px;font-weight:600;margin:.5em 0 .8em;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px 22px;}}
  @media(max-width:620px){{.grid{{grid-template-columns:1fr;}}}}
  .lbl{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--accent);
    font-weight:700;margin-bottom:2px;}}
  .grid ul{{margin:.2em 0;padding-left:1.1em;}} .grid li{{margin:.15em 0;}}
  .grid p{{margin:.2em 0;}}
  .notes{{margin:.9em 0 0;padding-top:.7em;border-top:1px dashed var(--border);font-size:14px;}}
  .notes .lbl{{display:inline;margin-right:.4em;}}
  .dim{{color:var(--dim);}} .empty{{color:var(--dim);text-align:center;padding:40px;}}
</style></head>
<body>
<header><h1>Session log</h1><p class="sub">{len(entries)} session(s) · newest first · generated {generated}</p></header>
<main>
{body}
</main>
</body></html>"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
