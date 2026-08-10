#!/usr/bin/env python3
"""Verify annotated paper quotations and configured paper-body absence terms.

Why this exists
---------------
Repeatedly in this project, assertions about what Shu, Yu and Mulvey (2024)
"say" turned out to be reconstructions rather than quotations: a clipping recipe
that lives only in the authors' example notebook, and an arithmetic
"contradiction" between "3000 days" and "12 years" that evaporates once you read
the word "approximately" sitting between them. Both survived because nobody
re-opened the paper at the cited line.

This script turns that discipline into a narrow, reproducible test. It scans
documents for annotated citations in the form

    [line 397] "given an observation sequence of D standardized features"

opens the extracted paper text at that line, and fails if the quote is not
there. It also runs a REFUTATION pass for five configured claims of absence
("the paper never mentions X"), scanning only the paper body and failing if
anything turns up.

It does not verify unannotated prose or semantically verify every claim in a
target. Targets with no annotated citations are reported as UNANNOTATED so a
successful run cannot be mistaken for full prose coverage.

The design follows Chain-of-Verification (Dhuliawala et al., ACL Findings 2024,
arXiv:2309.11495): checks are executed against the source in isolation from the
narrative that produced them, so a confident draft cannot bias its own audit.

Usage
-----
    uv run python scripts/audit/check_paper_claims.py                # all defaults
    uv run python scripts/audit/check_paper_claims.py FILE [FILE...]
    uv run python scripts/audit/check_paper_claims.py --paper /tmp/shu.txt

Regenerate the paper text with:
    pdftotext -layout data/external/inputs/2402.05272v3.pdf /tmp/shu.txt
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Pinned in the repository on purpose. A copy under /tmp was silently rewritten
# by a parallel process mid-audit, which shifted every line number by up to ten
# and made correct citations look fabricated. Line numbers are only meaningful
# against one fixed extraction, so that extraction lives here with a hash.
DEFAULT_PAPER = REPO / "data" / "external" / "inputs" / "shu_paper.txt"
PAPER_SHA256 = "dc81ad9d96767f86a4bb"  # first 20 hex chars; full check below
DEFAULT_TARGETS = [
    REPO / "docs" / "unspecified-choices.md",
    *sorted((REPO / "docs" / "audit").glob("*.md")),
    REPO / "CLAUDE.md",
    # The two documents a reader actually reads. An external review pointed out
    # that this checker claimed to "verify every claim" while skipping exactly
    # the manuscript and the README -- the only places a misquote reaches an
    # audience outside the repository.
    REPO / "paper" / "manuscript.tex",
    REPO / "README.md",
    # Frozen contracts justify their settings by quoting the paper, so those
    # quotes need checking for the same reason the prose does -- a config
    # comment is where a misquote does the most damage, because it is the
    # thing a run is sealed against.
    *sorted(REPO.glob("research*.toml")),
]

# Citations look like: [line 397] "quote"  or  [dòng 397] "quote"
CITATION = re.compile(
    r"\[(?:line|dòng|lines|dòng)\s+(\d+)(?:\s*[-–]\s*\d+)?\]\s*[\"“”']([^\"“”']{12,})[\"“”']",
    re.IGNORECASE,
)

# Claims of absence: the paper is asserted NOT to contain these. Each entry is
# (label, regex). A hit anywhere in the paper is a failure, because it means an
# absence claim in our documents is false.
ABSENCE_CLAIMS: list[tuple[str, str]] = [
    ("feature clipping / winsorising", r"\bclip\w*|\bwinsor\w*"),
    ("explicit annualisation factor", r"\b252\b|\b365\b|\b360\b|day[- ]count"),
    ("dividend construction of the total-return series", r"\bdividend\w*"),
    ("Bloomberg tickers or GFD series codes", r"\bticker\b|\bSPXT\b|\bNKYTR\b"),
    ("Saturday trading sessions", r"\bsaturday\b"),
]

# How far a quote may drift from the cited line before it counts as a bad
# citation. pdftotext wraps lines, so a quote legitimately spans a few of them.
LINE_TOLERANCE = 3


def normalise(text: str) -> str:
    """Collapse whitespace and unicode variants so quotes compare fairly."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("−", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def load_paper(path: Path) -> tuple[list[str], str]:
    # The extraction is not committed: the paper is copyrighted and
    # data/external/ is deliberately ignored. Rebuild it from the local PDF
    # instead, so the hash pin below still anchors every line number.
    if not path.exists() and path == DEFAULT_PAPER:
        pdf = REPO / "data" / "external" / "inputs" / "2402.05272v3.pdf"
        if pdf.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["pdftotext", "-layout", str(pdf), str(path)], check=True)
            print(f"regenerated {path.name} from {pdf.name}")
    if not path.exists():
        sys.exit(
            f"paper text not found: {path}\nregenerate with:\n"
            f"  pdftotext -layout {REPO}/data/external/inputs/2402.05272v3.pdf {path}"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path == DEFAULT_PAPER and not digest.startswith(PAPER_SHA256):
        sys.exit(
            f"paper text changed: {path}\n"
            f"  expected sha256 {PAPER_SHA256}..., found {digest[:20]}...\n"
            "Every line number in our documents refers to the pinned extraction. "
            "Re-cite before updating the pin."
        )
    # Split on newlines only. str.splitlines() also breaks on the form feeds
    # pdftotext emits at page boundaries, which shifts every index by one per
    # page and makes correct citations look wrong -- the first bug this script
    # found was its own.
    lines = raw.decode("utf-8", errors="replace").split("\n")
    return lines, normalise(" ".join(lines))


def window(lines: list[str], centre: int, radius: int) -> str:
    lo = max(0, centre - 1 - radius)
    hi = min(len(lines), centre + radius)
    return normalise(" ".join(lines[lo:hi]))


# A quote long enough to wrap gets a comment marker on every continuation
# line, which lands in the middle of the quoted string and makes a correct
# citation look fabricated. Strip leading markers before matching. Markdown
# headings lose their leading '#' too; that is harmless here, since nothing
# downstream reads anything but the quoted spans.
COMMENT_PREFIX = re.compile(r"^[ \t]*(?:#+|//+|\*)[ \t]?", re.MULTILINE)


def check_citations(
    target: Path, lines: list[str], whole: str
) -> tuple[list[str], int]:
    failures: list[str] = []
    text = target.read_text(encoding="utf-8", errors="replace")
    text = COMMENT_PREFIX.sub("", text)
    hits = list(CITATION.finditer(text))
    for match in hits:
        cited = int(match.group(1))
        raw_quote = match.group(2)
        quote = normalise(raw_quote)

        # Elided quotes ("first part ... second part") are standard practice.
        # Check each fragment separately: the first against the cited line, the
        # rest anywhere after it, since an ellipsis may span many lines.
        if "..." in raw_quote or "…" in raw_quote:
            parts = [
                normalise(p)
                for p in re.split(r"\.\.\.|…", raw_quote)
                if len(normalise(p)) >= 12
            ]
            head_ok = parts and parts[0] in window(lines, cited, LINE_TOLERANCE)
            tail_ok = all(p in whole for p in parts[1:])
            if head_ok and tail_ok:
                continue
            missing = [p for p in parts if p not in whole]
            failures.append(
                f"{target.name}: elided quote at line {cited} — "
                + (
                    f'fragment not in paper: "{missing[0][:50]}..."'
                    if missing
                    else "first fragment is not at the cited line"
                )
            )
            continue

        if quote in window(lines, cited, LINE_TOLERANCE):
            continue
        if quote in whole:
            found = next(
                (
                    i + 1
                    for i in range(len(lines))
                    if quote[:40] in window(lines, i + 1, LINE_TOLERANCE)
                ),
                None,
            )
            failures.append(
                f"{target.name}: quote cited at line {cited} actually appears "
                f'near line {found}: "{match.group(2)[:60]}..."'
            )
        else:
            failures.append(
                f"{target.name}: NOT IN PAPER, cited at line {cited}: "
                f'"{match.group(2)[:60]}..."'
            )
    return failures, len(hits)


def body_end(lines: list[str]) -> int:
    """First line of the reference list, so page numbers and volume/page
    ranges in citations cannot masquerade as method statements."""
    for i, line in enumerate(lines):
        if line.strip().lower() == "references":
            return i
    return len(lines)


def check_absences(lines: list[str]) -> list[str]:
    body = lines[: body_end(lines)]
    failures = []
    for label, pattern in ABSENCE_CLAIMS:
        rx = re.compile(pattern, re.IGNORECASE)
        hits = [(i + 1, ln.strip()) for i, ln in enumerate(body) if rx.search(ln)]
        # Line 170 states the paper does NOT process outliers; it is the
        # evidence for the absence claim, not a counterexample.
        hits = [(n, ln) for n, ln in hits if "do not process outlier" not in ln]
        if hits:
            where = "; ".join(f"line {n}: {ln[:70]}" for n, ln in hits[:3])
            failures.append(f"ABSENCE CLAIM BROKEN — {label}: {where}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify annotated line quotations and five configured paper-body "
            "absence terms. Unannotated prose is not checked."
        )
    )
    ap.add_argument("targets", nargs="*", type=Path, default=None)
    ap.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    args = ap.parse_args()
    targets = args.targets or DEFAULT_TARGETS

    lines, whole = load_paper(args.paper)
    print(f"paper: {args.paper} ({len(lines)} lines)")

    failures: list[str] = []
    total = 0
    unannotated = 0
    for target in targets:
        if not target.exists():
            print(f"  skip (missing): {target}")
            continue
        bad, n = check_citations(target, lines, whole)
        total += n
        failures.extend(bad)
        name = target.relative_to(REPO) if REPO in target.parents else target
        if n == 0:
            unannotated += 1
            print(f"  {name}: UNANNOTATED — 0 line quotation(s); prose not checked")
        else:
            print(f"  {name}: {n} annotated line quotation(s), {len(bad)} bad")

    absence = check_absences(lines)
    failures.extend(absence)
    print(
        "  paper-body absence terms: "
        f"{len(ABSENCE_CLAIMS)} configured, {len(absence)} broken"
    )

    print()
    if failures:
        print(f"FAIL — {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"OK — {total} annotated line quotation(s) verified; "
        f"{len(ABSENCE_CLAIMS)} configured paper-body absence terms checked; "
        f"{unannotated} target(s) contain unannotated prose that was not checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
