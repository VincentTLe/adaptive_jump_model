"""AUDIT step 5: fault injection on COPIES of the runner and the map script.

Nothing under audit is edited. Each fault is one edit to a copy; the copy is run
and the outcome recorded as CAUGHT (non-zero exit / refusal) or SILENT.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/tle/adaptive_jump_model")
HERE = Path(__file__).resolve().parent
SRC_RUNNER = ROOT / "scripts/run_frequency_ladder.py"
SRC_MAP = ROOT / "scripts/diagnose_penalty_frequency_map.py"
SRC_SPEC = ROOT / "research/frequency-ladder-001.toml"
LADDER_STATES = ROOT / "artifacts/frequency-ladder/01-run"

# Every copy gets these two neutral rewrites so it writes to a scratch output
# directory and reads the already-fitted states from there (no refitting, and
# the audited artifact is never touched). Applied to the CONTROL too.
NEUTRAL = [
    ('ROOT = Path(__file__).resolve().parents[1]',
     'ROOT = Path("/home/tle/adaptive_jump_model")'),
    ('OUT = ROOT / "artifacts/frequency-ladder/01-run"',
     'OUT = Path("{out}")'),
    ('SPEC = ROOT / "research/frequency-ladder-001.toml"',
     'SPEC = Path("{spec}")'),
]

FAULTS = {
    "control": [],
    "F1_cutoff_2000": [('CUTOFF = pd.Timestamp("1990-01-01")',
                        'CUTOFF = pd.Timestamp("2000-01-01")')],
    "F2_cutoff_removed": [('    pre = refits[refits["fit_date"] < CUTOFF]',
                           '    pre = refits')],
    "F3_mean_not_median": [('return [float(np.nanmedian(per_target[t])) for t in ladder]',
                            'return [float(np.nanmean(per_target[t])) for t in ladder]')],
    "F4_left_edge": [('        midpoints = (lam[:-1] + lam[1:]) / 2.0\n'
                      '        per_year = np.diff(obj) / np.diff(lam) / TRAINING_YEARS\n'
                      '        for target in ladder:',
                      '        midpoints = lam[:-1]\n'
                      '        per_year = np.diff(obj) / np.diff(lam) / TRAINING_YEARS\n'
                      '        for target in ladder:')],
    "F5_inverted_test": [('            reached = np.flatnonzero(per_year <= target)',
                          '            reached = np.flatnonzero(per_year >= target)')],
    # markets swapped at the point of USE, so the gate's own lookup is untouched
    "F6_swap_de_jp_menus": [
        ('            menu = [float(v) for v in arm[market]]',
         '            _swap = {"de": "jp", "jp": "de"}.get(market, market)\n'
         '            menu = [float(v) for v in arm[_swap]]')],
    "F7_armL_labelled_armM": [
        ('            menu = [float(v) for v in arm[market]]',
         '            menu = [float(v) for v in arms["L_full_ladder"][market]]')],
    "F8_gate_dropped": [
        ('        if not np.allclose(rederived, frozen, rtol=1e-9, atol=1e-9):',
         '        if False:')],
    "F9_sealed_v10_states": [
        ('        cache = OUT / f"states-{market}.csv"',
         '        cache = BASE / market / "jm-states.csv"')],
}
# faults applied to the frozen SPEC copy rather than to the runner
SPEC_FAULTS = {
    "F10_armM_truncated_further": [
        ('us = [0.55, 1.465348864441625, 5.459611390906074, 23.5, 37.5]\n'
         'de = [0.55, 2.829145724599095, 8.59842836500576, 23.5, 90.0]',
         'us = [0.55, 1.465348864441625, 5.459611390906074, 23.5]\n'
         'de = [0.55, 2.829145724599095, 8.59842836500576, 23.5]')],
    "F11_ladder_changed": [
        ('ladder_shifts_per_year = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]',
         'ladder_shifts_per_year = [10.0, 4.0, 2.0, 1.0, 0.5, 0.25]')],
    "F12_grading_bands_widened": [
        ('C = "20 to 40 percent"', 'C = "20 to 80 percent"')],
    "F13_turnover_dropped_from_cells": [
        ('"expected_shortfall_5pct", "turnover", "leverage"]',
         '"expected_shortfall_5pct", "leverage"]')],
    "F14_cutoff_prose_changed": [
        ('aggregation = "median across training windows whose fit_date is BEFORE '
         '1990-01-01,',
         'aggregation = "median across training windows whose fit_date is BEFORE '
         '2010-01-01,')],
}


def build(name, edits, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in ("us", "de", "jp"):
        tgt = out_dir / f"states-{m}.csv"
        if not tgt.exists():
            shutil.copy(LADDER_STATES / f"states-{m}.csv", tgt)
    spec_copy = out_dir / "spec.toml"
    spec_text = SRC_SPEC.read_text()
    for old, new in SPEC_FAULTS.get(name, []):
        assert old in spec_text, f"{name}: spec anchor not found:\n{old}"
        assert spec_text.count(old) == 1, f"{name}: spec anchor not unique"
        spec_text = spec_text.replace(old, new)
    spec_copy.write_text(spec_text)
    text = SRC_RUNNER.read_text()
    for old, new in NEUTRAL:
        assert old in text, f"neutral anchor missing: {old}"
        text = text.replace(old, new.format(out=out_dir, spec=spec_copy))
    for old, new in edits:
        assert old in text, f"{name}: anchor not found:\n{old}"
        assert text.count(old) == 1, f"{name}: anchor not unique"
        text = text.replace(old, new)
    script = out_dir / "runner.py"
    script.write_text(text)
    return script


def main():
    results = []
    only = sys.argv[1:] or list(FAULTS) + list(SPEC_FAULTS)
    for name in only:
        edits = FAULTS.get(name, [])
        out_dir = HERE / name
        script = build(name, edits, out_dir)
        p = subprocess.run([sys.executable, str(script), "4"], capture_output=True,
                           text=True, cwd=ROOT, timeout=3600)
        tail = (p.stdout + p.stderr).strip().splitlines()
        note = next((ln for ln in reversed(tail)
                     if "refusing" in ln or "not a column" in ln
                     or "Error" in ln or "no training window" in ln), "")
        caught = p.returncode != 0
        # a fault that runs to completion but changes the reported numbers is
        # still SILENT for the purposes of this audit
        summary = out_dir / "summary.csv"
        got = summary.read_text().strip() if summary.exists() else ""
        results.append((name, p.returncode, "CAUGHT" if caught else "SILENT",
                        note[:150], got))
        print(f"\n### {name}: rc={p.returncode} "
              f"{'CAUGHT' if caught else 'SILENT'}\n{note[:300]}")
        if got:
            print(got)
    print("\n\n===== MATRIX =====")
    for name, rc, verdict, note, _ in results:
        print(f"{name:<26}{verdict:<8}rc={rc:<4}{note[:90]}")


if __name__ == "__main__":
    main()
