# Submission packages — build & contents

Two independent submission packages produced from the retained experiments. No
reported number, sample, date, model definition, or claim was changed; where the
earlier drafts printed six decimals, the main tables here show three and the
exact values live in the per-paper supplementary CSV.

```
submission/
  paper1_jam/                 Journal of Asset Management (Chicago author-date)
    paper1_one_risk_measure.tex   full-author build (compile this)
    paper1_anon.tex               anonymous-review build
    references.bib                Crossref-verified, DOIs printed
    supplementary_S1_exact_values.csv
    cover_letter_paper1.md
    figures/ (fig_us_paths, fig_wealth_markets, fig_dd_scale_paths)
  paper2_anor/                Annals of Operations Research (Springer sn-jnl, APA)
    paper2_adaptive_switching.tex  OFFICIAL submission (documentclass sn-apa/sn-jnl)
    paper2_anon.tex                anonymous official build
    paper2_local_preview.tex       local-render twin (article + apacite)
    paper2_local_preview_anon.tex  anonymous local twin
    paper2_frontdata.tex           shared title/abstract/keywords
    paper2_content.tex             shared body (identical science in both builds)
    references.bib                 Crossref-verified
    supplementary_S2_exact_values.csv
    cover_letter_paper2.md
    sn-jnl.cls, sn-apacite.bst     vendored official Springer template files
    figures/ (fig_whipsaw.pdf, vector)
  audit/
    citation_audit.csv            per-reference verification log
    unresolved_references.md      markowitz1959 (no DOI) + aydinhan2024 (advance online)
    overlap_audit.md              >12-word identical-phrase audit
    claim_reference_matrix.csv    claim -> supporting reference(s)
```

## How to build

Engine used here: **Tectonic 0.16** (bundles TeX Live + an internal BibTeX).

### Paper 1 — builds locally and on Overleaf/TeX Live
```
cd paper1_jam
tectonic -X compile paper1_one_risk_measure.tex   # full author  (14 pp)
tectonic -X compile paper1_anon.tex               # anonymous    (13 pp)
```
Uses `natbib` + `chicagoa.bst` (Chicago author-date), `siunitx`, `booktabs`,
`threeparttable`, `cleveref`, `microtype`. Main tables are 3-decimal; no
`\resizebox`, no forced `[H]`.

### Paper 2 — official submission vs. local preview
- **Submission file:** `paper2_adaptive_switching.tex`,
  `\documentclass[sn-apa]{sn-jnl}` with APA (author-year) references via
  `sn-apacite.bst`. **Build it on Overleaf** ("Springer Nature LaTeX Template")
  **or a full TeX Live 2024+ install**, which is the environment Springer
  supports for this class. The class files are vendored here for convenience.
- **Known local limitation:** the vendored `sn-jnl.cls` does not compile under
  this repository's Tectonic sandbox — the class calls a `geometry` internal
  (`\Gm@savelength`) that the sandboxed TeX Live version clears before the
  class's begin-document hook. This is an environment mismatch, not an error in
  the manuscript. It is documented here so no one mistakes it for a content bug.
- **Local-render twin (for page-checking):** `paper2_local_preview.tex` reuses
  the *identical* `paper2_frontdata.tex` + `paper2_content.tex` through an
  `article` + `apacite` (APA author-year) wrapper and renders in Tectonic
  (10 pp; anonymous twin 9 pp). Its science, tables, figure, and reference list
  are the same as the submission file; only the class chrome differs.
```
cd paper2_anor
tectonic -X compile paper2_local_preview.tex        # local APA preview
# on Overleaf / full TeX Live:
#   pdflatex paper2_adaptive_switching ; bibtex ... ; pdflatex x2
```

## References & datasets

All 10 scholarly references were verified against Crossref on 2026-07-23 (see
`audit/citation_audit.csv`). DOIs are authoritative. Two items are flagged in
`audit/unresolved_references.md`: `markowitz1959` (1959 monograph, no Crossref
DOI — cited as a book) and `aydinhan2024` (advance online publication, no volume
or pages assigned — do not infer).

**Public datasets** are documented in each paper's *Data availability* statement,
identified by provider and ticker:
- Yahoo Finance: `^SP500TR`, `^GDAXI`, `^N225` (equity proxies).
- FRED (Federal Reserve Bank of St. Louis): `DTB3`, `IR3TIB01DEM156N` (short rates).
- Bank of Japan: `STRACLUC3M` (call-rate proxy).

**Software** materially used: Python 3.12, `jumpmodels` v0.1.1, `hmmlearn`,
NumPy, pandas — stated in each paper's *Code availability* statement.

> Remaining polish items (not blocking review): (1) regenerate the three Paper 1
> figures as vector PDF from the plotting source (currently high-resolution PNG);
> (2) if the target journals require datasets/software as formal reference-list
> entries rather than availability-statement prose, promote the items above to
> `.bib` entries.
