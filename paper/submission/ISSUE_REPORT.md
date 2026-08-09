# Production issue report

Compiled with Tectonic 0.16 from a clean output directory. Every page of every
locally-buildable target was rendered and inspected.

## Compile status

| Target | File | Pages | Overfull hbox | Undefined citations | Undefined refs |
|---|---|---:|---:|---:|---:|
| Paper 1 — full author | `paper1_jam/paper1_one_risk_measure.tex` | 14 | 0 | 0 | 0 |
| Paper 1 — anonymous | `paper1_jam/paper1_anon.tex` | 13 | 0 | 0 | 0 |
| Paper 2 — local preview | `paper2_anor/paper2_local_preview.tex` | 10 | 0 | 0 | 0 |
| Paper 2 — preview anon | `paper2_anor/paper2_local_preview_anon.tex` | 9 | 0 | 0 | 0 |
| Paper 2 — **Springer submission** | `paper2_anor/paper2_adaptive_switching.tex` | — | — | — | — |

## Clipped text / overfull boxes

None. All tables use `siunitx` decimal-aligned `S` columns inside `booktabs` +
`threeparttable`; no `\resizebox` and no forced `[H]` placement are used. No
overfull `\hbox` was reported in any target. Visual inspection of the title
pages, the results tables, and the vector figure confirmed no clipped text.

## Unresolved citations / missing metadata

- No undefined citations or references in any compiled target.
- Two reference-metadata items remain flagged (see
  `audit/unresolved_references.md`), neither invented:
  - `markowitz1959`: 1959 monograph, **no Crossref DOI**; cited as a book.
  - `aydinhan2024`: **advance online publication**, no volume/issue/pages
    assigned in Crossref — deliberately left as "advance online publication".

## Unsupported claims

None introduced. Every number in both manuscripts is unchanged from the retained
experiments and matches the supplementary CSVs; main tables were only re-rounded
from six to three decimals, with exact values preserved in
`supplementary_S1/S2_exact_values.csv`. The claim-to-reference mapping is in
`audit/claim_reference_matrix.csv`. Both papers state their results as
exploratory/development evidence, not performance claims.

## Known limitation (environment, not manuscript)

The official Springer `paper2_adaptive_switching.tex`
(`\documentclass[sn-apa]{sn-jnl}`) **does not compile in this repository's
Tectonic sandbox**: the vendored `sn-jnl.cls` invokes a `geometry` internal
(`\Gm@savelength`) that the sandbox's TeX Live version clears before the class's
begin-document hook fires. This is a package-version mismatch in the local
engine, **not** a defect in the manuscript. The file is the correct submission
source and compiles on Overleaf ("Springer Nature LaTeX Template") or a full
TeX Live 2024+ install. The `paper2_local_preview.tex` twin — identical science,
tables, figure, and references via `article` + `apacite` (APA author-year) —
renders locally and was used for the page-level checks above.

## Remaining polish (non-blocking)

1. **Paper 1 figures** are high-resolution PNG, not vector. The Paper 2 whipsaw
   figure was produced as vector PDF; the three Paper 1 panels should be
   regenerated as vector PDF from the plotting source before final submission.
2. **Datasets/software as formal reference entries.** Public datasets (Yahoo,
   FRED, Bank of Japan) and software (`jumpmodels`, `hmmlearn`, Python) are
   currently documented in the *Data-* and *Code-availability* statements. If a
   target journal requires them as reference-list entries, promote them to
   `.bib` items using each provider's own suggested citation (do not infer
   metadata).
