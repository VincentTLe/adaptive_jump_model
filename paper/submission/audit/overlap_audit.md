# Overlap audit — Paper 1 vs Paper 2 (final submission drafts)

Method: LaTeX commands, math, tables, figures, and the bibliography were stripped
from each manuscript; the remaining prose was lowercased and tokenised; every
identical word sequence **strictly longer than 12 words** (n = 13) shared between
the two papers was extracted and merged into maximal phrases.

Sources compared (final versions):
`paper1_jam/paper1_one_risk_measure.tex` vs
`paper2_anor/paper2_content.tex` + `paper2_anor/paper2_frontdata.tex`.

Approx. substantive body words: Paper 1 ≈ 3,500; Paper 2 ≈ 2,730.

## Identical phrases longer than 12 words: **1**

1. **[23 words]** "the author received no specific grant from any funding agency
   in the public, commercial, or not-for-profit sectors. The author declares no
   competing interests."
   - **Location:** the *Funding* and *Competing interests* declarations of both
     papers.
   - **Classification: EXEMPT.** This is the standard ICMJE / Springer Nature
     funding-and-competing-interests boilerplate. Journals expect this exact
     wording; deliberately paraphrasing a standard declaration to defeat a
     similarity check is discouraged. It is not narrative or scientific content.

## Scientific-content overlap: **0**

No shared abstract sentence, no shared theorem/proposition statement, no shared
table or figure caption, and no repeated paragraph of introduction, methods,
results, discussion, or conclusion were detected. The only shared 13+-word span
in the earlier draft (the protocol sentence "…walk-forward penalty selection, a
one-day trading delay, and 10 basis points…") was removed by rewording Paper 2's
empirical-design sentence ("monthly penalty reselection, next-day execution, and
a ten-basis-point one-way cost") and Paper 1's abstract ("monthly walk-forward
model selection … ten basis points").

## Separation checklist (brief)

- No shared abstract, principal figure, principal table, or theorem — OK.
- Paper 1 contains **no** adaptive derivations (arrival/lagged/pair-balanced
  costs, binary recursion, amplification identity) — OK.
- Paper 2 does **not** present DD-only as a principal contribution; DD-only is
  mentioned only as the fixed parent feature set — OK.
- Shared protocol description is concise and independently worded — OK.

Re-run command: see `submission/` overlap re-run script in the session log.
