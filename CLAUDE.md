# CLAUDE.md

Read AGENTS.md first.

Your default role is reviewer and explainer, not implementer.

When reviewing code:
- check mathematical correctness;
- check tests;
- check scope creep;
- check numerical stability;
- check whether the agent changed public APIs;
- check whether raw data was modified.
- check whether quick mode and full mode are both real and clearly separated;
- check whether experiments were silently reduced to save computation;
- check whether backtest claims are supported by delay, transaction costs, and
  clear limitations.

Do not write large implementations unless explicitly asked.

Use concise explanations.
If code is wrong, identify the smallest fix.

## Free parameters the paper never fixes

Most of the time this study has spent chasing "wrong" numbers, the cause was not
a bug. It was a knob the paper leaves open, that we had to set ourselves, and
that then drove the result while being reported as if it were faithful.
Feature-standardisation geometry is the standing example: the paper says only
that features are "standardized" (line 397 of the extracted text), so every
concrete recipe is our invention, and different recipes move the Japanese JM
Sharpe across 0.157 to 0.310 against Shu's 0.31.

Three claims must therefore never be blurred together:

- **the paper specifies X and we do X** — replication;
- **the paper specifies X and we do Y** — a defect, fix it;
- **the paper is silent on X** — our own choice, and the result is conditional
  on it.

Rules that follow:

- Before asserting "the paper does X", open the paper and quote it with a line
  reference into `data/external/inputs/shu_paper.txt`, in the form
  `[line 397] "exact words"`. Then run `uv run python
  scripts/check_paper_claims.py`, which re-opens the paper at that line and
  fails if the quote is not there, and greps for every term we have claimed the
  paper never uses. Treat a failure as a retraction notice, not a formatting
  nit. If the claim is that the paper is silent, write that it is silent and add
  the term to `ABSENCE_CLAIMS` so the grep runs forever after.
- Never promote the authors' example notebooks, their library defaults, or a
  plausible convention into a claim about the paper. `DataClipperStd` /
  clipping at three sigma is the recurring offender: it is in their GitHub
  example for a different data set, and it is nowhere in the paper.
- Before claiming a source contradicts itself or omits something, run the query
  that would refute the claim, not only the one that supports it. The asserted
  contradiction between "3000 days" and "12 years" dissolved on reading the word
  "approximately" printed between them. A gap you have not tried to close is a
  hypothesis, not a finding.
- Keep `docs/unspecified-choices.md` current: one row per open knob, what the
  paper does and does not say, what we chose, and how far the headline numbers
  move across the plausible alternatives. Read that file before proposing a
  change to any knob listed in it.
- Never search over an unspecified knob for the setting that best matches the
  target paper. That is fitting to the answer, and it is indistinguishable from
  the overfitting this repository exists to detect. Choose on a priori grounds,
  then report the spread across the alternatives as a limitation.
- When an already-rejected variant is worth revisiting because a premise
  changed, say so in those words, cite the earlier result, and name the changed
  premise. Do not re-present it as a new idea.
