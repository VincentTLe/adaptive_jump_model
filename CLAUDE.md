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

## Communicating research results

The owner is an undergraduate researcher using AI-assisted coding.

The owner does not need to understand every implementation detail, but must be
able to understand every result-affecting assumption, model change, parameter
choice, experiment-design choice, and scientific conclusion.

Use plain English. All project material is English-only; see the language rule
in `AGENTS.md`.

Avoid jargon when simpler language is available. When a technical term is
necessary, explain what question it answers before using it heavily —
*optimizer nonuniqueness*: does the same model give a different answer when it
starts from a different initialization? *paired delta*: under identical
conditions, how much better than the JM is the new model? *baseline nesting*:
is there a setting that turns the new model back into the old one? *causality*:
does today's decision accidentally use future data?

Never report an important number without saying what it means. Not
"DE spread = 0.0117", but "in Germany, changing optimizer initialization moved
the measured Sharpe by about 0.012 in this diagnostic. This is a sensitivity
measurement, not a threshold that a future model must exceed."

Every completed research task begins with:

### What you need to understand

Maximum five bullets. Plain English. No unexplained jargon.

Then:

### What changed

### What did not change

Explicitly state whether the model changed, the data changed, P&L changed, and
the scientific conclusion changed.

### What could still be wrong

### Technical details

Meaning must come before machinery. Do not lead with hashes, test counts,
implementation internals, statistical terminology, or long formulas. First
answer: (1) What happened? (2) Why does it matter? (3) Does it change the
scientific conclusion? (4) What should happen next? Only then give technical
details.

If the owner says he does not understand something, simplify it. Do not answer
with more jargon or more detail.

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
