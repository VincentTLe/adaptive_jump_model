# Task: Visual Textbook V2

## Identity

- `task_id`: `visual-textbook-002`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `2b62f4c61de168a8bd753a9dfba557aba5a7108a`
- `primary_class`: `ENGINEERING / DOCUMENTATION`
- `scientific_claim_change`: forbidden
- `adaptive_experiment`: forbidden
- `extension_access`: forbidden

The owner approved this task on 2026-07-13 after reviewing the completed
fourteen-chapter textbook and a research-backed visual redesign plan. The
completed predecessor is frozen at
`archive/completed-tasks/interactive-textbook-001.md`.

## Goal

Turn the existing English textbook into an explorable visual textbook without
removing its authored explanations. Add exactly 38 teaching figures across the
fourteen chapters: 23 interactive and 15 static. Each figure must answer a
specific finance, statistics, model, protocol, or evidence question.

The work begins with the mathematical chapters, then returns to the foundation
and evidence chapters. It is documentation work only. It must not alter market
data, model code, configuration, protocol, frozen v7 results, gates, or claims.

## Locked Content And Evidence Boundary

- Preserve at least the baseline visible word count for every chapter.
- Keep all course copy, captions, controls, and labels in English.
- Use only deterministic toy values already present in the chapters.
- Do not download data, inspect post-2023 evidence, or run an adaptive model.
- Do not manually copy empirical result values into authored HTML.
- Do not reproduce paper figures; create original teaching schematics.
- Keep the current conclusion: proxy non-replication and adaptive work blocked.

The locked paper remains Shu, Yu, and Mulvey, arXiv `2402.05272v3`, local
SHA-256 `141e48bd5ccaefe5d2c276a3c8772716b583b1df91517109d54a2452f4cb3af1`.

## Baseline And Visual Inventory

| Chapter | Baseline words | Static | Interactive |
| --- | ---: | ---: | ---: |
| 01 | 2,637 | 2 | 1 |
| 02 | 2,518 | 1 | 2 |
| 03 | 2,500 | 1 | 2 |
| 04 | 2,561 | 2 | 1 |
| 05 | 2,582 | 1 | 1 |
| 06 | 2,575 | 1 | 1 |
| 07 | 2,531 | 1 | 2 |
| 08 | 2,713 | 1 | 2 |
| 09 | 2,677 | 1 | 2 |
| 10 | 2,543 | 1 | 2 |
| 11 | 2,590 | 1 | 2 |
| 12 | 2,686 | 1 | 2 |
| 13 | 3,228 | 0 | 2 |
| 14 | 3,406 | 1 | 1 |
| **Total** | **37,747** | **15** | **23** |

## Learning Presentation Contract

Each important concept follows this reading sequence:

1. plain-language question and `$100` example;
2. teaching figure or direct manipulation;
3. caption and a short `What to notice` explanation;
4. native `details.deep-explanation` containing retained long-form prose;
5. mathematical layer, empirical evidence, limits, and advisor explanation.

The question, example, caption, takeaway, formula, evidence boundary,
limitation, and advisor explanation remain visible. Deep explanations are
closed by default on screen and expanded for print. Text may move into this
layer but must not be deleted, shortened, or silently rewritten.

Every visual must:

- carry a visible `Teaching example` label;
- have a semantic heading or accessible name and a `figcaption`;
- encode meaning with labels or shapes as well as color;
- remain useful with JavaScript and external requests disabled;
- expose deterministic state through accessible text or a table;
- avoid autoplay, random simulation, decoration, and result-bearing data.

Interactive visuals support keyboard use and `Play`, `Pause`, `Previous`,
`Next`, and `Reset` where a sequence is shown. Frames advance every 900 ms only
after `Play`. Reduced-motion mode removes tweening without removing controls.

## Browser Architecture

- Keep the static HTML, shared `course.css`, shared `course.js`, and inline
  chapter scripts.
- Keep exact MathJax `4.1.3` and Chart.js `4.5.1` pins and integrity hashes.
- Add no framework, D3, Mermaid, Node build system, image library, or runtime
  dependency.
- Use Chart.js for quantitative charts, semantic inline SVG for custom diagrams,
  and HTML grids/tables for state and evidence structures.
- Inline SVG contains a meaningful initial state before JavaScript runs.
- Canvas fallback tables remain visible unless chart rendering succeeds.
- `window.Course` may add a small stepper helper, a reduced-motion query, and
  shared semantic colors; existing `money()` and `percent()` remain stable.
- Keep `course.css` near 400 lines, `course.js` near 220 lines, and chapter
  files near 450 lines. Pause before exceeding those flags.

## Chapter Deliverables

- **01:** allocation split, before/after wealth, crash/rebound opportunity cost.
- **02:** corporate-action storyboard, ending wealth chart, compounding stepper.
- **03:** path comparison, wealth/peak/drawdown, interactive metric lens.
- **04:** observed/hidden layers, persistence graph, causal latency lanes.
- **05:** data identity comparison and exact-parity conjunction matrix.
- **06:** sealed evidence chain and tamper-point verifier response.
- **07:** causal clock, trailing-window scrubber, delayed net-return lanes.
- **08:** EWM decay chart, shock propagation, trailing scaler schematic.
- **09:** HMM anatomy, Gaussian emission explorer, Viterbi trellis stepper.
- **10:** objective anatomy, centroid feature space, objective/path response.
- **11:** path-tree compression, Bellman stepper, online/offline comparison.
- **12:** rolling windows, selection surface/gate, performance pipeline.
- **13:** paper claim-chain navigator and linked Equation (1) explanation.
- **14:** three workflow lanes and evidence/claim classification matrix.

## Write Boundary And Commit Order

Authorized substantive paths are `TASK.md`, its completed-task archive,
`docs/learning/`, and `tests/test_learning_docs.py`. Procedural handoff files
remain governed by `AGENTS.md`. Do not touch model, data, config, artifact, or
reporting code.

1. Freeze this task and archive the completed predecessor.
2. Repair Chapter 9 MathML in a dedicated commit.
3. Add shared visual primitives and their static contract.
4. Complete Chapters 08-12 one chapter per commit, then browser-review Part III.
5. Complete Chapters 01-04 one chapter per commit, then browser-review Part I.
6. Complete Chapters 05-07 one chapter per commit, then browser-review Part II.
7. Complete Chapters 13-14 one chapter per commit, then browser-review Part IV.
8. Update exact index counts, run full acceptance, close the task, and hand off.

Keep each commit below approximately 400 changed lines and 15 files. Stop at
each part checkpoint for owner review.

## Acceptance

- Static tests enforce per-chapter word floors and exact visual targets.
- Every teaching figure has a caption, accessible identity, stable dimensions,
  deterministic output, and an appropriate no-JavaScript fallback.
- MathML fixed-arity elements are valid; Chapter 9 has no `mjx-merror` and no
  formula overflow at 390 px.
- Browser checks cover 1440x900 and 390x844 in normal, external-blocked, and
  JavaScript-disabled modes.
- All controls, extrema, ties, steppers, resets, quizzes, navigation, canvas
  pixels, SVG bounds, keyboard paths, reduced motion, and print behavior pass.
- Full pytest, Ruff, lock, package, clean-archive, paper-hash, archive
  immutability, and frozen-v7 verification pass before closure.
