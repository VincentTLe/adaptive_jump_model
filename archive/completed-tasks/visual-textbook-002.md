# Task: Visual Textbook V2.1

## Identity

- `task_id`: `visual-textbook-002`
- `status`: `complete`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `2b62f4c61de168a8bd753a9dfba557aba5a7108a`
- `primary_class`: `ENGINEERING / DOCUMENTATION`
- `scientific_claim_change`: forbidden
- `adaptive_experiment`: forbidden
- `extension_access`: forbidden
- `revision`: `v2.1`

The owner approved this task on 2026-07-13 after reviewing the completed
fourteen-chapter textbook and a research-backed visual redesign plan. The
completed predecessor is frozen at
`archive/completed-tasks/interactive-textbook-001.md`.

The owner approved the V2.1 amendment on 2026-07-13 after reviewing Part III.
The amendment addresses cognitive load and the missing bridge between teaching
schematics and the source paper. It does not reopen any scientific work.

## Goal

Turn the existing English textbook into an explorable, visual-first textbook
without removing its authored explanations. Preserve exactly 38 core teaching
figures across the fourteen chapters: 23 interactive and 15 static. Add four
original concept illustrations and six original paper-reading lenses, for
exactly 48 primary visual objects. Every object must support a specific finance,
statistics, model, protocol, evidence, or source-reading objective.

The work begins with the mathematical chapters, then returns to the foundation
and evidence chapters. It is documentation work only. It must not alter market
data, model code, configuration, protocol, frozen v7 results, gates, or claims.

## Locked Content And Evidence Boundary

- Preserve at least the baseline visible word count for every chapter.
- Keep all course copy, captions, controls, and labels in English.
- Use only deterministic toy values already present in the chapters.
- Do not download data, inspect post-2023 evidence, or run an adaptive model.
- Do not manually copy empirical result values into authored HTML.
- Do not extract or reproduce a paper figure without explicit reuse permission
  recorded in the media credits before the image is committed.
- Paper-reading lenses must be original schematics, cite the locked paper and
  exact figure/page, link to the source PDF, and state that they are not copied
  data. Direct reproduction may replace a lens only after permission is recorded.
- External images require a verified CC BY, CC BY-SA, CC0, public-domain, or
  equivalent compatible license and complete attribution. Do not rely on search
  result license filters without checking the source page.
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

The table above is the immutable 38-figure core. V2.1 adds these separately
tested primary objects:

| Additional object | Count | Primary placement |
| --- | ---: | --- |
| Original concept illustration | 4 | Chapters 01, 05, 08, and 13 |
| Original paper-reading lens | 6 | Chapters 05, 09, 10, 11, 13, and 14 |
| **Additional total** | **10** | **All static** |
| **Course total** | **48** | **25 static, 23 interactive** |

An illustration reused as a thumbnail on the course index remains one unique
asset and is not counted as another primary object.

## Learning Presentation Contract

Each important concept follows this reading sequence:

1. plain-language question and `$100` example;
2. teaching figure or direct manipulation;
3. caption and a short `What to notice` explanation;
4. native `details.deep-explanation` containing retained long-form prose;
5. mathematical layer, empirical evidence, limits, and advisor explanation.

Each chapter defaults to a keyboard-accessible `Visual path`. It keeps the
question, `$100` example, primary visuals, captions, takeaways, central formula,
evidence boundary, limitation, and advisor explanation visible. A `Full chapter`
mode reveals every retained explanation. The mode is page-local and is not
stored in `localStorage`. With JavaScript disabled, all authored content remains
available; print always expands the full chapter.

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
- avoid autoplay, random simulation, irrelevant decoration, and unlabeled
  result-bearing data;
- label editorial images `Concept illustration - not data or evidence`;
- label source lenses `Paper reading guide - original schematic` and keep them
  visually distinct from project evidence and teaching examples.

Interactive visuals support keyboard use and `Play`, `Pause`, `Previous`,
`Next`, and `Reset` where a sequence is shown. Frames advance every 900 ms only
after `Play`. Reduced-motion mode removes tweening without removing controls.
Animation may clarify causality, state change, or timing. Decorative looping and
motion that competes with reading remain forbidden.

## Media And Source Contract

- Store authored media locally under `docs/learning/assets/`; do not hotlink.
- Keep a human-readable English record in `docs/learning/media-credits.html`.
- For generated media, record purpose, generation date, tool, prompt summary,
  SHA-256, and the statement that the image is not empirical evidence.
- For reused media, record title, creator, canonical source URL, license URL,
  modification status, local path, and SHA-256 using TASL attribution.
- Generic trading-floor, skyline, laptop, and chart stock photos are not
  instructional assets and must not be added merely to fill space.
- Concept illustrations share one restrained editorial style and directly map
  to the adjacent lesson. They must not depict fabricated market values.
- Treat the course as potentially public. Fair-use assumptions are not a
  substitute for permission or a compatible license.

## Browser Architecture

- Keep the static HTML, shared `course.css`, shared `course.js`, and inline
  chapter scripts.
- Keep exact MathJax `4.1.3` and Chart.js `4.5.1` pins and integrity hashes.
- Add no framework, D3, Mermaid, Node build system, image library, or runtime
  dependency.
- Add no external image request. Bitmap concept illustrations must have explicit
  width/height, responsive constraints, useful alt text, and an aggregate local
  asset budget no greater than 6 MB.
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

The four concept illustrations anchor the parts without claiming evidence:

- **01 / Part I:** one $100 decision moving between market and cash through a
  calm period, loss, and recovery.
- **05 / Part II:** a sealed data-to-claim research chain with visible audit
  locks and a separated future.
- **08 / Part III:** observed returns above latent regimes, with features feeding
  HMM and JM inference paths.
- **13 / Part IV:** a paper claim passing through data parity, protocol,
  replication, and evidence gates.

The six source lenses connect the locked paper without copying its figures:

- **Figure 1 / PDF page 4 -> Chapter 05:** series identity and cumulative-return
  comparison.
- **Figure 2 / PDF page 9 -> Chapter 09:** rolling HMM conditional parameters.
- **Figure 3 / PDF page 12 -> Chapter 10:** rolling JM feature centers by state.
- **Figure 4 / PDF page 14 -> Chapter 11:** in-sample traceback versus online
  terminal states.
- **Figure 5 / PDF page 17 -> Chapter 13:** JM regimes, allocation, and strategy
  curves.
- **Figure 6 / PDF page 18 -> Chapter 14:** HMM regimes and the evidence
  comparison boundary.

## Write Boundary And Commit Order

Authorized substantive paths are `TASK.md`, its completed-task archive,
`docs/learning/`, and `tests/test_learning_docs.py`. Procedural handoff files
remain governed by `AGENTS.md`. Do not touch model, data, config, artifact, or
reporting code.

1. Freeze this task, archive the completed predecessor, repair Chapter 9, add
   shared visual primitives, and complete the first Part III visual checkpoint.
2. Commit this V2.1 contract amendment and stop for owner review.
3. Add the shared Visual/Full mode, media credits, static contracts, and browser
   acceptance before adding media.
4. Revisit Chapters 08-12 one chapter per commit for the Part III concept image
   and Figure 2-4 reading lenses, then browser-review Part III again.
5. Complete Chapters 01-04 one chapter per commit, including the Part I concept
   illustration, then browser-review Part I.
6. Complete Chapters 05-07 one chapter per commit, including the Part II concept
   illustration and Figure 1 lens, then browser-review Part II.
7. Complete Chapters 13-14 one chapter per commit, including the Part IV concept
   illustration and Figure 5-6 lenses, then browser-review Part IV.
8. Update exact index and media-credit counts, run full acceptance, close the
   task, and hand off.

Keep each commit below approximately 400 changed lines and 15 files. Stop at
each part checkpoint for owner review.

## Acceptance

- Static tests enforce per-chapter word floors and exact visual targets.
- Static tests separately enforce 38 core teaching figures, four concept
  illustrations, six source lenses, and 48 total primary visual objects.
- Every teaching figure has a caption, accessible identity, stable dimensions,
  deterministic output, and an appropriate no-JavaScript fallback.
- Every bitmap has nonzero dimensions, meaningful alt text, a local source,
  matching credit/hash metadata, and no broken or external request.
- Every source lens links the exact locked-PDF page, names its paper figure,
  distinguishes source claims from project findings, and copies no empirical
  curve or value without recorded permission.
- Visual/Full mode passes mouse, keyboard, no-JavaScript, reduced-motion, print,
  reload-default, and history-navigation checks without deleting text.
- MathML fixed-arity elements are valid; Chapter 9 has no `mjx-merror` and no
  formula overflow at 390 px.
- Browser checks cover 1440x900 and 390x844 in normal, external-blocked, and
  JavaScript-disabled modes.
- All controls, extrema, ties, steppers, resets, quizzes, navigation, canvas
  pixels, SVG bounds, keyboard paths, reduced motion, and print behavior pass.
- Full pytest, Ruff, lock, package, clean-archive, paper-hash, archive
  immutability, and frozen-v7 verification pass before closure.

## Outcome

Completed on 2026-07-13 without changing market data, model code, research
configuration, protocol, frozen v7 results, gates, or scientific claims.

- The fourteen chapters contain 43,571 visible words, 15 static and 23
  interactive core teaching figures, four concept illustrations, and six
  paper-reading lenses: exactly 48 primary visual objects.
- The course index reports the current chapter, part, word, and visual totals.
  The media registry reports all four local concept assets, all six original
  source lenses, their hashes, and the 336,344-byte aggregate asset size.
- The permanent learning-document contract passes 22 tests. The complete suite
  passes 119 tests; Ruff check and format check pass.
- Chromium passes 70 chapter page-modes: normal desktop and mobile,
  external-network blocked, JavaScript disabled, and print with reduced motion.
  Controls, extrema, ties, steppers, resets, quizzes, fallbacks, canvas pixels,
  SVG bounds, MathJax rendering, keyboard paths, and navigation pass.
- The package source distribution and wheel build successfully. The lockfile,
  installed-package compatibility, paper SHA-256, pre-existing archive
  immutability, and generated-file boundaries pass.
- The official through-2023 frozen v7 artifact still verifies 131 inventory
  files, 18 boundary rows, and 27 metric rows, with maximum metric difference
  `7.327471962526033e-15`. Its conclusion remains proxy non-replication, so
  adaptive and post-2023 work remain unopened and blocked.
