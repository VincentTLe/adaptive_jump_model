# Task: Interactive Adaptive Jump Model Textbook

## Identity

- `task_id`: `interactive-textbook-001`
- `status`: `active`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `bd130224da212a6934b006e7e7f525982ed71f49`
- `primary_class`: `ENGINEERING / DOCUMENTATION`
- `scientific_claim_change`: forbidden
- `adaptive_experiment`: forbidden
- `extension_access`: forbidden

The owner approved this task after reviewing the short four-lesson sequence.
The completed communication task is frozen at
`archive/completed-tasks/author-contact-and-english-docs-001.md`.

## Goal

Replace the short lessons with a long, progressive English textbook for a
reader who knows no finance, statistics, machine learning, or software. The
course must remain simple at each step without becoming a summary: fourteen
chapters, each approximately 2,500-3,500 visible words, should build from a
$100 example to the exact mathematics, protocol, paper, and verified project
evidence.

The course is one authored documentation product. It must not create a second
research pipeline, rerun a market experiment, or change the frozen v7 result.

## Locked Source And Evidence Boundary

The named paper is exactly:

- Shu, Yu, and Mulvey, *Downside Risk Reduction Using Regime-Switching
  Signals: A Statistical Jump Model Approach*;
- arXiv `2402.05272v3`, 17 September 2024, 22 pages;
- local file `2402.05272v3.pdf`;
- SHA-256 `141e48bd5ccaefe5d2c276a3c8772716b583b1df91517109d54a2452f4cb3af1`.

Use the local PDF as the protocol source of truth. Permitted project evidence
is limited to the frozen v7 config and artifacts, active source/tests, Git
history, and archived provenance used only to describe earlier work.

No new market data may be downloaded. No post-2023 evidence may be inspected.
No model, feature, grid, metric, gate, result, or conclusion may change. The
course may use deterministic toy data that is visibly labeled as teaching data.

## Course Map

### Part I: Money, Returns, And Risk

1. Money, Assets, And Cash
2. Prices, Dividends, And Returns
3. Risk And Downside Losses
4. Market Regimes And Persistence

### Part II: Honest Research

5. Data Parity And Proxy Replication
6. Reproducibility And Sealed Evidence
7. Backtesting Without Seeing The Future

### Part III: Features And Models

8. From Returns To Model Features
9. Hidden Markov Models From Zero
10. Clustering And Statistical Jump Models
11. Dynamic Programming And Online Inference
12. Walk-Forward Selection And Performance Measurement

### Part IV: The Paper And Our Evidence

13. A Guided Reading Of Shu, Yu, And Mulvey
14. The Paper, The Legacy Project, And The Verified V7 Run

Every chapter includes a paper connection. Chapter 13 supplies the integrated
guided tour of Sections 1-5, Equation (1), Tables 1-5, and Figures 1-6.

## Teaching Contract

Each chapter must include, in order:

1. one plain-language question and a hand-worked `$100` example;
2. one idea to remember before notation;
3. definitions for every new term, symbol, unit, index, and state convention;
4. at least two worked examples, including one common failure;
5. hand-built diagrams or deterministic interactive teaching visuals;
6. a precise paper connection and project code/artifact map;
7. layered mathematics: intuition, formula, symbol table, then derivation;
8. separate empirical evidence and limitations sections;
9. recall, calculation, error-diagnosis, and advisor-explanation exercises;
10. complete answers with feedback and cumulative recall from earlier chapters.

Use short annotated code excerpts only after the theory. Link to the exact
implementation and artifact. Do not reproduce whole modules. Do not copy
rounded empirical values into authored HTML; link current project values to the
verified generated report and describe paper results with exact table/figure
citations rather than copied result tables.

## Interactive Presentation Contract

JavaScript and browser dependencies are authorized only where they improve
understanding. The approved browser libraries are:

- MathJax `4.1.3` for accessible equation rendering;
- Chart.js `4.5.1` for deterministic toy-data charts.

Load exact versions, record source URLs and integrity hashes, and never use a
rolling `latest` URL. Do not add a JavaScript framework, Node build system, or
project runtime dependency. Shared authored CSS and JavaScript must stay small,
direct, and documented.

Every interaction must have a reset action, deterministic output, keyboard
operation, accessible labels, and visible non-canvas fallback content. If the
network, MathJax, Chart.js, or all JavaScript is unavailable, the chapter's
text, native MathML, data table, exercises, and conclusion must remain usable.

Interactions teach the concept; they must not add decoration, random market
simulation, gamification, live prices, or result-bearing calculations.

## Milestones And Commit Rules

1. Freeze this task and archive the completed predecessor.
2. Add the shared course design, interaction layer, and dependency manifest.
3. Complete and browser-review Part I before beginning Part II.
4. Complete and browser-review Part II before beginning Part III.
5. Complete and browser-review Part III before beginning Part IV.
6. Complete Part IV, full acceptance, task close, and handoff.

Keep each commit below approximately 400 changed lines and 15 files. Write one
chapter per commit. Update index and previous/next navigation atomically. Do
not retain the old short course as a parallel active stack or add redirects.

## Acceptance

- Fourteen chapters each contain approximately 2,500-3,500 visible words.
- Static checks validate English language, required sections, chapter order,
  internal links, unique IDs, dependency pins, and absence of Mermaid.
- Chromium opens every page at 1440x900 and 390x844 with no console error,
  missing local link, text overlap, blank equation/chart, or page overflow.
- Browser tests exercise each slider, stepper, quiz, reset, and navigation path.
- A JavaScript-disabled and external-network-blocked pass confirms fallbacks.
- Full pytest, Ruff, lock, package, clean-archive, and archive-immutability
  acceptance pass before task completion.
- End each part with a handoff and stop for owner review.
