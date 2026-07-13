# Task: English Research Communication And Reproducible Reporting

## Identity

- `task_id`: `author-contact-and-english-docs-001`
- `status`: `complete`
- `target_branch`: `cleanup/research-protocol`
- `starting_ref`: `4812f230696748158eb173ef97cdb6a996c1bf90`
- `primary_class`: `ENGINEERING / DOCUMENTATION`
- `scientific_claim_change`: forbidden
- `adaptive_work`: forbidden
- `extension_access`: forbidden

The owner approved this task after the fixed-baseline proxy replication closed.
The completed protocol remains frozen in
`archive/completed-tasks/fixed-baselines-001.md` and `research.toml` v7.

## Goal

Make the project understandable to a first-time reader and reproducible without
changing its scientific result. All active user-facing documentation and newly
generated reports must be English.

The central communication artifact must compare three workflows:

1. the archived legacy implementation, including the minute stack and the
   pre-audit daily P0/P1/P2 pivot;
2. the current verified v7 proxy-replication pipeline;
3. the protocol documented in Shu, Yu, and Mulvey.

Every criticism of earlier work must cite repository evidence. Distinguish a
verified defect, an accepted approximation, and an unresolved ambiguity. Do not
attribute intent or unsupported claims to a person or agent.

## Deliverables

- `docs/research-workflow-comparison.html`: three complete workflow diagrams,
  an evidence-based comparison, discrepancy scenarios, and advisor talking
  points.
- `docs/author-data-request.txt`: a ready-to-send English email plus the exact
  series and implementation checklist requested from the paper authors.
- `docs/learning/index.html` and rewritten lessons `01` through `04`: progressive
  English explanations that assume no finance, statistics, or coding knowledge.
- A canonical in-package English report generator that reads only a verified
  sealed run and emits `report.html` without a hidden local script.
- A regenerated English v7 technical report with browser verification on
  desktop and mobile.

Historical files under `archive/` and append-only `.agent/` logs are provenance,
not active documentation, and must not be translated or rewritten.

## Teaching Contract

Each lesson must:

- begin with one plain-language question and one tiny numerical example;
- state the one idea the reader should remember before introducing notation;
- keep optional mathematics behind a clearly labeled deeper section;
- define every finance, statistics, and software term on first use;
- avoid assuming familiarity with returns, regimes, training, validation,
  Sharpe ratio, HMMs, dynamic programming, or backtesting;
- contain no manually maintained experiment result table;
- link empirical conclusions to the verified English report;
- render without overlap or horizontal page overflow at desktop and mobile
  widths.

## Evidence Boundary

Permitted evidence is limited to:

- `2402.05272v3.pdf`;
- the frozen source audit and official v7 artifacts already present locally;
- `research.toml`, active package code, tests, Git history, and handoff records;
- archived legacy source and documentation used only to describe prior work.

No new market data may be downloaded. No post-2023 result may be inspected. No
model, feature, grid, metric, gate, or conclusion may be changed.

## Commit And Acceptance Rules

- Keep each commit below approximately 400 changed lines and 15 files.
- Freeze this task before editing the report or lessons.
- Browser-check every HTML deliverable in Chromium at desktop and mobile widths.
- Run the complete test, Ruff, lock, package, and clean-archive acceptance suite.
- End with a sealed handoff. The next scientific action is author/data access,
  not adaptive-model implementation.

## Outcome

Completed on 2026-07-12 without changing the frozen scientific protocol,
market data, metrics, or conclusion.

- The advisor-facing comparison documents the legacy, current, and paper
  workflows with repository evidence and three complete visual diagrams.
- The author email requests the undisclosed paid-series identifiers and
  implementation conventions without treating proxy non-replication as a
  refutation.
- The four lessons are now a short English beginner sequence with no manually
  copied result tables.
- `adaptive-jump report --run <run_dir>` verifies the sealed run before writing
  a deterministic English report outside the immutable inventory.
- Chromium rendered all seven HTML deliverables at 1440x900 and 390x844 with no
  console errors, missing local links, or page-level horizontal overflow.
- The official v7 run still verifies 131 immutable files, 18 boundary rows, and
  27 recomputed metric rows. Its conclusion remains proxy non-replication, so
  adaptive and extension work remain blocked.
- The project environment and a clean Git archive passed 97 tests, Ruff check
  and format check, package build, dependency compatibility, lock validation,
  and CLI discovery.
