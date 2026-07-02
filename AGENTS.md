# AGENTS.md

## Project

This is a mathematical finance research project:
Duration-Calibrated Adaptive Jump Models for intraday regime detection.

The goal is not to build a large software product.
The goal is to implement small, testable research modules.

## Core Rule

Do one task at a time.
Do not expand scope.
Do not implement modules that were not explicitly requested.

## Research Execution Rule

This is a serious research project, not a toy demo project.

- Do not silently reduce requested experiments to save computation.
- Do not assume the owner wants the lightest possible demo.
- Quick mode is for debugging only.
- Full mode must perform real validation over available local data, relevant
  hyperparameter grids, sufficient seeds/initializations for stochastic models,
  and delay/cost sensitivity when backtesting.
- If full mode is slower, document expected runtime; do not fake a full mode.
- Backtesting is required when claiming market regimes are economically
  meaningful.
- Static dashboards/reports are useful research artifacts and should not be
  dismissed as unnecessary.
- Be honest about limitations and results, but do not under-test.

## Allowed Default Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- pytest
- ruff

Ask before adding any other dependency.

## Data Rules

- `data/raw/` is read-only.
- Never delete, overwrite, or modify raw data.
- Generated outputs must go to `data/processed/`.
- Never download large datasets unless explicitly requested.
- Never commit credentials, API keys, tokens, or paid-data files.

## Coding Rules

- Prefer simple, explicit code.
- Add type hints and docstrings for public functions.
- No silent fallbacks.
- No broad `except Exception: pass`.
- Raise explicit errors for invalid input.
- Keep notebooks thin. Core logic belongs in `src/adaptive_jump/`.

## Testing Rules

Every implementation task must include tests.

For mathematical algorithms:
- include small deterministic examples;
- include edge cases;
- include brute-force oracle tests when possible;
- test numerical invariants.

Do not claim completion until tests pass.

## Reporting Format

At the end of every task, report:

1. Files changed
2. What changed
3. Tests added
4. Commands run
5. Exact test result
6. Remaining risks

## Forbidden Without Explicit Approval

- changing public function names;
- changing file structure;
- adding dependencies;
- deleting files;
- rewriting unrelated code;
- implementing trading strategy logic;
- implementing HMM;
- implementing jump model before penalty and DP modules are complete;
- implementing dynamic programming before penalty rules are complete;
- implementing backtests before model correctness is verified.

When the current `TASK.md` explicitly allows HMMs, Jump Models, adaptive models,
backtests, reports, dashboards, or additional dependencies, that task-specific
approval is sufficient. Continue to obey the data, security, testing, and
reporting rules above.

## Session Handoff (mọi agent: Claude / Codex / Cursor)

Nguồn sự thật là `.agent/session-log.jsonl` (append-only, mỗi phiên đúng 1 dòng JSON).

- **Đầu phiên:** đọc dòng cuối của `.agent/session-log.jsonl` + chạy `git status` trước
  khi làm; nêu tình hình hiện tại và bước kế tiếp.
- **Cuối phiên:** append đúng 1 dòng JSON với các field
  `ts, agent, model, goal, files, verification, commit, next, notes`, rồi chạy
  `python3 .agent/render_log.py .agent` để dựng lại `session-log.html`.
- Đặt `agent` = `claude` / `codex` / `cursor`. Không sửa tay file HTML, không xoá
  dòng cũ, không bịa kết quả verification hay commit sha. Tiện ích: `bash .agent/handoff.sh`.
