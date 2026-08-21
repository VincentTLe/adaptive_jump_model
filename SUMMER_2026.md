# Summer 2026 Research Story

This file is the short human history of the project.

## 1. Rebuild the Shu-style Jump Model

The project started by trying to reproduce the regime-switching method and equity/cash results from Shu, Yu, and Mulvey.

Work completed:

- rebuilt market data from public sources;
- implemented the feature pipeline;
- implemented the fixed Jump Model walk-forward process;
- implemented monthly lambda selection;
- implemented the delayed equity/cash backtest with transaction costs (a signal
  at day `t` earns the return at `t+2`, at 10 basis points one way);
- compared results for the US, Germany, and Japan.

Main lesson:

> The paper cannot be reproduced exactly from public information alone.

Important missing or uncertain pieces include the original source data and some implementation choices such as the exact lambda grid and feature standardization details.

## 2. Try to improve the Jump Model

Several changes were tested, including simple challengers, adaptive switching penalties, confirmation rules, lagged evidence, and related controls.

Main lesson:

> No tested change has yet produced a clear improvement that should be treated as a final result.

Some variants changed regime behavior in interesting ways, but better-looking regime paths did not automatically produce better investment results.

## 3. Audit the foundation

Later checks found real problems in the research process and implementation.

Examples:

- a US index splice had deleted/misaligned one trading session;
- one holdout turnover calculation used the wrong scale;
- the German baseline grid-selection process was not clean;
- increasing the number of optimizer starts changed some grid conclusions;
- some tests looked stronger than they actually were;
- some earlier scientific claims had to be corrected or withdrawn.

Main lesson:

> Passing tests and producing sealed artifacts did not mean the research was scientifically settled.

## 4. Build too much research infrastructure

To prevent more mistakes, the repository accumulated contracts, hashes, registries, audit receipts, agent logs, and many tests.

This improved traceability, but it also created a new problem:

> The repository became difficult for the owner to understand.

By August, the project contained more process and historical machinery than a human needed for everyday research.

## 5. Current reset

As of 2026-08-21, the project is deliberately paused before any new model work.

The plan is:

1. independently understand and verify the core pipeline;
2. inventory the important experiments already run;
3. decide which results are still trustworthy;
4. only then choose the paper question.

DA-JM and other new model ideas are parked for now.

## What counts as progress now

Progress is not another experiment, another audit document, or another AI-generated framework.

Progress means being able to answer, in plain language:

- What data are used?
- How are features computed?
- How does the Jump Model produce states?
- How is lambda chosen?
- How do states become equity/cash positions?
- Which past experiments are trustworthy?
- What repeated scientific question remains after the audit?
