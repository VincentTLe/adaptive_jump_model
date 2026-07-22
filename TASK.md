# Active Task: None in progress

## Last completed: holdout-2026-001 (window not supported; full walk-forward unchanged)

The public proxies were extended to 30 June 2026 and the previously untouched
2024-01-02 to 2026-06-30 window was opened once for a pre-declared variant,
DD-only JM.

Every evaluation here is walk-forward causal: each monthly decision uses only
trailing data, so the whole 2008/2009--2026 span is out-of-sample per decision.
There is no future leakage on any window. The only thing special about
2024-2026 is that it is free of *selection bias*: DD-only was chosen after
inspecting the through-2023 sample, and this window was not.

### Net Sharpe, both evaluation windows (both walk-forward)

| Market | window | B&H | HMM | Fixed JM | DD-only | DD-only beats both? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| US | full 2008-2026 | 0.5678 | 0.6326 | 0.5684 | **0.8903** | Yes (`+0.258`) |
| DE | full 2008-2026 | 0.3499 | 0.1096 | 0.2596 | 0.3102 | No |
| JP | full 2008-2026 | 0.6708 | 0.5596 | 0.5115 | 0.5597 | No |
| US | 2024-2026 only | **1.0521** | 0.5316 | 0.5737 | 0.7750 | No |
| DE | 2024-2026 only | 0.9041 | 0.9041 | 0.9041 | 0.9041 | No (exact tie) |
| JP | 2024-2026 only | **1.2701** | 1.2701 | 1.2701 | 1.1696 | No |

On the full walk-forward through 2026, DD-only still beats both controls in the
US (`1/3`, unchanged from development); adding 2.5 years barely moved the US
number. On the isolated 2024-2026 window the frozen binary rule returns
`not_supported` (`0/3`), but that window is short (~620 days), its paired
bootstrap intervals include zero, and it was a broad bull that penalizes any
cash rotation. So the window **fails to confirm** the US edge on
selection-independent data and **does not refute** the 18-year result. It is
weak evidence, not a clean negative.

## Provenance

- Precondition met: a fresh v7 replication at HEAD byte-matched the sealed
  baseline on all 123 scientific files before any post-2023 output was read.
- Frozen contract `research/holdout-2026-001.toml` (`7618924a8e67...`),
  registered before the window was opened.
- Evidence: `artifacts/holdout-2026-001/holdout-20260722T111757Z`.
- Extended acquisition: `data/raw/shu-proxy-holdout-2026-20260722T093350Z`
  (byte-identical overlap with sealed v7).

## Next candidates (each needs a fresh frozen question)

- A longer or regime-stratified holdout: 2.5 bull years cannot separate the
  variants. A stronger test needs either more post-selection time or a
  drawdown-conditional readout that does not credit cash rotation only when the
  market falls.
- Lagged-log4 batch 2 on the same window (available under the current contract).
- Semi-Markov dwell cost: the standing mathematical candidate that escapes the
  zero-diagonal degeneracy. Not yet frozen.
