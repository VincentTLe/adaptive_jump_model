# DRAFT — NOT SENT. Owner reviews, signs, and sends from their own address.

Supersedes `author-data-request.txt` (v7-era; its "does not reproduce the
directional ordering" framing is obsolete — v8.3 reproduced JM>HMM>B&H in all
three markets, and the accounting proof below is far stronger than anything
that draft could offer). The ask is narrowed from that draft's full checklist
to the two items that match the measured residual exactly.

RECIPIENTS

Yizhan Shu <yizhans@princeton.edu>
Chenyu Yu <chenyu@princeton.edu>
John M. Mulvey <mulvey@princeton.edu>

SUBJECT

Replication of arXiv:2402.05272 — request for the Figure 5/6 state sequences
(or the standardization recipe + λ grid)

EMAIL

Dear Dr. Shu, Mr. Yu, and Professor Mulvey,

I am completing an independent replication of "Downside Risk Reduction Using
Regime-Switching Signals: A Statistical Jump Model Approach" on free public
data (Kenneth French, Stooq, and central-bank series reconstructed back to
1970). Where the replication stands:

- Applying the bear-regime paths printed in your Figures 5 and 6 — extracted
  losslessly from the PDF vector data and validated against the annotations
  printed on each panel — to my independently constructed data reproduces
  Table 4 almost exactly: 8/8 cells (S&P 500), 8/8 (DAX) and 7/8 (Nikkei)
  for the JM row within half a printed unit, and the Figure-6 HMM turnover
  to 0.002. Conditional on your state sequences, my data, cost and timing
  conventions therefore agree with yours nearly perfectly.
- Reproducing the state sequences themselves succeeds for the S&P 500 (my
  CV-selected path matches your printed 30 shifts / 19.7% bear, and a fixed
  λ = 35 path agrees with your Figure-5 shading on ~96% of days) but not for
  the DAX and Nikkei. Before writing to you I tried to close this from
  public information alone: an exhaustive search over all 6.47 million
  λ-candidate subsets of a 29-value menu assembled from your papers and the
  surrounding literature reaches no grid that reproduces the DAX or Nikkei
  Table-4 rows at the one-day delay, and the three feature-standardization
  variants the word "standardized" can denote (expanding, per-refit-window,
  frozen-initial) bracket but never match the λ = 0 row of Table 3. The
  remaining difference appears to live in implementation details the paper
  does not pin down, not in the data.

Two small items would let me close this cleanly, and I would gladly
acknowledge your help:

1. the online-inferred state sequences plotted in Figures 5–6, in any
   format — even the raw plotting arrays; or, failing that,
2. the exact feature-standardization recipe (the scaler's fitting window and
   refresh cadence) and the λ candidate grid behind the reported tables.

I will report the outcome either way, including a negative one, and can
share my full replication report, the figure-level comparison atlas, and the
audit trail on request.

Thank you for your time and for a paper precise enough to make this level of
replication possible at all.

Best regards,

[Tan Le]
[program / contact]
