# Email to advisor — status update, week of 2026-08-01

---

Dear Professor,

A follow-up to what I told you after the 07-31 meeting: I said replication was
essentially done and only the confirmatory holdout test was left before I
could call it finished. Here is the honest state a week later — the
confirmatory test ran, and it failed its own pre-registered gate. That is the
real news, not "no progress"; I want to walk through what actually happened
rather than let the headline stand alone.

## 1. Replication is closed, but not as a clean pass

The regime-shading in the paper's Figure 5 is lossless vector data, so I can
apply the authors' own daily state sequence to my data. Doing that reproduces
their Table-4 JM row almost exactly (8/8 cells US, 8/8 Germany, 7/8 Japan),
which proves my data, cost accounting, and trading conventions are correct.
The problem is upstream of that: *regenerating* their state sequence myself.

I ran the exhaustive search I described last time to completion: every subset
of size 2-8 from a 29-value candidate menu (their withdrawn arXiv-v1 grid,
Table 3's illustrative values, and every other sourced grid I could find) —
6,474,511 combinations, scored against Table 4/5 at three delays. Result:

- **US: solved.** 36,657 grids reach all 14 published cells; example
  {0, 21.5, 70}, worst deviation 0.011.
- **Germany / Japan: not solvable from public information.** Best available
  is 13 of 14 cells (366 / 2,948 grids). Zero of the 6.47 million combinations
  reach all 14, even including their own withdrawn grid. A frontier analysis
  pins the blocking cell precisely — Germany's turnover tops out at 1.47
  against a target of 1.70 once the other seven cells hold; Japan's leverage
  tops out at 0.74 against 0.75. It is not menu coverage; it is the shape of
  the state sequence my pipeline generates, which depends on the
  standardization recipe the paper never discloses.

So I am closing the replication track with honest, per-market labels: US ≈
replicated, Germany/Japan bounded-with-known-causes, not reproducible from
anything public. I also ran a follow-up (jm-disagreement-anatomy-010) testing
whether the gap concentrates in the eras where I had to reconstruct data
(pre-2012 Japan, pre-2000 Germany) — it does not (r=0.57 DE, r=0.76 JP,
against a 1.5 threshold), which rules out data vintage as the explanation and
points at the authors' undisclosed state-sequence geometry instead. Reopening
needs a new primary source — Yu's dissertation is not yet in Princeton
DataSpace as of my last check; the author e-mail (cc'd, sent 07-31) has had
no reply.

## 2. The confirmatory holdout test — this is the one I told you was left

AJM-EXT-001: the lagged-evidence transition-cost challenger (β = ln4), tested
against the paired fixed-JM baseline on three public Fama-French transport
regions (North America, Europe, Japan), with a market-disjoint Asia-Pacific
region held out as the actual confirmation set, opened only if transport
passed.

It did not pass. Min-over-specs Sharpe delta was negative in all three
transport regions (Europe -0.167, Japan -0.162, North America -0.176), so the
confirmation region never opened, per its own pre-registered stopping rule —
which is the right outcome for a rule I wrote specifically to prevent
retrying into a positive result. Independently re-verified end to end by an
agent that did not write the runner (receipt in
`docs/audit/2026-08-06-ajm-ext-d1-receipt.md`). I also traced the mechanism:
the failure is driven by the short grid, not by whipsaw behavior at the
binding cells, which the correction record documents precisely.

I then tried one owner-directed follow-up restricted to the US development
data — a "cap-guard" composition that falls back to the fixed JM specifically
in the months the paired fixed model's own cross-validation is pinned at the
grid's top penalty. Also NOT SUPPORTED (certified 7/7): it made the
worst-case delta worse, not better. But this one earned its keep as a
diagnostic — drawing the actual regime paths (not just reading the Sharpe
numbers) localized the failure to a specific, dated mechanism: the lagged
challenger re-enters the market mid-way through the August 2022 bear-market
rally and then rides the October leg down, while the fixed JM stays in cash
through the whole episode. That is now the concrete motivation for the next
candidate (a semi-Markov dwell cost that penalizes exactly that kind of
early re-entry) rather than another unmotivated variant.

## 3. What is still open — a validity question underneath all of the above

While auditing the week's work I found that a frozen, owner-approved
diagnostic from 08-01 (grid-selection-rule-001) was never finished. It asks:
among the German/Japanese candidate grids that already reach 13 of 14
published cells, which one best matches the authors' own regime path — rather
than just taking the first row of an examples file, which is how the
currently-adopted baseline was actually chosen. The partial result already on
record is uncomfortable: on a complete enumeration, the adopted German grid
ranks **dead last**, 366th of 366, by that criterion. If that holds up, every
"the mechanism fails in Germany" conclusion from the past month — including
the AJM-EXT-001 transport result's German-adjacent reasoning — was measured
against a baseline that is not the best one available, and needs rerunning
before I trust the verdict. I am completing that enumeration today (the
earlier partial run silently truncated its rankings; also spent part of today
rebuilding some intermediate computation caches that did not survive on disk
between sessions — an infrastructure problem, not a scientific one, now
fixed and documented).

## Where this leaves the week

Not the clean "replication done, confirmation next" I described on the 31st.
Instead: replication is closed with an honest label rather than a full match;
the confirmation test I said was pending actually ran and failed its own
gate, which is a real result, not a stall; and a rigor check surfaced a
baseline-selection issue that has to be resolved before I re-report anything
downstream of it. I'd rather bring you the accurate picture than the
optimistic one from a week ago. Happy to walk through any of the audit trail
at our next meeting — everything above is registered, frozen before running,
and independently re-verified on branch `cleanup/research-protocol`
(`research/experiment_registry.jsonl`; audit notes under `docs/audit/`).

Best regards,
Tan
