# Unresolved / partially-resolved references

Verification pass: 2026-07-23, against Crossref REST API (`api.crossref.org`).
9 of 10 scholarly references resolved to an authoritative DOI with full
metadata. The two items below could not be *fully* resolved and must be settled
manually before submission. No metadata has been invented for either.

---

## 1. `markowitz1959` — no DOI (UNRESOLVED)

- **Item:** Markowitz, H. M. (1959). *Portfolio Selection: Efficient
  Diversification of Investments*. Yale University Press.
- **Cited in:** Paper 1 (downside-risk tradition, Data section).
- **Problem:** Crossref returns no book/monograph record for this 1959
  monograph (a Crossref search surfaces only unrelated journal items). Old
  monographs frequently predate DOI assignment.
- **What is asserted vs. unverified:**
  - Asserted from the manuscript: author, year (1959), publisher (Yale
    University Press). These are widely-known and were **not** invented here.
  - Unverified by an authority: exact title/subtitle (the manuscript gave only
    "Portfolio Selection"; the full monograph title includes ": Efficient
    Diversification of Investments"), edition, and pagination.
- **Action required (choose one):**
  1. Confirm the exact title/edition from the Yale University Press catalogue
     or a library record (e.g. WorldCat) and cite as a book without a DOI
     (legitimate — books need not carry DOIs); **or**
  2. Replace with a DOI-bearing equivalent if the intent is only to anchor the
     "safety-first / downside-risk" lineage (already covered by `roy1952`), and
     drop `markowitz1959`.

---

## 2. `aydinhan2024` — resolved DOI, missing volume/pages (PARTIAL)

- **Item:** Aydınhan, A. O., Kolm, P. N., Mulvey, J. M., & Shu, Y. (2024).
  Identifying patterns in financial markets: extending the statistical jump
  model for regime identification. *Annals of Operations Research*. Advance
  online publication. https://doi.org/10.1007/s10479-024-06035-z
- **Cited in:** Paper 2 (related work).
- **Status:** DOI verified. Crossref shows **online publication 2024-05-14** and
  **no assigned volume, issue, or page/article number** at verification time.
- **Action required:** Before final submission, re-check the DOI. If a volume /
  issue / article number has since been assigned, update the `.bib`; otherwise
  keep the APA/Chicago "advance online publication" form. **Do not infer** a
  volume or page range.

---

## Not yet added — dataset & software citations (pending, this phase)

The brief requires formal citations for public datasets and materially-used
software. These are **not** invented here; each will be sourced from its
provider's own suggested citation before inclusion:

- FRED series `DTB3` (3-Month Treasury Bill Secondary Market Rate) — St. Louis Fed suggested citation.
- FRED series `IR3TIB01DEM156N` (Germany 3-month interbank rate, OECD-sourced) — St. Louis Fed suggested citation.
- Bank of Japan `STRACLUC3M` (uncollateralized overnight/3-month call rate) — BoJ statistics.
- Yahoo Finance series `^SP500TR`, `^GDAXI`, `^N225` — data-source note.
- `jumpmodels` Python package (v0.1.1) — software citation (authors Shu & Mulvey; PyPI/Apache-2.0).
- `hmmlearn`, NumPy, pandas, Python 3.12 — software note in code-availability.

These belong with the data- and code-availability statements and will be added
in that phase; recorded here so they are not forgotten.
