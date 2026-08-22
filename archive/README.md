# Archive

Old material, kept for traceability. Nothing here is on the current reading
path, and nothing here is current.

Start at `README.md`, `CURRENT.md`, and `SUMMER_2026.md` in the repository root
instead. Open a file here only when checking a specific historical claim.

| Folder | What it is |
| --- | --- |
| `paper/split-v1/` | The earlier split of the work into two papers, from the public-proxy era. |
| `paper/submission/` | Journal submission bundles (JAM, ANOR) built from that split: cover letters, anonymised versions, citation audits. |
| `research/history/` | Development status and worktree notes frozen 2026-07-22, v7 proxy era. Their own headers say not to quote them as current. |
| `research/SCIENTIFIC_LEDGER.md` | The long-form ledger of mathematical ideas, frozen experiments, and corrections. Superseded as a reading path by `CURRENT.md`; `research/experiment_registry.jsonl` remains the authority on experiment lifecycle. |
| `research/TASK.md` | Detailed task state as of 2026-08-09, superseded by `CURRENT.md`. |
| `docs/correspondence/` | Advisor status emails and the data request sent to the paper's authors. |
| `scripts/audit/` | Four one-shot verification scripts. Each checked a closed experiment, or the v9.4/v10 baselines that v11 replaced. None had a test; none checks the current pipeline. `gate_v10_reseal.py` cannot run any more — the v9.4 run it compares against was deleted from disk — but two live scripts cite it as the provenance for a substitution they make, so it is kept readable. |
| `docs/atlas/` | The replication atlas and its independent verifier receipt. The HTML is self-contained: every figure is embedded in it, so the loose PNG copies were dropped. |
| `docs/capguard-diagnosis/` | The lagged-capguard visual autopsy, same arrangement. |
| `docs/LEARN_*.html` | Vietnamese teaching pages about the pipeline, data flow, and degenerate fits, written during the AI-assisted phase. |

Two caveats worth knowing before you read anything here.

These files were written when they were current, so they can contradict
`CURRENT.md`. Where they do, `CURRENT.md` wins. They have deliberately not been
edited to agree with it.

Relative links inside them were written against the old locations and may not
resolve from here.

One thing to know before re-running anything here. `render_replication_atlas.py`
is still in `scripts/rendering/`, and it looks for the verifier receipt at the
old `docs/atlas/` path. Re-rendering the atlas now would produce a page saying
it has not been certified, which is not true — the certified page and its
receipt are the copies in this folder.
