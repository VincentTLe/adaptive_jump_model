"""jm-disagreement-anatomy-010: where the ours-vs-Fig5 disagreement lives.

Frozen spec: research/contracts/jm-disagreement-anatomy-010.toml (hash pinned in
the registry FROZEN row; this probe refuses to run on hash drift). Question: does
the daily disagreement between our replication-contract CV path (v9.4-recon,
anchor-gated) and the authors' Figure-5 paths concentrate in the eras where
our equity data rests on a different source than theirs — JP reconstructed TR
(pre 2011-12-19), DE pre-Xetra fixings (pre 2000) — with the US as placebo at
the same cut dates? Readout rule and threshold are frozen in the spec.

Everything here is EXPLORATORY and descriptive of where disagreement sits;
nothing may seed a grid or config.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from rendering.render_replication_atlas import (  # noqa: E402
    C_CONST,
    C_THEIRS,
    C_V94,
    FG,
    GRIDC,
    NAMES,
    PANEL,
    dark_style,
    footer,
    git_head,
    their_bear,
    utc_now,
)

from adaptive_jump.regime_comparison import (  # noqa: E402
    concordance,
    disagreement_decomposition,
    era_slices,
    switch_f1,
)
from experiments.probe_jm_effective_lambda_inversion import load_fig5  # noqa: E402

SPEC = ROOT / "research" / "contracts" / "jm-disagreement-anatomy-010.toml"
REGISTRY = ROOT / "research" / "experiment_registry.jsonl"
RECON = ROOT / "artifacts" / "jm-residual" / "atlas" / "v94-recon-selected-signal.csv"
ANCHORS = ROOT / "artifacts" / "jm-residual" / "01-grid-identification" / (
    "selected-anchors.csv"
)
OUT = ROOT / "artifacts" / "jm-residual" / "10-disagreement-anatomy"
DOCS = ROOT / "docs" / "atlas"
KINDS = ("timing", "missing", "extra")
KIND_COLORS = {"timing": C_V94, "missing": C_THEIRS, "extra": C_CONST}


def load_spec() -> dict:
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    frozen = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("experiment_id") == "jm-disagreement-anatomy-010"
        and json.loads(line).get("status") == "FROZEN"
    ]
    if not frozen:
        raise SystemExit("no FROZEN registry row for jm-disagreement-anatomy-010")
    pinned = frozen[-1]["frozen_spec_hash"]
    if digest != pinned:
        raise SystemExit(
            f"spec hash drift: file {digest[:12]}… vs registry pin {pinned[:12]}…"
        )
    print(f"spec hash verified: {digest[:12]}… matches the FROZEN registry row")
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


def load_recon() -> pd.DataFrame:
    recon = pd.read_csv(RECON, parse_dates=["date"]).set_index("date")
    anchors = pd.read_csv(ANCHORS, index_col=0)
    for market in ("us", "de", "jp"):
        oos = recon[market].dropna().loc["1990-01-01":"2023-12-31"]
        shifts = int((oos.diff().abs() > 0).sum())
        bear = float(1 - oos.mean())
        want = anchors.loc[market]
        if (
            shifts != int(want["shifts"])
            or len(oos) != int(want["days"])
            or abs(bear - float(want["bear_share"])) > 1e-12
        ):
            raise SystemExit(f"{market}: recon anchor re-verification FAILED")
    print("recon anchors re-verified against sealed selected-anchors.csv (3/3)")
    return recon


def pair(market: str, recon: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    theirs = their_bear(load_fig5(market))
    ours = (1 - recon[market].dropna()).astype(float)
    joint = theirs.index.intersection(ours.index)
    return theirs.loc[joint], ours.loc[joint]


def era_rows(
    market: str,
    partition: str,
    eras: list[tuple[str, str, str]],
    theirs: pd.Series,
    ours: pd.Series,
    margin: int,
) -> list[dict]:
    slices = era_slices(
        theirs.index,
        [(name, pd.Timestamp(lo), pd.Timestamp(hi)) for name, lo, hi in eras],
    )
    rows = []
    for name, index in slices:
        a, b = theirs.loc[index], ours.loc[index]
        disagree = int((a != b).sum())
        stats = switch_f1(a, b, margin=margin)
        rows.append(
            {
                "market": market,
                "partition": partition,
                "era": name,
                "start": index[0].date().isoformat(),
                "end": index[-1].date().isoformat(),
                "days": len(index),
                "disagree_days": disagree,
                "rate": disagree / len(index),
                "concordance": concordance(a, b),
                "f1": stats["f1"],
            }
        )
    return rows


def sub_span_row(
    market: str, name: str, lo: str, hi: str, theirs: pd.Series, ours: pd.Series
) -> dict:
    index = theirs.loc[lo:hi].index
    a, b = theirs.loc[index], ours.loc[index]
    disagree = int((a != b).sum())
    return {
        "market": market,
        "partition": "descriptive_span",
        "era": name,
        "start": index[0].date().isoformat(),
        "end": index[-1].date().isoformat(),
        "days": len(index),
        "disagree_days": disagree,
        "rate": disagree / len(index),
        "concordance": concordance(a, b),
        "f1": float("nan"),
    }


def fig_era(market: str, rows: list[dict], decomp: pd.DataFrame,
            placebo_note: str, footer_text: str) -> Path:
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    eras = [r for r in rows if r["market"] == market]
    positions = range(len(eras))
    bottoms = [0.0] * len(eras)
    for kind in KINDS:
        shares = []
        for row in eras:
            days = decomp[
                (decomp["market"] == market)
                & (decomp["era"] == row["era"])
                & (decomp["kind"] == kind)
            ]["days"]
            shares.append(
                100.0 * float(days.iloc[0]) / row["days"] if len(days) else 0.0
            )
        axis.bar(positions, shares, bottom=bottoms, color=KIND_COLORS[kind],
                 label=kind, width=0.55)
        bottoms = [b + s for b, s in zip(bottoms, shares, strict=True)]
    for pos, row in zip(positions, eras, strict=True):
        axis.annotate(
            f"{row['rate']:.1%}",
            xy=(pos, bottoms[pos]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=FG,
        )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(
        [f"{r['era']}\n{r['start'][:4]}–{r['end'][:4]}" for r in eras], fontsize=8
    )
    axis.set_ylabel("disagreement, % of era days")
    axis.set_title(f"{NAMES[market]} — where the disagreement lives{placebo_note}")
    axis.legend(fontsize=8, facecolor=PANEL, edgecolor=GRIDC)
    footer(fig, footer_text)
    out = DOCS / f"fig-era-decomposition-{market}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    dark_style()
    spec = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    recon = load_recon()
    margin = int(spec["readout"]["margin_trading_days"])
    eras_cfg = spec["eras"]

    pairs = {market: pair(market, recon) for market in ("us", "de", "jp")}
    all_rows: list[dict] = []
    decomp_rows: list[dict] = []

    primaries = {
        "de": [tuple(e) for e in eras_cfg["de_primary"]],
        "jp": [tuple(e) for e in eras_cfg["jp_primary"]],
    }
    for market, eras in primaries.items():
        theirs, ours = pairs[market]
        rows = era_rows(market, "primary", eras, theirs, ours, margin)
        all_rows.extend(rows)
        labelled = disagreement_decomposition(theirs, ours)
        for name, lo, hi in eras:
            inside = labelled[
                (labelled["date"] >= pd.Timestamp(lo))
                & (labelled["date"] <= pd.Timestamp(hi))
            ]
            for kind in KINDS:
                decomp_rows.append(
                    {
                        "market": market,
                        "era": name,
                        "kind": kind,
                        "days": int((inside["kind"] == kind).sum()),
                    }
                )

    theirs_us, ours_us = pairs["us"]
    cut_2000, cut_2012 = eras_cfg["us_placebo_cuts"]
    lo_us = theirs_us.index[0].date().isoformat()
    hi_us = theirs_us.index[-1].date().isoformat()
    placebo = {
        "placebo_2000": [
            ("pre_cut", lo_us, "1999-12-31"),
            ("post_cut", cut_2000, hi_us),
        ],
        "placebo_2012": [
            ("pre_cut", lo_us, "2011-12-18"),
            ("post_cut", cut_2012, hi_us),
        ],
    }
    for partition, eras in placebo.items():
        all_rows.extend(era_rows("us", partition, eras, theirs_us, ours_us, margin))
    labelled_us = disagreement_decomposition(theirs_us, ours_us)
    for name, lo, hi in [
        ("pre_2000", lo_us, "1999-12-31"),
        ("2000_2011", cut_2000, "2011-12-18"),
        ("post_2012", cut_2012, hi_us),
    ]:
        all_rows.extend(
            era_rows("us", "placebo_3way", [(name, lo, hi)],
                     theirs_us.loc[lo:hi], ours_us.loc[lo:hi], margin)
        )
        inside = labelled_us[
            (labelled_us["date"] >= pd.Timestamp(lo))
            & (labelled_us["date"] <= pd.Timestamp(hi))
        ]
        for kind in KINDS:
            decomp_rows.append(
                {
                    "market": "us",
                    "era": name,
                    "kind": kind,
                    "days": int((inside["kind"] == kind).sum()),
                }
            )

    for name, lo, hi in [tuple(e) for e in eras_cfg["jp_sub_era_descriptive"]]:
        all_rows.append(sub_span_row("jp", name, lo, hi, *pairs["jp"]))
    splices = eras_cfg["secondary_descriptive_cuts"]
    for market, cut in (
        ("de", splices["de_cash_splice"]),
        ("jp", splices["jp_cash_splice"]),
    ):
        theirs, ours = pairs[market]
        lo = theirs.index[0].date().isoformat()
        hi = theirs.index[-1].date().isoformat()
        all_rows.extend(
            era_rows(market, "cash_splice",
                     [("pre_splice", lo, (pd.Timestamp(cut) - pd.Timedelta(days=1)
                                          ).date().isoformat()),
                      ("post_splice", cut, hi)],
                     theirs, ours, margin)
        )

    era_metrics = pd.DataFrame(all_rows)
    decomposition = pd.DataFrame(decomp_rows)

    def rate(market: str, partition: str, era: str) -> float:
        row = era_metrics[
            (era_metrics["market"] == market)
            & (era_metrics["partition"] == partition)
            & (era_metrics["era"] == era)
        ].iloc[0]
        return float(row["rate"])

    r_de = rate("de", "primary", "pre_xetra_fixings") / rate(
        "de", "primary", "harmonized_closes"
    )
    r_jp = rate("jp", "primary", "reconstructed_tr") / rate(
        "jp", "primary", "official_n225tr"
    )
    r_us_2000 = rate("us", "placebo_2000", "pre_cut") / rate(
        "us", "placebo_2000", "post_cut"
    )
    r_us_2012 = rate("us", "placebo_2012", "pre_cut") / rate(
        "us", "placebo_2012", "post_cut"
    )
    threshold = float(spec["readout"]["threshold_ratio"])
    supported = (
        r_de >= threshold
        and r_jp >= threshold
        and r_de > r_us_2000
        and r_jp > r_us_2012
    )
    inverted = r_de <= 1.0 / threshold and r_jp <= 1.0 / threshold
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"

    readout = {
        "experiment_id": "jm-disagreement-anatomy-010",
        "rule": spec["readout"]["rule"],
        "threshold_ratio": threshold,
        "r_de": r_de,
        "r_jp": r_jp,
        "r_us_placebo_2000": r_us_2000,
        "r_us_placebo_2012": r_us_2012,
        "supported": supported,
        "inverted": inverted,
        "verdict": verdict + (" (INVERTED)" if inverted else ""),
        "computed_utc": utc_now(),
        "git_head": git_head(),
    }

    era_metrics.to_csv(OUT / "era-metrics.csv", index=False, lineterminator="\n")
    decomposition.to_csv(OUT / "decomposition.csv", index=False, lineterminator="\n")
    (OUT / "readout.json").write_text(
        json.dumps(readout, indent=2) + "\n", encoding="utf-8"
    )

    footer_text = (
        f"jm-disagreement-anatomy-010; commit {readout['git_head'][:12]}; "
        f"rendered {readout['computed_utc']} UTC"
    )
    for market, note in (("de", ""), ("jp", ""),
                         ("us", " (placebo: no source boundary)")):
        rows = [
            r for r in all_rows
            if r["market"] == market
            and r["partition"] in ("primary", "placebo_3way")
        ]
        fig_era(market, rows, decomposition, note, footer_text)

    lines = [
        "jm-disagreement-anatomy-010 — kết quả (EXPLORATORY, descriptive)",
        "",
        f"Readout (rule đóng băng trước khi tính, ngưỡng {threshold}):",
        f"  r_DE  = {r_de:.3f}   (pre-Xetra / harmonized)",
        f"  r_JP  = {r_jp:.3f}   (reconstructed / official)",
        f"  r_US placebo 2000 cut = {r_us_2000:.3f}; 2012 cut = {r_us_2012:.3f}",
        f"  VERDICT: {readout['verdict']}",
        "",
        "Per-era (primary + placebo):",
    ]
    for _, row in era_metrics[
        era_metrics["partition"].isin(["primary", "placebo_2000", "placebo_2012"])
    ].iterrows():
        lines.append(
            f"  {row['market']} {row['partition']}/{row['era']}: "
            f"rate {row['rate']:.1%} trên {row['days']} ngày, F1 {row['f1']:.2f}"
        )
    lines += [
        "",
        "Phân rã (timing/missing/extra, ngày):",
    ]
    for (market, era), group in decomposition.groupby(["market", "era"]):
        parts = ", ".join(
            f"{k} {int(group[group['kind'] == k]['days'].iloc[0])}" for k in KINDS
        )
        lines.append(f"  {market}/{era}: {parts}")
    lines += [
        "",
        "Disclosure mang theo (từ spec): era bị lẫn với market regime (era tái dựng",
        "của JP chứa trọn thập kỷ bear hậu-bubble); placebo cut trên US giảm nhưng",
        "không loại bỏ điều đó — verdict nói về CONCENTRATION, không nói về CAUSE.",
        "Prior ngược chiều phải trích dẫn: episode-shape-13 — flicker thừa của HMM US",
        "KHÔNG dồn vào era tái dựng pre-1988 (12/16 episode ngắn nằm sau 2000).",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
