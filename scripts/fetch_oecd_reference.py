"""Fetch the OECD reference series this project validates against, via DBnomics.

FRED serves these series but refuses connections from this machine entirely --
the TLS handshake completes and the read then times out, on every series and
every retry, inside and outside the sandbox. DBnomics mirrors the same OECD MEI
tables and does respond, so it is used as the transport. The data is OECD's
either way; only the delivery differs, and the sha256 pin makes that checkable.

These are REFERENCE series. Nothing here feeds a model. They exist to check
series that do, which is why they live beside the build inputs but are never
read by scripts/build_external_sources.py:

  SPASTT01, Germany   the OECD share-price index, monthly from 1960. The audit
                      ledger cited it as the independent check on the pre-1988
                      DAX backcast and the file had gone missing, leaving that
                      claim unreproducible.
  IRSTCB01, Japan     the central bank discount rate, monthly from 1955. Not a
                      candidate risk-free series -- it is administered, not a
                      market rate -- but it identifies WHICH kind of rate our
                      IMF T-bill series behaves like, which is the open question
                      for Japan.

Run: uv run python scripts/fetch_oecd_reference.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INP = ROOT / "data" / "external" / "inputs"
API = ("https://api.db.nomics.world/v22/series/OECD/MEI/{code}"
       "?observations=1")
UA = {"User-Agent": "Mozilla/5.0 (research replication; contact via repository)"}

SERIES = {
    "oecd_de_share_price_monthly.csv": (
        "DEU.SPASTT01.IXOB.M",
        "OECD MEI share price index, Germany, all shares, index 2015=100"),
    "oecd_jp_central_bank_rate_monthly.csv": (
        "JPN.IRSTCB01.ST.M",
        "OECD MEI central bank policy rate, Japan, percent per annum"),
}


def fetch(code: str) -> list[tuple[str, float]]:
    request = urllib.request.Request(API.format(code=code), headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    docs = payload["series"]["docs"]
    if len(docs) != 1:
        raise SystemExit(f"{code}: expected one series, got {len(docs)}")
    doc = docs[0]
    rows = [(period, value)
            for period, value in zip(doc["period"], doc["value"])
            if isinstance(value, (int, float))]
    if not rows:
        raise SystemExit(f"{code}: no numeric observations")
    return rows


def main() -> None:
    INP.mkdir(parents=True, exist_ok=True)
    for name, (code, description) in SERIES.items():
        path = INP / name
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{name:<40} đã có, sha256={digest[:16]}… (không ghi đè)")
            continue
        rows = fetch(code)
        # Monthly OECD periods are YYYY-MM; write them as the first of the month
        # so every consumer parses them the same way.
        lines = ["date,value"]
        lines += [f"{period}-01,{value!r}" for period, value in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name:<40} {len(rows)} dòng  {rows[0][0]}..{rows[-1][0]}")
        print(f"{'':<40} {description}")
        print(f"{'':<40} sha256={digest}")


if __name__ == "__main__":
    sys.exit(main())
