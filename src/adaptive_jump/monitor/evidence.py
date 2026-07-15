"""Path-allowlisted, verifier-gated access to sealed research evidence."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump import artifacts
from adaptive_jump.window_verifier import verify_window_run

Verifier = Callable[[str | Path], dict[str, Any]]
_FIXED_RUN_ID = re.compile(r"fixed-baselines-(?:[0-9a-f]{12}-){2}[0-9a-f]{12}\Z")
_MARKET_ID = re.compile(r"[a-z]{2}\Z")


class EvidenceError(RuntimeError):
    """Raised when sealed evidence is unavailable, invalid, or unauthorized."""


class OutcomeLocked(EvidenceError):
    """Raised when a valid run has not opened conclusion-bearing outcomes."""


@dataclass(frozen=True)
class EvidenceDefinition:
    run_id: str
    title: str
    relative_path: Path
    verifier: Verifier


SEALED_RUNS = {
    definition.run_id: definition
    for definition in (
        EvidenceDefinition(
            "fixed-baselines-8adb330565d6-3636939b525d-e9614112b234",
            "Fixed baseline proxy replication",
            Path(
                "artifacts/fixed-baselines/"
                "fixed-baselines-8adb330565d6-3636939b525d-e9614112b234"
            ),
            artifacts.verify_run,
        ),
        EvidenceDefinition(
            "jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad",
            "JM 4,000-day training-window sensitivity",
            Path(
                "artifacts/jm-train-window-sensitivity/"
                "jm-window-cd9ac0b9d7a6-3636939b525d-6c19911401ad"
            ),
            verify_window_run,
        ),
    )
}


class EvidenceStore:
    """Verify exact code-registered runs before exposing any artifact content."""

    def __init__(
        self,
        project_root: Path,
        definitions: Mapping[str, EvidenceDefinition] = SEALED_RUNS,
    ) -> None:
        self.project_root = project_root.resolve()
        self.artifact_root = self.project_root / "artifacts"
        self.definitions = dict(definitions)
        self._receipts: dict[str, tuple[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        for run_id, definition in self.definitions.items():
            run_dir = (self.project_root / definition.relative_path).resolve()
            if (
                run_id != definition.run_id
                or run_dir.name != run_id
                or not run_dir.is_relative_to(self.artifact_root)
            ):
                raise EvidenceError(
                    "sealed evidence catalog violates its path allowlist"
                )

    def catalog(self) -> tuple[dict[str, Any], ...]:
        """List registered identities without reading unverified result content."""
        return tuple(
            {
                "run_id": definition.run_id,
                "title": definition.title,
                "available": self._run_dir(definition).is_dir(),
            }
            for definition in self.definitions.values()
        )

    def evidence(self, run_id: str) -> dict[str, Any]:
        """Return verified identity and boundary evidence, never outcome metrics."""
        definition, run_dir, receipt, seal = self._verify(run_id)
        metadata = artifacts.read_json(run_dir / "run.json")
        boundaries = _read_records(run_dir / "boundaries.csv")
        safe_receipt = {
            key: value for key, value in receipt.items() if key not in {"conclusion"}
        }
        result = {
            "run_id": definition.run_id,
            "title": definition.title,
            "status": metadata.get("status"),
            "metrics_opened": metadata.get("metrics_opened") is True,
            "claim_label": metadata.get("claim_label"),
            "verification": safe_receipt,
            "boundaries": boundaries,
        }
        self._require_unchanged(run_dir, seal)
        return result

    def outcome(self, run_id: str) -> dict[str, Any]:
        """Return verified outcomes only when the run explicitly opened them."""
        definition, run_dir, receipt, seal = self._verify(run_id)
        metadata = artifacts.read_json(run_dir / "run.json")
        if (
            metadata.get("metrics_opened") is not True
            or metadata.get("status") != "complete"
        ):
            raise OutcomeLocked(f"outcomes remain locked for {definition.run_id}")
        metrics_path = run_dir / "metrics.csv"
        claim_path = run_dir / "claim.json"
        if not metrics_path.is_file() or not claim_path.is_file():
            raise EvidenceError("opened outcome files are incomplete")
        result = {
            "run_id": definition.run_id,
            "title": definition.title,
            "verification": receipt,
            "metrics": _read_records(metrics_path),
            "claim": artifacts.read_json(claim_path),
        }
        self._require_unchanged(run_dir, seal)
        return result

    def market_data(self, run_id: str, market: str) -> dict[str, Any]:
        """Return hash-checked raw OHLCV for one verified fixed-baseline run."""
        if not isinstance(run_id, str) or _FIXED_RUN_ID.fullmatch(run_id) is None:
            raise EvidenceError("verified fixed-baseline run identity is invalid")
        if not isinstance(market, str) or _MARKET_ID.fullmatch(market) is None:
            raise EvidenceError(f"market is unavailable: {market}")
        run_dir = (self.artifact_root / "fixed-baselines" / run_id).resolve()
        if run_dir.name != run_id or not run_dir.is_relative_to(self.artifact_root):
            raise EvidenceError("verified run path is invalid")

        receipt, metadata, seal = self._verify_monitor_run(run_id, run_dir)
        manifest = artifacts.read_json(run_dir / "data-manifest.json")
        source = _equity_source(manifest, market)
        payload = self._read_raw_source(source)
        rows, observed = _parse_ohlcv(payload)
        manifest_quality = source.get("quality")
        if not isinstance(manifest_quality, dict):
            raise EvidenceError("market source quality metadata is invalid")
        expected_rows = manifest_quality.get("rows")
        if isinstance(expected_rows, int) and expected_rows != observed["rows"]:
            raise EvidenceError("raw row count disagrees with the data manifest")

        result = {
            "run_id": run_id,
            "run_status": receipt["status"],
            "market": market,
            "acquisition_run_id": manifest.get("run_id"),
            "data_manifest_sha256": metadata.get("data_manifest_sha256"),
            "source": {
                "provider": source.get("provider"),
                "source_id": source.get("source_id"),
                "currency": source.get("currency"),
                "frequency": source.get("frequency"),
                "classification": source.get("source_classification"),
                "value_field": source.get("value_field"),
                "deviations": source.get("deviations", []),
                "raw_sha256": source["raw"]["sha256"],
                "raw_bytes": source["raw"]["bytes"],
            },
            "coverage": {
                "first_date": rows[0]["date"] if rows else None,
                "last_date": rows[-1]["date"] if rows else None,
                "rows": len(rows),
            },
            "quality": {
                **observed,
                "candles_available": observed["distinct_ohlc_rows"] > 0,
                "volume_available": observed["nonzero_volume_rows"] > 0,
                "price_rendering": (
                    "candlestick_with_close_fallback"
                    if observed["distinct_ohlc_rows"] > 0
                    else "close_line"
                ),
                "volume_rendering": (
                    "bars" if observed["nonzero_volume_rows"] > 0 else "hidden"
                ),
                "manifest": manifest_quality,
            },
            "rows": rows,
        }
        self._require_unchanged(run_dir, seal)
        return result

    def _verify_monitor_run(
        self, run_id: str, run_dir: Path
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        if not run_dir.is_dir():
            raise EvidenceError(f"verified run is unavailable: {run_id}")
        with self._lock:
            try:
                seal = self._seal_identity(run_dir)
                metadata = artifacts.read_json(run_dir / "run.json")
                cached = self._receipts.get(run_id)
                if cached is not None and cached[0] == seal:
                    receipt = cached[1]
                else:
                    receipt = artifacts.verify_run(run_dir)
                    receipt = json.loads(json.dumps(receipt, allow_nan=False))
                    self._receipts[run_id] = (seal, receipt)
            except (artifacts.ArtifactError, OSError, ValueError) as exc:
                raise EvidenceError(
                    f"verified run validation failed: {run_id}"
                ) from exc
        if (
            metadata.get("run_id") != run_id
            or receipt.get("run_id") != run_id
            or receipt.get("status") != metadata.get("status")
        ):
            raise EvidenceError("verifier returned a different run identity")
        return receipt, metadata, seal

    def _read_raw_source(self, source: dict[str, Any]) -> bytes:
        raw = source.get("raw")
        if not isinstance(raw, dict):
            raise EvidenceError("market source raw metadata is invalid")
        relative = raw.get("path")
        expected_hash = raw.get("sha256")
        expected_bytes = raw.get("bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not isinstance(expected_bytes, int)
        ):
            raise EvidenceError("market source raw identity is invalid")
        raw_root = (self.project_root / "data/raw").resolve()
        path = (self.project_root / relative).resolve()
        if not path.is_relative_to(raw_root) or not path.is_file():
            raise EvidenceError("market source path is outside the raw-data root")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise EvidenceError("market source cannot be read") from exc
        if len(payload) != expected_bytes:
            raise EvidenceError("market source byte count does not match its manifest")
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise EvidenceError("market source hash does not match its manifest")
        return payload

    def _verify(
        self, run_id: str
    ) -> tuple[EvidenceDefinition, Path, dict[str, Any], str]:
        definition = self._definition(run_id)
        run_dir = self._run_dir(definition)
        inventory = run_dir / "inventory.json"
        metadata_path = run_dir / "run.json"
        if (
            not run_dir.is_dir()
            or not inventory.is_file()
            or not metadata_path.is_file()
        ):
            raise EvidenceError(f"sealed evidence is unavailable: {run_id}")
        with self._lock:
            try:
                seal = self._seal_identity(run_dir)
                metadata = artifacts.read_json(metadata_path)
            except (artifacts.ArtifactError, OSError, ValueError) as exc:
                raise EvidenceError(
                    f"sealed evidence verification failed: {run_id}"
                ) from exc
            cached = self._receipts.get(run_id)
            if cached is not None and cached[0] == seal:
                return definition, run_dir, cached[1], seal
            try:
                receipt = definition.verifier(run_dir)
            except (artifacts.ArtifactError, OSError, ValueError) as exc:
                raise EvidenceError(
                    f"sealed evidence verification failed: {run_id}"
                ) from exc
            if (
                metadata.get("run_id") != run_id
                or receipt.get("run_id") != run_id
                or receipt.get("status") != metadata.get("status")
            ):
                raise EvidenceError("verifier returned a different run identity")
            safe_receipt = json.loads(json.dumps(receipt, allow_nan=False))
            self._receipts[run_id] = (seal, safe_receipt)
            return definition, run_dir, safe_receipt, seal

    def _definition(self, run_id: str) -> EvidenceDefinition:
        try:
            return self.definitions[run_id]
        except (KeyError, TypeError) as exc:
            raise EvidenceError(f"run is not registered: {run_id}") from exc

    def _run_dir(self, definition: EvidenceDefinition) -> Path:
        return (self.project_root / definition.relative_path).resolve()

    @staticmethod
    def _seal_identity(run_dir: Path) -> str:
        artifacts.verify_inventory(run_dir)
        return ":".join(
            artifacts.sha256_file(run_dir / name)
            for name in ("inventory.json", "run.json")
        )

    def _require_unchanged(self, run_dir: Path, expected: str) -> None:
        try:
            actual = self._seal_identity(run_dir)
        except (artifacts.ArtifactError, OSError) as exc:
            raise EvidenceError("sealed evidence changed while being read") from exc
        if actual != expected:
            raise EvidenceError("sealed evidence changed while being read")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceError(f"verified evidence file is missing: {path.name}")
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _equity_source(manifest: dict[str, Any], market: str) -> dict[str, Any]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise EvidenceError("data manifest sources are invalid")
    matches = [
        source
        for source in sources
        if isinstance(source, dict)
        and source.get("market") == market
        and source.get("kind") == "equity"
    ]
    if len(matches) != 1:
        raise EvidenceError(f"market is unavailable: {market}")
    return matches[0]


def _parse_ohlcv(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise EvidenceError("market source CSV is invalid") from exc
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required):
        raise EvidenceError("market source does not contain OHLCV columns")

    dates = []
    for value in frame["Date"]:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("market source contains an invalid date") from exc
        if pd.isna(timestamp):
            raise EvidenceError("market source contains an invalid date")
        dates.append(timestamp.date().isoformat())

    numeric = frame[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = numeric.map(lambda value: math.isfinite(float(value)))
    clean = numeric.where(finite)
    ohlc = clean[["Open", "High", "Low", "Close"]]
    complete = ohlc.notna().all(axis=1)
    valid = complete & (ohlc["Low"] <= ohlc[["Open", "Close"]].min(axis=1))
    valid &= ohlc["High"] >= ohlc[["Open", "Close"]].max(axis=1)
    distinct = valid & (ohlc.nunique(axis=1) > 1)
    output = clean.rename(columns=str.lower)
    output.insert(0, "date", dates)
    rows = json.loads(output.to_json(orient="records"))
    return rows, {
        "rows": len(rows),
        "duplicate_dates": len(dates) - len(set(dates)),
        "dates_sorted": dates == sorted(dates),
        "complete_ohlc_rows": int(complete.sum()),
        "valid_ohlc_rows": int(valid.sum()),
        "distinct_ohlc_rows": int(distinct.sum()),
        "missing_close_rows": int(clean["Close"].isna().sum()),
        "nonzero_volume_rows": int((clean["Volume"] > 0).sum()),
    }
