"""Bounded acquisition adapters for the frozen proxy data sources."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from adaptive_jump.config import ResearchConfig, SourceConfig


class AcquisitionError(RuntimeError):
    """Raised when a provider payload violates the acquisition contract."""


@dataclass(frozen=True)
class HttpResult:
    content: bytes
    url: str
    status: int
    content_type: str | None


@dataclass(frozen=True)
class SourcePayload:
    raw: bytes
    payload_type: str
    canonical: pd.DataFrame
    retrieval: dict[str, Any]


YahooLoader = Callable[[SourceConfig, date, date], tuple[pd.DataFrame, dict[str, Any]]]
HttpGetter = Callable[[str, dict[str, str]], HttpResult]


def fetch_source(
    source: SourceConfig,
    start: date,
    cutoff: date,
    *,
    yahoo_loader: YahooLoader | None = None,
    http_get: HttpGetter | None = None,
) -> SourcePayload:
    """Fetch one configured source without applying research transformations."""
    if source.provider == "yahoo":
        return _fetch_yahoo(source, start, cutoff, yahoo_loader or _download_yahoo)
    getter = http_get or _get_http
    if source.provider == "fred":
        return _fetch_fred(source, start, cutoff, getter)
    if source.provider == "boj":
        return _fetch_boj(source, start, cutoff, getter)
    raise AcquisitionError(f"Unsupported provider: {source.provider}")


def canonical_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize canonical observations deterministically for hashing."""
    return frame.to_csv(index=False, lineterminator="\n", na_rep="").encode()


def quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Return auditable quality facts for a validated canonical series."""
    valid = frame.loc[frame["value"].notna()]
    return {
        "rows": len(frame),
        "valid_rows": len(valid),
        "missing_values": int(frame["value"].isna().sum()),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
        "nonfinite_values": int(
            valid["value"].map(lambda value: not math.isfinite(value)).sum()
        ),
        "first_valid_date": valid["date"].min() if not valid.empty else None,
        "last_valid_date": valid["date"].max() if not valid.empty else None,
    }


def acquire(
    config: ResearchConfig,
    *,
    repo_root: str | Path | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
    git_sha: str | None = None,
    yahoo_loader: YahooLoader | None = None,
    http_get: HttpGetter | None = None,
) -> Path:
    """Acquire all configured sources and write a complete manifest last."""
    root = Path(repo_root or config.path.parent).resolve()
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise AcquisitionError("created_at must be timezone-aware")
    identifier = run_id or f"{config.config_id}-{timestamp:%Y%m%dT%H%M%SZ}"
    raw_dir = root / config.raw_root / identifier
    canonical_dir = root / config.processed_root / identifier
    if raw_dir.exists() or canonical_dir.exists():
        raise AcquisitionError(f"Acquisition run already exists: {identifier}")
    raw_dir.mkdir(parents=True)
    canonical_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    for market in config.markets:
        for kind, source in (("equity", market.equity), ("cash", market.cash)):
            payload = fetch_source(
                source,
                config.sample_start,
                config.replication_cutoff,
                yahoo_loader=yahoo_loader,
                http_get=http_get,
            )
            stem = f"{market.id}_{kind}"
            raw_path = raw_dir / f"{stem}.csv"
            canonical_path = canonical_dir / f"{stem}.csv"
            canonical_payload = canonical_bytes(payload.canonical)
            raw_path.write_bytes(payload.raw)
            canonical_path.write_bytes(canonical_payload)
            records.append(
                {
                    "market": market.id,
                    "kind": kind,
                    "currency": market.currency,
                    "market_classification": market.classification,
                    "deviations": list(market.deviations),
                    "provider": source.provider,
                    "source_id": source.source_id,
                    "source_classification": source.classification,
                    "frequency": source.frequency,
                    "value_field": source.value_field,
                    "payload_type": payload.payload_type,
                    "retrieval": payload.retrieval,
                    "raw": _file_record(root, raw_path, payload.raw),
                    "canonical": _file_record(root, canonical_path, canonical_payload),
                    "quality": quality(payload.canonical),
                }
            )

    manifest = {
        "schema_version": 1,
        "run_id": identifier,
        "claim_class": "ENGINEERING / SMOKE",
        "scientific_claim_allowed": False,
        "config_id": config.config_id,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "git_sha": git_sha or _git_sha(root),
        "created_at_utc": timestamp.astimezone(UTC).isoformat(),
        "sample_start": config.sample_start.isoformat(),
        "replication_cutoff": config.replication_cutoff.isoformat(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "sources": records,
    }
    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _fetch_yahoo(
    source: SourceConfig,
    start: date,
    cutoff: date,
    loader: YahooLoader,
) -> SourcePayload:
    frame, retrieval = loader(source, start, cutoff + timedelta(days=1))
    if frame.empty:
        raise AcquisitionError(f"{source.source_id}: Yahoo returned no rows")
    if source.value_field not in frame.columns:
        raise AcquisitionError(
            f"{source.source_id}: missing Yahoo field {source.value_field}"
        )
    raw = frame.to_csv(index=True, lineterminator="\n", na_rep="").encode()
    index = pd.to_datetime(frame.index, errors="raise")
    timezone = source.settings.get("timezone")
    if index.tz is not None and isinstance(timezone, str):
        index = index.tz_convert(timezone).tz_localize(None)
    canonical = _canonical(
        pd.Series(index.strftime("%Y-%m-%d")),
        frame[source.value_field].reset_index(drop=True),
        source,
        start,
        cutoff,
    )
    return SourcePayload(raw, "adapter_output", canonical, retrieval)


def _download_yahoo(
    source: SourceConfig, start: date, end_exclusive: date
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise AcquisitionError(
            "Yahoo acquisition requires: uv sync --extra data"
        ) from exc

    arguments = {
        "tickers": source.source_id,
        "start": start.isoformat(),
        "end": end_exclusive.isoformat(),
        "interval": "1d",
        "actions": False,
        "auto_adjust": bool(source.settings.get("auto_adjust", False)),
        "repair": False,
        "keepna": True,
        "progress": False,
        "threads": False,
        "ignore_tz": False,
        "multi_level_index": False,
        "timeout": 30,
    }
    frame = yf.download(**arguments)
    return frame, {"adapter": "yfinance.download", "arguments": arguments}


def _fetch_fred(
    source: SourceConfig, start: date, cutoff: date, getter: HttpGetter
) -> SourcePayload:
    url = _setting(source, "retrieval_url")
    params = {"cosd": start.isoformat(), "coed": cutoff.isoformat()}
    response = getter(url, params)
    rows = pd.read_csv(io.BytesIO(response.content), dtype=str, keep_default_na=False)
    if "observation_date" not in rows or source.value_field not in rows:
        raise AcquisitionError(f"{source.source_id}: unexpected FRED columns")
    canonical = _canonical(
        rows["observation_date"], rows[source.value_field], source, start, cutoff
    )
    return SourcePayload(
        response.content,
        "provider_response",
        canonical,
        _http_metadata(response, params),
    )


def _fetch_boj(
    source: SourceConfig, start: date, cutoff: date, getter: HttpGetter
) -> SourcePayload:
    url = _setting(source, "retrieval_url")
    params = {
        "startDate": start.strftime("%Y%m"),
        "endDate": cutoff.strftime("%Y%m"),
    }
    response = getter(url, params)
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
    if not rows or rows[0] != ["STATUS", "200"]:
        raise AcquisitionError(f"{source.source_id}: BOJ status is not 200")
    for row in rows:
        if row and row[0] == "NEXTPOSITION" and len(row) > 1 and row[1]:
            raise AcquisitionError(f"{source.source_id}: paginated BOJ response")
    try:
        header_index = next(
            index for index, row in enumerate(rows) if row and row[0] == "SERIES_CODE"
        )
    except StopIteration as exc:
        raise AcquisitionError(f"{source.source_id}: missing BOJ header") from exc
    header = rows[header_index]
    records = [dict(zip(header, row, strict=True)) for row in rows[header_index + 1 :]]
    if not records or any(row["SERIES_CODE"] != source.source_id for row in records):
        raise AcquisitionError(f"{source.source_id}: BOJ source ID mismatch")
    dates = [
        f"{row['SURVEY_DATES'][:4]}-{row['SURVEY_DATES'][4:6]}-01" for row in records
    ]
    values = [row[source.value_field] for row in records]
    canonical = _canonical(dates, values, source, start, cutoff)
    return SourcePayload(
        response.content,
        "provider_response",
        canonical,
        _http_metadata(response, params),
    )


def _canonical(
    dates: Any,
    values: Any,
    source: SourceConfig,
    start: date,
    cutoff: date,
) -> pd.DataFrame:
    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.date
    raw_values = pd.Series(values).replace({"": None, ".": None, "NA": None})
    numeric = pd.to_numeric(raw_values, errors="coerce")
    invalid = raw_values.notna() & numeric.isna()
    if invalid.any():
        token = raw_values.loc[invalid].iloc[0]
        raise AcquisitionError(f"{source.source_id}: non-numeric value {token!r}")
    frame = pd.DataFrame(
        {"date": date_values.map(date.isoformat), "value": numeric.astype(float)}
    )
    if frame.empty or frame["value"].notna().sum() == 0:
        raise AcquisitionError(f"{source.source_id}: no valid observations")
    if frame["date"].duplicated().any():
        raise AcquisitionError(f"{source.source_id}: duplicate dates")
    if date_values.min() < start or date_values.max() > cutoff:
        raise AcquisitionError(
            f"{source.source_id}: observation outside frozen interval"
        )
    finite = frame["value"].dropna().map(math.isfinite)
    if not finite.all():
        raise AcquisitionError(f"{source.source_id}: non-finite values")
    return frame


def _get_http(url: str, params: dict[str, str]) -> HttpResult:
    try:
        import requests
    except ImportError as exc:
        raise AcquisitionError(
            "HTTP acquisition requires: uv sync --extra data"
        ) from exc
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "adaptive-jump-model/0.1 research acquisition"},
        timeout=60,
    )
    response.raise_for_status()
    return HttpResult(
        content=response.content,
        url=response.url,
        status=response.status_code,
        content_type=response.headers.get("Content-Type"),
    )


def _http_metadata(response: HttpResult, params: dict[str, str]) -> dict[str, Any]:
    return {
        "url": response.url,
        "status": response.status,
        "content_type": response.content_type,
        "params": params,
    }


def _setting(source: SourceConfig, key: str) -> str:
    value = source.settings.get(key)
    if not isinstance(value, str) or not value:
        raise AcquisitionError(f"{source.source_id}: missing setting {key}")
    return value


def _file_record(root: Path, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("adaptive-jump-model", "pandas", "requests", "yfinance"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
