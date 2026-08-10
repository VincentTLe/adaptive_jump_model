"""Bounded acquisition adapters for the frozen proxy data sources."""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
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


HttpGetter = Callable[[str, dict[str, str]], HttpResult]


def fetch_source(
    source: SourceConfig,
    start: date,
    cutoff: date,
    *,
    repo_root: str | Path | None = None,
    http_get: HttpGetter | None = None,
) -> SourcePayload:
    """Fetch one configured source without applying research transformations."""
    if source.provider == "localfile":
        if repo_root is None:
            raise AcquisitionError(f"{source.source_id}: localfile requires repo_root")
        return _fetch_localfile(source, start, cutoff, repo_root)
    if source.provider == "fred":
        return _fetch_fred(source, start, cutoff, http_get or _get_http)
    # The yahoo and boj adapters were retired on 2026-08-06. They built the
    # pinned files under data/external/ once; re-running them could never
    # reproduce a sealed run anyway, because those vendors revise history.
    raise AcquisitionError(
        f"{source.source_id}: provider {source.provider!r} is retired; "
        "acquire from a hash-pinned localfile instead"
    )


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
    http_get: HttpGetter | None = None,
) -> Path:
    """Acquire all configured sources and write a complete manifest last."""
    root = Path(repo_root or config.repo_root).resolve()
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise AcquisitionError("created_at must be timezone-aware")
    identifier = run_id or f"{config.config_id}-{timestamp:%Y%m%dT%H%M%SZ}"
    revision = git_sha or research_git_sha(root)
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
                repo_root=root,
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
        "git_sha": revision,
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


def _fetch_localfile(
    source: SourceConfig,
    start: date,
    cutoff: date,
    repo_root: str | Path,
) -> SourcePayload:
    """Load a hash-pinned, pre-built canonical date,value file from the repo."""
    relative = _setting(source, "file_path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise AcquisitionError(f"{source.source_id}: unsafe localfile path")
    root = Path(repo_root).resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(f"{source.source_id}: unsafe localfile path") from exc
    if not resolved.is_file():
        raise AcquisitionError(f"{source.source_id}: missing localfile {relative}")
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = _setting(source, "sha256")
    if digest != expected:
        raise AcquisitionError(
            f"{source.source_id}: localfile sha256 mismatch "
            f"(expected {expected}, found {digest})"
        )
    rows = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    if list(rows.columns) != ["date", "value"]:
        raise AcquisitionError(f"{source.source_id}: localfile must be date,value")
    dates = pd.to_datetime(rows["date"], errors="raise").dt.date
    keep = (dates >= start) & (dates <= cutoff)
    canonical = _canonical(
        rows.loc[keep, "date"], rows.loc[keep, "value"], source, start, cutoff
    )
    return SourcePayload(
        raw,
        "local_file",
        canonical,
        {
            "adapter": "localfile",
            "arguments": {"file_path": relative},
            "sha256": digest,
            "construction": _setting(source, "construction"),
        },
    )


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
        {
            "url": response.url,
            "status": response.status,
            "content_type": response.content_type,
            "params": params,
        },
    )


def _canonical(
    dates: Any,
    values: Any,
    source: SourceConfig,
    start: date,
    cutoff: date,
) -> pd.DataFrame:
    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.date
    raw_values = pd.Series(values).replace(
        {"": None, ".": None, "NA": None, "null": None}
    )
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


def research_git_sha(root: Path) -> str:
    scope = [
        "research.toml",
        # Glob magic so every root research contract is covered, including ones
        # added later. ':(glob)' keeps '*' from crossing '/', so experiment
        # configs under research/ stay out of scope.
        ":(glob)research-*.toml",
        # Where protocol configs are moving. Listed now, while it still matches
        # nothing, so the move itself cannot quietly drop configs out of the
        # guard: a dirty config must fail the run before and after the move.
        "configs",
        "pyproject.toml",
        "uv.lock",
        "scripts",
        "src",
        "tests",
    ]
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *scope], cwd=root, check=False
    )
    if tracked.returncode == 1:
        raise AcquisitionError("result-affecting tracked files are dirty")
    if tracked.returncode != 0:
        raise AcquisitionError("could not inspect tracked research files")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *scope],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if untracked.stdout.strip():
        raise AcquisitionError("result-affecting untracked files exist")
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
    for package in ("adaptive-jump-model", "pandas", "requests"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions
