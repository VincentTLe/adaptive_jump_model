"""Materialize local raw Kibot data into processed research datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from adaptive_jump.data_kibot import load_kibot_adjusted_ohlcv_csv
from adaptive_jump.features import make_minute_features_from_minute_bidask, zscore_series


TICK_COLUMNS = ["date", "time", "price", "bid", "ask", "size"]
RAW_FILES = {
    "IBM": ("IBM.txt", "adjusted_ohlcv"),
    "OIH": ("OIH.txt", "adjusted_ohlcv"),
    "IVE": ("IVE_tickbidask.txt", "tick_bidask"),
    "WDC": ("WDC_tickbidask.txt", "tick_bidask"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--symbols", nargs="+", default=["all"], help="Symbols to process, or 'all'.")
    parser.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per chunk for tick files.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed files.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    symbols = _resolve_symbols(args.symbols)
    inventory = []
    started = perf_counter()
    for symbol in symbols:
        filename, data_type = RAW_FILES[symbol]
        raw_path = raw_dir / filename
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw file for {symbol}: {raw_path}")
        output_path = processed_dir / _output_name(symbol, data_type)
        if output_path.exists() and not args.force:
            print(f"SKIP {symbol}: {output_path} exists; pass --force to overwrite")
            inventory.append(_inventory_row(symbol, data_type, raw_path, output_path, skipped=True, seconds=0.0))
            continue

        print(f"PROCESS {symbol} ({data_type}) from {raw_path}")
        symbol_started = perf_counter()
        if data_type == "adjusted_ohlcv":
            df = _process_adjusted_ohlcv(raw_path, symbol)
        elif data_type == "tick_bidask":
            df = _process_tick_bidask_chunked(raw_path, symbol, args.chunksize)
        else:
            raise ValueError(f"Unknown data type for {symbol}: {data_type}")
        _write_processed_csv(df, output_path)
        seconds = perf_counter() - symbol_started
        row = _inventory_row(symbol, data_type, raw_path, output_path, skipped=False, seconds=seconds, df=df)
        _write_metadata(processed_dir, row)
        inventory.append(row)
        print(
            f"DONE {symbol}: rows={row['rows']} start={row['start']} end={row['end']} "
            f"seconds={seconds:.1f} output={output_path}"
        )

    inventory_df = _merge_inventory(processed_dir, inventory)
    inventory_path = processed_dir / "data_inventory.csv.gz"
    inventory_df.to_csv(inventory_path, index=False, compression="gzip")
    manifest = {
        "run_symbols": symbols,
        "chunksize": args.chunksize,
        "seconds": perf_counter() - started,
        "outputs": inventory_df.to_dict(orient="records"),
    }
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, allow_nan=False), encoding="utf-8")
    print(f"INVENTORY {inventory_path}")
    print(f"MANIFEST {manifest_path}")
    print(f"TOTAL seconds={manifest['seconds']:.1f}")


def _resolve_symbols(values: list[str]) -> list[str]:
    requested = [value.upper() for value in values]
    if requested == ["ALL"]:
        return list(RAW_FILES)
    unknown = sorted(set(requested) - set(RAW_FILES))
    if unknown:
        raise ValueError(f"Unknown symbols: {unknown}; valid symbols are {sorted(RAW_FILES)}")
    return requested


def _output_name(symbol: str, data_type: str) -> str:
    if data_type == "adjusted_ohlcv":
        return f"{symbol}_ohlcv_features.csv.gz"
    return f"{symbol}_tick_minute_features.csv.gz"


def _process_adjusted_ohlcv(path: Path, symbol: str) -> pd.DataFrame:
    bars = load_kibot_adjusted_ohlcv_csv(str(path))
    out = bars.copy()
    out["symbol"] = symbol
    out["data_type"] = "adjusted_ohlcv"
    out["has_bidask"] = 0
    out["price"] = out["close"]
    same_date = _same_date_as_previous(out.index)
    out["return"] = out["close"].pct_change().where(same_date)
    out["log_return"] = np.log(out["close"] / out["close"].shift(1)).where(same_date)
    out["mid_return"] = out["log_return"]
    out["abs_mid_return"] = out["mid_return"].abs()
    out["log_volume"] = np.log1p(out["volume"])
    out["rolling_vol_5"] = out["mid_return"].rolling(5).std()
    out["rolling_vol_20"] = out["mid_return"].rolling(20).std()
    out["realized_var"] = out["mid_return"].pow(2)
    out["spread_close"] = np.nan
    out["rel_spread_close"] = np.nan

    z_log_volume = zscore_series(out["log_volume"])
    z_mid_return = zscore_series(out["mid_return"])
    z_rolling_vol_20 = zscore_series(out["rolling_vol_20"])
    out["noise_score_raw"] = -z_log_volume
    out["shock_score_raw"] = z_mid_return.abs() + z_rolling_vol_20
    out["noise_proxy"] = "low_volume_zscore"
    return _ordered_output(out)


def _process_tick_bidask_chunked(path: Path, symbol: str, chunksize: int) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    dropped: dict[str, int] = {}
    prev_price = np.nan
    prev_timestamp = pd.NaT
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(path, header=None, names=TICK_COLUMNS, chunksize=chunksize), start=1):
        total_rows += len(chunk)
        cleaned = _clean_tick_chunk(chunk, dropped)
        if cleaned.empty:
            print(f"  chunk {i}: raw={len(chunk)} cleaned=0 total_raw={total_rows}")
            continue
        _raise_if_chunk_moves_backward(cleaned.index[0], prev_timestamp, path, i)
        agg, prev_price, prev_timestamp = _aggregate_tick_chunk(cleaned, prev_price, prev_timestamp)
        partials.append(agg)
        print(
            f"  chunk {i}: raw={len(chunk)} cleaned={len(cleaned)} minute_parts={len(agg)} "
            f"total_raw={total_rows}"
        )
    if not partials:
        raise ValueError(f"No valid tick rows found in {path}")

    minute = _combine_minute_partials(partials)
    features = make_minute_features_from_minute_bidask(minute)
    out = minute.copy()
    for column in ["mid_return", "abs_mid_return", "log_volume", "rolling_vol_5", "rolling_vol_20", "noise_score_raw", "shock_score_raw"]:
        out[column] = features[column]
    out["symbol"] = symbol
    out["data_type"] = "tick_bidask"
    out["has_bidask"] = 1
    out["price"] = out["close"]
    out["return"] = out["close"].pct_change().where(_same_date_as_previous(out.index))
    out["log_return"] = out["mid_return"]
    out["noise_proxy"] = "bidask_spread_minus_log_volume"
    for reason, count in dropped.items():
        out.attrs[f"dropped_{reason}"] = count
    return _ordered_output(out)


def _clean_tick_chunk(chunk: pd.DataFrame, dropped: dict[str, int]) -> pd.DataFrame:
    timestamp = pd.to_datetime(chunk["date"].astype(str) + " " + chunk["time"].astype(str), errors="raise")
    df = pd.DataFrame(index=pd.DatetimeIndex(timestamp))
    df.index.name = "timestamp"
    for column in ["price", "bid", "ask", "size"]:
        df[column] = pd.to_numeric(chunk[column], errors="raise").to_numpy()

    finite = pd.Series(np.isfinite(df[["price", "bid", "ask", "size"]].to_numpy()).all(axis=1), index=df.index)
    df = _drop_counted(df, finite, "nonfinite_tick_values", dropped)
    df = _drop_counted(df, (df[["price", "bid", "ask", "size"]] > 0).all(axis=1), "nonpositive_tick_values", dropped)
    df = _drop_counted(df, df["ask"] >= df["bid"], "crossed_tick_quotes", dropped)
    before_session = len(df)
    df = df.between_time("09:30", "16:00")
    _record_drop(dropped, "outside_regular_session", before_session - len(df))
    if df.empty:
        return df
    df = df.sort_index(kind="mergesort")
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid"]
    return df


def _aggregate_tick_chunk(
    df: pd.DataFrame, prev_price: float, prev_timestamp: pd.Timestamp | pd.NaT
) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    minute = df.index.floor("min")
    grouped = df.groupby(minute, sort=True)
    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "timestamp"
    out["open"] = grouped["price"].first()
    out["high"] = grouped["price"].max()
    out["low"] = grouped["price"].min()
    out["close"] = grouped["price"].last()
    out["volume"] = grouped["size"].sum()
    out["trade_count"] = grouped["price"].size()
    out["bid_close"] = grouped["bid"].last()
    out["ask_close"] = grouped["ask"].last()
    out["mid_close"] = grouped["mid"].last()
    out["spread_close"] = grouped["spread"].last()
    out["rel_spread_close"] = grouped["rel_spread"].last()

    previous_prices = df["price"].shift(1)
    previous_timestamps = pd.Series(df.index, index=df.index).shift(1)
    if np.isfinite(prev_price):
        previous_prices.iloc[0] = prev_price
        previous_timestamps.iloc[0] = prev_timestamp
    same_date = pd.Series(df.index.normalize(), index=df.index) == previous_timestamps.dt.normalize()
    log_returns = np.log(df["price"] / previous_prices).where(same_date)
    out["realized_var"] = log_returns.pow(2).groupby(minute).sum(min_count=1).fillna(0.0)
    return out, float(df["price"].iloc[-1]), df.index[-1]


def _combine_minute_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(partials)
    grouped = combined.groupby(level=0, sort=True)
    out = pd.DataFrame(index=grouped.size().index)
    out.index.name = "timestamp"
    out["open"] = grouped["open"].first()
    out["high"] = grouped["high"].max()
    out["low"] = grouped["low"].min()
    out["close"] = grouped["close"].last()
    out["volume"] = grouped["volume"].sum()
    out["trade_count"] = grouped["trade_count"].sum()
    out["bid_close"] = grouped["bid_close"].last()
    out["ask_close"] = grouped["ask_close"].last()
    out["mid_close"] = grouped["mid_close"].last()
    out["spread_close"] = grouped["spread_close"].last()
    out["rel_spread_close"] = grouped["rel_spread_close"].last()
    out["realized_var"] = grouped["realized_var"].sum()
    return out


def _ordered_output(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _add_session_fields(df)
    columns = [
        "symbol",
        "data_type",
        "has_bidask",
        "session_date",
        "bar_index",
        "gap_minutes",
        "open",
        "high",
        "low",
        "close",
        "price",
        "volume",
        "trade_count",
        "bid_close",
        "ask_close",
        "mid_close",
        "spread_close",
        "rel_spread_close",
        "return",
        "log_return",
        "mid_return",
        "abs_mid_return",
        "log_volume",
        "rolling_vol_5",
        "rolling_vol_20",
        "realized_var",
        "noise_score_raw",
        "shock_score_raw",
        "noise_proxy",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[columns].sort_index(kind="mergesort")


def _add_session_fields(df: pd.DataFrame) -> None:
    timestamps = pd.Series(pd.DatetimeIndex(df.index), index=np.arange(len(df)))
    dates = timestamps.dt.normalize()
    same_session = dates == dates.shift(1)
    df["session_date"] = dates.dt.strftime("%Y-%m-%d").to_numpy()
    df["bar_index"] = dates.groupby(dates).cumcount().to_numpy()
    df["gap_minutes"] = ((timestamps - timestamps.shift(1)).dt.total_seconds() / 60.0).where(same_session).to_numpy()


def _inventory_row(
    symbol: str,
    data_type: str,
    raw_path: Path,
    output_path: Path,
    skipped: bool,
    seconds: float,
    df: pd.DataFrame | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": symbol,
        "data_type": data_type,
        "raw_path": str(raw_path),
        "raw_bytes": int(raw_path.stat().st_size),
        "raw_mtime_ns": int(raw_path.stat().st_mtime_ns),
        "output_path": str(output_path),
        "skipped": skipped,
        "seconds": seconds,
        "timestamp_note": "Kibot Eastern time stored as timezone-naive pandas timestamps",
    }
    if df is not None:
        row.update(
            {
                "rows": int(len(df)),
                "start": str(df.index.min()),
                "end": str(df.index.max()),
                "has_bidask": int(df["has_bidask"].iloc[0]),
                "finite_return_rows": int(np.isfinite(df["return"]).sum()),
                "finite_feature_rows": int(np.isfinite(df[["mid_return", "rolling_vol_20", "noise_score_raw", "shock_score_raw"]].to_numpy()).all(axis=1).sum()),
            }
        )
        _add_drop_counts(row, df.attrs)
    return row


def _add_drop_counts(row: dict[str, object], attrs: dict[object, object]) -> None:
    for key, value in attrs.items():
        if key == "dropped_rows" and isinstance(value, dict):
            for reason, count in value.items():
                row[f"dropped_{reason}"] = int(count)
        elif isinstance(key, str) and key.startswith("dropped_"):
            row[key] = int(value)


def _merge_inventory(processed_dir: Path, new_rows: list[dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    inventory_path = processed_dir / "data_inventory.csv.gz"
    if not inventory_path.exists():
        return new_df

    existing = pd.read_csv(inventory_path)
    if existing.empty:
        return new_df
    if new_df.empty:
        return existing

    processed = new_df.loc[~new_df["skipped"].astype(bool)]
    replacement_keys = set(zip(processed["symbol"], processed["data_type"]))
    if replacement_keys:
        keep_existing = [
            (symbol, data_type) not in replacement_keys
            for symbol, data_type in zip(existing["symbol"], existing["data_type"])
        ]
        existing = existing.loc[keep_existing]

    existing_keys = set(zip(existing["symbol"], existing["data_type"]))
    keep_new = [
        not (bool(row.skipped) and (row.symbol, row.data_type) in existing_keys)
        for row in new_df.itertuples(index=False)
    ]
    return pd.concat([existing, new_df.loc[keep_new]], ignore_index=True, sort=False)


def _write_metadata(processed_dir: Path, row: dict[str, object]) -> None:
    metadata_path = processed_dir / f"{row['symbol']}_{row['data_type']}_metadata.json"
    metadata_path.write_text(json.dumps(_json_ready(row), indent=2, allow_nan=False), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def _write_processed_csv(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    df.to_csv(temp_path, compression="gzip")
    temp_path.replace(output_path)


def _raise_if_chunk_moves_backward(
    first_timestamp: pd.Timestamp,
    previous_timestamp: pd.Timestamp | pd.NaT,
    path: Path,
    chunk_number: int,
) -> None:
    if pd.notna(previous_timestamp) and first_timestamp < previous_timestamp:
        raise ValueError(
            "Tick timestamps moved backward across chunks in "
            f"{path} at chunk {chunk_number}: first={first_timestamp}, previous={previous_timestamp}"
        )


def _same_date_as_previous(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.Series(index.normalize(), index=index)
    return dates == dates.shift(1)


def _drop_counted(df: pd.DataFrame, valid: pd.Series, reason: str, dropped: dict[str, int]) -> pd.DataFrame:
    count = int((~valid).sum())
    _record_drop(dropped, reason, count)
    if count == 0:
        return df
    return df.loc[valid].copy()


def _record_drop(dropped: dict[str, int], reason: str, count: int) -> None:
    if count > 0:
        dropped[reason] = dropped.get(reason, 0) + int(count)


if __name__ == "__main__":
    main()
