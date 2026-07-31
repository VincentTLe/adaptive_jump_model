"""Per-market candidate grids shared by the rerun harnesses.

Every study restored here was written when one global JM lambda grid governed
all markets, so a single ``spec.lambdas`` tuple could name the candidate-state
columns, the sealed ``lambda0`` set and the event grid at once. Under the
calibrated-v10 contract each market carries its own grid
(``config.jm_protocol_for(market).lambda_grid``), so those three things are
per-market tables. This module is the one place that resolves them, so the
studies never re-derive a grid from a literal.

The event grid keeps its original meaning -- the positive lambdas -- because a
lambda0 of zero admits no transition-penalty discount at all: the arrival and
lagged rules both scale ``lambda0`` by a factor in ``(0, 1]``, so at
``lambda0 = 0`` every rule collapses onto fixed JM and the event tables would be
empty by construction. "The positive lambdas" is now read per market rather than
as ``LAMBDAS[1:]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

MarketGrids = dict[str, tuple[float, ...]]


class GridScopeError(ValueError):
    """Raised when a per-market grid table is missing or malformed."""


def market_grids(config: Any, markets: Iterable[str]) -> MarketGrids:
    """The JM lambda grid governing each market under the loaded contract."""
    return {
        market: tuple(config.jm_protocol_for(market).lambda_grid) for market in markets
    }


def positive_grids(grids: Mapping[str, tuple[float, ...]]) -> MarketGrids:
    """Restrict every market grid to its strictly positive lambdas."""
    return {
        market: tuple(value for value in values if value > 0.0)
        for market, values in grids.items()
    }


def parse_grid_table(value: Any, markets: Iterable[str]) -> MarketGrids | None:
    """Read a ``{market = [lambda, ...]}`` TOML table, or ``None`` if invalid."""
    expected = tuple(markets)
    if not isinstance(value, dict) or set(value) != set(expected):
        return None
    parsed: MarketGrids = {}
    for market in expected:
        values = value[market]
        if not isinstance(values, list) or not values:
            return None
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in values
        ):
            return None
        parsed[market] = tuple(float(item) for item in values)
    return parsed


def grids_equal(
    left: Mapping[str, tuple[float, ...]] | None,
    right: Mapping[str, tuple[float, ...]] | None,
) -> bool:
    """Exact per-market equality of two grid tables."""
    if left is None or right is None:
        return False
    if set(left) != set(right):
        return False
    return all(tuple(left[market]) == tuple(right[market]) for market in left)


def grid_for(grids: Mapping[str, tuple[float, ...]], market: str) -> tuple[float, ...]:
    """The grid governing one market, or a hard error naming the market."""
    try:
        return tuple(grids[market])
    except KeyError as exc:
        raise GridScopeError(f"no candidate grid for market: {market}") from exc
