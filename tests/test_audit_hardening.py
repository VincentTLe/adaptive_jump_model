"""Regression tests for the 2026-07 full-audit hardening fixes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import BacktestError, apply_signal, performance_metrics
from adaptive_jump.cli import RunError, _verify_manifest
from adaptive_jump.config import ConfigError, load_config
from adaptive_jump.features import make_features, prepare_market

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research-expanding-v8-2.toml"


def test_variant_v8_2_anchor() -> None:
    config = load_config(VARIANT)
    assert config.config_id == "shu-replication-expanding-v8-2"
    assert config.model_protocol.standardizer == "expanding_full_history_ddof1"
    jp = next(market for market in config.markets if market.id == "jp")
    assert jp.equity.settings["sha256"] == (
        "e8717952ed760b33cb3bd5ecb597e1aae1201c029983bd5bbf61bc731785fc11"
    )
    assert jp.equity.settings["source_start_observed"] == "1965-01-05"


VARIANT_V8_3 = ROOT / "research-expanding-v8-3.toml"


def test_variant_v8_3_anchor() -> None:
    config = load_config(VARIANT_V8_3)
    assert config.config_id == "shu-replication-expanding-v8-3"
    assert str(config.sample_start) == "1970-01-01"
    assert config.model_protocol.standardizer == "expanding_full_history_ddof1"
    assert config.model_protocol.standardizer_min_observations == 63
    assert config.hmm_protocol.smoothing_grid == (0, 2, 4, 8, 20)
    assert config.hmm_protocol.covars_prior == 0.0


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "covars_prior = 0.0",
            "covars_prior = 0.005",
            "covariance prior must be",
        ),
        (
            "smoothing_grid = [0, 2, 4, 8, 20]",
            "smoothing_grid = [0, 2, 4, 8, 10]",
            "invalid HMM smoothing grid",
        ),
        (
            "standardizer_min_observations = 63",
            "standardizer_min_observations = 62",
            "at least one quarter of warm-up",
        ),
    ],
)
def test_variant_v8_3_rejects_unfrozen_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = VARIANT_V8_3.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


VARIANT_V8_5 = ROOT / "research-expanding-v8-5.toml"


def test_variant_v8_5_differs_from_v8_4_only_in_the_smoothing_grid() -> None:
    """v8.5 must be a clean read on one edit, so nothing else may move.

    The grid gains k = 6 because line 390 of the paper names it as the
    method's inherited default. If any other frozen value drifts alongside
    it, the run stops measuring that edit and the comparison is worthless.
    """
    v8_4 = load_config(ROOT / "research-expanding-v8-4.toml")
    v8_5 = load_config(VARIANT_V8_5)

    assert v8_5.config_id == "shu-replication-expanding-v8-5"
    assert v8_5.hmm_protocol.smoothing_grid == (0, 2, 4, 6, 8, 20)
    assert v8_4.hmm_protocol.smoothing_grid == (0, 2, 4, 8, 20)

    # The anchor that puts every market out of sample on the paper's 1990 date.
    assert v8_5.sample_start == v8_4.sample_start
    assert v8_5.replication_cutoff == v8_4.replication_cutoff
    assert v8_5.model_protocol == v8_4.model_protocol
    assert v8_5.jm_protocol == v8_4.jm_protocol
    assert v8_5.selection_protocol == v8_4.selection_protocol
    assert v8_5.metrics_protocol == v8_4.metrics_protocol

    # Same data, so the acquisition manifest is shared and the two runs are
    # comparable row for row.
    assert v8_5.markets == v8_4.markets

    rest_5 = dict(vars(v8_5.hmm_protocol))
    rest_4 = dict(vars(v8_4.hmm_protocol))
    rest_5.pop("smoothing_grid")
    rest_4.pop("smoothing_grid")
    assert rest_5 == rest_4


def test_variant_v8_5_rejects_a_grid_the_paper_does_not_justify() -> None:
    """The allowlist is what stops the grid being tuned toward Table 4."""
    payload = VARIANT_V8_5.read_text(encoding="utf-8")
    old = "smoothing_grid = [0, 2, 4, 6, 8, 20]"
    assert old in payload
    candidate = ROOT / "tests" / "_tmp_v8_5_grid.toml"
    try:
        candidate.write_text(
            payload.replace(old, "smoothing_grid = [0, 2, 4, 6, 8, 12, 20]", 1),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="invalid HMM smoothing grid"):
            load_config(candidate)
    finally:
        candidate.unlink(missing_ok=True)


VARIANT_V9 = ROOT / "research-expanding-v9.toml"


def test_variant_v9_differs_from_v8_5_only_in_the_us_equity_series() -> None:
    """v9 must isolate the index substitution, or the run measures nothing.

    The audit traced the US HMM deviation to using the CRSP value-weighted
    total market where the paper names the S&P 500 (line 153-155). If any
    other frozen value drifts alongside the series, the run stops being a
    clean read on that one change -- and Germany and Japan, which already
    match Shu on 7 of 8 metrics, must be untouched.
    """
    v8_5 = load_config(VARIANT_V8_5)
    v9 = load_config(VARIANT_V9)

    assert v9.config_id == "shu-replication-expanding-v9"
    assert v9.sample_start == v8_5.sample_start
    assert v9.replication_cutoff == v8_5.replication_cutoff
    assert v9.model_protocol == v8_5.model_protocol
    assert v9.jm_protocol == v8_5.jm_protocol
    assert v9.hmm_protocol == v8_5.hmm_protocol
    assert v9.selection_protocol == v8_5.selection_protocol
    assert v9.metrics_protocol == v8_5.metrics_protocol
    assert v9.backtest_protocol == v8_5.backtest_protocol

    old = {m.id: m for m in v8_5.markets}
    new = {m.id: m for m in v9.markets}
    assert set(old) == set(new)

    # Germany and Japan already use the paper's own indices; they must be
    # byte-for-byte the same market definition.
    for market in ("de", "jp"):
        assert new[market] == old[market], market

    # The US changes in exactly one place: the equity source.
    assert new["us"].cash == old["us"].cash
    assert new["us"].equity != old["us"].equity
    assert new["us"].equity.source_id == "SP500_TR"
    assert old["us"].equity.source_id == "FRENCH_US_TR"
    assert new["us"].equity.settings["file_path"].endswith("us_equity_tr_sp500.csv")


VARIANT_V9_1 = ROOT / "research-expanding-v9-1.toml"


def test_variant_v9_1_differs_from_v9_only_in_the_drawdown_basis() -> None:
    """v9.1 must isolate the drawdown definition, or the run measures nothing.

    Every fitted state is identical to v9 by construction -- selection scores
    candidates on validation Sharpe and never reads a drawdown -- so this
    variant changes what is reported and nothing that is computed.
    """
    v9 = load_config(VARIANT_V9)
    v9_1 = load_config(VARIANT_V9_1)

    assert v9_1.config_id == "shu-replication-expanding-v9-1"
    assert v9.metrics_protocol.drawdown_basis == "total_wealth"
    assert v9_1.metrics_protocol.drawdown_basis == "risky_leg_wealth_flat_in_cash"

    assert v9_1.sample_start == v9.sample_start
    assert v9_1.replication_cutoff == v9.replication_cutoff
    assert v9_1.model_protocol == v9.model_protocol
    assert v9_1.jm_protocol == v9.jm_protocol
    assert v9_1.hmm_protocol == v9.hmm_protocol
    assert v9_1.selection_protocol == v9.selection_protocol
    assert v9_1.backtest_protocol == v9.backtest_protocol
    assert v9_1.markets == v9.markets

    rest_new = dict(vars(v9_1.metrics_protocol))
    rest_old = dict(vars(v9.metrics_protocol))
    rest_new.pop("drawdown_basis")
    rest_old.pop("drawdown_basis")
    assert rest_new == rest_old


def test_variant_v9_1_rejects_an_unnamed_drawdown_basis(tmp_path: Path) -> None:
    payload = VARIANT_V9_1.read_text(encoding="utf-8")
    old = 'maximum_drawdown = "risky_leg_wealth_flat_in_cash"'
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(
        payload.replace(old, 'maximum_drawdown = "excess_wealth"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="invalid metric definition"):
        load_config(candidate)


def test_configs_written_before_the_field_existed_keep_the_legacy_basis() -> None:
    """A sealed run must keep replaying to the numbers it recorded."""
    for variant in (VARIANT, VARIANT_V8_3, VARIANT_V8_5, VARIANT_V9):
        config = load_config(variant)
        assert config.metrics_protocol.drawdown_basis == "total_wealth", variant.name


def test_paper_drawdown_basis_is_flat_while_the_strategy_sits_in_cash() -> None:
    """The defining property, checked on a case where the two bases differ.

    Equity falls hard while invested, the strategy sits in cash for a while, then
    equity falls again. Under the legacy basis the cash yield lifts the wealth
    path between the two declines, so the second one starts from higher ground
    and the combined drawdown comes out shallower. Under the paper's basis the
    path is flat through the cash stretch and the two declines compound.
    """
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=10),
            "equity_simple": [0.0, -0.10, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              -0.10],
            "cash_return": [0.002] * 10,
        }
    )
    signal = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    path = apply_signal(returns, signal, delay_trading_days=0, one_way_cost_bps=0)
    path = path.dropna(subset=["position"])

    legacy = performance_metrics(path, drawdown_basis="total_wealth")
    paper = performance_metrics(
        path, drawdown_basis="risky_leg_wealth_flat_in_cash"
    )

    # Three -10% days while invested and nothing in between: 0.9 ** 3 - 1.
    assert paper["maximum_drawdown"] == pytest.approx(0.9**3 - 1.0)
    assert legacy["maximum_drawdown"] > paper["maximum_drawdown"]
    # Everything that does not touch the drawdown is untouched.
    for field in ("cagr", "volatility", "sharpe", "turnover", "leverage"):
        assert legacy[field] == pytest.approx(paper[field])
    # Calmar rides on the drawdown, so it moves with it and only with it.
    assert paper["calmar"] == pytest.approx(
        legacy["calmar"] * legacy["maximum_drawdown"] / paper["maximum_drawdown"]
    )


def test_paper_drawdown_basis_matches_the_legacy_one_when_never_in_cash() -> None:
    """Buy-and-hold cannot tell the two bases apart, which is why it settles nothing."""
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=10),
            "equity_simple": [0.01, -0.03, 0.02, -0.05, 0.01, 0.0, 0.02, -0.01,
                              0.03, -0.02],
            "cash_return": [0.0002] * 10,
        }
    )
    path = apply_signal(returns, pd.Series([1.0] * 10), delay_trading_days=0,
                        one_way_cost_bps=0).dropna(subset=["position"])
    legacy = performance_metrics(path, drawdown_basis="total_wealth")
    paper = performance_metrics(path,
                                drawdown_basis="risky_leg_wealth_flat_in_cash")
    assert legacy["maximum_drawdown"] == pytest.approx(paper["maximum_drawdown"])


def test_paper_drawdown_basis_requires_the_equity_column() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6),
            "equity_simple": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "cash_return": [0.0001] * 6,
        }
    )
    path = apply_signal(returns, pd.Series([1.0] * 6), delay_trading_days=0)
    path = path.dropna(subset=["position"]).drop(columns=["equity_simple"])
    with pytest.raises(BacktestError, match="missing metric columns"):
        performance_metrics(path, drawdown_basis="risky_leg_wealth_flat_in_cash")
    with pytest.raises(BacktestError, match="unknown drawdown basis"):
        performance_metrics(path, drawdown_basis="something_else")


VARIANT_V9_2 = ROOT / "research-expanding-v9-2.toml"


def test_variant_v9_2_differs_from_v9_1_only_in_the_de_equity_series() -> None:
    """v9.2 must isolate the German dividend repair.

    The repair restores dividends missing from the DAX before 1988. It touches
    only the training window -- Table 4's German column covers 1990-2023, which
    is inside the untouched official segment -- so if any other frozen value
    moves alongside it the run stops measuring that repair.
    """
    v9_1 = load_config(VARIANT_V9_1)
    v9_2 = load_config(VARIANT_V9_2)

    assert v9_2.config_id == "shu-replication-expanding-v9-2"
    assert v9_2.sample_start == v9_1.sample_start
    assert v9_2.replication_cutoff == v9_1.replication_cutoff
    assert v9_2.model_protocol == v9_1.model_protocol
    assert v9_2.jm_protocol == v9_1.jm_protocol
    assert v9_2.hmm_protocol == v9_1.hmm_protocol
    assert v9_2.selection_protocol == v9_1.selection_protocol
    assert v9_2.metrics_protocol == v9_1.metrics_protocol
    assert v9_2.backtest_protocol == v9_1.backtest_protocol

    old = {m.id: m for m in v9_1.markets}
    new = {m.id: m for m in v9_2.markets}
    assert set(old) == set(new)
    for market in ("us", "jp"):
        assert new[market] == old[market], market
    assert new["de"].cash == old["de"].cash
    assert new["de"].equity != old["de"].equity
    assert new["de"].equity.source_id == "STOOQ_DAX_TR"
    assert new["de"].equity.settings["file_path"].endswith(
        "de_equity_tr_dividend_adjusted.csv")


def test_de_repair_only_touches_the_backcast_era() -> None:
    """The official segment must come through byte-identical.

    From 1987-12-30 the DAX performance index is used as published in both
    files. If the repair leaked into that segment it would change results the
    paper reports on, and the repair would no longer be a training-window fix.
    """
    external = ROOT / "data" / "external"
    old_path = external / "de_equity_tr.csv"
    new_path = external / "de_equity_tr_dividend_adjusted.csv"
    if not new_path.is_file():
        pytest.skip("repaired German series not built")
    old = pd.read_csv(old_path, parse_dates=["date"]).set_index("date")["value"]
    new = pd.read_csv(new_path, parse_dates=["date"]).set_index("date")["value"]

    official = old.index >= pd.Timestamp("1987-12-30")
    pd.testing.assert_series_equal(old[official], new[official])

    # And the backcast era must have moved, in the right direction: adding a
    # dividend makes past values lower once the series is chained backwards.
    early = old.index < pd.Timestamp("1987-12-30")
    assert (new[early] < old[early]).mean() > 0.99
    ratio = float(old[early].iloc[0] / new[early].iloc[0])
    years = (pd.Timestamp("1987-12-30") - old.index[0]).days / 365.25
    implied = ratio ** (1 / years) - 1
    assert 0.02 < implied < 0.05, f"implied dividend {implied:.4f}"


def test_manifest_rejects_duplicated_source_entry() -> None:
    config = load_config(VARIANT)
    sources = [
        {"market": market.id, "kind": kind, "source_id": source.source_id}
        for market in config.markets
        for kind, source in (("equity", market.equity), ("cash", market.cash))
    ]
    document = {
        "config_id": config.config_id,
        "config_sha256": config.sha256,
        "replication_cutoff": config.replication_cutoff.isoformat(),
        "sources": sources + [sources[0]],
    }
    with pytest.raises(RunError, match="duplicated or missing source"):
        _verify_manifest(config, document, ROOT)


JP_SHA = "e8717952ed760b33cb3bd5ecb597e1aae1201c029983bd5bbf61bc731785fc11"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            f'sha256 = "{JP_SHA}"',
            f'sha256 = "{JP_SHA.upper()}"',
            "64 lowercase hex",
        ),
        (
            'file_path = "data/external/jp_equity_tr.csv"',
            'file_path = "data/../secrets/jp_equity_tr.csv"',
            "inside the repository",
        ),
    ],
)
def test_localfile_gate_rejects_crafted_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    assert old in payload
    candidate = tmp_path / "research.toml"
    candidate.write_text(payload.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(candidate)


def test_whitespace_only_documentation_is_rejected(tmp_path: Path) -> None:
    payload = VARIANT.read_text(encoding="utf-8")
    start = payload.index('splice_documentation = """')
    end = payload.index('"""', start + 30) + 3
    doped = (
        payload[:start]
        + 'splice_documentation = "' + " " * 60 + '"'
        + payload[end:]
    )
    candidate = tmp_path / "research.toml"
    candidate.write_text(doped, encoding="utf-8")
    with pytest.raises(ConfigError, match="splice_documentation"):
        load_config(candidate)


def test_apply_signal_treats_series_positionally() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6),
            "equity_simple": [0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
            "cash_return": [0.0001] * 6,
        }
    )
    plain = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    shifted_index = pd.Series(plain.to_numpy(), index=range(10, 16))
    dated_index = pd.Series(plain.to_numpy(), index=returns["date"])
    base = apply_signal(returns, plain)
    for variant in (shifted_index, dated_index):
        result = apply_signal(returns, variant)
        pd.testing.assert_series_equal(result["position"], base["position"])
    assert base["position"].iloc[2] == 1.0  # signal day 0 first held at t+2


def test_prepare_market_applies_expanding_standardization() -> None:
    config = load_config(VARIANT)
    market = config.markets[0]
    rng = np.random.default_rng(5)
    n = 900
    dates = pd.bdate_range("1980-01-01", periods=n)
    level = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    equity = pd.DataFrame({"date": dates.date.astype(str), "value": level})
    cash = pd.DataFrame(
        {"date": dates.date.astype(str), "value": np.full(n, 3.0)}
    )
    frame = prepare_market(equity, cash, market, config)
    raw = make_features(frame["excess_return"])
    min_obs = config.model_protocol.standardizer_min_observations
    assert frame["dd_10"].iloc[: min_obs - 1].isna().all()
    tail = frame["dd_10"].iloc[min_obs + 50 :].dropna()
    raw_tail = raw["dd_10"].iloc[min_obs + 50 :].dropna()
    assert not np.allclose(tail.to_numpy(), raw_tail.to_numpy())
    assert tail.abs().max() < 25  # standardized magnitudes, not raw levels
