"""Regression tests for the 2026-07 full-audit hardening fixes."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from adaptive_jump.backtest import BacktestError, apply_signal, performance_metrics
from adaptive_jump.cli import RunError, _verify_manifest
from adaptive_jump.config import ConfigError, load_config
from adaptive_jump.features import make_features, prepare_market

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "research-expanding-v8-5.toml"


def load_claim_checker() -> ModuleType:
    path = ROOT / "scripts" / "check_paper_claims.py"
    spec = importlib.util.spec_from_file_location("check_paper_claims", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_checker_accepts_a_quote_at_the_annotated_line(tmp_path: Path) -> None:
    checker = load_claim_checker()
    lines = ["heading", "the quoted claim is present in the paper", "tail"]
    target = tmp_path / "claim.md"
    target.write_text(
        '[line 2] "the quoted claim is present in the paper"',
        encoding="utf-8",
    )

    assert checker.check_citations(
        target, lines, checker.normalise(" ".join(lines))
    ) == ([], 1)


def test_claim_checker_rejects_a_quote_annotated_at_the_wrong_line(
    tmp_path: Path,
) -> None:
    checker = load_claim_checker()
    lines = ["filler"] * 8 + ["the real paper quote appears down here"]
    target = tmp_path / "claim.md"
    target.write_text(
        '[line 1] "the real paper quote appears down here"',
        encoding="utf-8",
    )

    failures, count = checker.check_citations(
        target, lines, checker.normalise(" ".join(lines))
    )

    assert count == 1
    assert len(failures) == 1
    assert "actually appears near line" in failures[0]


def test_claim_checker_rejects_a_fabricated_quote(tmp_path: Path) -> None:
    checker = load_claim_checker()
    lines = ["the actual paper says something else"]
    target = tmp_path / "claim.md"
    target.write_text(
        '[line 1] "this quotation was never in the source paper"',
        encoding="utf-8",
    )

    failures, count = checker.check_citations(
        target, lines, checker.normalise(" ".join(lines))
    )

    assert count == 1
    assert len(failures) == 1
    assert "NOT IN PAPER" in failures[0]


def test_claim_checker_scans_absence_terms_only_in_the_paper_body() -> None:
    checker = load_claim_checker()
    references_only = ["method text", "References", "A dividend study"]
    assert checker.check_absences(references_only) == []

    failures = checker.check_absences(
        ["the method uses a dividend adjustment", "References"]
    )
    assert len(failures) == 1
    assert "dividend construction" in failures[0]


def test_claim_checker_defaults_include_reader_facing_documents() -> None:
    checker = load_claim_checker()
    assert ROOT / "README.md" in checker.DEFAULT_TARGETS
    assert ROOT / "paper" / "manuscript.tex" in checker.DEFAULT_TARGETS


def test_claim_checker_discloses_zero_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checker = load_claim_checker()
    paper = tmp_path / "paper.txt"
    paper.write_text("plain paper body\nReferences\n", encoding="utf-8")
    target = tmp_path / "unannotated.md"
    target.write_text("An unannotated prose claim.", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["check_paper_claims.py", "--paper", str(paper), str(target)],
    )

    assert checker.main() == 0
    output = capsys.readouterr().out
    assert "UNANNOTATED" in output
    assert "prose not checked" in output
    assert "1 target(s) contain unannotated prose that was not checked" in output


def test_variant_v8_5_rejects_a_grid_the_paper_does_not_justify() -> None:
    """The allowlist is what stops the grid being tuned toward Table 4."""
    payload = VARIANT.read_text(encoding="utf-8")
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
            "equity_simple": [0.0, -0.10, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.10],
            "cash_return": [0.002] * 10,
        }
    )
    signal = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    path = apply_signal(returns, signal, delay_trading_days=0, one_way_cost_bps=0)
    path = path.dropna(subset=["position"])

    legacy = performance_metrics(path, drawdown_basis="total_wealth")
    paper = performance_metrics(path, drawdown_basis="risky_leg_wealth_flat_in_cash")

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
            "equity_simple": [
                0.01,
                -0.03,
                0.02,
                -0.05,
                0.01,
                0.0,
                0.02,
                -0.01,
                0.03,
                -0.02,
            ],
            "cash_return": [0.0002] * 10,
        }
    )
    path = apply_signal(
        returns, pd.Series([1.0] * 10), delay_trading_days=0, one_way_cost_bps=0
    ).dropna(subset=["position"])
    legacy = performance_metrics(path, drawdown_basis="total_wealth")
    paper = performance_metrics(path, drawdown_basis="risky_leg_wealth_flat_in_cash")
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
        payload[:start] + 'splice_documentation = "' + " " * 60 + '"' + payload[end:]
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
    cash = pd.DataFrame({"date": dates.date.astype(str), "value": np.full(n, 3.0)})
    frame = prepare_market(equity, cash, market, config)
    raw = make_features(frame["excess_return"])
    min_obs = config.model_protocol.standardizer_min_observations
    assert frame["dd_10"].iloc[: min_obs - 1].isna().all()
    tail = frame["dd_10"].iloc[min_obs + 50 :].dropna()
    raw_tail = raw["dd_10"].iloc[min_obs + 50 :].dropna()
    assert not np.allclose(tail.to_numpy(), raw_tail.to_numpy())
    assert tail.abs().max() < 25  # standardized magnitudes, not raw levels
