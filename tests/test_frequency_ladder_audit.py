"""Audit regressions for the frequency-ladder experiment (frequency-ladder-001).

Written by the independent auditor of that experiment, who wrote neither the
spec, the map script nor the runner. Each test pins a property that the headline
claim rests on and that NOTHING in the repository checked at the time of the
audit:

* the runner's re-derivation gate covers arm L only, so arm M -- the arm the
  headline German number comes from -- is never re-derived and never checked to
  be arm L truncated (fault F7 passed silently);
* nothing binds the spec's declared cells, delay, cost or grading bands to the
  code that implements them;
* nothing asserts that the pre-1990 restriction is what actually makes the menu
  causal, i.e. that post-cutoff windows cannot move it;
* nothing states that the achievable menu is quantised to midpoints of the
  union penalty grid, so the ladder is not the only free choice;
* the objective column is assumed to be the PENALISED objective and its
  finite-difference slope assumed to be the jump count, and neither was ever
  checked against a refit; and
* the frozen spec asserts that no window with a decreasing objective is dated
  before 1990. One is (jp 1989-07-03), and it sets Japan's slowest rung.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/frequency-ladder-001.toml"
REGISTRY = ROOT / "research/experiment_registry.jsonl"
UNION = ROOT / "artifacts/jm-residual/01-grid-identification"
RUN = ROOT / "artifacts/frequency-ladder/01-run"
BASE = (
    ROOT / "artifacts/fixed-baselines/"
    "fixed-baselines-36ca1ace131c-ed7abd7daea3-f9f3e0a93736"
)
MARKETS = ("us", "de", "jp")
FEATURE_COLUMNS = ("dd_10", "sortino_20", "sortino_60")
TRAINING_YEARS = 3000 / 252

# The run outputs these audits measure. Every one of them is ignored by
# .gitignore -- the sealed baseline is a hash-named run with a per-file
# inventory, and the rest are megabyte state matrices and refit tables that are
# regenerable -- so a clean checkout has none of them.
UNION_REFITS = tuple(UNION / market / "union-refits.csv" for market in MARKETS)
METRICS = BASE / "metrics.csv"
DE_FEATURES = BASE / "de/features.csv"
DE_SEALED_STATES = BASE / "de/jm-states.csv"
DE_LADDER_STATES = RUN / "states-de.csv"
LADDER_CELLS = RUN / "cells.csv"


def _require(*paths: Path) -> None:
    """Skip, naming the file, when a local-only run artifact is absent.

    Each audit below re-derives or recomputes a number from the artifact it
    names, so there is no smaller stand-in that still audits anything. Without
    the artifact the test says which one is missing rather than dying on
    FileNotFoundError -- or, for the xfail below, appearing to fail for the
    reason it documents when it actually failed on a missing file.
    """
    for path in paths:
        if not path.exists():
            pytest.skip(f"{path.relative_to(ROOT)} not built in this checkout")


def _module(relative: str):
    """Import a script by its path under ``scripts/`` without running ``main()``."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    path = ROOT / "scripts" / f"{relative}.py"
    spec = importlib.util.spec_from_file_location(f"{path.stem}_under_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec() -> dict:
    return tomllib.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runner():
    return _module("experiments/run_frequency_ladder")


def _refits(market: str) -> pd.DataFrame:
    frame = pd.read_csv(UNION / market / "union-refits.csv")
    frame["fit_date"] = pd.to_datetime(frame["fit_date"])
    return frame


def _derive(refits: pd.DataFrame, ladder, cutoff: pd.Timestamp) -> list[float]:
    """The auditor's own inversion, written from the spec's prose."""
    pre = refits[refits["fit_date"] < cutoff]
    per: dict[float, list[float]] = {target: [] for target in ladder}
    for _, group in pre.groupby("fit_date"):
        group = group.sort_values("lambda")
        lam = group["lambda"].to_numpy(float)
        obj = group["objective"].to_numpy(float)
        midpoints = (lam[:-1] + lam[1:]) / 2.0
        rate = np.diff(obj) / np.diff(lam) / TRAINING_YEARS
        for target in ladder:
            hit = np.flatnonzero(rate <= target)
            per[target].append(midpoints[hit[0]] if hit.size else np.nan)
    return [float(np.nanmedian(per[target])) for target in ladder]


# --------------------------------------------------------------------------
# 1. what was run is what was frozen
# --------------------------------------------------------------------------


def test_frozen_spec_matches_its_registry_hash():
    digest = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("experiment_id") == "frequency-ladder-001"
    ]
    assert rows, "frequency-ladder-001 has no registry row"
    latest = [row for row in rows if row.get("spec_sha256")][-1]
    assert latest["spec_sha256"] == digest, (
        f"spec on disk hashes to {digest}; the last hashed registry row "
        f"({latest.get('status')}) froze {latest['spec_sha256']}"
    )


def test_the_amendment_is_recorded_before_the_result_event():
    """SPEC_AMENDED_BEFORE_RESULTS has to sit between the freeze and the result."""
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("experiment_id") == "frequency-ladder-001"
    ]
    statuses = [row["status"] for row in rows]
    # The property is the ORDER, not the count. Pinning the exact list made a
    # later, legitimate event - the post-audit retraction - fail a test about
    # something else. The amendment must come after the freeze and before any
    # event that reports a number, and it may happen only once.
    assert statuses[0] == "FROZEN_BEFORE_RESULTS", statuses
    assert statuses.count("SPEC_AMENDED_BEFORE_RESULTS") == 1, statuses
    amended = statuses.index("SPEC_AMENDED_BEFORE_RESULTS")
    reporting = [
        i for i, s in enumerate(statuses) if s.startswith("EXPERIMENT_COMPLETE")
    ]
    assert reporting, statuses
    assert amended < min(reporting), statuses
    stamps = [row["event_utc"] for row in rows]
    assert stamps == sorted(stamps), f"registry events are out of order: {stamps}"
    assert spec_amended_at(rows) == tomllib.loads(SPEC.read_text())["amended_at_utc"]


def spec_amended_at(rows) -> str:
    return next(r["event_utc"] for r in rows
                if r["status"] == "SPEC_AMENDED_BEFORE_RESULTS")


def test_the_amendment_changed_only_what_it_says_it_changed():
    """Reconstruct the pre-amendment spec and hash it against the freeze.

    The amendment claims it corrected four rounded transcriptions and touched
    nothing else. That is checkable without the old file: delete the two lines
    the amendment added, put the rounded values back, and the result must hash
    to the sha256 the FROZEN_BEFORE_RESULTS registry row recorded. If any other
    byte had changed -- a ladder rung, the cutoff, a grading band, a decision
    rule -- this hash could not match.
    """
    rows = [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("experiment_id") == "frequency-ladder-001"
    ]
    frozen = next(r for r in rows if r["status"] == "FROZEN_BEFORE_RESULTS")
    current = SPEC.read_text(encoding="utf-8")
    stripped = "".join(
        line
        for line in current.splitlines(keepends=True)
        if not line.startswith(("amended_at_utc", "amendment ="))
    )
    for exact, rounded in (
        ("1.465348864441625", "1.47"),
        ("5.459611390906074", "5.46"),
        ("2.829145724599095", "2.83"),
        ("8.59842836500576", "8.60"),
    ):
        assert exact in stripped, f"{exact} is no longer in the spec"
        stripped = stripped.replace(exact, rounded)
    digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
    assert digest == frozen["spec_sha256"], (
        "the pre-amendment spec cannot be reconstructed from the amendment's "
        f"own description: rebuilt hash {digest}, frozen hash "
        f"{frozen['spec_sha256']}. Something other than the four transcribed "
        "values changed between the freeze and the run."
    )


def test_the_spec_declares_the_cells_delay_cost_and_bands_the_code_uses(spec):
    """A spec that declares a protocol its code does not implement is prose."""
    scorer = _module("score_grid")
    table5 = _module("_shu_table5")
    assert tuple(spec["protocol"]["cells"]) == scorer.CELLS
    assert spec["protocol"]["delay_trading_days"] == scorer.DELAY == 1
    assert spec["protocol"]["one_way_cost_bps"] == scorer.COST == 10.0
    bands = dict(table5.GRADE_BANDS)
    assert spec["grading"]["A"] == "<= 2 percent" and bands["A"] == 0.02
    assert spec["grading"]["B"] == "2 to 20 percent" and bands["B"] == 0.20
    assert spec["grading"]["C"] == "20 to 40 percent" and bands["C"] == 0.40
    assert spec["grading"]["D"] == "40 to 60 percent" and bands["D"] == 0.60


# --------------------------------------------------------------------------
# 2. the menu: every arm re-derived, not only the one the gate checks
# --------------------------------------------------------------------------


def test_every_arm_is_the_menu_the_refit_objectives_derive(spec):
    """The runner re-derives arm L only. Arm M carries the headline number."""
    _require(*UNION_REFITS)
    ladder = spec["derivation"]["ladder_shifts_per_year"]
    cutoff = pd.Timestamp("1990-01-01")
    for market in MARKETS:
        derived = _derive(_refits(market), ladder, cutoff)
        for arm_id, arm in spec["arms"].items():
            frozen = [float(value) for value in arm[market]]
            assert np.allclose(frozen, derived[: len(frozen)], rtol=1e-12), (
                f"{arm_id} {market}: frozen {frozen} is not the first "
                f"{len(frozen)} rungs of the derived menu {derived}"
            )


def test_arm_m_is_arm_l_truncated_at_the_rung_the_spec_names(spec):
    ladder = spec["derivation"]["ladder_shifts_per_year"]
    assert ladder == [8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
    for market in MARKETS:
        full = [float(v) for v in spec["arms"]["L_full_ladder"][market]]
        truncated = [float(v) for v in spec["arms"]["M_no_freeze"][market]]
        assert len(full) == len(ladder)
        # "truncated at 0.5 shifts per year" == every rung down to and including
        # 0.5, and nothing slower
        assert len(truncated) == ladder.index(0.5) + 1
        assert truncated == full[: len(truncated)]


# --------------------------------------------------------------------------
# 3. no lookahead
# --------------------------------------------------------------------------


def test_no_window_dated_on_or_after_the_cutoff_can_move_the_menu(
    runner, spec, tmp_path, monkeypatch
):
    """Corrupt every post-cutoff objective; the derived menu must not move."""
    _require(*UNION_REFITS)
    ladder = spec["derivation"]["ladder_shifts_per_year"]
    rng = np.random.default_rng(4)
    clean = {market: runner.derive(market, ladder) for market in MARKETS}
    for market in MARKETS:
        frame = _refits(market)
        post = frame["fit_date"] >= runner.CUTOFF
        assert post.any(), f"{market}: no post-cutoff window to corrupt"
        poisoned = frame.copy()
        poisoned.loc[post, "objective"] = rng.uniform(-1e6, 1e6, int(post.sum()))
        target = tmp_path / market
        target.mkdir(parents=True, exist_ok=True)
        poisoned.to_csv(target / "union-refits.csv", index=False)
    monkeypatch.setattr(runner, "UNION", tmp_path)
    for market in MARKETS:
        assert runner.derive(market, ladder) == pytest.approx(clean[market]), (
            f"{market}: poisoning post-{runner.CUTOFF.date()} objectives moved "
            "the menu, so the menu is not causal"
        )


def test_contributing_windows_end_before_the_evaluation_period(runner):
    """Every window that feeds the menu must be over before scoring starts."""
    _require(*UNION_REFITS, METRICS)
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    for market in MARKETS:
        frame = _refits(market)
        pre = frame[frame["fit_date"] < runner.CUTOFF]
        assert not pre.empty, f"{market}: no pre-cutoff window exists"
        latest = pd.to_datetime(pre["training_end"]).max()
        start = reported[
            (reported.market == market)
            & (reported.model == "fixed_jm")
            & (reported.delay == 1)
        ].iloc[0]["start"]
        assert latest < start, (
            f"{market}: a contributing window ends {latest.date()}, on or after "
            f"the evaluation window opens {start.date()}"
        )
        assert set(pre["observations"]) == {3000}


def test_the_cutoff_is_the_start_of_the_evaluation_period(runner):
    _require(METRICS)
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start"])
    earliest = reported[reported.model == "fixed_jm"]["start"].min()
    assert runner.CUTOFF <= earliest


def test_the_spec_prose_names_the_cutoff_the_runner_uses(runner, spec):
    """The cutoff is a runner constant; the spec only describes it in prose."""
    prose = spec["derivation"]["aggregation"]
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", prose)
    assert dates, f"the spec's aggregation prose names no cutoff date: {prose!r}"
    assert pd.Timestamp(dates[0]) == runner.CUTOFF, (
        f"the spec says the menu is aggregated before {dates[0]}, the runner "
        f"uses {runner.CUTOFF.date()}"
    )


def test_the_runner_refuses_a_menu_that_does_not_re_derive(runner, monkeypatch):
    """The gate itself, exercised. Deleting it is otherwise undetectable."""
    called: list[str] = []

    def wrong(market, ladder):
        called.append(market)
        return [float(value) + 1.0 for value in ladder]

    monkeypatch.setattr(runner, "derive", wrong)
    monkeypatch.setattr(runner, "fixed_jm_states", _must_not_run)
    monkeypatch.setattr(sys, "argv", ["run_frequency_ladder.py", "1"])
    with pytest.raises(SystemExit, match="refusing to run"):
        runner.main()
    assert called, "the runner never called derive(), so there is no gate"


def _must_not_run(*args, **kwargs):  # pragma: no cover - the point is not to
    raise AssertionError("the runner fitted states despite a failed gate")


# --------------------------------------------------------------------------
# 4. the ladder is not the only free choice: the menu is quantised
# --------------------------------------------------------------------------


def test_every_menu_value_is_a_midpoint_of_the_union_penalty_grid(spec):
    """The reachable menu is a lattice fixed by an artifact, not by the ladder.

    The union grid was assembled by jm-grid-identification-001 out of published
    and author-supplied candidate sets, so a menu value can only ever be the
    midpoint of two of ITS neighbours (or, when the median falls between two
    windows' answers, the mean of two such midpoints). Recording that here so a
    later reader cannot mistake "the ladder is the only free choice" for a
    statement about the values.
    """
    _require(*UNION_REFITS)
    for market in MARKETS:
        lam = np.array(sorted(_refits(market)["lambda"].unique()), dtype=float)
        midpoints = (lam[:-1] + lam[1:]) / 2.0
        pairs = {
            (a + b) / 2.0 for i, a in enumerate(midpoints) for b in midpoints[i:]
        }
        for value in spec["arms"]["L_full_ladder"][market]:
            assert any(abs(float(value) - p) < 1e-9 for p in pairs), (
                f"{market}: {value} is not a midpoint of the union grid, nor the "
                "median of two such midpoints"
            )


# --------------------------------------------------------------------------
# 5. the inversion: the objective column and its slope
# --------------------------------------------------------------------------


def test_the_recorded_objective_is_penalised_and_its_slope_brackets_jumps():
    """Refit one contributing window and check both halves of the claim.

    The spec asserts L(lambda) = min over state sequences of the squared-error
    loss PLUS lambda times the jump count, so the recorded value must be that
    number, and the chord slope between two penalties must lie between the jump
    counts at the two ends. Both are checked on the first German pre-1990
    window, at three consecutive union penalties.
    """
    _require(DE_FEATURES, UNION / "de/union-refits.csv")
    from jumpmodels.jump import JumpModel

    market, penalties = "de", (1.0, 1.93069772888325, 3.72759372031494)
    frame = pd.read_csv(BASE / market / "features.csv", parse_dates=["date"])
    complete = (
        frame.loc[:, ["date", *FEATURE_COLUMNS, "excess_return"]]
        .dropna(subset=[*FEATURE_COLUMNS, "excess_return"])
        .reset_index(drop=True)
    )
    refits = _refits(market)
    fit_date = refits.loc[refits.fit_date < pd.Timestamp("1990-01-01"),
                          "fit_date"].min()
    end = int(complete.index[complete["date"] == fit_date][0])
    window = complete.iloc[end - 3000 + 1 : end + 1]
    recorded, jumps = [], []
    for penalty in penalties:
        model = JumpModel(
            n_components=2, jump_penalty=penalty, random_state=0,
            max_iter=1000, tol=1e-8, n_init=10,
        ).fit(window.loc[:, list(FEATURE_COLUMNS)],
              ret_ser=window.loc[:, "excess_return"], sort_by="cumret")
        row = refits[(refits.fit_date == fit_date)
                     & np.isclose(refits["lambda"], penalty)].iloc[0]
        assert float(model.val_) == pytest.approx(row["objective"], rel=1e-9), (
            f"{market} {fit_date.date()} lambda {penalty}: the artifact records "
            f"{row['objective']}, a refit gives {float(model.val_)}"
        )
        recorded.append(float(row["objective"]))
        jumps.append(int((np.diff(np.asarray(model.labels_, dtype=int)) != 0).sum()))
    for i in range(len(penalties) - 1):
        slope = (recorded[i + 1] - recorded[i]) / (penalties[i + 1] - penalties[i])
        assert jumps[i + 1] <= slope <= jumps[i], (
            f"chord slope {slope:.3f} on ({penalties[i]}, {penalties[i+1]}) is "
            f"outside the jump counts {jumps[i]} and {jumps[i+1]}; the objective "
            "is not the concave envelope the derivation assumes"
        )
    assert jumps[0] > jumps[-1], (
        "the jump count is NOT constant across the intervals the derivation "
        f"averages over ({jumps}); the midpoint rule's stated justification is "
        "false, so the mapping from rung to penalty is an approximation"
    )


@pytest.mark.xfail(
    strict=True,
    reason="frequency-ladder-001 decision_rule 5 claims no window with a "
           "decreasing objective is dated before 1990. jp 1989-07-03 is, and "
           "it contributes lambda 75.0 to Japan's 0.25/yr rung off an "
           "impossible negative slope on (70, 80). Refitting that window with "
           "n_init=60 lowers the objective at BOTH lambda 60 and lambda 70 "
           "(by 1.12% and 1.79%) and makes every rate 0.168, so the recorded "
           "map is not converged. Remove this xfail only when the fit is "
           "repaired or the spec is corrected.",
)
def test_no_contributing_window_has_a_decreasing_objective(runner):
    _require(*UNION_REFITS)
    offenders = []
    for market in MARKETS:
        frame = _refits(market)
        pre = frame[frame["fit_date"] < runner.CUTOFF]
        for fit_date, group in pre.groupby("fit_date"):
            group = group.sort_values("lambda")
            steps = np.diff(group["objective"].to_numpy(float))
            if (steps < -1e-9).any():
                offenders.append((market, str(pd.Timestamp(fit_date).date())))
    assert not offenders, (
        "these windows feed the menu and have an objective that FALLS with the "
        f"penalty, which is impossible at the optimum: {offenders}"
    )


# --------------------------------------------------------------------------
# 6. the committed cells reproduce, and the German comparison is like for like
# --------------------------------------------------------------------------


def test_the_headline_german_cells_reproduce_through_the_library(spec):
    """Recompute arm M / Germany without the runner and match the artifact."""
    _require(DE_FEATURES, METRICS, DE_LADDER_STATES, LADDER_CELLS)
    sys.path.insert(0, str(ROOT / "src"))
    from adaptive_jump.backtest import apply_signal, performance_metrics
    from adaptive_jump.config import load_config
    from adaptive_jump.walkforward import select_monthly_candidate

    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    menu = [float(v) for v in spec["arms"]["M_no_freeze"]["de"]]
    frame = pd.read_csv(BASE / "de/features.csv", parse_dates=["date"])
    states = pd.read_csv(RUN / "states-de.csv", index_col=0, parse_dates=[0])
    states.columns = [float(c) for c in states.columns]
    columns = [
        next(c for c in states.columns if abs(c - v) < 1e-9) for v in menu
    ]
    selection = select_monthly_candidate(
        frame[["date", "equity_simple", "cash_return"]],
        states.loc[:, columns],
        config.selection_protocol,
        delay_trading_days=spec["protocol"]["delay_trading_days"],
        one_way_cost_bps=float(spec["protocol"]["one_way_cost_bps"]),
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
    )
    signal = selection.signal.rename("s").reset_index()
    signal.columns = ["date", "s"]
    merged = frame.merge(signal, on="date", how="left")
    path = apply_signal(
        merged[["date", "equity_simple", "cash_return"]],
        merged["s"],
        delay_trading_days=spec["protocol"]["delay_trading_days"],
        one_way_cost_bps=float(spec["protocol"]["one_way_cost_bps"]),
    )
    reported = pd.read_csv(BASE / "metrics.csv", parse_dates=["start", "end"])
    row = reported[
        (reported.market == "de")
        & (reported.model == "fixed_jm")
        & (reported.delay == 1)
    ].iloc[0]
    kept = path[(path.date >= row["start"]) & (path.date <= row["end"])].dropna(
        subset=["cash_return", "position", "one_way_turnover", "strategy_return"]
    )
    scored = performance_metrics(
        kept,
        periods_per_year=config.metrics_protocol.periods_per_year,
        volatility_ddof=config.metrics_protocol.volatility_ddof,
        expected_shortfall_quantile=(
            config.metrics_protocol.expected_shortfall_quantile
        ),
        turnover_scale=config.metrics_protocol.turnover_scale,
        drawdown_basis="total_wealth",
    )
    committed = pd.read_csv(RUN / "cells.csv")
    committed = committed[
        (committed.arm == "M_no_freeze") & (committed.market == "de")
    ]
    assert len(committed) == len(spec["protocol"]["cells"])
    for _, cell in committed.iterrows():
        assert float(scored[cell["cell"]]) == pytest.approx(
            float(cell["ours"]), abs=1e-12
        ), f"de arm M {cell['cell']}: recomputed {scored[cell['cell']]}, "\
           f"committed {cell['ours']}"


def test_the_german_comparison_changes_only_the_menu():
    """v10 and the ladder must differ in the menu and in nothing else.

    The two German runs share the features file byte for byte, the evaluation
    window, and the state grid geometry; only the candidate penalties differ.
    """
    _require(DE_SEALED_STATES, DE_LADDER_STATES)
    sealed = pd.read_csv(BASE / "de/jm-states.csv", index_col=0, parse_dates=[0])
    ladder = pd.read_csv(RUN / "states-de.csv", index_col=0, parse_dates=[0])
    assert ladder.index.equals(sealed.index)
    assert ladder.notna().any(axis=1).eq(sealed.notna().any(axis=1)).all(), (
        "the ladder states become available on different dates from the sealed "
        "states, so the two German runs do not see the same history"
    )
    sys.path.insert(0, str(ROOT / "src"))
    from adaptive_jump.config import load_config

    config = load_config(ROOT / "configs/baselines/legacy/research-calibrated-v10.toml")
    sealed_menu = set(config.jm_protocol_for("de").lambda_grid)
    ladder_menu = {float(c) for c in ladder.columns}
    assert sealed_menu.isdisjoint(ladder_menu), (
        "the two German menus overlap; the comparison is then not clean"
    )
