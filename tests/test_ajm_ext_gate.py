import pytest

from adaptive_jump.ajm_ext_gate import (
    GateError,
    SpecPair,
    evaluate_transport_gate,
    region_verdict,
)

SPECS = ("shu_v1|trailing", "shu_v1|expanding", "table3|trailing", "table3|expanding")


def _pairs(deltas, *, mdd_gap=0.0, turnover_gap=0.0):
    return tuple(
        SpecPair(
            spec=spec,
            adaptive_sharpe=0.5 + delta,
            fixed_sharpe=0.5,
            adaptive_mdd=-0.30 - mdd_gap,
            fixed_mdd=-0.30,
            adaptive_turnover=1.0 + turnover_gap,
            fixed_turnover=1.0,
        )
        for spec, delta in zip(SPECS, deltas, strict=True)
    )


def test_estimand_is_the_minimum_paired_delta_and_names_its_spec() -> None:
    verdict = region_verdict("eu", _pairs((0.10, 0.02, -0.03, 0.08)))

    assert verdict.estimand == pytest.approx(-0.03)
    assert verdict.binding_spec == "table3|trailing"
    assert not verdict.positive


def test_gate_needs_two_positive_regions_including_europe() -> None:
    pairs = {
        "Fama-French Europe": _pairs((0.05, 0.04, 0.03, 0.02)),
        "Fama-French Japan": _pairs((0.05, 0.04, 0.03, 0.01)),
        "Fama-French North America": _pairs((-0.01, 0.04, 0.03, 0.02)),
    }

    result = evaluate_transport_gate(
        pairs,
        required_passing_region="Fama-French Europe",
        minimum_positive_regions=2,
    )

    assert result.positive_regions == ("Fama-French Europe", "Fama-French Japan")
    assert result.passed


def test_two_near_copy_regions_cannot_open_the_gate_without_europe() -> None:
    pairs = {
        "Fama-French Europe": _pairs((-0.02, 0.04, 0.03, 0.02)),
        "Fama-French Japan": _pairs((0.05, 0.04, 0.03, 0.01)),
        "Fama-French North America": _pairs((0.06, 0.04, 0.03, 0.02)),
    }

    result = evaluate_transport_gate(
        pairs,
        required_passing_region="Fama-French Europe",
        minimum_positive_regions=2,
    )

    assert len(result.positive_regions) == 2
    assert not result.required_region_positive
    assert not result.passed


def test_guardrail_breach_fails_the_gate_regardless_of_sharpe() -> None:
    breaching = _pairs((0.05, 0.04, 0.03, 0.02), mdd_gap=0.05, turnover_gap=0.5)
    pairs = {
        "Fama-French Europe": _pairs((0.05, 0.04, 0.03, 0.02)),
        "Fama-French Japan": _pairs((0.05, 0.04, 0.03, 0.01)),
        "Fama-French North America": breaching,
    }

    result = evaluate_transport_gate(
        pairs,
        required_passing_region="Fama-French Europe",
        minimum_positive_regions=2,
    )

    # Regions come back sorted; North America is the breaching one.
    assert [v.guardrail_breached for v in result.regions] == [False, False, True]
    assert not result.passed


def test_one_clearing_spec_lifts_the_guardrail() -> None:
    pairs = list(_pairs((0.05, 0.04, 0.03, 0.02), mdd_gap=0.05, turnover_gap=0.5))
    cleared = pairs[2]
    pairs[2] = SpecPair(
        spec=cleared.spec,
        adaptive_sharpe=cleared.adaptive_sharpe,
        fixed_sharpe=cleared.fixed_sharpe,
        adaptive_mdd=-0.25,  # better drawdown than the fixed leg
        fixed_mdd=cleared.fixed_mdd,
        adaptive_turnover=cleared.adaptive_turnover,
        fixed_turnover=cleared.fixed_turnover,
    )

    verdict = region_verdict("na", tuple(pairs))

    assert not verdict.guardrail_breached


def test_gate_rejects_wrong_spec_count_and_positive_drawdowns() -> None:
    with pytest.raises(GateError, match="expected 4 spec pairs"):
        evaluate_transport_gate(
            {"Fama-French Europe": _pairs((0.05, 0.04, 0.03, 0.02))[:3]},
            required_passing_region="Fama-French Europe",
            minimum_positive_regions=2,
        )
    bad = list(_pairs((0.05, 0.04, 0.03, 0.02)))
    bad[0] = SpecPair(
        spec=bad[0].spec,
        adaptive_sharpe=0.5,
        fixed_sharpe=0.5,
        adaptive_mdd=0.10,
        fixed_mdd=-0.30,
        adaptive_turnover=1.0,
        fixed_turnover=1.0,
    )
    with pytest.raises(GateError, match="non-positive"):
        region_verdict("eu", tuple(bad))
