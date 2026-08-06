"""Transport-gate arithmetic for AJM-EXT-001.

Pure functions from per-(region, spec) metrics to the frozen gate verdict.
Nothing here reads disk or runs models, so a separate verifier can re-derive
the verdict from the sealed metric rows alone.

Guardrail interpretation, disclosed: the contract says "no region may have
both worse MDD and higher turnover than every paired fixed specification".
This module reads that pairwise — a region breaches only if, for EVERY spec,
its paired challenger has strictly worse (deeper) maximum drawdown than that
spec's fixed leg AND strictly higher turnover than that spec's fixed leg.
A single spec where either side holds clears the region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class GateError(ValueError):
    """Raised when the metric rows cannot support the frozen verdict."""


@dataclass(frozen=True)
class SpecPair:
    """One paired specification's headline cells for one region."""

    spec: str
    adaptive_sharpe: float
    fixed_sharpe: float
    adaptive_mdd: float
    fixed_mdd: float
    adaptive_turnover: float
    fixed_turnover: float


@dataclass(frozen=True)
class RegionVerdict:
    region: str
    estimand: float
    binding_spec: str
    positive: bool
    guardrail_breached: bool


@dataclass(frozen=True)
class GateResult:
    regions: tuple[RegionVerdict, ...]
    positive_regions: tuple[str, ...]
    required_region_positive: bool
    passed: bool


def region_verdict(region: str, pairs: tuple[SpecPair, ...]) -> RegionVerdict:
    if not pairs:
        raise GateError(f"{region}: no spec pairs")
    for pair in pairs:
        cells = (
            pair.adaptive_sharpe,
            pair.fixed_sharpe,
            pair.adaptive_mdd,
            pair.fixed_mdd,
            pair.adaptive_turnover,
            pair.fixed_turnover,
        )
        if not all(math.isfinite(cell) for cell in cells):
            raise GateError(f"{region}/{pair.spec}: non-finite metric cell")
        if pair.adaptive_mdd > 0 or pair.fixed_mdd > 0:
            raise GateError(f"{region}/{pair.spec}: drawdowns must be non-positive")
    deltas = {pair.spec: pair.adaptive_sharpe - pair.fixed_sharpe for pair in pairs}
    binding_spec = min(deltas, key=lambda spec: deltas[spec])
    estimand = deltas[binding_spec]
    breached = all(
        abs(pair.adaptive_mdd) > abs(pair.fixed_mdd)
        and pair.adaptive_turnover > pair.fixed_turnover
        for pair in pairs
    )
    return RegionVerdict(
        region=region,
        estimand=estimand,
        binding_spec=binding_spec,
        positive=estimand > 0,
        guardrail_breached=breached,
    )


def evaluate_transport_gate(
    region_pairs: dict[str, tuple[SpecPair, ...]],
    *,
    required_passing_region: str,
    minimum_positive_regions: int,
    expected_specs: int = 4,
) -> GateResult:
    """Apply the frozen transport rule to all regions at once."""
    if required_passing_region not in region_pairs:
        raise GateError(f"required region is missing: {required_passing_region}")
    verdicts = []
    for region, pairs in sorted(region_pairs.items()):
        if len(pairs) != expected_specs:
            raise GateError(
                f"{region}: expected {expected_specs} spec pairs, got {len(pairs)}"
            )
        if len({pair.spec for pair in pairs}) != len(pairs):
            raise GateError(f"{region}: duplicate spec names")
        verdicts.append(region_verdict(region, pairs))
    positive = tuple(v.region for v in verdicts if v.positive)
    required_positive = required_passing_region in positive
    breach = any(v.guardrail_breached for v in verdicts)
    passed = (
        len(positive) >= minimum_positive_regions and required_positive and not breach
    )
    return GateResult(
        regions=tuple(verdicts),
        positive_regions=positive,
        required_region_positive=required_positive,
        passed=passed,
    )
