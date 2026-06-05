from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from evocore.core.errors import ConfigurationError


@dataclass(frozen=True)
class DEStrategySpec:
    """Internal Differential Evolution strategy metadata."""

    name: str
    min_population_size: int
    is_adaptive: bool = False
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_state_schema: int | None = None


@dataclass(frozen=True)
class TrialProposal:
    """Strategy output before ask/tell wraps it as a Candidate."""

    genes: list[float | int | bool]
    metadata: dict[str, object]


SUPPORTED_DE_STRATEGIES: dict[str, DEStrategySpec] = {
    "rand1bin": DEStrategySpec(name="rand1bin", min_population_size=4),
    "best1bin": DEStrategySpec(name="best1bin", min_population_size=4),
    "rand2bin": DEStrategySpec(name="rand2bin", min_population_size=6),
    "current-to-best1bin": DEStrategySpec(
        name="current-to-best1bin",
        min_population_size=4,
    ),
    "jde-rand1bin": DEStrategySpec(
        name="jde-rand1bin",
        min_population_size=4,
        is_adaptive=True,
        checkpoint_state_schema=1,
    ),
}


def supported_strategy_names():
    """Return strategy names in a stable display order."""
    return tuple(SUPPORTED_DE_STRATEGIES)


def strategy_spec_for(strategy: str) -> DEStrategySpec:
    """Return the internal strategy spec or raise a user-facing config error."""
    try:
        return SUPPORTED_DE_STRATEGIES[str(strategy)]
    except KeyError as exc:
        accepted = "', '".join(supported_strategy_names())
        raise ConfigurationError(
            f"DifferentialEvolutionOptimizer strategy must be one of '{accepted}'."
        ) from exc


def validate_strategy_population_size(strategy: str, population_size: int) -> None:
    """Validate population size against the selected strategy."""
    spec = strategy_spec_for(strategy)
    if int(population_size) < spec.min_population_size:
        raise ConfigurationError(
            "population_size must be at least "
            f"{spec.min_population_size} for strategy={spec.name!r}."
        )


__all__ = [
    "SUPPORTED_DE_STRATEGIES",
    "DEStrategySpec",
    "TrialProposal",
    "strategy_spec_for",
    "supported_strategy_names",
    "validate_strategy_population_size",
]
