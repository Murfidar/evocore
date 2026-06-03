from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.search_space import GeneSpace


@dataclass(frozen=True)
class DEStrategySpec:
    """Internal Differential Evolution strategy metadata."""

    name: str
    min_population_size: int
    is_adaptive: bool = False
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_state_schema: int | None = None


@dataclass(frozen=True)
class TrialContext:
    """Inputs needed to build one DE trial vector."""

    strategy_name: str
    gene_space: GeneSpace
    population: Sequence[Candidate]
    target_slot: int
    generation: int
    seed: int
    mutation_factor: float
    crossover_rate: float
    strategy_state: object | None = None


@dataclass(frozen=True)
class TrialProposal:
    """Strategy output before ask/tell wraps it as a Candidate."""

    genes: list[float | int | bool]
    metadata: dict[str, object]


SUPPORTED_DE_STRATEGIES: dict[str, DEStrategySpec] = {
    "rand1bin": DEStrategySpec(name="rand1bin", min_population_size=4),
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


def rng_for_de_trial(seed: int, generation: int, target_slot: int, op: int) -> random.Random:
    """Return deterministic per-trial RNG matching the original DE implementation."""

    derived = int(_core.py_derive_seed(int(seed), int(generation), int(target_slot), int(op)))
    return random.Random(derived)  # noqa: S311 - deterministic optimizer sampling.


def repair_de_gene_value(value: float, gene) -> float | int | bool:
    """Repair one DE gene value according to the mixed GeneSpace contract."""

    if gene.kind == "bool":
        return bool(float(value) >= 0.5)
    low = float(gene.low)
    high = float(gene.high)
    clamped = min(max(float(value), low), high)
    if gene.kind == "int":
        return int(round(clamped))
    return float(clamped)


def _target_candidate(context: TrialContext) -> Candidate:
    return context.population[context.target_slot]


def _rand1bin_donor_slots(context: TrialContext) -> tuple[int, int, int]:
    choices = [slot for slot in range(len(context.population)) if slot != context.target_slot]
    rng = rng_for_de_trial(context.seed, context.generation, context.target_slot, _core.OP_SELECTION)
    selected = rng.sample(choices, 3)
    return int(selected[0]), int(selected[1]), int(selected[2])


def _forced_variable_index(context: TrialContext, rng: random.Random) -> int:
    variable_indices = context.gene_space.variable_indices
    return variable_indices[rng.randrange(len(variable_indices))] if variable_indices else 0


def _rand1bin_trial(context: TrialContext) -> TrialProposal:
    if len(context.population) < strategy_spec_for("rand1bin").min_population_size:
        validate_strategy_population_size("rand1bin", len(context.population))

    target = _target_candidate(context)
    a_slot, b_slot, c_slot = _rand1bin_donor_slots(context)
    donor_a = context.population[a_slot]
    donor_b = context.population[b_slot]
    donor_c = context.population[c_slot]
    mask_rng = rng_for_de_trial(
        context.seed, context.generation, context.target_slot, _core.OP_CROSSOVER
    )
    bool_rng = rng_for_de_trial(
        context.seed, context.generation, context.target_slot, _core.OP_MUTATION
    )
    forced_index = _forced_variable_index(context, mask_rng)
    values: list[float | int | bool] = []

    for index, gene in enumerate(context.gene_space.genes):
        if gene.is_fixed:
            values.append(repair_de_gene_value(float(gene.low), gene))
            continue
        selected = index == forced_index or mask_rng.random() < context.crossover_rate
        if not selected:
            values.append(target.genes[index])
            continue
        if gene.kind == "bool":
            trial_bool = bool(donor_a.genes[index])
            if bool(donor_b.genes[index]) != bool(
                donor_c.genes[index]
            ) and bool_rng.random() < min(1.0, context.mutation_factor):
                trial_bool = not trial_bool
            values.append(trial_bool)
            continue
        mutant = float(donor_a.genes[index]) + context.mutation_factor * (
            float(donor_b.genes[index]) - float(donor_c.genes[index])
        )
        values.append(repair_de_gene_value(mutant, gene))

    context.gene_space.validate_genes(values)
    return TrialProposal(
        genes=values,
        metadata={
            "strategy": "rand1bin",
            "target_slot": context.target_slot,
            "donor_slots": (a_slot, b_slot, c_slot),
        },
    )


def trial_proposal_for_strategy(context: TrialContext) -> TrialProposal:
    """Build a trial proposal for the selected internal strategy."""

    spec = strategy_spec_for(context.strategy_name)
    validate_strategy_population_size(spec.name, len(context.population))
    if spec.name == "rand1bin":
        return _rand1bin_trial(context)
    raise ConfigurationError(f"Unsupported DE strategy implementation: {spec.name!r}.")


__all__ = [
    "DEStrategySpec",
    "SUPPORTED_DE_STRATEGIES",
    "TrialContext",
    "TrialProposal",
    "repair_de_gene_value",
    "rng_for_de_trial",
    "strategy_spec_for",
    "supported_strategy_names",
    "trial_proposal_for_strategy",
    "validate_strategy_population_size",
]
