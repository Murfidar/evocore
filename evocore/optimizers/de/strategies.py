from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import Candidate
from evocore.optimizers.de.adaptive import JDEAdaptiveState
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
    direction: str = "maximize"
    strategy_state: object | None = None


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
    rng = rng_for_de_trial(
        context.seed, context.generation, context.target_slot, _core.OP_SELECTION
    )
    selected = rng.sample(choices, 3)
    return int(selected[0]), int(selected[1]), int(selected[2])


def _forced_variable_index(context: TrialContext, rng: random.Random) -> int:
    variable_indices = context.gene_space.variable_indices
    return variable_indices[rng.randrange(len(variable_indices))] if variable_indices else 0


def _best_slot(context: TrialContext) -> int:
    return max(
        range(len(context.population)),
        key=lambda slot: context.population[slot].state_comparison_score(context.direction),
    )


def _sample_slots(
    *,
    context: TrialContext,
    count: int,
    excluded: set[int],
    op_offset: int = 0,
) -> tuple[int, ...]:
    choices = [slot for slot in range(len(context.population)) if slot not in excluded]
    rng = rng_for_de_trial(
        context.seed,
        context.generation,
        context.target_slot + op_offset,
        _core.OP_SELECTION,
    )
    selected = rng.sample(choices, count)
    return tuple(int(slot) for slot in selected)


def _selected_gene(
    context: TrialContext,
    index: int,
    forced_index: int,
    mask_rng: random.Random,
) -> bool:
    return index == forced_index or mask_rng.random() < context.crossover_rate


def _bool_from_difference_pairs(
    *,
    base: Candidate,
    pairs: Sequence[tuple[Candidate, Candidate]],
    gene_index: int,
    mutation_factor: float,
    bool_rng: random.Random,
) -> bool:
    value = bool(base.genes[gene_index])
    for left, right in pairs:
        if bool(left.genes[gene_index]) != bool(
            right.genes[gene_index]
        ) and bool_rng.random() < min(1.0, mutation_factor):
            value = not value
    return value


def _mutant_value(
    *,
    context: TrialContext,
    gene_index: int,
    base: Candidate,
    pairs: Sequence[tuple[Candidate, Candidate]],
    target: Candidate | None = None,
    best: Candidate | None = None,
    bool_rng: random.Random,
) -> float | int | bool:
    gene = context.gene_space.genes[gene_index]
    if gene.kind == "bool":
        if (
            target is not None
            and best is not None
            and bool(best.genes[gene_index]) != bool(target.genes[gene_index])
        ):
            return bool(best.genes[gene_index])
        return _bool_from_difference_pairs(
            base=base,
            pairs=pairs,
            gene_index=gene_index,
            mutation_factor=context.mutation_factor,
            bool_rng=bool_rng,
        )

    mutant = float(base.genes[gene_index])
    if target is not None and best is not None:
        mutant = float(target.genes[gene_index]) + context.mutation_factor * (
            float(best.genes[gene_index]) - float(target.genes[gene_index])
        )
    for left, right in pairs:
        mutant += context.mutation_factor * (
            float(left.genes[gene_index]) - float(right.genes[gene_index])
        )
    return repair_de_gene_value(mutant, gene)


def _proposal_from_recipe(
    *,
    context: TrialContext,
    strategy: str,
    base_slot: int,
    difference_pairs: Sequence[tuple[int, int]],
    best_slot: int | None = None,
    current_to_best: bool = False,
) -> TrialProposal:
    target = _target_candidate(context)
    base = context.population[base_slot]
    best = context.population[best_slot] if best_slot is not None else None
    pairs = [
        (context.population[left], context.population[right]) for left, right in difference_pairs
    ]
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
        if not _selected_gene(context, index, forced_index, mask_rng):
            values.append(target.genes[index])
            continue
        values.append(
            _mutant_value(
                context=context,
                gene_index=index,
                base=base,
                pairs=pairs,
                target=target if current_to_best else None,
                best=best if current_to_best else None,
                bool_rng=bool_rng,
            )
        )

    context.gene_space.validate_genes(values)
    donor_slots = (base_slot, *[slot for pair in difference_pairs for slot in pair])
    metadata: dict[str, object] = {
        "strategy": strategy,
        "target_slot": context.target_slot,
        "donor_slots": tuple(donor_slots),
        "base_slot": base_slot,
        "difference_pairs": tuple(tuple(pair) for pair in difference_pairs),
    }
    if best_slot is not None:
        metadata["best_slot"] = best_slot
    return TrialProposal(genes=values, metadata=metadata)


def _rand1bin_trial(context: TrialContext) -> TrialProposal:
    a_slot, b_slot, c_slot = _rand1bin_donor_slots(context)
    return _proposal_from_recipe(
        context=context,
        strategy="rand1bin",
        base_slot=a_slot,
        difference_pairs=((b_slot, c_slot),),
    )


def _jde_rand1bin_trial(context: TrialContext) -> TrialProposal:
    if not isinstance(context.strategy_state, JDEAdaptiveState):
        raise ConfigurationError("strategy_state is required for strategy='jde-rand1bin'.")
    params = context.strategy_state.propose_parameters(
        seed=context.seed,
        generation=context.generation,
        target_slot=context.target_slot,
    )
    proposal = _rand1bin_trial(
        TrialContext(
            strategy_name="rand1bin",
            gene_space=context.gene_space,
            population=context.population,
            target_slot=context.target_slot,
            generation=context.generation,
            seed=context.seed,
            mutation_factor=params.mutation_factor,
            crossover_rate=params.crossover_rate,
            direction=context.direction,
            strategy_state=None,
        )
    )
    metadata = dict(proposal.metadata)
    metadata.update(
        {
            "strategy": "jde-rand1bin",
            "adaptive_slot": context.target_slot,
            "mutation_factor": params.mutation_factor,
            "crossover_rate": params.crossover_rate,
        }
    )
    return TrialProposal(genes=proposal.genes, metadata=metadata)


def _best1bin_trial(context: TrialContext) -> TrialProposal:
    best_slot = _best_slot(context)
    b_slot, c_slot = _sample_slots(
        context=context,
        count=2,
        excluded={context.target_slot, best_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="best1bin",
        base_slot=best_slot,
        difference_pairs=((b_slot, c_slot),),
        best_slot=best_slot,
    )


def _rand2bin_trial(context: TrialContext) -> TrialProposal:
    a_slot, b_slot, c_slot, d_slot, e_slot = _sample_slots(
        context=context,
        count=5,
        excluded={context.target_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="rand2bin",
        base_slot=a_slot,
        difference_pairs=((b_slot, c_slot), (d_slot, e_slot)),
    )


def _current_to_best1bin_trial(context: TrialContext) -> TrialProposal:
    best_slot = _best_slot(context)
    b_slot, c_slot = _sample_slots(
        context=context,
        count=2,
        excluded={context.target_slot, best_slot},
    )
    return _proposal_from_recipe(
        context=context,
        strategy="current-to-best1bin",
        base_slot=context.target_slot,
        difference_pairs=((b_slot, c_slot),),
        best_slot=best_slot,
        current_to_best=True,
    )


def trial_proposal_for_strategy(context: TrialContext) -> TrialProposal:
    """Build a trial proposal for the selected internal strategy."""
    spec = strategy_spec_for(context.strategy_name)
    validate_strategy_population_size(spec.name, len(context.population))
    if spec.name == "rand1bin":
        return _rand1bin_trial(context)
    if spec.name == "best1bin":
        return _best1bin_trial(context)
    if spec.name == "rand2bin":
        return _rand2bin_trial(context)
    if spec.name == "current-to-best1bin":
        return _current_to_best1bin_trial(context)
    if spec.name == "jde-rand1bin":
        return _jde_rand1bin_trial(context)
    raise ConfigurationError(f"Unsupported DE strategy implementation: {spec.name!r}.")


__all__ = [
    "SUPPORTED_DE_STRATEGIES",
    "DEStrategySpec",
    "TrialContext",
    "TrialProposal",
    "repair_de_gene_value",
    "rng_for_de_trial",
    "strategy_spec_for",
    "supported_strategy_names",
    "trial_proposal_for_strategy",
    "validate_strategy_population_size",
]
