"""Genetic algorithm optimizer configuration helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
    stable_object_identity,
)
from evocore.optimizers.operators import validate_operator_set
from evocore.search_space import GeneSpace


class _GAOptimizerLike(Protocol):
    population_size: int
    max_generations: int
    seed: int | None
    direction: str
    elitism: int
    max_evaluations: int | None
    track_diversity: bool
    parallel: str
    n_workers: int
    crossover: str
    crossover_prob: float
    crossover_eta: float
    crossover_alpha: float
    mutation: str
    mutation_prob: float
    mutation_individual_prob: float
    mutation_sigma: float
    mutation_sigma_schedule: str
    mutation_sigma_end: float
    selection: str
    tournament_size: int
    callbacks: Sequence[object]
    process_initializer: object | None
    process_initargs: Sequence[object]
    gene_space: GeneSpace | None
    crossover_operator: object
    mutation_operator: object
    selection_operator: object
    bounds_policy: object


def build_ga_config(optimizer: _GAOptimizerLike) -> OptimizerConfig:
    """Build the canonical GA optimizer config."""
    return OptimizerConfig(
        optimizer_type="GeneticAlgorithmOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "max_generations": optimizer.max_generations,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "elitism": optimizer.elitism,
            "max_evaluations": optimizer.max_evaluations,
            "track_diversity": optimizer.track_diversity,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
        },
        components={
            "bounds_policy": optimizer.bounds_policy.signature(),
            "crossover": optimizer.crossover_operator.signature(),
            "mutation": optimizer.mutation_operator.signature(),
            "mutation_schedule": {
                "type": optimizer.mutation_sigma_schedule,
                "parameters": {"sigma_end": optimizer.mutation_sigma_end},
            },
            "selection": optimizer.selection_operator.signature(),
        },
    )


def ga_runtime_hooks(optimizer: _GAOptimizerLike) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a GA optimizer."""
    hooks = list(callback_hook_signatures(optimizer.callbacks))
    hooks.extend(operator_runtime_hook_signatures(optimizer))
    if optimizer.process_initializer is not None:
        hooks.append(
            RuntimeHookSignature(
                hook_type="environment",
                identity=stable_object_identity(optimizer.process_initializer),
                config={"process_initargs": optimizer.process_initargs},
                reproducibility="partial",
                notes=("process_initializer is opaque.",),
            )
        )
    return tuple(hooks)


def ga_reproducibility_status(
    optimizer: _GAOptimizerLike,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a GA optimizer."""
    return reproducibility_from_hooks(ga_runtime_hooks(optimizer))


def validate_ga_compatibility(optimizer: _GAOptimizerLike) -> None:
    """Validate GA optimizer, operator, and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for GeneticAlgorithmOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    validate_operator_set(
        gene_space=optimizer.gene_space,
        crossover=optimizer.crossover_operator,
        mutation=optimizer.mutation_operator,
        selection=optimizer.selection_operator,
        bounds_policy=optimizer.bounds_policy,
    )
    if optimizer.parallel not in ("none", "thread", "process"):
        raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
    if optimizer.population_size < 2:
        raise ConfigurationError("population_size must be at least 2.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if optimizer.max_evaluations is not None and optimizer.max_evaluations <= 0:
        raise ConfigurationError("max_evaluations must be positive when provided.")
    if optimizer.elitism < 0 or optimizer.elitism >= optimizer.population_size:
        raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
    if not (0.0 <= optimizer.mutation_individual_prob <= 1.0):
        raise ConfigurationError("mutation_individual_prob must be in [0, 1].")
    if optimizer.mutation_sigma_schedule not in ("constant", "linear_decay", "cosine_decay"):
        raise ConfigurationError(
            "mutation_sigma_schedule must be 'constant', 'linear_decay', or 'cosine_decay'."
        )


def operator_runtime_hook_signatures(
    optimizer: _GAOptimizerLike,
) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for custom Python operators."""
    hooks: list[RuntimeHookSignature] = []
    for component_name, operator in (
        ("crossover", optimizer.crossover_operator),
        ("mutation", optimizer.mutation_operator),
        ("selection", optimizer.selection_operator),
    ):
        if getattr(operator, "custom", False):
            hooks.append(
                RuntimeHookSignature(
                    hook_type="environment",
                    identity=stable_object_identity(operator.implementation),
                    config={"component": component_name, "operator": operator.signature()},
                    reproducibility="partial",
                    notes=(f"custom {component_name} operator executes Python code.",),
                )
            )
    return tuple(hooks)


__all__ = [
    "build_ga_config",
    "ga_reproducibility_status",
    "ga_runtime_hooks",
    "operator_runtime_hook_signatures",
    "validate_ga_compatibility",
]
