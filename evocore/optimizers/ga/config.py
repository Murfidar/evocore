"""Genetic algorithm optimizer configuration helpers."""

from __future__ import annotations

from typing import Any

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
    stable_object_identity,
)
from evocore.search_space import OperatorCodec


def build_ga_config(optimizer: Any) -> OptimizerConfig:
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
            "crossover": {
                "type": optimizer.crossover,
                "parameters": {
                    "probability": optimizer.crossover_prob,
                    "eta": optimizer.crossover_eta,
                    "alpha": optimizer.crossover_alpha,
                },
            },
            "mutation": {
                "type": optimizer.mutation,
                "parameters": {
                    "probability": optimizer.mutation_prob,
                    "individual_probability": optimizer.mutation_individual_prob,
                    "sigma": optimizer.mutation_sigma,
                },
            },
            "mutation_schedule": {
                "type": optimizer.mutation_sigma_schedule,
                "parameters": {"sigma_end": optimizer.mutation_sigma_end},
            },
            "selection": {
                "type": optimizer.selection,
                "parameters": {"tournament_size": optimizer.tournament_size},
            },
        },
    )


def ga_runtime_hooks(optimizer: Any) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a GA optimizer."""
    hooks = list(callback_hook_signatures(optimizer.callbacks))
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
    optimizer: Any,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a GA optimizer."""
    return reproducibility_from_hooks(ga_runtime_hooks(optimizer))


def validate_ga_compatibility(optimizer: Any) -> None:
    """Validate GA optimizer, operator, and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for GeneticAlgorithmOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    OperatorCodec(optimizer.gene_space, optimizer.crossover, optimizer.mutation)
    if optimizer.parallel not in ("none", "thread", "process"):
        raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
    if optimizer.selection not in ("tournament", "roulette", "rank"):
        raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
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


__all__ = [
    "build_ga_config",
    "ga_reproducibility_status",
    "ga_runtime_hooks",
    "validate_ga_compatibility",
]
