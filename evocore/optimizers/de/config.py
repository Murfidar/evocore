"""Differential Evolution optimizer configuration helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from evocore.core.errors import ConfigurationError
from evocore.optimizers.config import (
    OptimizerConfig,
    ReproducibilityStatus,
    RuntimeHookSignature,
    callback_hook_signatures,
    reproducibility_from_hooks,
)
from evocore.search_space import GeneSpace


class _DEOptimizerLike(Protocol):
    population_size: int
    max_generations: int
    mutation_factor: float
    crossover_rate: float
    strategy: str
    seed: int
    direction: str
    parallel: str
    n_workers: int | None
    track_diversity: bool
    callbacks: Sequence[object]
    max_evaluations: int | None
    gene_space: GeneSpace | None


def build_de_config(optimizer: _DEOptimizerLike) -> OptimizerConfig:
    """Build the canonical Differential Evolution optimizer config."""
    return OptimizerConfig(
        optimizer_type="DifferentialEvolutionOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "max_generations": optimizer.max_generations,
            "mutation_factor": optimizer.mutation_factor,
            "crossover_rate": optimizer.crossover_rate,
            "strategy": optimizer.strategy,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
            "max_evaluations": optimizer.max_evaluations,
            "track_diversity": optimizer.track_diversity,
        },
        components={
            "strategy": {
                "type": optimizer.strategy,
                "parameters": {
                    "mutation_factor": optimizer.mutation_factor,
                    "crossover_rate": optimizer.crossover_rate,
                },
            }
        },
    )


def de_runtime_hooks(optimizer: _DEOptimizerLike) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a DE optimizer."""
    return callback_hook_signatures(optimizer.callbacks)


def de_reproducibility_status(
    optimizer: _DEOptimizerLike,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a DE optimizer."""
    return reproducibility_from_hooks(de_runtime_hooks(optimizer))


def validate_de_compatibility(optimizer: _DEOptimizerLike) -> None:
    """Validate DE optimizer and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for DifferentialEvolutionOptimizer. "
            "Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    if optimizer.strategy != "rand1bin":
        raise ConfigurationError("DifferentialEvolutionOptimizer strategy must be 'rand1bin'.")
    if optimizer.population_size < 4:
        raise ConfigurationError("population_size must be at least 4 for strategy='rand1bin'.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if not math.isfinite(float(optimizer.mutation_factor)) or optimizer.mutation_factor < 0.0:
        raise ConfigurationError("mutation_factor must be finite and >= 0.")
    if not 0.0 <= float(optimizer.crossover_rate) <= 1.0:
        raise ConfigurationError("crossover_rate must be in [0, 1].")
    if optimizer.max_evaluations is not None and optimizer.max_evaluations <= 0:
        raise ConfigurationError("max_evaluations must be positive when provided.")
    if optimizer.parallel not in ("none", "thread", "process"):
        raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
    if optimizer.direction not in ("maximize", "minimize"):
        raise ConfigurationError("direction must be 'maximize' or 'minimize'.")


__all__ = [
    "build_de_config",
    "de_reproducibility_status",
    "de_runtime_hooks",
    "validate_de_compatibility",
]
