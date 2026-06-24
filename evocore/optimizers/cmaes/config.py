"""CMA-ES optimizer configuration helpers."""

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
)
from evocore.search_space import GeneSpace


class _CMAESOptimizerLike(Protocol):
    population_size: int
    initial_mean: Sequence[float] | None
    initial_sigma: float
    max_generations: int
    seed: int | None
    direction: str
    parallel: str
    n_workers: int
    track_diversity: bool
    integer_strategy: str
    integer_min_probability: float
    callbacks: Sequence[object]
    gene_space: GeneSpace | None


def build_cmaes_config(optimizer: _CMAESOptimizerLike) -> OptimizerConfig:
    """Build the canonical CMA-ES optimizer config."""
    distribution_parameters = {"initial_sigma": optimizer.initial_sigma}
    if optimizer.integer_strategy != "round" or float(optimizer.integer_min_probability) != 0.02:
        distribution_parameters["integer_strategy"] = optimizer.integer_strategy
        distribution_parameters["integer_min_probability"] = optimizer.integer_min_probability
    return OptimizerConfig(
        optimizer_type="CMAESOptimizer",
        parameters={
            "population_size": optimizer.population_size,
            "initial_mean": optimizer.initial_mean,
            "initial_sigma": optimizer.initial_sigma,
            "max_generations": optimizer.max_generations,
            "seed": optimizer.seed,
            "direction": optimizer.direction,
            "parallel": optimizer.parallel,
            "n_workers": optimizer.n_workers,
            "track_diversity": optimizer.track_diversity,
        },
        components={
            "distribution": {
                "type": "cma_es",
                "parameters": distribution_parameters,
            }
        },
    )


def cmaes_runtime_hooks(optimizer: _CMAESOptimizerLike) -> tuple[RuntimeHookSignature, ...]:
    """Return runtime hook signatures for a CMA-ES optimizer."""
    return callback_hook_signatures(optimizer.callbacks)


def cmaes_reproducibility_status(
    optimizer: _CMAESOptimizerLike,
) -> tuple[ReproducibilityStatus, tuple[str, ...]]:
    """Return reproducibility status and notes for a CMA-ES optimizer."""
    return reproducibility_from_hooks(cmaes_runtime_hooks(optimizer))


def _validate_integer_strategy_config(optimizer: _CMAESOptimizerLike) -> None:
    if optimizer.integer_strategy not in ("round", "margin"):
        raise ConfigurationError("integer_strategy must be 'round' or 'margin'.")
    if not (0.0 < float(optimizer.integer_min_probability) < 1.0):
        raise ConfigurationError("integer_min_probability must be in (0, 1).")
    if optimizer.integer_strategy != "margin":
        return
    for gene in optimizer.gene_space.genes:
        if gene.kind != "int":
            continue
        range_size = int(gene.high) - int(gene.low) + 1
        if float(optimizer.integer_min_probability) * range_size >= 1.0:
            raise ConfigurationError(
                "integer_min_probability is too large for integer gene range."
            )


def validate_cmaes_compatibility(optimizer: _CMAESOptimizerLike) -> None:
    """Validate CMA-ES optimizer and gene-space compatibility."""
    if optimizer.gene_space is None:
        raise ConfigurationError(
            "gene_space required for CMAESOptimizer. Pass GeneSpace.uniform(-5.0, 5.0, length)."
        )
    if "bool" in optimizer.gene_space.kinds:
        raise ConfigurationError(
            "CMAESOptimizer does not support bool genes; use float/int genes only."
        )
    if optimizer.gene_space.fixed_count:
        raise ConfigurationError(
            "CMAESOptimizer does not support fixed numeric genes yet. "
            "Use GeneticAlgorithmOptimizer for full-genome fixed genes, or remove fixed genes from the CMA-ES GeneSpace."
        )
    if optimizer.parallel == "process":
        raise ConfigurationError(
            "CMAESOptimizer does not support parallel='process'.\n"
            "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable.\n"
            "  Fix: use parallel='thread' if your objective function releases the GIL, or parallel='none'.\n"
            "  Note: parallel='process' is supported by GeneticAlgorithmOptimizer, not CMAESOptimizer."
        )
    if optimizer.parallel not in ("none", "thread"):
        raise ConfigurationError("CMAESOptimizer parallel must be 'none' or 'thread'.")
    if optimizer.population_size < 2:
        raise ConfigurationError("population_size must be at least 2.")
    if optimizer.max_generations < 0:
        raise ConfigurationError("max_generations must be >= 0.")
    if not (optimizer.initial_sigma > 0.0):
        raise ConfigurationError("initial_sigma must be > 0.")
    if (
        optimizer.initial_mean is not None
        and len(optimizer.initial_mean) != optimizer.gene_space.length
    ):
        raise ConfigurationError("initial_mean length must match gene_space.length.")
    _validate_integer_strategy_config(optimizer)


__all__ = [
    "build_cmaes_config",
    "cmaes_reproducibility_status",
    "cmaes_runtime_hooks",
    "validate_cmaes_compatibility",
]
