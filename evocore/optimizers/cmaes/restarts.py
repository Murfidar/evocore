"""Restart planning helpers for CMA-ES optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.lifecycle import derive_child_seed


@dataclass(frozen=True)
class CMAESRestartDecision:
    """Describe one planned fresh CMA-ES restart."""

    restart_index: int
    reason: str
    population_size: int
    seed: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.restart_index) < 0:
            raise ConfigurationError("restart_index must be >= 0.")
        if not self.reason:
            raise ConfigurationError("restart reason must be non-empty.")
        if int(self.population_size) < 2:
            raise ConfigurationError("restart population_size must be at least 2.")
        object.__setattr__(self, "restart_index", int(self.restart_index))
        object.__setattr__(self, "population_size", int(self.population_size))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class CMAESRestartPolicy(Protocol):
    """Protocol for CMA-ES restart decision policies."""

    def decide(
        self,
        *,
        parent,
        restart_index: int,
        reason: str,
    ) -> CMAESRestartDecision:
        """Return a fresh restart decision for a parent optimizer."""
        raise NotImplementedError


@dataclass(frozen=True)
class FixedCMAESRestartPolicy:
    """Restart with a fixed population size."""

    population_size: int

    def __post_init__(self) -> None:
        if int(self.population_size) < 2:
            raise ConfigurationError("population_size must be at least 2.")
        object.__setattr__(self, "population_size", int(self.population_size))

    def decide(
        self,
        *,
        parent,
        restart_index: int,
        reason: str,
    ) -> CMAESRestartDecision:
        """Return a restart decision with the configured population size."""
        _reject_pending_restart(parent)
        return _restart_decision(
            parent=parent,
            restart_index=restart_index,
            reason=reason,
            population_size=self.population_size,
            policy_type="fixed",
        )


@dataclass(frozen=True)
class IPOPCMAESRestartPolicy:
    """Restart with IPOP-style population growth."""

    base_population_size: int
    growth_factor: int = 2

    def __post_init__(self) -> None:
        if int(self.base_population_size) < 2:
            raise ConfigurationError("base_population_size must be at least 2.")
        if int(self.growth_factor) < 2:
            raise ConfigurationError("growth_factor must be at least 2.")
        object.__setattr__(self, "base_population_size", int(self.base_population_size))
        object.__setattr__(self, "growth_factor", int(self.growth_factor))

    def decide(
        self,
        *,
        parent,
        restart_index: int,
        reason: str,
    ) -> CMAESRestartDecision:
        """Return a restart decision with IPOP population growth."""
        _reject_pending_restart(parent)
        population_size = self.base_population_size * (self.growth_factor ** int(restart_index))
        return _restart_decision(
            parent=parent,
            restart_index=restart_index,
            reason=reason,
            population_size=population_size,
            policy_type="ipop",
        )


def create_cmaes_restart(*, parent, decision: CMAESRestartDecision):
    """Create a fresh CMA-ES optimizer from a restart decision."""
    _reject_pending_restart(parent)
    from evocore.optimizers.cmaes.engine import CMAESOptimizer

    return CMAESOptimizer(
        parent.gene_space,
        population_size=decision.population_size,
        initial_mean=None,
        initial_sigma=parent.initial_sigma,
        max_generations=parent.max_generations,
        parallel=parent.parallel,
        n_workers=parent.n_workers,
        callbacks=list(parent.callbacks),
        seed=decision.seed,
        direction=parent.direction,
        track_diversity=parent.track_diversity,
        integer_strategy=parent.integer_strategy,
        integer_min_probability=parent.integer_min_probability,
    )


def _reject_pending_restart(parent) -> None:
    pending_batch_ids = parent.state_summary().pending_batch_ids
    if pending_batch_ids:
        raise ConfigurationError(
            f"CMAES restart requires no pending batches, got {pending_batch_ids!r}."
        )


def _restart_decision(
    *,
    parent,
    restart_index: int,
    reason: str,
    population_size: int,
    policy_type: str,
) -> CMAESRestartDecision:
    seed = derive_child_seed(
        parent_seed=parent.seed,
        candidate_hash=parent.gene_space.hash(),
        stage=f"cma_restart:{restart_index}:{reason}",
    )
    return CMAESRestartDecision(
        restart_index=restart_index,
        reason=reason,
        population_size=population_size,
        seed=seed,
        metadata={"policy_type": policy_type},
    )


__all__ = [
    "CMAESRestartDecision",
    "CMAESRestartPolicy",
    "FixedCMAESRestartPolicy",
    "IPOPCMAESRestartPolicy",
    "create_cmaes_restart",
]
