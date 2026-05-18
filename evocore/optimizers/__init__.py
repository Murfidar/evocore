"""Optimization algorithm implementations."""

# config has no heavy deps — safe to import eagerly
from evocore.optimizers.config import (
    ConfigurableComponent,
    OptimizerConfig,
    RuntimeHookSignature,
    config_hash,
)


# Defer engine imports to break the circular dependency:
#   evocore.results → optimizers.config → optimizers.__init__
#                                          → cmaes.engine → evocore.results 💥
def __getattr__(name: str) -> object:
    if name == "CMAESOptimizer":
        from evocore.optimizers.cmaes import CMAESOptimizer

        return CMAESOptimizer
    if name == "GeneticAlgorithmOptimizer":
        from evocore.optimizers.ga import GeneticAlgorithmOptimizer

        return GeneticAlgorithmOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CMAESOptimizer",
    "ConfigurableComponent",
    "GeneticAlgorithmOptimizer",
    "OptimizerConfig",
    "RuntimeHookSignature",
    "config_hash",
]
