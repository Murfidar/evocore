"""Optimization algorithm implementations."""

# config has no heavy deps — safe to import eagerly
from evocore.optimizers.config import (
    ConfigurableComponent,
    OptimizerConfig,
    RuntimeHookSignature,
    config_hash,
)
from evocore.optimizers.operators import (
    BoundsPolicy,
    CrossoverOperator,
    MutationOperator,
    SelectionOperator,
)


# Defer engine imports to break the circular dependency:
#   evocore.results → optimizers.config → optimizers.__init__
#                                          → cmaes.engine → evocore.results 💥
def __getattr__(name: str) -> object:
    if name == "CMAESOptimizer":
        from evocore.optimizers.cmaes import CMAESOptimizer

        return CMAESOptimizer
    if name == "DifferentialEvolutionOptimizer":
        from evocore.optimizers.de import DifferentialEvolutionOptimizer

        return DifferentialEvolutionOptimizer
    if name == "GeneticAlgorithmOptimizer":
        from evocore.optimizers.ga import GeneticAlgorithmOptimizer

        return GeneticAlgorithmOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BoundsPolicy",
    "CMAESOptimizer",
    "ConfigurableComponent",
    "CrossoverOperator",
    "DifferentialEvolutionOptimizer",
    "GeneticAlgorithmOptimizer",
    "MutationOperator",
    "OptimizerConfig",
    "RuntimeHookSignature",
    "SelectionOperator",
    "config_hash",
]
