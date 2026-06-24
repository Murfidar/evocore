import pytest

from evocore import CMAESOptimizer, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import (
    FixedCMAESRestartPolicy,
    IPOPCMAESRestartPolicy,
    create_cmaes_restart,
)


def test_fixed_restart_derives_fresh_child_seed() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=4).decide(
        parent=parent,
        restart_index=1,
        reason="stall",
    )

    assert decision.restart_index == 1
    assert decision.reason == "stall"
    assert decision.population_size == 4
    assert decision.seed != parent.seed


def test_ipop_restart_grows_population() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = IPOPCMAESRestartPolicy(base_population_size=4, growth_factor=2).decide(
        parent=parent,
        restart_index=2,
        reason="stall",
    )

    assert decision.population_size == 16


def test_restart_rejects_pending_batch() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    parent.ask()

    with pytest.raises(ConfigurationError, match="pending"):
        FixedCMAESRestartPolicy(population_size=4).decide(
            parent=parent,
            restart_index=1,
            reason="stall",
        )


def test_create_restart_returns_fresh_optimizer() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=6).decide(
        parent=parent,
        restart_index=1,
        reason="stall",
    )
    child = create_cmaes_restart(parent=parent, decision=decision)

    assert child.population_size == 6
    assert child.seed == decision.seed
    assert child.generation == 0
