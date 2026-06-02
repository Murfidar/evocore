import pytest

from evocore import DifferentialEvolutionOptimizer, Gene, GeneSpace
from evocore.core.errors import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def test_de_constructor_sets_public_configuration() -> None:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=8,
        max_generations=12,
        mutation_factor=0.7,
        crossover_rate=0.6,
        seed=123,
        direction="minimize",
    )

    assert engine.population_size == 8
    assert engine.max_generations == 12
    assert engine.mutation_factor == pytest.approx(0.7)
    assert engine.crossover_rate == pytest.approx(0.6)
    assert engine.strategy == "rand1bin"
    assert engine.seed == 123
    assert engine.direction == "minimize"
    assert engine.state_summary().trusted_count == 0


def test_de_config_signature_is_stable_and_hash_changes_with_parameters() -> None:
    left = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.5)
    right = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.9)

    assert left.config_signature()["optimizer_type"] == "DifferentialEvolutionOptimizer"
    assert left.config_hash() != right.config_hash()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gene_space": None}, "gene_space required"),
        ({"population_size": 3}, "population_size"),
        ({"mutation_factor": -0.1}, "mutation_factor"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"parallel": "gpu"}, "parallel"),
        ({"direction": "lowest"}, "direction"),
    ],
)
def test_de_rejects_invalid_configuration(kwargs, message) -> None:
    params = {"gene_space": _space(), **kwargs}
    with pytest.raises(ConfigurationError, match=message):
        DifferentialEvolutionOptimizer(**params)
