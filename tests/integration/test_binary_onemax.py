from evocore import GAEngine, GeneDef, GeneSpace
from tests.vnext_helpers import IndividualEvaluator


def onemax(ind):
    return sum(1 for value in ind.genes if value)


def test_binary_onemax_smoke():
    space = GeneSpace([GeneDef(f"bit_{index}", "bool") for index in range(50)])
    engine = GAEngine(
        space,
        population_size=80,
        generations=80,
        crossover="one_point",
        mutation="bit_flip",
        seed=42,
    )

    result = engine.run(IndividualEvaluator(onemax))

    assert result.best_fitness >= 40
