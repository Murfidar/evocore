from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def onemax(ind):
    return sum(1 for value in ind.genes if value)


def test_binary_onemax_smoke():
    space = GeneSpace([Gene(f"bit_{index}", "bool") for index in range(50)])
    engine = GeneticAlgorithmOptimizer(
        space,
        population_size=80,
        max_generations=80,
        crossover="one_point",
        mutation="bit_flip",
        seed=42,
    )

    result = engine.run(IndividualEvaluator(onemax))

    assert result.best_score >= 40
