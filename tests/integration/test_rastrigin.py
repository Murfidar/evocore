import math

from evocore import GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def rastrigin(ind):
    n = len(ind.genes)
    value = 10 * n + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in ind.genes)
    return -value


def test_ga_rastrigin_smoke():
    engine = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.12, 5.12, 6), population_size=100, max_generations=120, seed=7
    )

    result = engine.run(IndividualEvaluator(rastrigin))

    assert result.best_score > -40.0
