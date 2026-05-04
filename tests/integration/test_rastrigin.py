import math

from evocore import GAEngine, GeneSpace


def rastrigin(ind):
    n = len(ind.genes)
    value = 10 * n + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in ind.genes)
    return -value


def test_ga_rastrigin_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.12, 5.12, 6), population_size=100, generations=120, seed=7)

    result = engine.run(rastrigin)

    assert result.best_fitness > -40.0
