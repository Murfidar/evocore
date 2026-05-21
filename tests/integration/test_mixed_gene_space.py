from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator


def mixed_target(ind):
    params = ind.params
    return -((params["period"] - 20) ** 2) - ((params["threshold"] - 0.3) ** 2)


def test_mixed_gene_space_keeps_ints_typed():
    space = GeneSpace(
        [
            Gene("period", "int", 5, 50, sigma=0.05),
            Gene("threshold", "float", 0.0, 1.0),
        ]
    )
    engine = GeneticAlgorithmOptimizer(space, population_size=60, max_generations=50, seed=42)

    result = engine.run(IndividualEvaluator(mixed_target))

    assert isinstance(result.best_solution.values[0], int)
    assert result.best_score > -10.0
