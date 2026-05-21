from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer
from tests.vnext_helpers import IndividualEvaluator, full_policy


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


def mixed_bool_target(ind):
    params = ind.params
    enabled_bonus = 5.0 if params["enabled"] else 0.0
    return enabled_bonus - ((params["period"] - 20) ** 2) - ((params["threshold"] - 0.3) ** 2)


def test_mixed_bool_gene_space_runs_with_default_operators():
    space = GeneSpace(
        [
            Gene("threshold", "float", 0.0, 1.0),
            Gene("period", "int", 2, 50),
            Gene("enabled", "bool"),
        ]
    )
    engine = GeneticAlgorithmOptimizer(
        space,
        population_size=12,
        max_generations=4,
        seed=42,
    )

    result = engine.run(
        IndividualEvaluator(mixed_bool_target),
        policy=full_policy(48, batch_size=12),
    )

    assert result.n_evaluations == 48
    assert type(result.best_solution.params["enabled"]) is bool
    assert all(type(solution.values[2]) is bool for solution in result.final_solutions)
