from evocore import CMAESOptimizer, GeneSpace


def rosenbrock(ind):
    xs = ind.genes
    value = sum(
        100 * (xs[index + 1] - xs[index] ** 2) ** 2 + (1 - xs[index]) ** 2
        for index in range(len(xs) - 1)
    )
    return -value


def test_cmaes_rosenbrock_smoke():
    engine = CMAESOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, max_generations=80, seed=42
    )

    result = engine.run(rosenbrock)

    assert result.best_score > -50.0
