from evocore import CMAESEngine, GeneSpace


def rosenbrock(ind):
    xs = ind.genes
    return -sum(
        100 * (xs[index + 1] - xs[index] ** 2) ** 2 + (1 - xs[index]) ** 2
        for index in range(len(xs) - 1)
    )


engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, generations=80, seed=42)
result = engine.run(rosenbrock)
print(result.best_fitness, result.best_individual.genes)
