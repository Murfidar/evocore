from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=80, generations=80, seed=42)
result = engine.run(sphere)
print(result.best_fitness, result.best_individual.genes)
