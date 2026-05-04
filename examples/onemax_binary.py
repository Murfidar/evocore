from evocore import GAEngine, GeneDef, GeneSpace


def onemax(ind):
    return sum(ind.genes)


space = GeneSpace([GeneDef(f"bit_{index}", "bool") for index in range(50)])
engine = GAEngine(
    space,
    population_size=80,
    generations=80,
    crossover="one_point",
    mutation="bit_flip",
    seed=42,
)
result = engine.run(onemax)
print(result.best_fitness, result.best_individual.genes)
