from evocore import GAEngine, GeneDef, GeneSpace


def objective(ind):
    params = ind.params
    return -abs(params["period"] - 21) - abs(params["threshold"] - 0.35)


space = GeneSpace(
    [
        GeneDef("period", "int", 5, 50, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
    ]
)
result = GAEngine(space, population_size=60, generations=50, seed=7).run(objective)
print(result.best_fitness, result.best_individual.params)
