from evocore import GAEngine, GeneDef, GeneSpace


def mixed_target(ind):
    params = ind.params
    return -((params["period"] - 20) ** 2) - ((params["threshold"] - 0.3) ** 2)


def test_mixed_gene_space_keeps_ints_typed():
    space = GeneSpace(
        [
            GeneDef("period", "int", 5, 50, sigma=0.05),
            GeneDef("threshold", "float", 0.0, 1.0),
        ]
    )
    engine = GAEngine(space, population_size=60, generations=50, seed=42)

    result = engine.run(mixed_target)

    assert isinstance(result.best_individual.genes[0], int)
    assert result.best_fitness > -10.0
