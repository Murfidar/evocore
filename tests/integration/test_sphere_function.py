from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_ga_sphere_converges_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=80, generations=80, seed=42)

    result = engine.run(sphere)

    assert result.best_fitness > -2.0
