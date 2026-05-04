from evocore.individual import Individual, Population


def test_individual_params_property():
    ind = Individual([10, 0.5], metadata={"params": {"fast": 10, "threshold": 0.5}})
    assert ind.params == {"fast": 10, "threshold": 0.5}


def test_population_best_ignores_none_fitness():
    pop = Population(
        [
            Individual([1.0], fitness=None),
            Individual([2.0], fitness=5.0),
            Individual([3.0], fitness=2.0),
        ]
    )
    assert pop.best()[0].genes == [2.0]


def test_population_mean_and_std():
    pop = Population([Individual([0], fitness=1.0), Individual([1], fitness=3.0)])
    assert pop.mean_fitness() == 2.0
    assert pop.std_fitness() == 1.0


def test_population_diversity_bool_as_numeric():
    pop = Population([Individual([False, 0.0]), Individual([True, 2.0])])
    div = pop.diversity()
    assert len(div) == 2
    assert div[0] > 0.0
    assert div[1] > 0.0
