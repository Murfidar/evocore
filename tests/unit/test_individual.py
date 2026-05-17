from evocore.search_space import Solution, SolutionSet


def test_individual_params_property():
    ind = Solution([10, 0.5], metadata={"params": {"fast": 10, "threshold": 0.5}})
    assert ind.params == {"fast": 10, "threshold": 0.5}


def test_population_best_ignores_none_fitness():
    pop = SolutionSet(
        [
            Solution([1.0], fitness=None),
            Solution([2.0], fitness=5.0),
            Solution([3.0], fitness=2.0),
        ]
    )
    assert pop.best()[0].genes == [2.0]


def test_population_mean_and_std():
    pop = SolutionSet([Solution([0], fitness=1.0), Solution([1], fitness=3.0)])
    assert pop.mean_fitness() == 2.0
    assert pop.std_fitness() == 1.0


def test_population_diversity_bool_as_numeric():
    pop = SolutionSet([Solution([False, 0.0]), Solution([True, 2.0])])
    div = pop.diversity()
    assert len(div) == 2
    assert div[0] > 0.0
    assert div[1] > 0.0
