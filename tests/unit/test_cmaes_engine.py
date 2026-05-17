import pytest

from evocore import ConfigurationError, FitnessError, GeneDef, GeneSpace
from evocore.cmaes import CMAESEngine


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_cmaes_requires_gene_space():
    with pytest.raises(ConfigurationError, match="gene_space required"):
        CMAESEngine(gene_space=None)


def test_cmaes_rejects_bool_genes():
    with pytest.raises(ConfigurationError, match="bool"):
        CMAESEngine(GeneSpace([GeneDef("flag", "bool")]))


def test_cmaes_process_parallel_raises_at_construction():
    with pytest.raises(ConfigurationError) as exc:
        CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), parallel="process")

    message = str(exc.value)
    assert "parallel='process'" in message
    assert "not picklable" in message
    assert "parallel='thread'" in message


def test_apply_bounds_and_round_for_int_genes():
    space = GeneSpace([GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)])
    engine = CMAESEngine(space, population_size=6, max_generations=1, seed=42)

    assert engine._apply_bounds_and_round([20.8, 1.5]) == [20.0, 1.0]
    assert engine._decode_individual([10.2, 0.25]).genes == [10, 0.25]


def test_cmaes_run_returns_result():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3), population_size=10, max_generations=5, seed=42
    )

    result = engine.run(sphere)

    assert result.best_fitness <= 0.0
    assert len(result.logbook) == 5
    assert len(result.final_population) == 10
    assert result.seed == 42


def test_cma_generation_loop_result_attaches_history_and_reproducibility():
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    engine = CMAESEngine(space, population_size=6, max_generations=2, seed=42)

    result = engine.run(lambda ind: -sum(float(v) ** 2 for v in ind.genes))

    assert result.engine_type == "CMAESEngine"
    assert result.direction == "maximize"
    assert result.best_score == pytest.approx(result.best_fitness)
    assert [event.event_type for event in result.history] == [
        "generation",
        "generation",
        "run_stop",
    ]
    assert result.reproducibility is not None
    assert result.reproducibility.engine_type == "CMAESEngine"
    assert result.reproducibility.gene_space_signature == space.signature()
    assert result.reproducibility.gene_space_hash == space.hash()
    assert result.reproducibility.optimizer_config["population_size"] == 6
    assert result.reproducibility.optimizer_config["max_generations"] == 2
    assert "generations" not in result.reproducibility.optimizer_config


def test_cmaes_run_minimize_direction_returns_lowest_raw_fitness():
    engine = CMAESEngine(
        GeneSpace.uniform(-5.0, 5.0, 2),
        population_size=8,
        max_generations=1,
        seed=1,
        direction="minimize",
    )

    result = engine.run(lambda ind: sum(float(x) ** 2 for x in ind.genes))

    population_fitnesses = [individual.fitness for individual in result.final_population]
    assert result.best_fitness == pytest.approx(min(population_fitnesses))


def test_cmaes_thread_parallel_allowed():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=10,
        max_generations=2,
        parallel="thread",
    )

    assert engine.run(sphere).best_fitness <= 0.0


def test_cmaes_integer_fitness_receives_ints():
    seen_types = []
    space = GeneSpace([GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)])

    def fitness(ind):
        seen_types.append(type(ind.genes[0]))
        return -abs(ind.genes[0] - 10) - ind.genes[1] ** 2

    CMAESEngine(space, population_size=12, max_generations=3, seed=42).run(fitness)

    assert seen_types
    assert all(seen_type is int for seen_type in seen_types)


def test_cmaes_rejects_fixed_numeric_genes_until_reconstruction_is_supported():
    space = GeneSpace([GeneDef("signal_mode", "int", 2, 2), GeneDef("x", "float", -1.0, 1.0)])

    with pytest.raises(ConfigurationError) as exc:
        CMAESEngine(space)

    message = str(exc.value)
    assert "fixed numeric genes" in message
    assert "GAEngine" in message


def test_cmaes_non_finite_fitness_raises() -> None:
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), population_size=6, max_generations=1)

    with pytest.raises(FitnessError, match="finite"):
        engine.run(lambda _ind: float("nan"))


def test_cmaes_uses_max_generations_and_rejects_generations_keyword():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    assert engine.max_generations == 2
    assert not hasattr(engine, "generations")

    with pytest.raises(ConfigurationError, match="max_generations"):
        CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), generations=2)


def test_cmaes_run_reports_max_generations_and_run_stop_event():
    engine = CMAESEngine(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine.run(sphere)

    assert result.max_generations == 2
    assert result.max_evaluations is None
    assert result.stop_reason == "max_generations"
    assert [event.event_type for event in result.history][-1] == "run_stop"
    assert result.history.to_rows()[-1]["metadata"] == {
        "max_evaluations": None,
        "max_generations": 2,
        "n_evaluations": result.n_evaluations,
        "stop_reason": "max_generations",
    }
