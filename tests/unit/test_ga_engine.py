import warnings

import pytest

from evocore import (
    Callback,
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    FitnessError,
    FitnessWarning,
    GAEngine,
    GeneDef,
    GenerationInfo,
    GeneSpace,
)
from evocore.ga import MultiRunResult, RunResult
from evocore.individual import Individual, Population
from evocore.stats import Logbook


def make_result(seed: int, fitness: float) -> RunResult:
    ind = Individual([fitness], fitness=fitness, fitness_valid=True)
    return RunResult(
        best_individual=ind,
        best_fitness=fitness,
        final_population=Population([ind]),
        logbook=Logbook(),
        wall_time_seconds=0.01,
        n_evaluations=1,
        elite_history=[ind],
        diversity_history=[],
        seed=seed,
        stopped_early=False,
    )


def test_multi_run_best_n_and_summary():
    r1 = make_result(1, 1.0)
    r2 = make_result(2, 3.0)
    r3 = make_result(3, 2.0)

    multi = MultiRunResult(best=r2, all_runs=[r2, r3, r1], n_runs=3, wall_time_seconds=0.03)

    assert multi.best_n(2) == [r2, r3]
    assert multi.fitness_summary() == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}


def test_ga_engine_requires_gene_space():
    with pytest.raises(ConfigurationError, match="gene_space required"):
        GAEngine(gene_space=None)


def test_invalid_parallel_mode_rejected():
    with pytest.raises(ConfigurationError, match="parallel"):
        GAEngine(gene_space=GeneSpace.uniform(-1.0, 1.0, 2), parallel="gpu")


def test_binary_space_default_operators_work():
    engine = GAEngine(
        gene_space=GeneSpace([GeneDef("a", "bool"), GeneDef("b", "bool")]),
        crossover="one_point",
        mutation="bit_flip",
    )
    assert engine.population_size == 100


def test_large_int_without_sigma_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GAEngine(
            gene_space=GeneSpace([GeneDef("ema_slow", "int", 10, 500)]),
            population_size=10,
            generations=2,
            mutation_sigma=0.2,
        )

    warnings_of_type = [
        warning for warning in caught if issubclass(warning.category, ConfigurationWarning)
    ]
    assert len(warnings_of_type) == 1
    assert "ema_slow" in str(warnings_of_type[0].message)


def test_tuple_fitness_stores_metrics():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    ind = Individual([0.5, 0.25])

    fitnesses, nan_count = engine._evaluate_all(
        [ind],
        lambda _ind: (1.5, {"sharpe": 2.0}),
        gen=0,
    )

    assert fitnesses == [1.5]
    assert nan_count == 0
    assert ind.metadata["metrics"] == {"sharpe": 2.0}
    assert ind.fitness_valid is True


def test_nan_fitness_warns_once_and_sanitizes():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    ind = Individual([0.0, 0.0])

    with pytest.warns(FitnessWarning):
        fitnesses, nan_count = engine._evaluate_all([ind], lambda _ind: float("nan"), gen=0)

    assert fitnesses == [float("-inf")]
    assert nan_count == 1

    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        engine._evaluate_all([Individual([0.0, 0.0])], lambda _ind: float("nan"), gen=1)

    assert len(second) == 0


def test_fitness_exception_wrapped():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)

    with pytest.raises(FitnessError, match="ZeroDivisionError"):
        engine._evaluate_all([Individual([0.0, 0.0])], lambda _ind: 1 / 0, gen=0)


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_ga_run_returns_result_with_logbook_length():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 3), population_size=20, generations=5, seed=42)

    result = engine.run(sphere)

    assert result.best_fitness <= 0.0
    assert len(result.final_population) == 20
    assert len(result.logbook) == 5
    assert result.seed == 42
    assert result.n_evaluations > 0


def test_ga_run_reports_default_generation_stop_diagnostics():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=2, seed=42)

    result = engine.run(module_sphere)

    assert result.max_evaluations is None
    assert result.stop_reason == "generations"
    assert result.budget_reached is False
    assert result.stopped_early is False


def test_on_generation_end_receives_generation_info():
    received = []

    class Capture(Callback):
        def on_generation_end(self, gen, pop, info):
            received.append(info)

    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=3,
        callbacks=[Capture()],
    )

    engine.run(sphere)

    assert len(received) == 3
    assert all(isinstance(info, GenerationInfo) for info in received)


def test_track_diversity_false_and_true():
    off = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=2,
        track_diversity=False,
    )
    on = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=2,
        track_diversity=True,
    )

    assert off.run(sphere).diversity_history == []
    assert len(on.run(sphere).diversity_history) == 2


def test_elitism_caches_best_individual():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2), population_size=12, generations=3, elitism=2, seed=7
    )

    result = engine.run(sphere)

    assert any(entry.cached_count == 2 for entry in result.logbook)


def module_sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_run_multiple_sequential_returns_sorted_runs():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)

    multi = engine.run_multiple(module_sphere, n_runs=3, run_parallel=False)

    assert multi.n_runs == 3
    assert len(multi.all_runs) == 3
    assert multi.all_runs == sorted(multi.all_runs, key=lambda run: run.best_fitness, reverse=True)
    assert multi.wall_time_seconds > 0.0


def test_run_multiple_parallel_rejects_lambda():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)

    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        engine.run_multiple(lambda _ind: 1.0, n_runs=2, run_parallel=True)


def test_resume_missing_checkpoint_lists_available(tmp_path):
    (tmp_path / "checkpoint_gen_1.pkl").write_bytes(b"bad")
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=2)

    with pytest.raises(CheckpointError, match="Available checkpoints"):
        engine.resume(lambda _ind: 1.0, str(tmp_path / "checkpoint_gen_9.pkl"))


def test_ga_run_preserves_fixed_numeric_genes_in_full_genome():
    space = GeneSpace(
        [
            GeneDef("signal_mode", "int", 2, 2),
            GeneDef("threshold", "float", 0.5, 0.5),
            GeneDef("period", "int", 5, 20),
            GeneDef("x", "float", -1.0, 1.0),
        ]
    )
    engine = GAEngine(
        space,
        population_size=20,
        generations=5,
        crossover_prob=1.0,
        mutation_prob=1.0,
        mutation="uniform",
        seed=42,
    )

    result = engine.run(lambda ind: -abs(ind.params["period"] - 12) - ind.genes[3] ** 2)

    assert result.n_evaluations == 20 + (19 * 5)
    for individual in result.final_population:
        assert individual.genes[0] == 2
        assert individual.genes[1] == 0.5
        assert individual.params["signal_mode"] == 2
        assert individual.params["threshold"] == 0.5


def test_ga_max_evaluations_can_stop_during_initial_population():
    calls = []
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=5,
        max_evaluations=4,
        seed=42,
    )

    result = engine.run(lambda ind: calls.append(tuple(ind.genes)) or module_sphere(ind))

    assert len(calls) == 4
    assert result.n_evaluations == 4
    assert len(result.final_population) == 4
    assert all(ind.fitness_valid for ind in result.final_population)
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert result.stopped_early is True


def test_ga_max_evaluations_stops_exactly_after_partial_generation():
    calls = []
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        generations=5,
        elitism=2,
        max_evaluations=11,
        seed=42,
    )

    result = engine.run(lambda ind: calls.append(tuple(ind.genes)) or module_sphere(ind))

    assert len(calls) == 11
    assert result.n_evaluations == 11
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert result.stopped_early is True
    assert all(ind.fitness_valid for ind in result.final_population)
    assert len(result.final_population) <= engine.population_size


def test_ga_rejects_non_positive_max_evaluations():
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), max_evaluations=0)
