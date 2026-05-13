import warnings

import pytest

from evocore import (
    Callback,
    CheckpointError,
    ConfigurationError,
    ConfigurationWarning,
    EvaluationContext,
    EvaluationRecord,
    FitnessError,
    FitnessWarning,
    GAEngine,
    GeneDef,
    GenerationInfo,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)
from evocore.ga import MultiRunResult, RunResult
from evocore.individual import Individual, Population
from evocore.stats import Logbook


class CallableEvaluator:
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(self.fn(candidate.genes)),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


def full_policy(budget: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        full_evaluation_budget=budget,
        batch_size=batch_size,
    )


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


def test_invalid_mutation_individual_probability_rejected():
    with pytest.raises(ConfigurationError, match="mutation_individual_prob"):
        GAEngine(
            gene_space=GeneSpace.uniform(-1.0, 1.0, 2),
            mutation_individual_prob=1.5,
        )


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


def test_ga_run_returns_result_with_evaluations():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 3), population_size=20, generations=5, seed=42)

    result = engine.run(
        CallableEvaluator(lambda genes: -sum(float(value) ** 2 for value in genes)),
        policy=full_policy(20),
    )

    assert result.best_fitness <= 0.0
    assert len(result.final_population) == 20
    assert result.seed == 42
    assert result.n_evaluations == 20


def test_ga_run_accepts_uniform_crossover_for_mixed_numeric_space():
    space = GeneSpace(
        [
            GeneDef("mode", "int", 0, 4),
            GeneDef("threshold", "float", -1.0, 1.0),
        ]
    )
    engine = GAEngine(
        space,
        population_size=8,
        generations=2,
        crossover="uniform",
        mutation_prob=0.0,
        seed=42,
    )

    result = engine.run(
        CallableEvaluator(lambda genes: -abs(genes[0] - 2)),
        policy=full_policy(16, batch_size=8),
    )

    assert result.n_evaluations == 16


def test_initial_population_uses_budget_cap_when_smaller_than_population():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=5,
        max_evaluations=4,
        seed=42,
    )

    assert len(engine._initial_population()) == 4


def test_ga_run_reports_vnext_stop_diagnostics():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=2, seed=42)

    result = engine.run(
        CallableEvaluator(lambda genes: -sum(float(v) ** 2 for v in genes)),
        policy=full_policy(12, batch_size=6),
    )

    assert result.max_evaluations == 12
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert result.n_evaluations == 12


def test_ga_run_with_callback_generation_tracking():
    """Callbacks are supported on the _run_from_population path, not the vNext run() path.

    This test verifies that the old _run_from_population internal path still works with callbacks.
    """
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

    def sphere_fn(ind):
        return -sum(x * x for x in ind.genes)

    pop = engine._initial_population()
    engine._run_from_population(pop, sphere_fn, start_generation=0)

    assert len(received) == 3
    assert all(isinstance(info, GenerationInfo) for info in received)


def test_track_diversity_via_internal_run():
    """Diversity tracking is a generation-loop feature on the old _run_from_population path."""
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

    def sphere_fn(ind):
        return -sum(x * x for x in ind.genes)

    off_result = off._run_from_population(off._initial_population(), sphere_fn, start_generation=0)
    on_result = on._run_from_population(on._initial_population(), sphere_fn, start_generation=0)

    assert off_result.diversity_history == []
    assert len(on_result.diversity_history) == 2


def test_elitism_caches_best_individual_via_internal_run():
    """Elitism caching is a generation-loop feature on the old _run_from_population path."""
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2), population_size=12, generations=3, elitism=2, seed=7
    )

    def sphere_fn(ind):
        return -sum(x * x for x in ind.genes)

    result = engine._run_from_population(
        engine._initial_population(), sphere_fn, start_generation=0
    )

    assert any(entry.cached_count == 2 for entry in result.logbook)


class ModuleSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(v) ** 2 for v in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


def test_run_multiple_sequential_returns_sorted_runs():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)

    multi = engine.run_multiple(ModuleSphereEvaluator(), n_runs=3, run_parallel=False)

    assert multi.n_runs == 3
    assert len(multi.all_runs) == 3
    assert multi.all_runs == sorted(multi.all_runs, key=lambda run: run.best_fitness, reverse=True)
    assert multi.wall_time_seconds > 0.0


class ModuleMinimizeSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(float(v) ** 2 for v in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


def test_run_multiple_minimize_returns_lowest_scoring_run_as_best():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=2,
        seed=42,
        direction="minimize",
    )

    multi = engine.run_multiple(ModuleMinimizeSphereEvaluator(), n_runs=3, run_parallel=False)

    assert multi.best.best_fitness == pytest.approx(
        min(run.best_fitness for run in multi.all_runs)
    )
    assert multi.all_runs == sorted(multi.all_runs, key=lambda run: run.best_fitness)


def test_run_multiple_parallel_rejects_lambda():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)

    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        engine.run_multiple(lambda _ind: 1.0, n_runs=2, run_parallel=True)


def test_run_multiple_applies_max_evaluations_per_child_run():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        generations=5,
        seed=42,
    )

    result = engine.run_multiple(
        ModuleSphereEvaluator(),
        n_runs=3,
        run_parallel=False,
    )

    assert result.n_runs == 3
    assert all(run.n_evaluations > 0 for run in result.all_runs)
    assert all(run.stop_reason == "max_evaluations" for run in result.all_runs)
    assert all(run.budget_reached is True for run in result.all_runs)


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

    result = engine.run(
        CallableEvaluator(lambda genes: -abs(genes[2] - 12) - genes[3] ** 2),
        policy=full_policy(40, batch_size=20),
    )

    assert result.n_evaluations == 40
    for individual in result.final_population:
        assert individual.genes[0] == 2
        assert individual.genes[1] == 0.5


def test_ga_max_evaluations_stops_at_budget():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=10,
        generations=5,
        seed=42,
    )

    result = engine.run(
        CallableEvaluator(lambda genes: -sum(float(v) ** 2 for v in genes)),
        policy=full_policy(4, batch_size=4),
    )

    assert result.n_evaluations == 4
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True


def test_ga_max_evaluations_stops_exactly_at_partial_batch():
    engine = GAEngine(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=6,
        generations=5,
        seed=42,
    )

    result = engine.run(
        CallableEvaluator(lambda genes: -sum(float(v) ** 2 for v in genes)),
        policy=full_policy(11, batch_size=6),
    )

    assert result.n_evaluations == 11
    assert result.stop_reason == "max_evaluations"
    assert result.budget_reached is True
    assert all(ind.fitness_valid for ind in result.final_population)


def test_ga_rejects_non_positive_max_evaluations():
    with pytest.raises(ConfigurationError, match="max_evaluations"):
        GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), max_evaluations=0)
