import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    GeneSpace,
    OptimizationBatchResult,
    _core,
)
from evocore.core.errors import ConfigurationError


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class MinimizeSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


def _optimizer(seed: int = 42) -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=seed,
    )


def test_de_run_multiple_returns_sorted_batch_result() -> None:
    result = _optimizer().run_multiple(SphereEvaluator(), n_runs=4)

    assert isinstance(result, OptimizationBatchResult)
    assert result.n_runs == 4
    assert result.best is result.all_runs[0]
    assert result.direction == "maximize"
    assert [run.best_score for run in result.all_runs] == sorted(
        [run.best_score for run in result.all_runs],
        reverse=True,
    )


def test_de_run_multiple_uses_deterministic_child_seeds() -> None:
    first = _optimizer(seed=7).run_multiple(SphereEvaluator(), n_runs=3)
    second = _optimizer(seed=7).run_multiple(SphereEvaluator(), n_runs=3)
    expected_seeds = {
        int(_core.py_derive_seed(7, 0, run_idx, _core.OP_MULTI_RUN)) for run_idx in range(3)
    }

    assert {run.seed for run in first.all_runs} == expected_seeds
    assert [run.seed for run in first.all_runs] == [run.seed for run in second.all_runs]
    assert [run.best_score for run in first.all_runs] == pytest.approx(
        [run.best_score for run in second.all_runs]
    )


def test_de_run_multiple_supports_non_default_strategy() -> None:
    result = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        strategy="rand2bin",
        seed=7,
    ).run_multiple(SphereEvaluator(), n_runs=3)

    assert result.n_runs == 3
    assert result.best is result.all_runs[0]
    assert {
        run.reproducibility.optimizer_config["parameters"]["strategy"] for run in result.all_runs
    } == {"rand2bin"}


def test_de_run_multiple_supports_rust_backed_strategy() -> None:
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        strategy="current-to-best1bin",
        seed=42,
    )

    batch = optimizer.run_multiple(SphereEvaluator(), n_runs=2)

    assert batch.n_runs == 2
    assert len(batch.all_runs) == 2
    assert batch.best in batch.all_runs
    assert all(run.optimizer_type == "DifferentialEvolutionOptimizer" for run in batch.all_runs)


def test_de_run_multiple_minimize_uses_direction_aware_child_best_scores() -> None:
    result = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=7,
        direction="minimize",
    ).run_multiple(MinimizeSphereEvaluator(), n_runs=3)

    assert result.best.best_score == pytest.approx(min(run.best_score for run in result.all_runs))
    assert [run.best_score for run in result.all_runs] == sorted(
        [run.best_score for run in result.all_runs]
    )
    for run in result.all_runs:
        assert run.best_score == pytest.approx(
            min(solution.score for solution in run.final_solutions)
        )


def test_de_run_multiple_rejects_invalid_arguments() -> None:
    engine = _optimizer()

    with pytest.raises(ConfigurationError, match="n_runs must be positive"):
        engine.run_multiple(SphereEvaluator(), n_runs=0)
    with pytest.raises(ConfigurationError, match="aggregate must be 'best' or 'all'"):
        engine.run_multiple(SphereEvaluator(), aggregate="median")


def test_de_run_multiple_parallel_requires_picklable_evaluator() -> None:
    class NestedEvaluator:
        def evaluate(self, candidates, context):
            return SphereEvaluator().evaluate(candidates, context)

    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        _optimizer().run_multiple(NestedEvaluator(), n_runs=2, run_parallel=True)
