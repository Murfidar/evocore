import pytest

from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    Gene,
    GeneSpace,
)
from evocore.callbacks import Callback
from evocore.core.errors import ConfigurationError, FitnessError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 1.5, 1.5),
        ]
    )


def test_de_constructor_sets_public_configuration() -> None:
    engine = DifferentialEvolutionOptimizer(
        _space(),
        population_size=8,
        max_generations=12,
        mutation_factor=0.7,
        crossover_rate=0.6,
        seed=123,
        direction="minimize",
    )

    assert engine.population_size == 8
    assert engine.max_generations == 12
    assert engine.mutation_factor == pytest.approx(0.7)
    assert engine.crossover_rate == pytest.approx(0.6)
    assert engine.strategy == "rand1bin"
    assert engine.seed == 123
    assert engine.direction == "minimize"
    assert engine.state_summary().trusted_count == 0


def test_de_config_signature_is_stable_and_hash_changes_with_parameters() -> None:
    left = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.5)
    right = DifferentialEvolutionOptimizer(_space(), population_size=8, mutation_factor=0.9)

    assert left.config_signature()["optimizer_type"] == "DifferentialEvolutionOptimizer"
    assert left.config_hash() != right.config_hash()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gene_space": None}, "gene_space required"),
        ({"population_size": 3}, "population_size"),
        ({"mutation_factor": -0.1}, "mutation_factor"),
        ({"crossover_rate": 1.1}, "crossover_rate"),
        ({"parallel": "gpu"}, "parallel"),
        ({"direction": "lowest"}, "direction"),
    ],
)
def test_de_rejects_invalid_configuration(kwargs, message) -> None:
    params = {"gene_space": _space(), **kwargs}
    with pytest.raises(ConfigurationError, match=message):
        DifferentialEvolutionOptimizer(**params)


def test_de_rejects_unknown_strategy_with_supported_names() -> None:
    with pytest.raises(ConfigurationError, match="strategy must be one of 'rand1bin'"):
        DifferentialEvolutionOptimizer(_space(), population_size=8, strategy="best1bin")


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(
                    float(value) ** 2 for value in candidate.genes if type(value) is not bool
                ),
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


def _two_stage_policy(max_evaluations: int = 12, batch_size: int = 6) -> BudgetPolicy:
    return BudgetPolicy(
        stages=[
            EvaluationStage(
                "cheap",
                budget=0.10,
                promote_fraction=0.50,
                confidence="partial",
            ),
            EvaluationStage(
                "full",
                budget=1.00,
                promote_fraction=1.00,
                confidence="trusted_full",
            ),
        ],
        max_evaluations=max_evaluations,
        batch_size=batch_size,
        exploration_fraction=0.0,
        audit_fraction=0.0,
    )


class TwoStageSphereEvaluator:
    def __init__(self) -> None:
        self.stage_calls: list[tuple[str, int]] = []

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        stage_name = context.stage.name
        self.stage_calls.append((stage_name, len(candidates)))
        scale = 0.25 if stage_name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale
                * sum(float(value) ** 2 for value in candidate.genes if type(value) is not bool),
                confidence=context.stage.confidence,
                stage=stage_name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CachedFinalEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        confidence = "cached" if context.stage.name == "full" else context.stage.confidence
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(
                    float(value) ** 2 for value in candidate.genes if type(value) is not bool
                ),
                confidence=confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CachedCheapEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        confidence = "cached" if context.stage.name == "cheap" else context.stage.confidence
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(
                    float(value) ** 2 for value in candidate.genes if type(value) is not bool
                ),
                confidence=confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class HalfPromotionEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        if context.stage.name == "cheap":
            return [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(index),
                    confidence="partial",
                    stage="cheap",
                    cost=context.stage.budget,
                )
                for index, candidate in enumerate(candidates)
            ]
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=100.0,
                confidence="trusted_full",
                stage="full",
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


class CountingCallback(Callback):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.completed = False

    def on_generation_start(self, generation, population) -> None:
        self.starts += 1

    def on_generation_end(self, generation, population, info) -> None:
        self.ends += 1

    def on_run_end(self, result) -> None:
        self.completed = True


def test_de_run_returns_optimization_result_with_events_and_generations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.optimizer_type == "DifferentialEvolutionOptimizer"
    assert result.best_candidate_id is not None
    assert result.n_evaluations >= 6
    assert len(result.generations) == 2
    assert len(result.events) > 0
    assert result.reproducibility is not None
    assert result.reproducibility.optimizer_type == "DifferentialEvolutionOptimizer"


def test_de_run_is_reproducible_for_same_seed_and_config() -> None:
    left = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        seed=42,
    )
    right = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        seed=42,
    )

    left_result = left.run(SphereEvaluator())
    right_result = right.run(SphereEvaluator())

    assert left_result.best_score == pytest.approx(right_result.best_score)
    assert left_result.best_candidate_id == right_result.best_candidate_id
    assert [record.best_score for record in left_result.generations] == [
        record.best_score for record in right_result.generations
    ]
    assert [tuple(event.genes or ()) for event in left_result.events] == [
        tuple(event.genes or ()) for event in right_result.events
    ]


def test_de_run_invokes_callbacks() -> None:
    callback = CountingCallback()
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
        callbacks=[callback],
    )

    engine.run(SphereEvaluator())

    assert callback.starts == 2
    assert callback.ends == 2
    assert callback.completed is True


def test_de_run_minimize_reports_lowest_scoring_solution_as_best() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        seed=42,
        direction="minimize",
    )

    result = engine.run(MinimizeSphereEvaluator())
    final_scores = [solution.score for solution in result.final_solutions]

    assert result.best_score == pytest.approx(min(final_scores))
    assert result.best_solution.metadata["candidate_id"] == result.best_candidate_id
    assert result.generations[-1].best_score == pytest.approx(min(final_scores))


def test_de_run_honors_max_evaluations() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        max_evaluations=8,
        seed=42,
    )

    result = engine.run(SphereEvaluator())

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations <= 12


def test_de_run_accepts_explicit_single_full_policy() -> None:
    policy = BudgetPolicy.single_full(max_evaluations=8, batch_size=4)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        seed=42,
    )

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations == 8
    assert result.max_evaluations == 8
    assert result.telemetry.candidates_full_evaluated == 8


def test_de_run_prefers_explicit_policy_over_constructor_max_evaluations() -> None:
    policy = BudgetPolicy.single_full(max_evaluations=9, batch_size=3)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=10,
        max_evaluations=4,
        seed=42,
    )

    result = engine.run(SphereEvaluator(), policy=policy)

    assert result.max_evaluations == 9
    assert result.n_evaluations == 9


def test_de_run_rejects_non_policy_argument() -> None:
    engine = DifferentialEvolutionOptimizer(GeneSpace.uniform(-2.0, 2.0, 3), seed=42)

    with pytest.raises(ConfigurationError, match="policy must be a BudgetPolicy"):
        engine.run(SphereEvaluator(), policy=object())


def test_de_run_two_stage_policy_screens_and_closes_batches() -> None:
    evaluator = TwoStageSphereEvaluator()
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(evaluator, policy=_two_stage_policy(max_evaluations=9, batch_size=6))

    assert result.stop_reason == "max_evaluations"
    assert result.n_evaluations == 9
    assert result.telemetry.candidates_partial_evaluated > 0
    assert result.telemetry.candidates_full_evaluated == 9
    assert engine.state_summary().pending_batch_ids == ()
    assert any(stage_name == "cheap" for stage_name, _ in evaluator.stage_calls)
    assert any(stage_name == "full" for stage_name, _ in evaluator.stage_calls)


def test_de_run_cached_final_records_update_state_without_spending_fresh_budget() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(
        CachedFinalEvaluator(),
        policy=_two_stage_policy(max_evaluations=6, batch_size=6),
    )

    assert result.n_evaluations == 0
    assert result.telemetry.candidates_cached > 0
    assert result.best_candidate_id is not None
    assert len(result.final_solutions) == 6


def test_de_run_rejects_state_eligible_non_final_policy_records() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    with pytest.raises(FitnessError, match="state-eligible records before final stage"):
        engine.run(
            CachedCheapEvaluator(),
            policy=_two_stage_policy(max_evaluations=6, batch_size=6),
        )


class MissingRecordEvaluator:
    def evaluate(self, candidates, context):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=1.0,
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in list(candidates)[:-1]
        ]


def test_de_run_rejects_missing_evaluator_records() -> None:
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    with pytest.raises(FitnessError, match="missing evaluation records"):
        engine.run(MissingRecordEvaluator(), policy=BudgetPolicy.single_full(max_evaluations=6))


def test_de_policy_screened_out_trials_leave_targets_unchanged() -> None:
    policy = _two_stage_policy(max_evaluations=9, batch_size=6)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=1,
        seed=42,
    )

    result = engine.run(HalfPromotionEvaluator(), policy=policy)
    rejected_trial_events = [
        event
        for event in result.events
        if event.event_type == "tell"
        and event.origin == "mutation"
        and event.confidence == "rejected"
        and event.metadata.get("reason") == "not_promoted"
    ]
    final_candidate_ids = {
        solution.metadata["candidate_id"] for solution in result.final_solutions
    }

    assert rejected_trial_events
    assert engine.state_summary().pending_batch_ids == ()
    for event in rejected_trial_events:
        assert event.metadata["target_candidate_id"] in final_candidate_ids


def test_de_policy_screened_trials_respect_max_generations() -> None:
    policy = _two_stage_policy(max_evaluations=18, batch_size=6)
    engine = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=3,
        seed=42,
    )

    result = engine.run(HalfPromotionEvaluator(), policy=policy)

    assert result.stop_reason == "max_generations"
    assert result.n_evaluations == 15
    assert result.telemetry.candidates_full_evaluated == 15
    assert len(result.generations) == 3
    assert [record.n_evaluations for record in result.generations] == [3, 3, 3]
    assert engine.generation == 3


def test_de_public_checkpoint_example_smoke(tmp_path) -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )
    optimizer = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
    candidates = optimizer.ask()
    checkpoint_path = tmp_path / "de-ask-tell.evocore-checkpoint.json"
    optimizer.save_checkpoint(
        checkpoint_path,
        optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
    )

    restored = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-sum(float(value) ** 2 for value in candidate.genes),
            confidence="trusted_full",
            stage="full",
        )
        for candidate in candidates
    ]
    result = restored.tell(records)

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.trusted_count == 6
    assert result.pending_batch_ids == ()
