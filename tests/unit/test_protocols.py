from evocore import (
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    Evaluator,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    Optimizer,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


class StructuralSphereEvaluator:
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


def test_ga_cma_and_de_satisfy_optimizer_protocol_at_runtime() -> None:
    assert isinstance(GeneticAlgorithmOptimizer(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(CMAESOptimizer(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(
        DifferentialEvolutionOptimizer(_space(), population_size=4, seed=1), Optimizer
    )


def test_structural_evaluator_satisfies_evaluator_protocol_at_runtime() -> None:
    evaluator = StructuralSphereEvaluator()
    stage = EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")
    context = EvaluationContext(
        stage=stage,
        batch_id="b-1",
        event_index=0,
        direction="minimize",
        budget=1.0,
    )

    assert isinstance(evaluator, Evaluator)
    assert evaluator.evaluate([], context) == []
