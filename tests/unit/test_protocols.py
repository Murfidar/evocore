from evocore import (
    CMAESEngine,
    EvaluationContext,
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneDef,
    GeneSpace,
    Optimizer,
    Rung,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            GeneDef("x", "float", -5.0, 5.0),
            GeneDef("period", "int", 2, 20),
        ]
    )


class StructuralSphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


def test_ga_and_cma_satisfy_optimizer_protocol_at_runtime() -> None:
    assert isinstance(GAEngine(_space(), population_size=4, seed=1), Optimizer)
    assert isinstance(CMAESEngine(_space(), population_size=4, seed=1), Optimizer)


def test_structural_evaluator_satisfies_evaluator_protocol_at_runtime() -> None:
    evaluator = StructuralSphereEvaluator()
    rung = Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")
    context = EvaluationContext(
        rung=rung,
        batch_id="b-1",
        event_index=0,
        direction="minimize",
        budget=1.0,
    )

    assert isinstance(evaluator, Evaluator)
    assert evaluator.evaluate([], context) == []
