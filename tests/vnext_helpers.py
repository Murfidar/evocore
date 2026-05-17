from evocore import BudgetPolicy, EvaluationContext, EvaluationRecord, EvaluationStage
from evocore.search_space import Solution


class IndividualEvaluator:
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        records = []
        for candidate in candidates:
            solution = Solution(
                list(candidate.genes),
                metadata={
                    "params": candidate.params,
                    "candidate_id": candidate.candidate_id,
                },
            )
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(self.fn(solution)),
                    confidence=context.stage.confidence,
                    stage=context.stage.name,
                    cost=context.stage.budget,
                )
            )
        return records


def full_policy(max_evaluations: int, batch_size: int = 8) -> BudgetPolicy:
    return BudgetPolicy(
        stages=[
            EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")
        ],
        max_evaluations=max_evaluations,
        batch_size=batch_size,
    )
