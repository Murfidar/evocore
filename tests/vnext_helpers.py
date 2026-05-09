from evocore import EvaluationRecord, Evaluator, MultiFidelityPolicy, Rung
from evocore.individual import Individual


class IndividualEvaluator(Evaluator):
    def __init__(self, fn):
        self.fn = fn

    def evaluate(self, candidates, rung):
        records = []
        for candidate in candidates:
            individual = Individual(
                list(candidate.genes),
                metadata={
                    "params": candidate.params,
                    "candidate_id": candidate.candidate_id,
                },
            )
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    score=float(self.fn(individual)),
                    confidence=rung.confidence,
                    rung=rung.name,
                    cost=rung.budget,
                )
            )
        return records


def full_policy(budget: int, batch_size: int = 8) -> MultiFidelityPolicy:
    return MultiFidelityPolicy(
        rungs=[Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")],
        full_evaluation_budget=budget,
        batch_size=batch_size,
    )
