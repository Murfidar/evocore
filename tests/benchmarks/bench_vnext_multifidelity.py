from evocore import (
    EvaluationRecord,
    Evaluator,
    GAEngine,
    GeneSpace,
    MultiFidelityPolicy,
    Rung,
)


class DeceptiveSphere(Evaluator):
    def evaluate(self, candidates, rung):
        records = []
        for candidate in candidates:
            true_score = -sum(float(value) ** 2 for value in candidate.genes)
            cheap_score = true_score + (0.1 if candidate.candidate_id.endswith("0") else 0.0)
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    score=cheap_score if rung.name == "cheap" else true_score,
                    confidence=rung.confidence,
                    rung=rung.name,
                    cost=rung.budget,
                )
            )
        return records


def test_vnext_multifidelity_benchmark_smoke() -> None:
    policy = MultiFidelityPolicy(
        rungs=[
            Rung("cheap", budget=0.1, promote_fraction=0.5, confidence="partial"),
            Rung("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full"),
        ],
        full_evaluation_budget=16,
        batch_size=8,
        audit_fraction=0.25,
    )
    result = GAEngine(GeneSpace.uniform(-5.0, 5.0, 3), population_size=8, seed=11).run(
        DeceptiveSphere(),
        policy=policy,
    )

    assert result.telemetry.candidates_full_evaluated == 16
    assert result.telemetry.candidates_partial_evaluated >= 16
    assert result.best_individual.fitness_valid
