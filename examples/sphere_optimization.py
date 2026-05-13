from evocore import EvaluationContext, EvaluationRecord, GAEngine, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
            )
            for candidate in candidates
        ]


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=80, generations=80, seed=42)
result = engine.run(SphereEvaluator())
print(result.best_fitness, result.best_individual.genes)
