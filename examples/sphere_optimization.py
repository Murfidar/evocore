from evocore import EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer


class SphereEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        if stage is None:
            raise ValueError("SphereEvaluator requires a scheduled stage.")
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


engine = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 5), population_size=80, max_generations=80, seed=42
)
result = engine.run(SphereEvaluator())
print(result.best_score, result.best_solution.genes)
