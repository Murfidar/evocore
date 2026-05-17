from evocore import EvaluationRecord, Gene, GeneSpace, GeneticAlgorithmOptimizer


class MixedEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        if stage is None:
            raise ValueError("MixedEvaluator requires a scheduled stage.")
        records = []
        for candidate in candidates:
            params = candidate.params or {}
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=-abs(params["period"] - 21) - abs(params["threshold"] - 0.35),
                    confidence=stage.confidence,
                    stage=stage.name,
                    cost=stage.budget,
                )
            )
        return records


space = GeneSpace(
    [
        Gene("period", "int", 5, 50, sigma=0.05),
        Gene("threshold", "float", 0.0, 1.0),
    ]
)
result = GeneticAlgorithmOptimizer(space, population_size=60, max_generations=50, seed=7).run(
    MixedEvaluator()
)
print(result.best_score, result.best_solution.params)
