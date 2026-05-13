from evocore import EvaluationRecord, GAEngine, GeneDef, GeneSpace


class MixedEvaluator:
    def evaluate(self, candidates, context):
        rung = context.rung
        if rung is None:
            raise ValueError("MixedEvaluator requires a scheduled rung.")
        records = []
        for candidate in candidates:
            params = candidate.params or {}
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=-abs(params["period"] - 21) - abs(params["threshold"] - 0.35),
                    confidence=rung.confidence,
                    rung=rung.name,
                    cost=rung.budget,
                )
            )
        return records


space = GeneSpace(
    [
        GeneDef("period", "int", 5, 50, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
    ]
)
result = GAEngine(space, population_size=60, generations=50, seed=7).run(MixedEvaluator())
print(result.best_fitness, result.best_individual.params)
