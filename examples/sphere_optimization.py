from evocore import EvaluationRecord, Evaluator, GAEngine, GeneSpace


class SphereEvaluator(Evaluator):
    def evaluate(self, candidates, rung):
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates
        ]


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=80, generations=80, seed=42)
result = engine.run(SphereEvaluator())
print(result.best_fitness, result.best_individual.genes)
