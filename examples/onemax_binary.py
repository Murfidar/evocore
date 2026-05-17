from evocore import EvaluationRecord, Gene, GeneSpace, GeneticAlgorithmOptimizer


class OneMaxEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        if stage is None:
            raise ValueError("OneMaxEvaluator requires a scheduled stage.")
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


space = GeneSpace([Gene(f"bit_{index}", "bool") for index in range(50)])
engine = GeneticAlgorithmOptimizer(
    space,
    population_size=80,
    max_generations=80,
    crossover="one_point",
    mutation="bit_flip",
    seed=42,
)
result = engine.run(OneMaxEvaluator())
print(result.best_score, result.best_solution.genes)
