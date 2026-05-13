from evocore import EvaluationRecord, GAEngine, GeneDef, GeneSpace


class OneMaxEvaluator:
    def evaluate(self, candidates, context):
        rung = context.rung
        if rung is None:
            raise ValueError("OneMaxEvaluator requires a scheduled rung.")
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(candidate.genes),
                confidence=rung.confidence,
                rung=rung.name,
                cost=rung.budget,
            )
            for candidate in candidates
        ]


space = GeneSpace([GeneDef(f"bit_{index}", "bool") for index in range(50)])
engine = GAEngine(
    space,
    population_size=80,
    generations=80,
    crossover="one_point",
    mutation="bit_flip",
    seed=42,
)
result = engine.run(OneMaxEvaluator())
print(result.best_fitness, result.best_individual.genes)
