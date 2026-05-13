from evocore import EvaluationContext, EvaluationRecord, GAEngine, GeneDef, GeneSpace


class OneMaxEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=sum(candidate.genes),
                confidence=context.rung.confidence,
                rung=context.rung.name,
                cost=context.rung.budget,
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
