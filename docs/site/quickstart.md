# Quickstart

```python
from evocore import EvaluationContext, EvaluationRecord, GeneticAlgorithmOptimizer, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.stage is not None
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=context.stage.confidence,
                stage=context.stage.name,
                cost=context.stage.budget,
            )
            for candidate in candidates
        ]


engine = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 10),
    population_size=100,
    max_generations=100,
    seed=42,
    direction="maximize",
)
result = engine.run(SphereEvaluator())

print(result.best_score)
print(result.best_solution.values)
```
