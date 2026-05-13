# Genetic Algorithms

`GAEngine` proposes deterministic candidate batches over a `GeneSpace` and updates state from
`EvaluationRecord` values.

Use `ask()` and `tell()` directly when an external system owns evaluation. Use `run()` with an
an evaluator object and optional `MultiFidelityPolicy` when EvoCore should drive the budget loop.

## Budgeted Evaluation

`GAEngine.run()` expects an evaluator object:

```python
from evocore import EvaluationContext, EvaluationRecord


class Objective:
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
```

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
