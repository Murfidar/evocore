# Genetic Algorithms

`GAEngine` proposes deterministic candidate batches over a `GeneSpace` and updates state from
`EvaluationRecord` values.

Use `ask()` and `tell()` directly when an external system owns evaluation. Use `run()` with an
`Evaluator` and optional `MultiFidelityPolicy` when EvoCore should drive the budget loop.

## Budgeted Evaluation

`GAEngine.run()` expects an evaluator object:

```python
from evocore import EvaluationRecord, Evaluator


class Objective(Evaluator):
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
```

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
