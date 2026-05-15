# Genetic Algorithms

`GAEngine` proposes deterministic candidate batches over a `GeneSpace` and updates state from
state-eligible `EvaluationRecord` values. `trusted_full` and `cached` records can update
best-candidate state; `partial` and `surrogate` records remain available for scheduling and
telemetry.

Full-evaluation budget accounting counts fresh `trusted_full` records only. Cached records
can update best-candidate state, but they do not spend fresh objective budget.

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

## Result Export

`RunResult` is the stable envelope for a completed run:

```python
result = engine.run(Objective())
payload = result.to_dict()
json_text = result.to_json(indent=2)
events = result.history.to_rows()
```

Runtime timing is excluded from deterministic exports by default. Pass
`include_runtime=True` to include `wall_time_seconds` under the `runtime` key.

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
