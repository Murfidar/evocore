# Budget-Aware Optimization

EvoCore vNext treats full objective calls as scarce resources.

Use `BudgetPolicy` and `EvaluationStage` to describe cheap, partial, and full evaluation
levels. Engines ask for candidates, schedulers assign stages, evaluators return
`EvaluationRecord` objects, and engines update state through `tell()`.

```python
from evocore import BudgetPolicy, EvaluationStage

policy = BudgetPolicy(
    stages=[
        EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    max_evaluations=64,
    batch_size=16,
)
```

`GeneticAlgorithmOptimizer` and `DifferentialEvolutionOptimizer` can drive this
policy directly through `run(evaluator, policy=policy)`. For DE, non-final
stages screen candidates and final state-eligible records initialize or replace
target slots. CMA-ES remains manual ask/tell for policy-shaped external
evaluation in this release line.

`max_evaluations` counts fresh `trusted_full` observations only. Cached records can
update optimizer state, but they do not spend fresh full-evaluation budget.
