# Budget-Aware Optimization

EvoCore vNext treats full fitness calls as scarce resources.

Use `MultiFidelityPolicy` and `Rung` to describe cheap, partial, and full evaluation
levels. Engines ask for candidates, schedulers assign rungs, evaluators return
`EvaluationRecord` objects, and engines update state through `tell()`.

```python
from evocore import MultiFidelityPolicy, Rung

policy = MultiFidelityPolicy(
    rungs=[
        Rung("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        Rung("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    full_evaluation_budget=64,
    batch_size=16,
)
```
