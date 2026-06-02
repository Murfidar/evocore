# Differential Evolution

`DifferentialEvolutionOptimizer` proposes one trial candidate per target slot and
keeps the trial when it is at least as good as the incumbent target for the
configured direction.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
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


optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=12,
    max_generations=20,
    seed=42,
)
result = optimizer.run(SphereEvaluator())
```

## Mixed Bool And Numeric Spaces

DE supports flat spaces containing `float`, `int`, and `bool` genes. Float genes
use arithmetic DE variation, integer genes are rounded and clamped, and bool
genes use a deterministic binary rule inspired by GA mixed-space mutation.

```python
from evocore import DifferentialEvolutionOptimizer, Gene, GeneSpace

space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("period", "int", 2, 50),
        Gene("enabled", "bool"),
    ]
)

optimizer = DifferentialEvolutionOptimizer(space, population_size=10, seed=7)
```

## Ask/Tell Acceptance Decisions

`tell()` returns `UpdateResult.acceptance_decisions`. For DE,
`accepted_for_state=True` means the trial replaced its target slot.
`accepted_count` still counts records accepted by the ask/tell ledger.
