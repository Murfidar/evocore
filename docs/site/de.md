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

## When To Choose DE

DE is a good fit for continuous or mostly numeric search spaces where objective
evaluations are expensive enough that stable candidate proposals matter more
than gradient information. It is often easier to tune than GA for numeric
parameters because the default `rand1bin` strategy uses scaled population
differences instead of custom crossover and mutation operators.

Use GA when the search is heavily discrete, operator design is central, or
multi-run utilities are required today. Use CMA-ES when the space is continuous
and covariance adaptation is the main advantage. DE currently supports flat
`float`, `int`, and `bool` `GeneSpace` values; CMA-ES continues to be the more
specialized continuous optimizer.

## Reproducibility

DE candidate IDs, batch IDs, initialization samples, trial target mappings, and
replacement decisions are deterministic for the same `GeneSpace`, optimizer
configuration, direction, and seed. Stable checkpoint files also include the
gene-space hash, optimizer config hash, seed, and direction so resume fails
early when the receiving optimizer does not match the saved state.

## Ask/Tell Checkpointing

Manual ask/tell checkpoints are the stable continuation boundary for DE.
Checkpointing preserves pending initialization candidates, target population
state, pending trial-to-target mappings, telemetry, and audit events.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, Gene, GeneSpace

space = GeneSpace(
    [
        Gene("x", "float", -5.0, 5.0),
        Gene("period", "int", 2, 20),
        Gene("enabled", "bool"),
    ]
)
optimizer = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "de-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
summary = restored.resume_ask_tell_checkpoint("de-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
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

## Current Limitations

DE does not yet expose `run_multiple(...)`, policy-aware `run(...)`, custom
strategy plugins, or a Rust-backed variation kernel. Those are future feature
and performance parity tracks; the current DE checkpoint contract focuses on
manual ask/tell continuation and synchronous evaluator-driven `run()`.
