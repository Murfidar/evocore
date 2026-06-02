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

## Budgeted Evaluation

`DifferentialEvolutionOptimizer.run(evaluator, policy=...)` uses the same
`BudgetPolicy` and `EvaluationStage` vocabulary as GA. Non-final stages can
screen candidates, while DE target slots are initialized or replaced only after
final state-eligible `trusted_full` or `cached` records.

```python
from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
    GeneSpace,
)


class TwoStageSphere:
    def evaluate(self, candidates, context):
        stage = context.stage
        scale = 0.5 if stage.name == "cheap" else 1.0
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-scale * sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


policy = BudgetPolicy(
    stages=[
        EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    max_evaluations=32,
    batch_size=8,
)

result = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 2),
    population_size=8,
    seed=42,
).run(TwoStageSphere(), policy=policy)
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

## Multi-Run Execution

Use `run_multiple(...)` when you want deterministic child runs from one DE
configuration:

```python
batch = optimizer.run_multiple(TwoStageSphere(), n_runs=5)
best = batch.best
scores = [run.best_score for run in batch.all_runs]
```

Child seeds are derived from the optimizer seed and results are sorted
best-first using the optimizer direction.

## Current Limitations

DE does not yet expose custom strategy plugins or a Rust-backed variation
kernel. Those remain future feature and performance tracks. Policy-driven
mid-loop checkpoint resume is also outside checkpoint v1; use manual ask/tell
checkpoints when evaluation work must survive process restarts.
