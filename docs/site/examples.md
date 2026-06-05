# Examples

This page shows common situations where EvoCore is useful and the smallest code
shape for each one. Use the optimizer-specific pages when you want deeper tuning
details.

## Pick An Optimizer

| Situation | Start with |
| --- | --- |
| Continuous numeric tuning with few assumptions about the landscape | `DifferentialEvolutionOptimizer` |
| Continuous smooth-ish numeric tuning where covariance adaptation helps | `CMAESOptimizer` |
| Mixed `float`, `int`, and `bool` settings | `GeneticAlgorithmOptimizer` or `DifferentialEvolutionOptimizer` |
| Binary on/off decisions | `GeneticAlgorithmOptimizer` |
| External jobs, queues, simulations, or cached evaluations | Any optimizer with `ask()` / `tell()` |
| Cheap screening before expensive full evaluation | `BudgetPolicy` with GA or DE |

EvoCore maximizes scores by default. For loss functions, return a negative loss
or pass `direction="minimize"` when that better matches the domain.

## Tune Continuous Parameters

Situation: you have numeric parameters for a simulation, controller, model, or
pipeline and only a black-box score.

Differential Evolution is a good first choice because it handles bounded numeric
spaces without gradients and has simple defaults.

```python
from evocore import DifferentialEvolutionOptimizer, EvaluationRecord, GeneSpace


class SphereEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=16,
    max_generations=40,
    seed=42,
)
result = optimizer.run(SphereEvaluator())

print(result.best_score)
print(result.best_solution.values)
```

Use `strategy="rand2bin"` when you want broader exploration and can afford a
larger population. Use `strategy="current-to-best1bin"` when the current best
candidate should pull the population more strongly.

```python
optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=24,
    strategy="rand2bin",
    seed=42,
)
```

## Optimize Mixed Product Or System Settings

Situation: you need to tune a configuration that has numeric thresholds,
integer counts, and boolean feature switches.

Named genes become `candidate.params`, which keeps the evaluator readable.

```python
from evocore import EvaluationRecord, Gene, GeneSpace, GeneticAlgorithmOptimizer


space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("batch_size", "int", 8, 128, sigma=0.03),
        Gene("use_cache", "bool"),
    ]
)


class ConfigEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        records = []
        for candidate in candidates:
            params = candidate.params
            score = 0.0
            score -= abs(float(params["threshold"]) - 0.62)
            score -= abs(int(params["batch_size"]) - 64) / 64.0
            score += 0.25 if params["use_cache"] else 0.0
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence=stage.confidence,
                    stage=stage.name,
                    cost=stage.budget,
                )
            )
        return records


optimizer = GeneticAlgorithmOptimizer(
    space,
    population_size=32,
    max_generations=30,
    seed=7,
)
result = optimizer.run(ConfigEvaluator())

print(result.best_solution.params)
```

GA is a natural default when discrete choices and operator behavior matter.
DE can also run the same flat `float` / `int` / `bool` space when numeric
variation is the better fit.

```python
from evocore import DifferentialEvolutionOptimizer

optimizer = DifferentialEvolutionOptimizer(
    space,
    population_size=12,
    max_generations=20,
    seed=7,
)
```

## Select A Set Of Binary Decisions

Situation: choose which items, rules, or switches to enable under a scoring
function.

Represent each decision as a `bool` gene and use GA bit-flip mutation.

```python
from evocore import EvaluationRecord, Gene, GeneSpace, GeneticAlgorithmOptimizer


space = GeneSpace([Gene(f"item_{index}", "bool") for index in range(20)])
weights = [1.0, 0.5, 2.0, 1.5] * 5
costs = [1.0, 1.0, 3.0, 2.0] * 5
budget = 18.0


class SelectionEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        records = []
        for candidate in candidates:
            selected = [index for index, enabled in enumerate(candidate.genes) if enabled]
            value = sum(weights[index] for index in selected)
            cost = sum(costs[index] for index in selected)
            penalty = max(0.0, cost - budget) * 4.0
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=value - penalty,
                    confidence=stage.confidence,
                    stage=stage.name,
                    cost=stage.budget,
                )
            )
        return records


optimizer = GeneticAlgorithmOptimizer(
    space,
    population_size=80,
    max_generations=80,
    crossover="one_point",
    mutation="bit_flip",
    seed=42,
)
result = optimizer.run(SelectionEvaluator())

chosen_items = [
    name
    for name, enabled in zip(space.names, result.best_solution.values, strict=True)
    if enabled
]
print(chosen_items)
```

## Screen Cheaply Before Full Evaluation

Situation: full evaluations are expensive, but you have a cheap proxy that can
discard weak candidates.

Use a `BudgetPolicy` with one partial stage and one final trusted stage. GA and
DE can drive this policy directly through `run(evaluator, policy=policy)`.

```python
from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    EvaluationStage,
    GeneSpace,
)


class TwoStageEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        records = []
        for candidate in candidates:
            x = [float(value) for value in candidate.genes]
            true_score = -sum(value * value for value in x)
            score = 0.5 * true_score if stage.name == "cheap" else true_score
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=score,
                    confidence=stage.confidence,
                    stage=stage.name,
                    cost=stage.budget,
                )
            )
        return records


policy = BudgetPolicy(
    stages=[
        EvaluationStage("cheap", budget=0.10, promote_fraction=0.50, confidence="partial"),
        EvaluationStage("full", budget=1.00, promote_fraction=1.00, confidence="trusted_full"),
    ],
    max_evaluations=40,
    batch_size=8,
)

optimizer = DifferentialEvolutionOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 3),
    population_size=8,
    seed=42,
)
result = optimizer.run(TwoStageEvaluator(), policy=policy)

print(result.n_evaluations)
print(result.telemetry.candidates_partial_evaluated)
```

`max_evaluations` counts fresh `trusted_full` records. Partial and surrogate
records are useful for scheduling and telemetry, but they do not become the
optimizer best.

## Send Work To An External Queue

Situation: candidates must be evaluated by another process, service, notebook, or
job queue.

Use manual ask/tell. Keep `candidate_id` and `batch_id` with each submitted job,
then return one `EvaluationRecord` for each completed candidate.

```python
from evocore import EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer


optimizer = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-2.0, 2.0, 3),
    population_size=10,
    seed=42,
)

candidates = optimizer.ask(5)

# Submit these candidates to your queue and persist candidate_id + batch_id.
jobs = [
    {
        "candidate_id": candidate.candidate_id,
        "batch_id": candidate.batch_id,
        "genes": candidate.genes,
    }
    for candidate in candidates
]

# Later, turn completed jobs into records. Records can arrive in any order.
records = [
    EvaluationRecord(
        candidate_id=job["candidate_id"],
        batch_id=job["batch_id"],
        score=-sum(float(value) ** 2 for value in job["genes"]),
        confidence="trusted_full",
        stage="full",
        cost=1.0,
    )
    for job in reversed(jobs)
]

update = optimizer.tell(records)
print(update.accepted_count)
```

Use `optimizer.ask_tell_checkpoint()` before handing work to an external system
when the process may restart before records return.

```python
optimizer.save_checkpoint(
    "queue-submit.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"queue": "submitted"}),
)
```

## Fit Continuous Parameters With CMA-ES

Situation: the space is continuous or integer-valued, the objective has useful
local structure, and covariance adaptation may be valuable.

CMA-ES accepts a simple callable that receives a `Solution`.

```python
from evocore import CMAESOptimizer, GeneSpace


def rosenbrock(solution):
    values = solution.values
    loss = sum(
        100.0 * (values[index + 1] - values[index] ** 2) ** 2
        + (1.0 - values[index]) ** 2
        for index in range(len(values) - 1)
    )
    return -loss


optimizer = CMAESOptimizer(
    GeneSpace.uniform(-2.0, 2.0, 4),
    population_size=30,
    max_generations=80,
    initial_sigma=0.4,
    seed=42,
)
result = optimizer.run(rosenbrock)

print(result.best_score)
print(result.best_solution.values)
```

CMA-ES rejects boolean genes. Use GA or DE when boolean switches are part of the
search space.

## Compare Multiple Seeded Runs

Situation: one optimizer run can be unlucky, and you want a deterministic batch
of child runs from one parent seed.

GA and DE expose `run_multiple(...)`. Results are sorted best-first using the
optimizer direction.

```python
from evocore import EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer


class SphereEvaluator:
    def evaluate(self, candidates, context):
        stage = context.stage
        return [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=-sum(float(value) ** 2 for value in candidate.genes),
                confidence=stage.confidence,
                stage=stage.name,
                cost=stage.budget,
            )
            for candidate in candidates
        ]


optimizer = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-5.0, 5.0, 4),
    population_size=24,
    max_generations=20,
    seed=123,
)
batch = optimizer.run_multiple(SphereEvaluator(), n_runs=5)

print(batch.best.best_score)
print([run.best_score for run in batch.all_runs])
```

Use `run_parallel=True` only when the evaluator and optimizer are pickle-safe.

## Inspect And Export Results

Situation: after a run, you want the best candidate, final population, event
audit rows, and reproducibility metadata.

```python
payload = result.to_dict()
json_text = result.to_json(indent=2)

best_values = result.best_solution.values
best_params = result.best_solution.params
events = result.events.to_rows()

print(payload["reproducibility"]["gene_space_hash"])
print(best_values)
print(best_params)
print(events[:3])
```

`OptimizationResult.to_dict()` is for analysis and reporting. Stable checkpoint
files are separate optimizer-state snapshots for continuation.

## What To Read Next

- [Quickstart](quickstart.md) for the smallest full run.
- [Gene Spaces](gene-space.md) for `float`, `int`, `bool`, repair, and hashing.
- [Genetic Algorithms](ga.md), [Differential Evolution](de.md), and
  [CMA-ES](cmaes.md) for optimizer-specific details.
- [Budget-Aware Optimization](budget-aware-optimization.md) for staged
  evaluation.
- [Ask/Tell Engines](ask-tell-engines.md) for external evaluation systems.
- [Callbacks And Checkpointing](callbacks-checkpointing.md) for stable resume
  workflows.
