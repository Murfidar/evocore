# evocore

Rust-backed expensive black-box optimization for Python.

## Install for Development

```bash
pip install maturin pytest
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

## Budget-Aware GA Quickstart

```python
from evocore import EvaluationContext, EvaluationRecord, GAEngine, GeneSpace


class SphereEvaluator:
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


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=100, generations=100, seed=42)
result = engine.run(SphereEvaluator())
print(result.best_fitness, result.best_individual.genes)
```

## vNext Optimizer Model

EvoCore vNext separates candidate proposal from evaluation. Engines call `ask()` to create
stable candidate IDs, evaluators return `EvaluationRecord` objects, and `tell()` updates
optimizer state from trusted or cached records. `MultiFidelityPolicy` and `Rung` let you
spend cheap, partial, and full budgets deliberately.

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

## Named Mixed Gene Space

```python
from evocore import EvaluationContext, EvaluationRecord, GAEngine, GeneDef, GeneSpace

space = GeneSpace(
    [
        GeneDef("period", "int", 5, 200, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
    ]
)


class MixedEvaluator:
    def evaluate(self, candidates, context):
        assert isinstance(context, EvaluationContext)
        assert context.rung is not None
        records = []
        for candidate in candidates:
            params = candidate.params
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=-abs(params["period"] - 21) - abs(params["threshold"] - 0.35),
                    confidence=context.rung.confidence,
                    rung=context.rung.name,
                    cost=context.rung.budget,
                )
            )
        return records


result = GAEngine(space, seed=42).run(MixedEvaluator())
```

## CMA-ES

```python
from evocore import CMAESEngine, GeneSpace

engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, generations=80, seed=42)
result = engine.run(lambda ind: -sum(x * x for x in ind.genes))
```

## Reproducibility

All randomness derives from `derive_seed(master_seed, generation, individual_idx, op)`.
There is no global RNG state. Re-running the same engine with the same seed gives the same result,
and thread worker count does not change the generated populations.

## Parallelism

- vNext `GAEngine.run()` delegates evaluation strategy to your structural evaluator.
- Legacy generation-loop helpers still support `parallel="none"`, `"thread"`, and `"process"`.
- `CMAESEngine` rejects `parallel="process"` because its Rust covariance state is not picklable.

## Evaluation Protocol

Evaluators receive `Candidate` values and an `EvaluationContext`, then return
`EvaluationRecord` values. Non-rejected records must include finite scores; rejected records
can carry only diagnostics in `metrics`.
