# evocore

Rust-native Genetic Algorithms and CMA-ES for Python.

## Install for Development

```bash
pip install maturin pytest
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

## Genetic Algorithm Quickstart

```python
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=100, generations=100, seed=42)
result = engine.run(sphere)
print(result.best_fitness, result.best_individual.genes)
```

## DEAP-Parity GA Configuration

For mixed numeric chromosomes that encode categorical choices as integers, use uniform
crossover and the optional per-offspring mutation gate to mirror DEAP-style GA pipelines:

```python
engine = GAEngine(
    space,
    crossover="uniform",
    crossover_prob=0.8,
    mutation="gaussian",
    mutation_prob=0.2,
    mutation_individual_prob=0.2,
    selection="tournament",
    tournament_size=4,
    seed=42,
)
```

## Named Mixed Gene Space

```python
from evocore import GAEngine, GeneDef, GeneSpace

space = GeneSpace(
    [
        GeneDef("period", "int", 5, 200, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
    ]
)


def objective(ind):
    return -abs(ind.params["period"] - 21) - abs(ind.params["threshold"] - 0.35)


result = GAEngine(space, seed=42).run(objective)
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

- `parallel="none"`: simplest and best for fast fitness functions.
- `parallel="thread"`: useful when the fitness function releases the GIL.
- `parallel="process"`: available on `GAEngine`; requires a module-level picklable fitness function.
- `CMAESEngine` rejects `parallel="process"` because its Rust covariance state is not picklable.

## Fitness Function Protocol

Fitness functions receive `Individual` and return either `float` or `(float, metrics_dict)`.
NaN and Inf are treated as `-inf` for selection and emit `FitnessWarning` once per run.
