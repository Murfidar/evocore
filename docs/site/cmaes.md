# CMA-ES

`CMAESOptimizer` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"`
because the Rust covariance state is not picklable.

`direction="maximize"` and `direction="minimize"` preserve raw user scores in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state. Cached records
reuse trusted previous full observations and do not spend fresh full-evaluation budget.

## Rust State Snapshots

`PyCMAESState` exposes a Rust-backed state snapshot primitive:

```python
from evocore._core import PyCMAESState

state = PyCMAESState([0.0, 0.0], 0.5, 6, [(-5.0, 5.0), (-5.0, 5.0)])
samples = state.ask(42, state.generation)
state.tell(samples, [-sum(value * value for value in sample) for sample in samples])

snapshot = state.to_dict()
restored = PyCMAESState.from_dict(snapshot)

assert restored.ask(42, restored.generation) == state.ask(42, state.generation)
```

The snapshot is a schema-versioned optimizer-state payload. It preserves the
CMA-ES adaptation state needed for deterministic continuation, including mean,
sigma, covariance, evolution paths, generation, bounds, and lazy
eigendecomposition state.

`CMAESOptimizer` checkpoint/resume is still unsupported in checkpoint v1. The
Rust state primitive is the foundation for that later optimizer-level work,
which also needs Python candidate ledgers, pending batches, telemetry, and event
indexes.

Use `OptimizationResult.to_dict()` for completed-run export and `engine.events`
for ask/tell audit rows. Those exports are not checkpoint files and are not
replayed to rebuild CMA-ES state.

## Result Export

Generation-oriented CMA runs attach generation events to `OptimizationResult.events` and keep
generation summaries in `OptimizationResult.generations`.

```python
result = engine.run(objective_fn)
payload = result.to_dict()
events = result.events.to_rows()
```

Ask/tell CMA usage records `ask` and `tell` events on `engine.events`.

::: evocore.optimizers.cmaes.CMAESOptimizer
    options:
      members:
        - run


## Configuration Identity

`CMAESOptimizer` exposes the same config export surface as GA:

```python
from evocore import CMAESOptimizer, GeneSpace

space = GeneSpace.uniform(-2.0, 2.0, 4)
optimizer = CMAESOptimizer(space, population_size=24, initial_sigma=0.25, seed=42)

signature = optimizer.config_signature()
config_hash = optimizer.config_hash()
optimizer.validate_compatibility()
```

The CMA-ES config hash covers public strategy inputs such as population size, initial
mean, initial sigma, maximum generations, seed, direction, and supported parallel mode.
Gene-space identity remains separate through `space.signature()` and `space.hash()`.
