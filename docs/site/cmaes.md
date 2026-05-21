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

`CMAESOptimizer` supports stable ask/tell checkpoints for manual external-evaluation
workflows:

```python
from evocore import CMAESOptimizer, EvaluationRecord, GeneSpace

space = GeneSpace.uniform(-2.0, 2.0, 3)
optimizer = CMAESOptimizer(space, population_size=8, seed=42)
candidates = optimizer.ask()

optimizer.save_checkpoint(
    "cmaes-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = CMAESOptimizer(space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("cmaes-ask-tell.evocore-checkpoint.json")

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

The checkpoint combines the Rust CMA-ES state snapshot with Python candidate
ledgers, pending batches, telemetry, and audit events. Stable JSON checkpoints
produced by EvoCore 0.8.0 are the checkpoint schema v1 compatibility baseline
for manual CMA-ES ask/tell resume. Generation-loop and policy-driven CMA-ES
resume remain unsupported.

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
