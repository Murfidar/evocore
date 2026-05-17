# CMA-ES

`CMAESOptimizer` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"`
because the Rust covariance state is not picklable.

`direction="maximize"` and `direction="minimize"` preserve raw user fitness values in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state. Cached records
reuse trusted previous full observations and do not spend fresh full-evaluation budget.


## Result Export

Generation-oriented CMA runs attach generation events to `OptimizationResult.events` and keep
generation summaries in `OptimizationResult.generations`.

```python
result = engine.run(fitness_fn)
payload = result.to_dict()
events = result.events.to_rows()
```

Ask/tell CMA usage records `ask` and `tell` events on `engine.events`.

::: evocore.optimizers.cmaes.CMAESOptimizer
    options:
      members:
        - run
