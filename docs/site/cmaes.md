# CMA-ES

`CMAESEngine` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"`
because the Rust covariance state is not picklable.

`direction="maximize"` and `direction="minimize"` preserve raw user fitness values in
results while using direction-aware comparison internally. In ask/tell mode, complete
batches of `trusted_full` or `cached` records update the covariance state.

## Result Export

Generation-oriented CMA runs attach generation events to `RunResult.history` and keep
generation summaries in `RunResult.logbook`.

```python
result = engine.run(fitness_fn)
payload = result.to_dict()
events = result.history.to_rows()
```

Ask/tell CMA usage records `ask` and `tell` events on `engine.history`.

::: evocore.cmaes.CMAESEngine
    options:
      members:
        - run
