# Parallelism

EvoCore vNext lets evaluators own expensive evaluation strategy.

`GAEngine.run()` calls your `Evaluator.evaluate(candidates, rung)` method for each scheduled
batch. Put thread pools, process pools, remote jobs, cached backtests, or exchange-specific
rate limits inside that evaluator.

The legacy generation-loop helpers still support three local modes:

- `parallel="none"`: simplest mode for cheap Python fitness functions.
- `parallel="thread"`: useful when the callable releases the GIL.
- `parallel="process"`: useful for CPU-bound module-level callables that are pickle-safe.

`CMAESEngine` supports only `parallel="none"` and `parallel="thread"`.
