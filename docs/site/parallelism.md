# Parallelism

EvoCore vNext lets evaluators own expensive evaluation strategy.

`GAEngine.run()` calls your `Evaluator.evaluate(candidates, rung)` method for each scheduled
batch. Put thread pools, process pools, remote jobs, cached simulations, or service rate limits
inside that evaluator.

The legacy generation-loop helpers still support three local modes:

- `parallel="none"`: simplest mode for cheap Python fitness functions.
- `parallel="thread"`: useful when the callable releases the GIL.
- `parallel="process"`: useful for CPU-bound module-level callables that are pickle-safe.

`ProcessParallel` keeps its process pool alive across repeated `evaluate(...)` calls on the
same helper. Use it as a context manager or call `close()` when an evaluator is finished.

`CMAESEngine` supports only `parallel="none"` and `parallel="thread"`.
