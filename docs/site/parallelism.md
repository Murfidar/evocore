# Parallelism

evocore supports three evaluation modes for `GAEngine`.

- `parallel="none"`: simplest mode and usually best for cheap fitness functions.
- `parallel="thread"`: useful when the fitness function releases the GIL.
- `parallel="process"`: useful for CPU-bound Python fitness functions that are pickle-safe.

Process mode requires module-level functions. Lambdas, nested functions, and closures are
rejected because they cannot be pickled reliably.

`CMAESEngine` supports only `parallel="none"` and `parallel="thread"`.
