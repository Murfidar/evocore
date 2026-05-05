# Genetic Algorithms

`GAEngine` runs deterministic genetic algorithm optimization over a `GeneSpace`.

Use `parallel="none"` for fast Python fitness functions, `parallel="thread"` when the fitness
function releases the GIL, and `parallel="process"` for pickle-safe module-level fitness
functions.

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
