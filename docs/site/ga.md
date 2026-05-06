# Genetic Algorithms

`GAEngine` runs deterministic genetic algorithm optimization over a `GeneSpace`.

Use `parallel="none"` for fast Python fitness functions, `parallel="thread"` when the fitness
function releases the GIL, and `parallel="process"` for pickle-safe module-level fitness
functions.

## DEAP-Parity Controls

EvoCore keeps deterministic seed derivation while matching common DEAP GA semantics when needed:

- `crossover="uniform"` is available for numeric spaces as well as binary spaces. For mixed
  float, integer, and categorical-by-integer chromosomes, this swaps alleles between parents
  instead of averaging them as SBX or BLX would.
- `mutation_prob` is the per-gene mutation probability after an offspring is selected for
  mutation.
- `mutation_individual_prob` is the per-offspring mutation gate. Leave it at `1.0` for legacy
  EvoCore behavior, or set it to the same outer mutation probability used by a DEAP pipeline.
- Tournament selection samples aspirants with replacement, matching DEAP `selTournament`.

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

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
