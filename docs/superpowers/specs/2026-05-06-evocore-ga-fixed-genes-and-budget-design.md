# Evocore GA Fixed Genes And Budget Design

**Date:** 2026-05-06
**Status:** Draft
**Scope:** GA-first compatibility improvement for full-genome fixed genes, exact evaluation budgets, and run diagnostics

## Purpose

Evocore should support optimization schemas where some numeric genes are intentionally fixed by a strategy family or search profile. A fixed gene remains part of the full genome and parameter contract, but operators must never change its value.

This design also adds exact GA evaluation-budget control so downstream benchmark harnesses can compare Evocore to other optimizers at the same number of fitness calls.

## Goals

- Allow `GeneDef(..., "float", value, value)` and `GeneDef(..., "int", value, value)`.
- Preserve fixed genes in `Individual.genes`, `Individual.params`, `GeneSpace.names`, `GeneSpace.bounds`, callback populations, logbook-facing populations, and population dataframes.
- Keep GA operators full-genome aware while ensuring fixed genes cannot drift during initialization, crossover, mutation, cloning, resume, or `run_multiple`.
- Add `GAEngine(max_evaluations=...)` as a hard cap on user fitness calls.
- Expose clear run diagnostics for why a GA run stopped.
- Keep CMA-ES behavior unchanged in this iteration, while shaping `GeneSpace` metadata so CMA-ES can adopt fixed-gene reconstruction later.

## Non-Goals

- Do not migrate CMA-ES to fixed-gene variable-dimension optimization in this change.
- Do not add conditional, derived, or dependent genes yet.
- Do not add a separate variable-only genome API for GA users.
- Do not change the public meaning of existing variable genes.
- Do not change existing bool-gene semantics.

## API Design

`GeneDef` accepts equal bounds for numeric genes:

```python
GeneDef("signal_mode", "int", 2, 2)
GeneDef("threshold", "float", 0.5, 0.5)
```

`low > high` remains invalid. Fixed integer genes still require integer bounds. Bool genes remain unbounded binary genes, and mixed bool plus numeric spaces keep the current operator restriction.

`GeneDef` gains:

```python
@property
def is_fixed(self) -> bool: ...
```

For numeric genes, `is_fixed` is true when `low == high`. For bool genes it is false in this iteration.

`GeneSpace` gains read-only metadata:

```python
fixed_indices: list[int]
variable_indices: list[int]
fixed_count: int
variable_count: int
```

These are diagnostic and future-engine helpers. GA remains full-genome: callers pass and receive complete gene vectors.

## GA Runtime Design

Fixed genes stay in the full-length Rust boundary. The encoded population still has one value per `GeneDef`.

Initialization must produce the fixed value for fixed numeric slots. Rust `init_population` must special-case `low == high` and return `low` instead of sampling from an empty float range.

Reproduction must preserve fixed slots. The safest implementation is:

- Set fixed gene mutation sigmas to `0.0`.
- Clamp and round against equal bounds after crossover and mutation.
- Decode full-length individuals as usual.
- Add targeted tests proving fixed slots remain unchanged under high crossover and mutation probabilities.

`OperatorSet.decode_genes` and `GeneSpace.params_for` continue to return complete decoded values, so downstream code sees full parameter dictionaries without adapter-side expansion.

## Exact Evaluation Budget

`GAEngine` gains:

```python
max_evaluations: int | None = None
```

When `max_evaluations` is `None`, current generation-based behavior remains unchanged.

When set, it is a hard cap on calls to the user fitness function. `0` and negative values are invalid.

The cap applies per run. `run_multiple` passes the same cap to each child run.

If `max_evaluations < population_size`, the engine evaluates only that many initial individuals and returns a final population containing evaluated individuals only. This avoids exposing unevaluated individuals in `final_population`.

For later generations, the engine may evaluate a partial offspring batch. It must stop before calling the user fitness function after the cap is reached. Cached elites do not count as new evaluations.

## Run Diagnostics

`RunResult` gains:

```python
max_evaluations: int | None
stop_reason: Literal["generations", "max_evaluations", "callback"]
budget_reached: bool
```

`budget_reached` is true when `max_evaluations` is set and `n_evaluations >= max_evaluations`. When no cap is configured, it is false.

`stopped_early` remains for compatibility. It is true when a callback stops the run, or when `max_evaluations` stops the run before the configured generation loop completes.

The precedence is:

1. `callback` if a callback requests stop.
2. `max_evaluations` if the hard cap is reached.
3. `generations` when the configured generation count is exhausted.

## Error Handling

- `GeneDef("x", "float", 1.0, 0.0)` raises `ConfigurationError`.
- `GeneDef("period", "int", 1.0, 1.0)` raises `ConfigurationError` because int bounds must be integers.
- `GAEngine(..., max_evaluations=0)` raises `ConfigurationError`.
- `CMAESEngine` raises a clear `ConfigurationError` if the provided `GeneSpace` contains fixed numeric genes, because CMA-ES fixed-dimension reconstruction is out of scope for this iteration.
- If a fitness function raises before the cap is reached, the existing `FitnessError` wrapping behavior remains unchanged.
- If all evaluated fitnesses are non-finite, existing sanitization to `-inf` remains unchanged.

## Testing

Required unit tests:

- Fixed float and int genes are valid, while reversed bounds are invalid.
- `GeneDef.is_fixed`, `GeneSpace.fixed_indices`, `variable_indices`, `fixed_count`, and `variable_count` are correct.
- `GeneSpace.rust_bounds` preserves equal bounds.
- `OperatorSet.decode_individual` includes fixed genes in `Individual.params`.
- Initial GA populations contain full-length individuals with fixed values intact.
- GA crossover and mutation cannot alter fixed values, including high variation settings.
- `GAEngine(max_evaluations=N)` performs exactly `N` fitness calls for caps smaller than, equal to, and larger than population size.
- `RunResult.stop_reason`, `budget_reached`, `stopped_early`, and `max_evaluations` are correct.
- `run_multiple` applies the cap per child run.
- `CMAESEngine` rejects fixed numeric genes with a clear error until CMA-ES fixed-gene reconstruction is implemented.
- Existing GA, callback, resume, and mixed numeric tests continue to pass.

Required Rust/PyO3 tests:

- `init_population` handles equal numeric bounds without panicking.
- `reproduce_population` preserves fixed numeric positions after crossover and mutation.

## Migration Notes

Existing users with only variable genes should see no behavior change.

Users who previously excluded fixed genes manually can simplify their adapters after upgrading: fixed genes can remain in `GeneSpace`, and `Individual.params` will already contain the full schema.

CMA-ES users should not pass fixed numeric genes expecting variable-dimension behavior yet. The new `GeneSpace` metadata prepares for that future work, but this iteration is GA-first.

## Implementation Order

1. Update `GeneDef` validation and add fixed metadata on `GeneDef` and `GeneSpace`.
2. Update Rust initialization and reproduction guards for equal numeric bounds.
3. Add fixed-gene tests at Python and Rust/PyO3 levels.
4. Add `max_evaluations` to `GAEngine`, `_copy_with_seed`, `RunResult`, and `run_multiple`.
5. Add exact-budget tests and diagnostics tests.
6. Run targeted Python and Rust tests, then the full verification suite if practical.
