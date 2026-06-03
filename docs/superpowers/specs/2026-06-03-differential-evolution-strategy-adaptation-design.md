# Differential Evolution Strategy And Adaptation Design

**Date:** 2026-06-03
**Status:** Design approved for specification
**Scope:** Add a long-term strategy architecture for
`DifferentialEvolutionOptimizer`, including stateless built-in strategies and a
first simple jDE-style adaptive strategy. SHADE-style adaptation remains a later
track.

## Summary

`DifferentialEvolutionOptimizer` is now a first-class EvoCore optimizer with
mixed `float`/`int`/`bool` search-space support, deterministic ask/tell,
checkpointing, policy-driven `run(...)`, and `run_multiple(...)`. Its remaining
long-term gap is not the optimizer lifecycle. It is strategy extensibility.

The current public strategy surface accepts only `strategy="rand1bin"`, and the
trial generation details live directly in the DE ask/tell implementation. The
next long-term step should split strategy math from lifecycle orchestration,
then add a small curated set of built-in DE strategies. A simple jDE-style
adaptive strategy should follow as the first stateful strategy. SHADE-style
adaptation should remain future work until jDE proves the strategy state,
checkpoint, and policy boundaries.

The work may proceed in parallel across three tracks:

1. Strategy contract and `rand1bin` refactor.
2. Stateless built-in strategies.
3. Simple jDE-style adaptation.

The merge order should still be sequential: contract first, stateless built-ins
second, jDE third.

## Current Context

The DE implementation currently lives in:

```text
evocore/optimizers/de/
  __init__.py
  ask_tell.py
  checkpointing.py
  config.py
  engine.py
  multi_run.py
```

`ask_tell.py` owns initialization candidates, trial candidates, target-slot
replacement, pending batches, telemetry, and events. `config.py` validates
`strategy="rand1bin"` and the global `mutation_factor` and `crossover_rate`.
`checkpointing.py` serializes the ask/tell state needed to resume pending
initialization or trial batches.

That structure is stable, but strategy growth would make `ask_tell.py` too
large and too strategy-specific. The new design keeps ask/tell lifecycle logic
where it is and moves strategy-specific trial math into focused DE strategy
modules.

## Goals

- Preserve existing `rand1bin` behavior for deterministic seeds, candidate IDs,
  trial values, replacement decisions, and ask/tell checkpoint continuation.
- Introduce an internal DE strategy contract so `ask_tell.py` delegates trial
  construction instead of hardcoding each strategy.
- Add stateless built-ins behind the existing string API:
  `best1bin`, `rand2bin`, and `current-to-best1bin`.
- Add `jde-rand1bin` as the first stateful adaptive strategy.
- Keep mixed `float`/`int`/`bool` repair behavior shared across all strategies.
- Make strategy validation, config signatures, config hashes, docs, and
  checkpoint state derive from one strategy specification source.
- Ensure policy-driven `run(...)` does not need strategy-specific branching.
- Prove jDE checkpoint restore from pending trials is deterministic.
- Leave SHADE-style adaptation as a later design and implementation track.

## Non-Goals

- Do not expose custom user strategy plugins in this slice.
- Do not add a Rust/PyO3 DE variation kernel in this slice.
- Do not implement SHADE-style memory-based adaptation in this slice.
- Do not change public optimizer names or old DE lifecycle semantics.
- Do not make partial, surrogate, screened-out, or rejected policy records update
  DE target slots or adaptive strategy state.
- Do not refactor GA, CMA-ES, or shared lifecycle primitives unless a small
  defect blocks the DE strategy work.

## Public API

The public API should remain string-based:

```python
from evocore import DifferentialEvolutionOptimizer, GeneSpace

space = GeneSpace.uniform(-5.0, 5.0, 4)

base = DifferentialEvolutionOptimizer(space, strategy="rand1bin", seed=42)
best = DifferentialEvolutionOptimizer(space, strategy="best1bin", seed=42)
rand2 = DifferentialEvolutionOptimizer(space, strategy="rand2bin", seed=42)
current = DifferentialEvolutionOptimizer(
    space,
    strategy="current-to-best1bin",
    seed=42,
)
adaptive = DifferentialEvolutionOptimizer(
    space,
    strategy="jde-rand1bin",
    mutation_factor=0.5,
    crossover_rate=0.9,
    seed=42,
)
```

`mutation_factor` and `crossover_rate` remain the global defaults for stateless
strategies and the initial per-slot jDE values. The first jDE slice should avoid
adding public tuning knobs unless implementation proves they are necessary.

If tuning knobs are later needed, they should be explicit and strategy-scoped:

```python
DifferentialEvolutionOptimizer(
    space,
    strategy="jde-rand1bin",
    jde_f_probability=0.1,
    jde_cr_probability=0.1,
    jde_f_bounds=(0.1, 1.0),
)
```

Those additional knobs are not required for the first implementation.

## Architecture

Keep DE under `evocore/optimizers/de/` and add focused strategy modules:

```text
evocore/optimizers/de/
  strategies.py      # strategy registry, protocol, stateless strategies
  adaptive.py        # jDE state and future adaptive strategy state
  ask_tell.py        # lifecycle orchestration, delegates trial generation
  checkpointing.py   # includes strategy state when required
  config.py          # validates strategy specs and config signatures
```

`ask_tell.py` remains responsible for candidate lifecycle state:

- candidate and batch ledgers;
- initialization candidates;
- target slots;
- pending trial candidate IDs;
- `tell(...)` validation;
- replacement decisions;
- telemetry and events.

`strategies.py` owns stateless donor selection and trial value construction.
It should expose strategy specifications with:

```text
name
min_population_size
is_adaptive
default_parameters
checkpoint_state_schema
```

`adaptive.py` owns stateful parameter adaptation. In the first slice, that means
jDE per-slot `F` and `CR` values plus pending trial parameters. Later SHADE
state can live in this module without reshaping the optimizer API again.

`config.py` should resolve the user-provided strategy string into a strategy
spec and use that spec for validation, config signatures, and config hashes.

## Strategy Contract

The internal contract should be small and deterministic. A strategy needs enough
context to build one trial for a target slot:

```text
gene_space
population candidates
target_slot
best_slot, when strategy needs it
generation
seed
mutation_factor
crossover_rate
strategy_state, when adaptive
```

It returns:

```text
trial genes
donor slot metadata
strategy metadata
pending adaptive metadata, when adaptive
```

The contract should not create `Candidate` objects. Candidate construction,
candidate IDs, batch IDs, origins, events, and pending trial maps remain in
`ask_tell.py`.

Shared repair helpers should handle all mixed-space values:

- floats are clamped to bounds;
- ints are rounded and clamped;
- bools follow the existing deterministic binary DE behavior.

That shared repair path is part of the strategy contract, so all strategies
honor the same `GeneSpace` semantics.

## Built-In Stateless Strategies

Add these stateless strategies after the contract lands:

- `rand1bin`: preserve the existing baseline behavior.
- `best1bin`: mutate from the current best target.
- `rand2bin`: use two difference vectors.
- `current-to-best1bin`: move from the target toward the current best plus one
  difference vector.

Population-size validation should be strategy-aware:

```text
rand1bin              -> population_size >= 4
best1bin              -> population_size >= 4
rand2bin              -> population_size >= 6
current-to-best1bin   -> population_size >= 4
jde-rand1bin          -> population_size >= 4
```

If the implementation chooses a stricter donor-exclusion rule for `best1bin`,
the validation may require a larger population, but the rule must be explicit in
tests and error messages.

All built-ins should use binomial crossover in this slice. Exponential crossover
and other naming families can be considered later if there is user demand.

## jDE Adaptation

`jde-rand1bin` is the first adaptive strategy. It should use the same donor and
crossover shape as `rand1bin`, but each target slot has its own adaptive
`mutation_factor` and `crossover_rate`.

The jDE state should contain:

```text
f_by_slot: list[float]
cr_by_slot: list[float]
pending_trial_params: dict[candidate_id, {target_slot, mutation_factor, crossover_rate}]
schema_version: int
```

Before trial generation for a slot, jDE probabilistically refreshes the slot's
trial `F` and `CR`. The trial candidate records the sampled values in metadata.

After `tell(...)` determines replacement:

- if the trial replaces its target, commit the trial `F` and `CR` to that slot;
- if the trial is rejected, keep the previous slot `F` and `CR`;
- if the record is partial, surrogate, screened out, or otherwise not
  state-eligible, do not adapt;
- if the record is state-eligible cached, adapt only if the normal DE
  replacement logic accepts the trial for state.

jDE should be documented as an early adaptive strategy. SHADE should remain a
future strategy family once jDE state and checkpoint behavior are stable.

## Data Flow

Trial generation should flow through `ask(...)`:

```text
ask()
  -> choose initialization or trial phase
  -> for trial phase, resolve active DE strategy
  -> strategy builds trial genes and metadata
  -> ask_tell.py creates Candidate records and batch ledger entries
```

For stateless strategies, candidate metadata should include:

```text
target_slot
target_candidate_id
donor_slots
strategy
```

For `jde-rand1bin`, candidate metadata should also include:

```text
mutation_factor
crossover_rate
adaptive_slot
```

Optimizer state changes should still flow through `tell(...)`:

```text
tell(records)
  -> validate records
  -> update candidate scores and batch state
  -> make DE replacement decision
  -> if strategy is adaptive and record is final state-eligible:
       accepted: commit trial F/CR to slot
       rejected: keep previous slot F/CR
  -> return UpdateResult with acceptance_decisions
```

Policy-driven `run(...)` should stay strategy-agnostic. It calls `ask(...)` and
`tell(...)` as usual. Since DE only replaces target slots from final
state-eligible records, jDE should only adapt from those same records.

## Checkpointing

Existing `rand1bin` checkpoints must continue to restore. Stateless strategies
do not require strategy state. The checkpoint payload may omit
`strategy_state`, or include `strategy_state: null`, for stateless strategies.

`jde-rand1bin` checkpoints must include committed per-slot values and pending
trial parameters. This is required for checkpoints saved after `ask(...)` and
before `tell(...)`.

The jDE checkpoint state should include a schema version:

```json
{
  "strategy": "jde-rand1bin",
  "strategy_state_schema_version": 1,
  "f_by_slot": [0.5, 0.7],
  "cr_by_slot": [0.9, 0.4],
  "pending_trial_params": {
    "candidate-id": {
      "target_slot": 0,
      "mutation_factor": 0.7,
      "crossover_rate": 0.4
    }
  }
}
```

Restoring a checkpoint into an incompatible strategy should fail through the
existing optimizer config hash and strategy validation. Additional errors should
be clear when adaptive state is missing or malformed:

```text
strategy_state is required for strategy='jde-rand1bin' checkpoints.
strategy_state_schema_version 2 is not supported for strategy='jde-rand1bin'.
```

## Error Handling

Configuration errors should be raised before state mutation where possible.

Unsupported strategy names should mention the accepted values. Strategy-specific
population errors should name the strategy:

```text
population_size must be at least 6 for strategy='rand2bin'.
DifferentialEvolutionOptimizer strategy must be one of 'rand1bin', 'best1bin',
'rand2bin', 'current-to-best1bin', or 'jde-rand1bin'.
```

Evaluator errors and record validation remain in the current DE lifecycle. New
strategies must not weaken existing checks for missing records, duplicate
records, unknown candidate IDs, wrong batch IDs, invalid confidence values, bad
scores, or stale pending batches.

jDE adaptation should never occur from non-final policy stages. If an evaluator
returns state-eligible records before the final stage, the existing policy
validation should still raise `FitnessError`.

## Testing Strategy

The implementation should be test-led.

Contract preservation tests:

- seeded `rand1bin` generates the same candidates before and after the strategy
  refactor;
- `rand1bin` replacement decisions remain deterministic;
- existing `rand1bin` checkpoint fixtures continue to restore.

Strategy validation tests:

- unsupported strategy names raise `ConfigurationError`;
- each built-in validates its population-size requirement;
- config signatures and hashes change when the strategy changes;
- `rand1bin` config hashing remains stable if the public config shape is
  unchanged.

Built-in strategy tests:

- donor slots are unique and exclude the target when required;
- forced crossover index is honored;
- bounds repair is shared for float and int genes;
- bool genes follow the existing deterministic mixed-space rule;
- minimize and maximize directions identify the current best correctly for
  best-based strategies.

jDE tests:

- initial `F` and `CR` values are deterministic;
- per-slot trial parameters are deterministic for the same seed;
- accepted trials commit `F` and `CR` to the target slot;
- rejected trials preserve the previous slot values;
- partial, surrogate, rejected, and screened-out policy records do not adapt;
- final cached records adapt only when accepted by normal DE replacement logic;
- pending jDE trial checkpoints resume identically to uninterrupted runs.

Integration tests:

- one mixed `float`/`int`/`bool` search-space run for a stateless new strategy;
- one mixed search-space `jde-rand1bin` run;
- one policy-driven `jde-rand1bin` run with a two-stage budget policy;
- one `run_multiple(...)` smoke test with a non-default strategy.

## Documentation And Changelog

Update `docs/site/de.md` to describe:

- supported strategy names;
- when `rand1bin`, `best1bin`, `rand2bin`, and `current-to-best1bin` are useful;
- `jde-rand1bin` as the first adaptive strategy;
- SHADE as future work;
- checkpoint limitations and guarantees for adaptive state.

Update `docs/site/api.md` only if the public constructor signature changes.

Update `CHANGELOG.md` because adding DE strategies and jDE adaptation changes
public behavior.

If new examples are added, prefer one concise example for strategy selection and
one concise example for jDE. Avoid adding a large benchmark suite in this slice;
benchmarks can follow once the strategy semantics are stable.

## Parallel Sequencing

The work can proceed simultaneously, but integration should use hard gates:

```text
Gate 1: Contract lands
  - rand1bin refactored through strategy contract
  - deterministic behavior preserved
  - no new public strategy behavior required yet

Gate 2: Stateless strategies land
  - best1bin
  - rand2bin
  - current-to-best1bin
  - docs explain when each is useful

Gate 3: jDE lands
  - strategy_state checkpointing
  - per-slot F/CR adaptation
  - deterministic restore from pending trials
  - marked early/adaptive in docs

Gate 4: SHADE deferred
  - start only after jDE state boundaries are stable
```

Suggested branches:

```text
feature/de-strategy-contract
feature/de-built-in-strategies
feature/de-jde-adaptation
```

`feature/de-strategy-contract` should merge first. The built-in and jDE work can
develop on top of it or rebase after it lands.

## Success Criteria

- `rand1bin` remains the deterministic baseline and does not regress.
- New strategies are opt-in by public string name.
- Strategy-specific validation, config signatures, and docs are consistent.
- jDE state is contained in the adaptive strategy layer.
- jDE pending trial checkpoints resume deterministically.
- Policy-driven runs remain strategy-agnostic.
- SHADE is clearly documented as later work, not partially implemented in this
  slice.
