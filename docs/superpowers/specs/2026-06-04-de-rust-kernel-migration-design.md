# Differential Evolution Rust Proposal Kernel Migration Design

**Date:** 2026-06-04
**Status:** Design approved for specification
**Scope:** Move Differential Evolution trial proposal math into a Rust/PyO3
kernel while keeping optimizer lifecycle, checkpoints, and public semantics in
Python.

## Summary

`DifferentialEvolutionOptimizer` is currently Python-owned. Python manages the
public class, ask/tell lifecycle, target population state, built-in strategy
math, jDE adaptation state, checkpointing, events, telemetry, policy-driven
execution, callbacks, and evaluator integration. Rust provides shared helpers
such as deterministic seed derivation, candidate IDs, and initialization
sampling.

The long-term target is to migrate DE proposal generation into Rust without
moving the optimizer state machine across the PyO3 boundary. Rust should own
the deterministic inner-loop math for all current built-in strategies. Python
should continue to own product semantics and lifecycle behavior.

The recommended architecture is a single Rust batch proposal kernel exposed
through `_core.de_generate_trials(...)`. The kernel returns trial proposals and
metadata, not `Candidate` objects. Python wraps those proposals into candidates,
stores trial-to-target mappings, records events, owns jDE pending state, and
applies replacement decisions in `tell(...)`.

## Goals

- Add one Rust DE proposal kernel for all current strategies:
  - `rand1bin`
  - `best1bin`
  - `rand2bin`
  - `current-to-best1bin`
  - `jde-rand1bin`
- Keep the public DE constructor, `ask(...)`, `tell(...)`, `run(...)`, and
  `run_multiple(...)` APIs stable.
- Keep Python as the owner of candidates, batches, target replacement,
  checkpoints, telemetry, events, policies, callbacks, and evaluator calls.
- Move batch trial generation, donor sampling, crossover masks, mixed repair,
  strategy recipes, and jDE trial-parameter proposal into Rust.
- Preserve deterministic behavior for the same EvoCore version, seed, config,
  generation, direction, and search space.
- Keep jDE committed and pending adaptive state in Python, with Rust only
  proposing trial-level `mutation_factor` and `crossover_rate`.
- Update `_core.pyi` for the new Rust-backed export.
- Document the kernel migration and any intentional seeded sequence change.

## Non-Goals

- Do not move `DifferentialEvolutionOptimizer` into Rust.
- Do not move `Candidate`, `CandidateBatch`, events, telemetry, policy
  execution, callbacks, or evaluator integration into Rust.
- Do not make Rust own target replacement decisions.
- Do not make Rust own checkpoint envelopes or checkpoint identity validation.
- Do not add a public runtime flag that permanently keeps two DE
  implementations.
- Do not expose a broad custom DE strategy plugin system in this slice.
- Do not implement SHADE-style adaptation.

## Architecture

Add a focused Rust module:

```text
src/de.rs
```

Expose one PyO3 function from `src/lib.rs`:

```python
_core.de_generate_trials(
    population,
    scores,
    gene_bounds,
    gene_kinds,
    strategy,
    mutation_factor,
    crossover_rate,
    seed,
    generation,
    target_slots,
    direction,
    jde_state=None,
)
```

For `jde-rand1bin`, `jde_state` should be a small mapping containing the
committed per-slot values:

```python
{
    "f_by_slot": [0.5, 0.7],
    "cr_by_slot": [0.9, 0.4],
}
```

The function returns a list of proposal dictionaries. Each proposal contains
encoded trial genes and metadata needed by Python:

```python
[
    {
        "target_slot": 0,
        "genes": [0.1, 4.0, 1.0],
        "metadata": {
            "strategy": "rand1bin",
            "target_slot": 0,
            "base_slot": 2,
            "donor_slots": [2, 4, 5],
            "difference_pairs": [[4, 5]],
        },
    },
]
```

For `jde-rand1bin`, Rust also returns the proposed adaptive parameters:

```python
{
    "metadata": {
        "strategy": "jde-rand1bin",
        "adaptive_slot": 0,
        "mutation_factor": 0.73,
        "crossover_rate": 0.41,
        "base_slot": 2,
        "donor_slots": [2, 4, 5],
        "difference_pairs": [[4, 5]],
    }
}
```

Rust owns proposal math. Python owns optimizer state.

## Rust Responsibilities

`src/de.rs` should own:

- strategy dispatch for every current built-in DE strategy;
- population-size validation at kernel level;
- input shape validation:
  - population rows match gene count;
  - scores match population size;
  - bounds and kinds match gene count;
  - target slots are valid;
- donor sampling without replacement;
- target exclusion rules;
- direction-aware best-slot selection;
- forced crossover index selection;
- binomial crossover masks;
- float mutation arithmetic;
- int mutation arithmetic followed by round and clamp;
- bool difference-pair flip behavior;
- fixed gene preservation;
- encoded repair of every generated value;
- jDE deterministic `mutation_factor` and `crossover_rate` proposal;
- metadata construction for Python events and debugging.

Rust should return metadata for:

- `strategy`
- `target_slot`
- `base_slot`
- `best_slot`, when used
- `donor_slots`
- `difference_pairs`
- `mutation_factor`, when strategy-specific or adaptive
- `crossover_rate`, when strategy-specific or adaptive
- `adaptive_slot`, for jDE

Rust should not own:

- `Candidate`
- `CandidateBatch`
- candidate IDs
- target replacement decisions
- accepted or rejected jDE commits
- telemetry
- events
- checkpoint envelopes
- `BudgetPolicy`
- callbacks
- evaluator calls
- public optimizer classes

## Python Integration

Keep the DE package shape:

```text
evocore/optimizers/de/
  __init__.py
  adaptive.py
  ask_tell.py
  checkpointing.py
  config.py
  engine.py
  multi_run.py
  strategies.py
```

`strategies.py` should remain the Python registry and compatibility layer. It
should keep:

- `DEStrategySpec`
- `SUPPORTED_DE_STRATEGIES`
- `supported_strategy_names`
- `strategy_spec_for`
- `validate_strategy_population_size`

The Python strategy math helpers should be removed from the runtime path once
the Rust kernel is integrated. If parity tests need the old behavior during the
migration, keep that comparison code in tests or private test fixtures rather
than preserving a second production implementation.

`ask_tell.py` should gain a focused wrapper that builds Rust inputs and converts
Rust outputs back into `TrialProposal`-like data for candidate construction.
The wrapper should:

- encode the current target population;
- build direction-aware target scores;
- pass the current strategy name and parameters;
- pass jDE committed `f_by_slot` and `cr_by_slot` when the strategy is
  `jde-rand1bin`;
- call `_core.de_generate_trials(...)`;
- decode and validate returned genes;
- preserve existing candidate construction;
- preserve `trial_candidate_id -> target_slot` mappings;
- preserve `trial_candidate_id -> target_candidate_id` mappings;
- register pending jDE trial parameters in Python.

`adaptive.py` remains Python-owned. Python passes committed jDE state to Rust,
Rust proposes trial parameters, and Python commits or discards them after
replacement in `tell(...)`.

## Data Flow

Initialization remains unchanged:

```text
ask()
  -> target population not full
  -> _core.init_population(...)
  -> Python decodes values
  -> Python creates Candidate objects
```

Trial generation changes:

```text
ask()
  -> target population full
  -> Python builds encoded population and scores
  -> Python calls _core.de_generate_trials(...)
  -> Rust returns encoded trial proposals and metadata
  -> Python decodes and validates genes
  -> Python creates Candidate objects
  -> Python stores trial-to-target mappings
  -> Python registers pending jDE parameters when needed
```

State updates remain unchanged:

```text
tell(records)
  -> validate candidate, batch, and record
  -> apply record to candidate
  -> compare trial against target
  -> accept or reject replacement
  -> commit or discard jDE params
  -> update telemetry and events
  -> return UpdateResult
```

Checkpointing remains Python-owned:

```text
ask_tell_checkpoint()
  -> stores Python candidate, batch, event, and telemetry state
  -> stores target population IDs
  -> stores pending trial mappings
  -> stores Python-owned jDE committed and pending params
```

## Compatibility And Determinism

The public API should remain stable:

- same public constructor;
- same strategy names;
- same `ask(...)` and `tell(...)` APIs;
- same `UpdateResult` fields;
- same checkpoint envelope shape unless the implementation intentionally bumps
  the DE checkpoint schema;
- same jDE checkpoint structure, because Python still owns jDE state.

The migration should preserve deterministic behavior, but it does not need to
preserve exact Python `random.Random` trial sequences. Duplicating Python's RNG
and sampling behavior in Rust would be brittle. The stronger long-term contract
should be deterministic output within the Rust kernel version for the same
seed, config, generation, direction, and search space.

If generated trial values change from prior Python implementation, document the
change in `CHANGELOG.md` and update affected golden fixtures intentionally.
Old checkpoints should still restore when their payload shape is valid. After
restore, the next generated trial batch may use the new Rust kernel sequence.

Keep the DE ask/tell checkpoint schema version unchanged if the checkpoint
payload shape is unchanged. Bump the schema only if implementation requires new
payload fields or changes the meaning of saved fields. Seeded trial sequence
drift alone should be documented in the changelog, not treated as a checkpoint
payload schema change.

## Error Handling

Rust should validate low-level kernel inputs and return clear PyO3 errors for:

- unknown strategy names;
- unsupported gene kinds;
- invalid population size for a strategy;
- mismatched population, score, bounds, and kind lengths;
- invalid target slots;
- non-finite scores where best-slot selection requires them;
- invalid jDE state payloads.

Python should continue to raise user-facing `ConfigurationError`,
`FitnessError`, and `CheckpointError` at lifecycle boundaries. Python wrapper
code should map Rust errors into existing public error conventions where
practical.

## Testing Strategy

Use a test-led migration.

Rust unit tests should cover:

- every built-in strategy;
- donor uniqueness and target exclusion;
- best-slot selection for maximize and minimize;
- forced crossover index behavior;
- binomial crossover mask behavior;
- bounds repair for float, int, bool, and fixed genes;
- deterministic output for the same inputs;
- jDE F/CR proposal determinism.

Python tests should cover:

- `_core.de_generate_trials(...)` signature and stub behavior;
- valid proposal conversion through the DE ask wrapper;
- metadata fields for every strategy;
- mixed-space decoded values as real `float`, `int`, and `bool`;
- jDE pending parameter registration after `ask(...)`;
- jDE commit/discard behavior after `tell(...)`;
- all strategies through ask/tell;
- checkpoint restore after trial ask;
- jDE checkpoint restore with pending params;
- policy-driven runs with Rust-generated trials;
- `run_multiple(...)` with non-default strategies;
- deterministic same-seed behavior under the new Rust kernel.

Fixture tests should be updated intentionally if exact trial continuation values
change. Tests should distinguish stable checkpoint payload compatibility from
old Python RNG sequence compatibility.

## Documentation And Changelog

Update docs because DE behavior and implementation guarantees are user-visible:

- `docs/site/de.md` should note that trial proposals are Rust-backed.
- Checkpoint docs should mention any checkpoint schema change if one occurs.
- `CHANGELOG.md` should describe the Rust DE proposal kernel migration.
- If seeded trial values change, the changelog should state that DE remains
  deterministic for the same EvoCore version/config/seed, but exact generated
  trial sequences may differ from the pre-kernel Python implementation.

Update `evocore/_core.pyi` when the Rust function is added.

## Acceptance Criteria

- DE trial generation for all current strategies is produced by the Rust kernel.
- Python remains the owner of candidate lifecycle, target replacement,
  checkpoints, events, telemetry, policies, callbacks, and evaluators.
- jDE committed and pending adaptive state remains checkpointed in Python.
- `ask(...)`, `tell(...)`, `run(...)`, and `run_multiple(...)` continue to work
  for all DE strategies.
- Mixed search spaces remain valid through trial generation and replacement.
- Same-seed runs are deterministic under the new Rust kernel.
- Checkpoint compatibility behavior is explicitly tested and documented.
- Rust checks, Python linting, maturin develop, and DE-focused tests pass before
  implementation is committed.
