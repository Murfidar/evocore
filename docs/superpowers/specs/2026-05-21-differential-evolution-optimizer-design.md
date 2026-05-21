# DifferentialEvolutionOptimizer And Acceptance Contract Design

## Summary

Add `DifferentialEvolutionOptimizer` as a modern EvoCore optimizer with the same
domain-oriented public shape as the existing GA and CMA-ES optimizers. The first
implementation should support mixed `float`, `int`, and `bool` `GeneSpace`
instances, manual ask/tell workflows, synchronous evaluator-driven runs, stable
ask/tell checkpointing, reproducibility metadata, telemetry, events, docs, and
tests.

This work also introduces a shared optimizer acceptance contract. Existing
`UpdateResult.accepted_count` keeps its current meaning: the number of
evaluation records accepted into the lifecycle ledger. A new per-record decision
describes whether a candidate was accepted into optimizer state. For DE, that
boolean is the central replacement decision: whether a trial replaced its target
slot.

## Goals

- Add `DifferentialEvolutionOptimizer` under `evocore/optimizers/de/`.
- Keep the public API aligned with domain vocabulary and existing optimizer
  conventions.
- Support flat `GeneSpace` values containing floats, ints, bools, and fixed
  numeric genes.
- Use a hybrid per-kind DE implementation:
  - floats use classic DE arithmetic variation,
  - ints use classic DE arithmetic variation followed by round and clamp,
  - bools use DE crossover masks with binary inheritance or deterministic flip
    behavior inspired by GA,
  - fixed numeric genes remain fixed.
- Add stable ask/tell checkpointing from the first release of the optimizer.
- Add a shared acceptance contract without breaking the existing
  `accepted_count` meaning.
- Keep the v1 implementation Python-only unless later performance work justifies
  Rust/PyO3 helpers.

## Non-Goals

- Do not add GA-style legacy pickle checkpoints for DE.
- Do not add `run_multiple(...)` in the first DE slice.
- Do not add custom DE strategy plugins yet.
- Do not add a Rust DE kernel in v1.
- Do not promise mid-loop resume for synchronous `run(...)`; stable manual
  ask/tell checkpoints are the v1 continuation boundary.

## Architecture

Add a new package:

```text
evocore/
  optimizers/
    de/
      __init__.py
      engine.py
      ask_tell.py
      checkpointing.py
      config.py
```

Responsibilities:

- `engine.py`: public constructor, state summary, config/reproducibility hooks,
  and synchronous run orchestration.
- `ask_tell.py`: DE proposal generation, record acceptance, target replacement,
  event emission, and update summaries.
- `checkpointing.py`: stable ask/tell checkpoint and resume helpers.
- `config.py`: stable optimizer config signatures, compatibility validation,
  and reproducibility hook summaries.
- `__init__.py`: convenience export only.

Public imports should expose `DifferentialEvolutionOptimizer` from
`evocore.optimizers.de`, `evocore.optimizers`, and `evocore`.

## Public Constructor

The constructor should use conservative DE defaults:

```python
DifferentialEvolutionOptimizer(
    gene_space,
    population_size=50,
    max_generations=300,
    mutation_factor=0.8,
    crossover_rate=0.9,
    strategy="rand1bin",
    parallel="none",
    n_workers=None,
    process_initializer=None,
    process_initargs=(),
    seed=0,
    direction="maximize",
    max_evaluations=None,
    track_diversity=False,
    callbacks=None,
)
```

Validation:

- `gene_space` is required.
- `population_size` must be large enough for the selected strategy. For
  `rand1bin`, require at least 4 targets: one target plus three distinct donor
  indices.
- `mutation_factor` must be finite and non-negative.
- `crossover_rate` must be in `[0, 1]`.
- `max_generations >= 0`.
- `max_evaluations` is positive when provided.
- `parallel` is one of `"none"`, `"thread"`, or `"process"`.
- `direction` is `"maximize"` or `"minimize"`.

## Algorithm And Data Flow

DE starts in an initialization phase. The first `ask(population_size)` proposes
random candidates using the existing `GeneSpace`/codec behavior. `tell()` stores
valid records and fills target slots in ask order when records are state
eligible (`trusted_full` or `cached`). Once the target population is full, later
`ask()` proposes one trial candidate per target slot.

Trial creation uses `rand/1/bin` by default:

```text
mutant = a + mutation_factor * (b - c)
trial[j] = mutant[j] when the crossover mask selects j, otherwise target[j]
```

For mixed spaces:

- Float genes use arithmetic DE and clamp to bounds.
- Int genes use arithmetic DE, then round and clamp.
- Bool genes use the crossover mask but produce actual boolean values through a
  binary rule. If the crossover mask does not select the gene, keep the target
  bool. If it does select the gene, start from donor `a`; when donors `b` and
  `c` differ, flip donor `a` with probability `min(1.0, mutation_factor)` using
  deterministic seeded sampling. This mirrors DE's base-plus-difference shape
  without treating booleans as continuous values.
- Fixed numeric genes remain fixed.

Each proposed trial records its target slot and target candidate ID in optimizer
state. That mapping is required for replacement and checkpoint resume.

## Acceptance Contract

Preserve current `UpdateResult.accepted_count` semantics: it is the number of
records validated and stored by `tell()`.

Add a shared lifecycle dataclass:

```python
@dataclass(frozen=True)
class AcceptanceDecision:
    candidate_id: str
    batch_id: str
    accepted_for_state: bool
    reason: str
    target_candidate_id: str | None = None
    target_slot: int | None = None
```

Extend `UpdateResult` with defaulted fields:

```python
acceptance_decisions: tuple[AcceptanceDecision, ...] = ()
state_accepted_count: int = 0
```

Semantics:

- `accepted_count`: records accepted into the API ledger.
- `state_accepted_count`: candidates accepted into optimizer state.
- `accepted_for_state`: per-candidate boolean state acceptance.
- For DE, `accepted_for_state=True` means a trial replaced its target slot.
- For DE, `accepted_for_state=False` means the record was valid but the target
  stayed.
- For GA and CMA-ES, populate decisions for state-eligible records where the
  optimizer state is updated, so the contract is consistent across optimizers.

DE tell events should include acceptance details in `metadata`:

```python
{
    "accepted_for_state": True,
    "acceptance_reason": "trial_replaced_target",
    "target_candidate_id": "...",
    "target_slot": 3,
}
```

Do not add new top-level event fields in this slice.

## Tell Semantics

`tell()` remains asynchronous-friendly:

- It accepts any subset of records for known candidates.
- It rejects unknown candidates, unknown explicit batch IDs, batch mismatches,
  duplicate candidate/stage records, invalid scores, and invalid rejected
  records with `FitnessError`.
- `tell([])` is a valid no-op.
- Partial and surrogate records update lifecycle observations and telemetry but
  do not replace targets.
- Rejected records eliminate the trial candidate but do not replace targets.
- Trusted and cached records can participate in target replacement.
- A DE replacement accepts the trial when its direction-aware comparison score is
  greater than or equal to the target score.

## Checkpointing

Add `DifferentialEvolutionCheckpointingMixin` modeled on GA and CMA-ES ask/tell
checkpointing.

The checkpoint envelope should use:

- `optimizer_type`: `"DifferentialEvolutionOptimizer"`
- `state_kind`: `"de_ask_tell"`
- `schema_version`: `1`
- standard stable identity fields: gene-space hash, optimizer config hash, seed,
  and direction

Shared runtime state:

- `event_index`
- `candidates_by_id`
- `batches_by_id`
- `best_candidate_id`
- `telemetry`
- `events`

DE-specific runtime state:

- ordered `target_candidate_ids`
- pending mapping from `trial_candidate_id` to `target_slot`
- pending mapping from `trial_candidate_id` to `target_candidate_id`
- `generation`, incremented after each completed replacement sweep

Resume must validate that every target ID and pending trial mapping references a
known candidate. A checkpoint taken after `ask()` but before every trial record
returns must restore enough state for later `tell()` calls to make the same
replacement decisions.

## Synchronous Run API

The v1 `run(...)` API should follow the newer evaluator style:

```python
result = optimizer.run(evaluator, policy=None)
```

The evaluator implements:

```python
def evaluate(candidates, context):
    return records
```

The run loop should use ask/tell internally, produce `OptimizationResult`, append
generation summaries, invoke callbacks, and honor `max_generations` and optional
`max_evaluations`. Manual ask/tell checkpointing remains the supported resume
boundary; synchronous mid-loop resume is out of scope.

Parallel evaluation should support `"none"`, `"thread"`, and `"process"` for the
synchronous run path, with the same pickling expectations as GA for process
mode.

## Events, Telemetry, And Results

DE should emit append-only `ask`, `tell`, generation, and `run_stop` events using
the existing `EventHistory` surface.

Telemetry should reuse `OptimizationTelemetry`. DE-specific replacement counts
can be represented through acceptance decisions and event metadata in v1. If
later dashboards need aggregate replacement counters, add explicit telemetry
fields in a follow-up.

`OptimizationResult` should include:

- `optimizer_type="DifferentialEvolutionOptimizer"`
- best solution and best candidate ID
- final target population as `final_solutions`
- generation history
- event history
- telemetry
- reproducibility metadata

## Documentation And Changelog

Update user-visible documentation because this adds public API and behavior:

- API docs for `DifferentialEvolutionOptimizer`.
- A new DE guide with a numeric example and mixed bool/numeric example.
- Ask/tell docs explaining `AcceptanceDecision`,
  `state_accepted_count`, and `accepted_for_state`.
- Checkpointing docs with DE ask/tell checkpoint/resume examples.
- Gene-space docs noting DE support for bool and mixed spaces.
- `CHANGELOG.md`.

Update `evocore/_core.pyi` only if the implementation adds or changes public
Rust-backed exports. The current design does not require that.

## Testing

Use test-driven implementation around the public contracts:

- `AcceptanceDecision` and `UpdateResult` defaults preserve existing callers.
- Public import surfaces expose `DifferentialEvolutionOptimizer`.
- Config validation, config signatures, and config hashes behave deterministically.
- Initialization ask/tell fills target slots in deterministic order.
- Trial ask/tell replaces targets on better or equal trusted/cached records.
- Trial ask/tell keeps targets on worse trusted/cached records.
- Minimize direction uses the correct comparison behavior.
- Partial, surrogate, cached, trusted, and rejected records produce expected
  lifecycle, telemetry, and pending-batch behavior.
- Unknown candidates, unknown batch IDs, batch mismatches, and duplicates raise
  `FitnessError`.
- Mixed spaces decode bool genes as real Python `bool`, ints as `int`, and fixed
  genes as fixed values.
- Same seed and config reproduce the same ask/tell sequence.
- Checkpoint resume works after initialization ask, partial initialization tell,
  trial ask, partial trial tell, and completed replacement sweeps.
- Integration tests cover numeric sphere/Rastrigin-style objectives and a mixed
  bool/numeric objective.

Verification for the implementation should include:

- `.venv` Python ruff format check and ruff check.
- `python -m maturin develop --release`.
- Unit and integration pytest.
- Rust checks only if the implementation touches Rust.
