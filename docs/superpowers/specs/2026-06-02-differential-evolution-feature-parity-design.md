# Differential Evolution Feature Parity Sprint Design

**Date:** 2026-06-02
**Status:** Design approved for specification
**Scope:** Add DE feature parity for multi-run execution, policy-driven
evaluation, and richer documentation examples while keeping strategy expansion
out of this sprint.

## Summary

`DifferentialEvolutionOptimizer` has been stabilized as a public optimizer with
deterministic ask/tell behavior, synchronous evaluator-driven `run()`, and
stable manual ask/tell checkpoints. The next sprint should move DE closer to the
long-term EvoCore optimizer lifecycle: policy-driven evaluation, reusable
multi-run execution, and examples that show how DE fits callbacks and
checkpointing.

The long-term direction is that `run(evaluator, policy=None)` should be the
canonical execution model for optimizers that EvoCore drives directly.
`policy=None` resolves to a single full-evaluation budget, while explicit
`BudgetPolicy` instances can screen candidates through cheap, partial, or
surrogate stages before final state-eligible evaluation. Manual `ask()`/`tell()`
remains the durable boundary for external evaluation systems and checkpoint
resume.

DE should follow that model now. It should support `run_multiple(...)`, make
`run(...)` policy-driven, and delay DE target-slot replacement until a candidate
reaches a state-eligible final stage. New DE strategies remain future work.

## Current Context

The stabilization design in
`docs/superpowers/specs/2026-06-02-differential-evolution-stabilization-design.md`
explicitly deferred:

- `DifferentialEvolutionOptimizer.run_multiple(...)`;
- policy-aware `DifferentialEvolutionOptimizer.run(...)`;
- richer callback and checkpoint examples;
- additional DE strategy options.

The current DE implementation lives in:

```text
evocore/optimizers/de/
  __init__.py
  ask_tell.py
  checkpointing.py
  config.py
  engine.py
```

`engine.py` currently rejects `run(..., policy=...)`. `config.py` only accepts
`strategy="rand1bin"`. The DE docs still list `run_multiple(...)`,
policy-aware `run(...)`, custom strategy plugins, and Rust-backed variation as
future limitations.

GA already exposes `run_multiple(...)` and has a policy-driven vNext `run()`
through its ask/tell mixin. CMA-ES has manual ask/tell checkpoints but its
public `run()` remains a classic generation loop. For this sprint, DE should
align with the cleaner long-term policy-driven model rather than copy every
historical difference between GA and CMA-ES.

## Goals

- Add `DifferentialEvolutionOptimizer.run_multiple(...)` with deterministic
  child seed derivation and `OptimizationBatchResult` aggregation.
- Make `DifferentialEvolutionOptimizer.run(evaluator, policy=None)` policy
  driven.
- Resolve `policy=None` to `BudgetPolicy.single_full(...)`.
- Support explicit multi-stage `BudgetPolicy` screening.
- Preserve DE target-slot semantics by allowing replacement only from final
  state-eligible records.
- Keep manual ask/tell checkpoints as the supported resume boundary.
- Add richer DE docs and examples for budget-aware runs, multi-run execution,
  callbacks, and manual checkpointing.
- Update user-visible docs and changelog entries for the public behavior change.

## Non-Goals

- Do not add new DE strategies in this sprint. `rand1bin` remains the only
  supported strategy.
- Do not add custom strategy plugins or a strategy abstraction.
- Do not add a Rust/PyO3 DE variation kernel.
- Do not refactor GA, CMA-ES, or shared lifecycle primitives unless DE policy
  integration exposes a small defect that must be fixed.
- Do not add policy-driven mid-loop checkpoint resume.
- Do not change public names away from `DifferentialEvolutionOptimizer`,
  `BudgetPolicy`, `EvaluationStage`, `UpdateResult`, or
  `OptimizationBatchResult`.

## Public API

DE should expose these public execution paths:

```python
from evocore import (
    BudgetPolicy,
    DifferentialEvolutionOptimizer,
    EvaluationStage,
    GeneSpace,
)

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = DifferentialEvolutionOptimizer(space, population_size=20, seed=42)

single = optimizer.run(SphereEvaluator())

policy = BudgetPolicy(
    stages=[
        EvaluationStage(
            "cheap",
            budget=0.1,
            promote_fraction=0.5,
            confidence="partial",
        ),
        EvaluationStage(
            "full",
            budget=1.0,
            promote_fraction=1.0,
            confidence="trusted_full",
        ),
    ],
    max_evaluations=80,
    batch_size=20,
)
budgeted = optimizer.run(SphereEvaluator(), policy=policy)

batch = optimizer.run_multiple(SphereEvaluator(), n_runs=5)
```

`run_multiple(...)` should follow the GA result shape:

```python
def run_multiple(
    self,
    evaluator: Evaluator,
    n_runs: int = 10,
    aggregate: str = "best",
    run_parallel: bool = False,
) -> OptimizationBatchResult: ...
```

`aggregate` should accept `"best"` and `"all"` for parity with GA. Results
should be sorted in direction-aware best-first order.

## Architecture

DE should stay under `evocore/optimizers/de/`.

Add a focused multi-run module:

```text
evocore/optimizers/de/multi_run.py
```

That module should own child seed derivation, `_copy_with_seed(...)`, spawned
process execution for `run_parallel=True`, pickling validation, sorting, and
`OptimizationBatchResult` construction. It can mirror GA's multi-run pattern
without importing GA internals.

Policy-driven `run()` can initially remain in `engine.py`, because that file
already owns evaluator orchestration, callbacks, generation history, result
construction, and DE-specific stop reasons. If the implementation becomes
bulky, split private helpers into:

```text
evocore/optimizers/de/policy_run.py
```

`ask_tell.py` should continue to own candidate lifecycle state, target-slot
mapping, telemetry updates, events, and replacement decisions. Policy helpers
may call `ask()` and `tell()`, but they should not duplicate replacement logic.

`config.py` should keep validation and config signatures stable. It should
continue to reject unsupported strategies.

## Budget Resolution

`run(evaluator, policy=None)` should resolve its execution budget as follows:

1. If `policy` is provided, it must be a `BudgetPolicy`; use
   `policy.max_evaluations` and `policy.batch_size`.
2. If `policy` is omitted and `self.max_evaluations` is set, create
   `BudgetPolicy.single_full(max_evaluations=self.max_evaluations,
   batch_size=self.population_size)`.
3. If both are omitted, create a single-full policy with enough budget for
   initialization plus configured trial generations:
   `population_size * (max_generations + 1)`.

The long-term model is policy-owned budgets. The constructor
`max_evaluations` remains a shorthand only when no explicit policy is supplied.
Docs should make this precedence clear.

Fresh `trusted_full` records spend the full-evaluation budget. `cached` records
can update optimizer state but do not spend fresh full-evaluation budget.
Partial, surrogate, and rejected records never spend full-evaluation budget.

## Policy-Driven Lifecycle

The DE run loop should use the same policy vocabulary as GA while preserving
DE's target-slot replacement semantics.

### Initialization

DE needs a trusted target population before it can produce trials.

The policy runner should:

1. Ask initialization candidates while fewer than `population_size` target
   members exist and fresh full-evaluation budget remains.
2. Evaluate each active candidate batch through policy stages in order.
3. Call `tell(records)` for each evaluator response so telemetry, scores,
   statuses, and events remain observable.
4. Use `BudgetScheduler.promote(...)` after non-final stages to select active
   candidates for the next stage.
5. Admit candidates into the target population only when they receive a
   state-eligible final record.

If screening eliminates initialization candidates before the final stage, those
candidates do not become target members. The runner should keep asking more
initialization candidates until the target population is full or budget is
exhausted.

### Trial Generations

Once initialized, each generation should:

1. Fire `on_generation_start(gen, population)`.
2. Ask trial candidates, generally up to `population_size` or remaining budget.
3. Evaluate trials through policy stages.
4. Promote candidates between non-final stages with `BudgetScheduler`.
5. Apply DE replacement only when final state-eligible records are told.
6. Leave a target slot unchanged when its trial is eliminated before final
   evaluation.
7. Append a `GenerationRecord` after final-stage trial records are applied.
8. Fire `on_generation_end(gen, population, info)`.

The key invariant is:

```text
partial, surrogate, and rejected records may affect telemetry and promotion,
but they must never replace a DE target slot.
```

### Candidate And Batch Closure

Policy screening can eliminate candidates after a non-final stage. The
implementation must close those candidates cleanly so the DE batch ledger does
not report stale pending batches after a synchronous `run()`.

The preferred design is to add a small private policy-run helper that marks
screened-out candidates terminal for the current run without creating a fake
state update. It should preserve auditability through events or metadata and
must not make eliminated candidates eligible for target population state.

If implementation uses synthetic rejection records, those records must avoid
duplicate `(candidate_id, stage)` collisions with the real evaluator records
already stored by `CandidateBatch`.

## Callbacks

DE callbacks should remain generation-oriented:

- `on_generation_start(gen, population)` fires after initialization is complete
  and before trials are proposed for generation `gen`.
- `on_generation_end(gen, population, info)` fires after final state-eligible
  trial records have been applied for generation `gen`.
- `on_run_end(result)` fires after `OptimizationResult` construction.

`GenerationInfo` should report fresh full evaluations for the generation and
preserve the existing callback shape. Cached, rejected, and screened counts
should continue to be available through telemetry and events where the existing
result model supports them.

DE should not advertise `CheckpointCallback` for policy-driven `run()` unless a
stable generation-loop checkpoint factory is intentionally added. This sprint
should instead show ordinary run callbacks plus manual ask/tell checkpoint
examples.

## Checkpointing

Manual ask/tell checkpoints remain the supported DE resume boundary:

```python
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "de-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)
restored = DifferentialEvolutionOptimizer(space, population_size=6, seed=42)
restored.resume_ask_tell_checkpoint("de-ask-tell.evocore-checkpoint.json")
restored.tell(records)
```

Policy-driven `run(evaluator, policy=...)` mid-loop resume remains out of
checkpoint v1. Docs should say that completed `OptimizationResult` exports and
event rows are audit/export data, not checkpoint replay inputs.

The new policy-driven run path must not weaken existing manual ask/tell
checkpoint identity validation, including optimizer type, state kind, schema
version, gene-space hash, config hash, seed, and direction.

## Error Handling

Configuration errors should raise `ConfigurationError` before state mutation
where possible.

- Unsupported strategies remain rejected.
- Invalid population size, generation count, mutation factor, crossover rate,
  parallel mode, direction, and budget values remain rejected.
- `policy` must be a `BudgetPolicy` when provided.
- Explicit `policy.max_evaluations` takes precedence over constructor
  `max_evaluations`.

Evaluator errors should raise `FitnessError`.

- `run()` should validate that evaluator records match the assigned candidates.
- Missing records, duplicate records, unknown candidate IDs, wrong batch IDs,
  invalid confidence values, bad scores, and incomplete final-stage records
  should fail explicitly.
- If a final stage returns no fresh `trusted_full` records and no state-eligible
  cached records, the run should fail instead of spinning forever.
- If a non-final stage eliminates all active trial candidates, DE should leave
  all target slots unchanged and proceed to the next ask/generation if budget
  remains.

If budget is exhausted during initialization, return a result that makes the
partial state clear. If at least one state-eligible candidate exists, report
the best known trusted candidate and `stop_reason="max_evaluations"`. If no
state-eligible candidate exists, return a defensive empty-result shape
consistent with existing EvoCore result conventions and cover it with tests.

If budget is exhausted after initialization but before any trial generation,
return the initialized target population as `final_solutions` with
`stop_reason="max_evaluations"`.

## Testing Strategy

The implementation should be test-led.

Focused unit tests should cover:

- `run_multiple(...)` derives deterministic child seeds.
- `run_multiple(...)` returns an `OptimizationBatchResult` sorted best-first.
- `run_multiple(..., run_parallel=True)` enforces picklability like GA.
- `run(policy=None)` resolves to a single-full policy.
- Constructor `max_evaluations` is used only when explicit policy is absent.
- Explicit `BudgetPolicy.single_full(...)` honors `policy.max_evaluations`.
- Multi-stage policy evaluates stages in order.
- Multi-stage policy promotes candidates according to `BudgetScheduler`.
- Partial, surrogate, and rejected records do not update target slots.
- Final `trusted_full` records can initialize targets and replace trial targets.
- Final `cached` records can update state without spending fresh full budget.
- Eliminated trial candidates leave their target slots unchanged.
- Initialization continues asking until the target population is full or budget
  is exhausted.
- Same seed, config, and policy produce deterministic result summaries.
- Callback order and generation counts remain stable.
- Invalid evaluator records raise `FitnessError`.
- Policy-screened runs do not leave stale pending batches in completed results.

Integration tests should cover at least one mixed `float`/`int`/`bool` DE run
with a two-stage budget policy.

Docs/API smoke tests should cover the public DE import surface and examples that
are intended to remain copyable.

## Documentation

Update `docs/site/de.md` to remove the current limitation for
`run_multiple(...)` and policy-aware `run(...)`. Keep the limitation for custom
strategy plugins and Rust-backed variation kernels.

Add or update examples:

- `examples/budgeted_de.py` for a two-stage budget-aware evaluator.
- A docs snippet for `run_multiple(...)`.
- A docs snippet for callbacks with policy-driven DE `run()`.
- A manual ask/tell checkpoint example with pending initialization or trial
  candidates.

Update `docs/site/budget-aware-optimization.md` so DE appears alongside GA as a
policy-driven optimizer. Keep CMA-ES wording accurate: CMA-ES manual ask/tell
can consume state-eligible records, but policy-driven `run(...)` is not part of
this sprint.

Update `docs/site/callbacks-checkpointing.md` to clarify:

- manual DE ask/tell checkpoints are supported;
- policy-driven DE mid-loop resume is unsupported;
- `CheckpointCallback` should not be shown as a DE synchronous-run feature
  unless implementation adds a stable checkpoint factory.

Update `CHANGELOG.md` because this sprint changes public DE behavior.

## Verification

Run the smallest reliable DE-focused tests first while implementing, then
broaden before commit.

Expected final verification commands are:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Rust verification is not expected unless implementation touches Rust, PyO3
stubs, or cross-language contracts. If Rust is touched, also run the Rust
formatting, linting, and test commands from `AGENTS.md`.

For the design-doc-only commit, `git diff --check` is sufficient verification.

## Acceptance Criteria

- `DifferentialEvolutionOptimizer.run_multiple(...)` is available and
  deterministic.
- `DifferentialEvolutionOptimizer.run(evaluator, policy=None)` uses
  `BudgetPolicy.single_full(...)`.
- Explicit multi-stage policies screen and promote DE candidates.
- DE target-slot replacement happens only from final state-eligible records.
- Screened-out candidates do not leave completed policy runs with stale pending
  batch state.
- Callback ordering remains generation-oriented and documented.
- Manual ask/tell checkpoints remain the documented resume path.
- DE docs and examples describe budget-aware runs, multi-run execution,
  callbacks, checkpointing, and remaining strategy limitations.
- `CHANGELOG.md` describes the user-visible DE feature-parity work.
- Relevant verification passes before the implementation branch is committed.

## Future Work

After this feature-parity sprint, plan DE strategy expansion separately. That
future design can consider fixed built-in strategies such as `best1bin` or
`rand2bin`, a public strategy abstraction, and any Rust/PyO3 kernels justified
by benchmark measurements.

CMA-ES can later receive a separate long-term lifecycle design for
policy-driven `run(evaluator, policy=...)`. That design must respect CMA-ES's
complete-batch covariance update contract and should not be bundled into the DE
feature-parity sprint.
