# Budget And Termination Contract Design

**Date:** 2026-05-15
**Status:** Draft approved for specification
**Scope:** Shared budget vocabulary, termination vocabulary, result diagnostics, and hard
public API cleanup for EvoCore optimizers

## Summary

EvoCore should use one budget and termination contract across managed optimizer runs.
The contract has two public limits: `max_evaluations` and `max_generations`.
`max_evaluations` means fresh full objective observations only, matching the approved
objective/evaluation semantics. `max_generations` means the maximum optimizer
iteration or generation count.

This slice is vocabulary and semantics first, but it intentionally performs a hard
public API cleanup. The old `generations` name is removed in favor of
`max_generations`. Policy-level `full_evaluation_budget` and
`single_full(budget=...)` are renamed to the shared `max_evaluations` vocabulary.
`RunResult.stop_reason` becomes the single authoritative final stop status; legacy
parallel booleans such as `stopped_early` and `budget_reached` are removed.

New controls such as `target_score`, patience-based early stopping, and wall-clock
limits are not added in this slice. Their stop reason names are reserved so future
features can compose with the same result and history vocabulary.

## Product Direction

Users should be able to answer three questions from a completed run without learning
optimizer-specific terminology:

```text
What limit was configured?
How much fresh objective work was consumed?
Why did the run stop?
```

The user-facing model should be:

```python
engine = GAEngine(
    gene_space,
    population_size=32,
    max_generations=100,
    max_evaluations=512,
)
result = engine.run(evaluator)

assert result.n_evaluations <= result.max_evaluations
print(result.stop_reason)
```

For policy-driven runs:

```python
policy = MultiFidelityPolicy.single_full(
    max_evaluations=64,
    batch_size=16,
)
result = engine.run(evaluator, policy=policy)
```

The same words should apply to GA, CMA-ES, policy-driven execution, result exports,
event history, and future island-style runners.

## Goals

- Define stable meanings for `max_evaluations`, `max_generations`, and
  `n_evaluations`.
- Align budget accounting with objective/evaluation semantics: only fresh
  `trusted_full` records consume full-evaluation budget.
- Rename generation limits to `max_generations` immediately and remove public
  `generations` constructor arguments.
- Rename policy-level full-evaluation budget fields to `max_evaluations`.
- Keep `population_size` as optimizer configuration, not budget.
- Define a shared stop reason vocabulary across optimizers.
- Make `RunResult.stop_reason` the single authoritative final stop status.
- Remove legacy result booleans that duplicate stop status.
- Add a final `run_stop` history event for completed managed runs.
- Reserve future stop reason names without adding their controls now.

## Non-Goals

- Do not add `target_score` as a public control in this slice.
- Do not add shared patience or early-stopping constructor controls.
- Do not add wall-clock budget enforcement.
- Do not promise island-model execution behavior.
- Do not add multi-objective termination semantics.
- Do not change constraint handling; constraints remain metadata-only.
- Do not add noisy-observation aggregation.
- Do not define checkpoint migration for old result payloads.

## Budget Vocabulary

### `max_evaluations`

`max_evaluations` is the maximum number of fresh full objective observations a managed
run may consume.

It counts accepted `EvaluationRecord` objects with:

```text
confidence == "trusted_full"
```

It does not count:

- `cached`
- `partial`
- `surrogate`
- `rejected`

This is a run-level full objective budget. It is not a candidate proposal count, ask
count, event count, population count, or total evaluation-record count.

`cached` records may update optimizer state and best state, but they do not spend
fresh objective budget because they represent previous full objective work.

### `n_evaluations`

`RunResult.n_evaluations` remains the number of fresh full objective observations
actually consumed by the completed run.

For a run with `max_evaluations` set:

```text
0 <= n_evaluations <= max_evaluations
```

For runs without an explicit full-evaluation cap, `n_evaluations` still uses the same
definition. It may be less than `population_size * max_generations` when callbacks,
manual interruption, optimizer convergence, or partial batches stop a run.

### `max_generations`

`max_generations` is the maximum number of optimizer iterations or generations the
managed run may execute.

For generation-oriented optimizers, one generation is the optimizer's natural outer-loop
step. For CMA-ES, one generation is one ask/tell distribution update cycle. For future
optimizers that are not strictly generation-oriented, `max_generations` should map to
the optimizer's stable top-level iteration count or be unsupported with a clear
configuration error.

### `population_size`

`population_size` is optimizer configuration only.

It may affect:

- proposal batch size
- generation shape
- covariance update shape
- selection pressure
- memory and runtime cost per iteration

It does not define a budget. It should not appear as a stop reason and should not be
used as a synonym for `max_evaluations`.

## Public API Cleanup

This stabilization slice performs hard cleanup instead of compatibility aliasing.

Constructor cleanup:

- `GAEngine(..., generations=...)` is removed.
- `CMAESEngine(..., generations=...)` is removed.
- Both constructors use `max_generations`.
- Passing `generations` should raise `ConfigurationError` with a clear message that
  names `max_generations`.

Policy cleanup:

- `MultiFidelityPolicy.full_evaluation_budget` becomes `max_evaluations`.
- `MultiFidelityPolicy.single_full(budget=...)` becomes
  `single_full(max_evaluations=...)`.
- Passing `budget` to `single_full(...)` should raise `ConfigurationError` with a clear
  message that names `max_evaluations`.

Rung vocabulary:

- `Rung.budget` remains unchanged in this slice.
- Rung budget describes rung/fidelity cost or intensity, not the run-level fresh full
  evaluation cap.

Result cleanup:

- `RunResult.stopped_early` is removed.
- `RunResult.budget_reached` is removed.
- `RunResult.stop_reason` is the only final stop-status field.

## Stop Reason Vocabulary

`stop_reason` is a stable string vocabulary shared by optimizers.

Implemented or immediately meaningful reasons:

| Reason | Meaning |
| --- | --- |
| `max_evaluations` | The run stopped because the fresh full-evaluation cap was reached. |
| `max_generations` | The run stopped because the generation or iteration cap was exhausted. |
| `callback` | A callback requested stop. Existing callback-based early stopping reports this reason. |
| `manual` | A user or runner explicitly interrupted the managed run. This is reserved until a public manual-stop flow exists. |
| `optimizer_converged` | The optimizer's own convergence criterion ended the run. This is reserved until an optimizer exposes such a criterion. |

Reserved future-control reasons:

| Reason | Future use |
| --- | --- |
| `target_score` | A future target-score control stops when the best raw score is good enough. |
| `patience` | A future shared patience control stops after insufficient improvement. |
| `wall_time` | A future wall-clock control stops after elapsed time reaches a configured limit. |

The reserved names are part of the vocabulary, but this slice does not add the controls
that emit them.

## Stop Precedence

When multiple stop conditions become true at the same boundary, use deterministic
precedence:

1. `manual`
2. `callback`
3. `optimizer_converged`
4. `max_evaluations`
5. `max_generations`

This keeps explicit user or callback decisions ahead of ordinary budget exhaustion.
Fresh evaluation budget wins over generation exhaustion when both limits are reached at
the same boundary.

If a higher-precedence reason wins while another condition is also true, `stop_reason`
still records only the winning reason. This slice does not add a triggered-conditions
list. A future result metadata extension may add one if users need that distinction.

## Result Export Contract

`RunResult` should include these stop and budget fields:

- `stop_reason`
- `max_evaluations`
- `max_generations`
- `n_evaluations`

`RunResult.to_dict()` should keep the established top-level envelope but update the
nested stop and budget fields:

```python
{
    "stop": {
        "reason": result.stop_reason,
    },
    "budget": {
        "max_evaluations": result.max_evaluations,
        "max_generations": result.max_generations,
        "n_evaluations": result.n_evaluations,
    },
}
```

`stopped_early` and `budget_reached` should not appear in the object or deterministic
export payload.

Runtime wall-clock observations remain runtime metadata, not stop budget controls.
`wall_time_seconds` may remain available through the existing runtime export path, but it
must not imply a wall-clock limit until a `wall_time` control exists.

## Event History Contract

Completed managed runs should append one final `run_stop` event to `EventHistory`.

`EventRecord.event_type` expands to include:

```text
run_stop
```

The stop event should not duplicate candidate-level score rows. It is a run-level audit
row. Its metadata should include:

```python
{
    "stop_reason": result.stop_reason,
    "max_evaluations": result.max_evaluations,
    "max_generations": result.max_generations,
    "n_evaluations": result.n_evaluations,
}
```

The stop event should be the final event for a completed managed run. Manual ask/tell
usage does not produce a `run_stop` event unless wrapped by a managed run controller.

Existing `ask`, `tell`, and `generation` event meanings remain unchanged.

## Engine State Summary

`EngineStateSummary` remains a live state snapshot, not a completed-run result.

It should continue to expose:

- best candidate ID
- best raw score
- event index
- pending batch IDs
- trusted count
- telemetry

It should not expose `stop_reason`. Stop status belongs to completed `RunResult`
objects and final `run_stop` history events.

## Optimizer Behavior

### GA

GA constructor configuration uses `max_generations`.

When GA is run through a policy, the policy's `max_evaluations` is the fresh
`trusted_full` cap. If the final rung returns both cached and fresh trusted records,
only the fresh `trusted_full` records increment `n_evaluations`.

If a final-rung batch returns no fresh `trusted_full` records, the run should fail with
`FitnessError` rather than loop forever. Cached-only final batches do not make progress
toward `max_evaluations`.

For GA generation-loop behavior, callback requests stop as `callback`; normal exhaustion
of the configured generation count reports `max_generations`.

### CMA-ES

CMA-ES constructor configuration uses `max_generations`.

CMA-ES should return `RunResult` values using the same stop and budget vocabulary as GA.
Normal exhaustion of the generation count reports `max_generations`. Callback requests
report `callback`.

If CMA-ES later exposes optimizer-native convergence criteria, those stops report
`optimizer_converged`.

### Manual Ask/Tell

Manual `ask/tell` flows update state and telemetry but do not automatically terminate.
Users may inspect telemetry to decide when to stop. Final stop diagnostics are produced
only by completed managed runs or a future managed manual-run controller.

## Compatibility And Migration

This branch intentionally favors a hard cleanup over deprecation aliases.

Breaking public changes:

- `generations` is no longer a public constructor argument.
- `max_generations` replaces `generations` in optimizer configuration and
  reproducibility metadata.
- `MultiFidelityPolicy.full_evaluation_budget` is removed.
- `MultiFidelityPolicy.max_evaluations` replaces `full_evaluation_budget`.
- `MultiFidelityPolicy.single_full(budget=...)` is removed.
- `MultiFidelityPolicy.single_full(max_evaluations=...)` replaces it.
- `RunResult.stopped_early` is removed.
- `RunResult.budget_reached` is removed.
- Result exports remove `stop.stopped_early` and `budget.budget_reached`.

Documentation, examples, changelog entries, tests, and reproducibility metadata must use
the new names consistently.

## Testing Guidance

Implementation should cover:

- `GAEngine(..., max_generations=N)` stores and exports `max_generations`.
- `CMAESEngine(..., max_generations=N)` stores and exports `max_generations`.
- Passing `generations` raises a clear `ConfigurationError`.
- `MultiFidelityPolicy(max_evaluations=N, ...)` validates and exports the configured cap.
- Passing `full_evaluation_budget` is no longer accepted.
- `MultiFidelityPolicy.single_full(max_evaluations=N, ...)` works.
- Passing `budget` to `single_full(...)` raises a clear `ConfigurationError`.
- `RunResult` no longer accepts or exports `stopped_early`.
- `RunResult` no longer accepts or exports `budget_reached`.
- `RunResult.stop_reason` reports `max_generations` for natural generation exhaustion.
- `RunResult.stop_reason` reports `max_evaluations` for fresh full-evaluation budget
  exhaustion.
- Callback stops report `callback`.
- Stop precedence chooses `callback` over budget exhaustion and `max_evaluations` over
  `max_generations`.
- `RunResult.n_evaluations` counts fresh `trusted_full` records only.
- Cached final-rung records can update best state without incrementing `n_evaluations`.
- A cached-only final-rung batch raises `FitnessError`.
- Completed managed runs append a final `run_stop` event.
- `EngineStateSummary` does not expose `stop_reason`.
- Reproducibility metadata uses `max_generations` and `max_evaluations` names.
- Docs and examples contain no public `generations`, `full_evaluation_budget`,
  `single_full(budget=...)`, `stopped_early`, or `budget_reached` references except
  migration notes.

## Deferred Follow-Ups

- Add `target_score` only after the shared contract defines raw-score thresholds,
  direction interaction, and result metadata.
- Add shared patience controls only after deciding whether they observe best raw score,
  direction-aware comparison score, generation summaries, or fresh full-evaluation
  events.
- Add wall-clock limits only after defining deterministic testing behavior and runtime
  export semantics.
- Add triggered stop-condition lists if users need to know which non-winning stop
  conditions were also true.
- Add island-model budget composition after island lifecycle and migration semantics are
  designed.

## Acceptance Criteria

- Budget vocabulary is consistent across constructors, policy objects, result fields,
  docs, tests, and reproducibility metadata.
- `max_evaluations` has one meaning: fresh `trusted_full` observations only.
- `max_generations` replaces public `generations` usage.
- `population_size` remains optimizer configuration, not budget.
- `stop_reason` is the only final stop-status field.
- Completed managed runs expose a final `run_stop` event.
- Future controls have reserved stop reason names but no public knobs in this slice.
