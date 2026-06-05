# Optimizer Lifecycle Helper Consolidation Design

**Date:** 2026-06-05
**Status:** Draft for review
**Scope:** Identify shared GA, DE, and CMA-ES lifecycle helper behavior that can
be consolidated after codec and kernel-boundary cleanup.

## Summary

GA and CMA-ES already have good optimizer architecture. DE now follows the same
general direction after the Rust proposal-kernel migration: Python owns
lifecycle, public records, evaluation, policies, events, telemetry, and
checkpoints; Rust owns deterministic kernels where useful.

There is still lifecycle duplication across optimizer ask/tell modules. Some of
it is healthy because optimizers differ in replacement, state updates, and
policy semantics. Some of it is mechanical: event append records, confidence
counting, evaluation record validation, candidate lookup, and context creation.

This design separates that cleanup from codec and Rust-kernel work so it can be
reviewed with the right risk level.

## Goals

- Consolidate only lifecycle helpers that clearly share behavior across
  optimizers.
- Preserve optimizer-specific replacement and state-update semantics.
- Preserve public optimizer APIs, event schemas, telemetry schemas, checkpoint
  payloads, and seeded behavior.
- Reduce duplicated validation and event-construction code in GA, DE, and
  CMA-ES ask/tell flows.
- Keep helper modules focused and domain-oriented.

## Non-Goals

- Do not move optimizer lifecycle into Rust.
- Do not introduce a generic base optimizer class that obscures GA, DE, or
  CMA-ES-specific behavior.
- Do not change how GA selection/reproduction, DE target replacement, or CMA-ES
  state `tell(...)` works.
- Do not redesign `BudgetPolicy`, `EvaluationStage`, or checkpoint envelopes.
- Do not combine this work with search-space codec or Rust gene-codec parity
  refactors.

## Candidate Shared Helpers

The likely shared helpers are:

- append ask events for a sequence of candidates;
- append tell events for candidate/evaluation-record pairs;
- count confidence categories for telemetry updates;
- validate evaluator records match assigned candidates;
- build evaluation contexts for stage-driven evaluation;
- look up candidates and batches for evaluation records;
- validate checkpoint payload fields that are common across optimizer families.

These helpers should live in lifecycle-focused modules, not optimizer-specific
packages, when they are truly optimizer-neutral.

## Boundaries

Shared helpers should accept explicit inputs and return simple values. They
should not mutate optimizer-specific state unless the mutation is the whole
point of the helper, such as appending to an `EventHistory`.

Optimizer-specific code should continue to own:

- GA generation advancement and reproduction state;
- DE target population replacement and jDE commit/discard;
- CMA-ES trusted-record filtering and continuous sample `tell(...)` state;
- optimizer-specific checkpoint restoration;
- optimizer-specific telemetry summary fields.

If a helper requires many optimizer-specific callbacks, it is probably not a
good shared helper.

## Proposed Module Shape

Potential modules:

```text
evocore/lifecycle/ask_tell_helpers.py
evocore/lifecycle/evaluation_helpers.py
evocore/lifecycle/checkpoint_helpers.py
```

The final names should follow existing lifecycle package conventions. Avoid a
large catch-all module.

## Testing

Start with characterization tests around current GA, DE, and CMA-ES behavior:

- ask events remain append-only and identical;
- tell events preserve confidence, stage, score, cost, and metadata fields;
- partial, surrogate, cached, rejected, and trusted confidence counts match
  current optimizer behavior;
- evaluator record mismatch errors remain clear;
- checkpoint resume behavior remains unchanged for partial pending batches.

Then add focused unit tests for any new shared helper.

## Compatibility

This should be strictly behavior-preserving. It has higher behavioral risk than
codec cleanup because lifecycle code interacts with policies, telemetry,
checkpoints, and evaluator records. It should happen after the search-space
codec and DE Rust adapter specs are implemented and verified.
