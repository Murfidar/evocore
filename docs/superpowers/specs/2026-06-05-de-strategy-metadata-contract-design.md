# DE Strategy Metadata Contract Design

**Date:** 2026-06-05
**Status:** Draft for review
**Scope:** Prevent drift between Python and Rust Differential Evolution
strategy metadata after the Rust proposal-kernel migration.

## Summary

DE now has strategy knowledge in two layers. Python keeps the public registry in
`evocore/optimizers/de/strategies.py`, while Rust parses strategy names and
validates strategy-specific population requirements in `src/de.rs`. That split
is reasonable because Python owns public configuration and Rust owns strategy
math. The risk is metadata drift: a strategy can be added or changed in one
layer without matching validation in the other.

This design defines how EvoCore should keep Python and Rust strategy metadata
aligned without prematurely building a full strategy plugin system.

## Goals

- Keep Python as the public strategy registry and compatibility surface.
- Keep Rust as the owner of built-in strategy proposal math.
- Ensure strategy names, minimum population sizes, and adaptive-state
  requirements agree across Python and Rust.
- Add tests that fail when Python and Rust metadata drift.
- Avoid broad public API expansion unless a metadata endpoint clearly reduces
  risk.

## Non-Goals

- Do not add user-defined Rust DE strategies.
- Do not add SHADE or other new adaptation families.
- Do not move the Python registry into Rust wholesale.
- Do not remove Python validation.
- Do not preserve duplicate Python strategy math as a fallback implementation.

## Contract

For every built-in strategy, Python and Rust must agree on:

- canonical strategy name;
- minimum population size;
- whether the strategy uses best-slot metadata;
- whether the strategy uses current target as base;
- donor slot count;
- difference-pair count;
- whether jDE adaptive F/CR state is required or returned.

The current strategy set is:

- `rand1bin`;
- `best1bin`;
- `rand2bin`;
- `current-to-best1bin`;
- `jde-rand1bin`.

## Options

### Option A: Parity Tests Only

Keep Python and Rust metadata as separate implementations, but add tests that
assert they agree. Tests can call the Rust kernel with minimal inputs for each
strategy and compare observed validation behavior with Python
`DEStrategySpec`.

This is the smallest option and avoids new PyO3 API surface.

### Option B: Rust Metadata Endpoint

Expose a small internal extension function such as `_core.de_strategy_specs()`
that returns Rust-known strategy metadata. Python tests compare that payload
with `SUPPORTED_DE_STRATEGIES`.

This adds API surface to `_core`, but it is more direct and easier to test than
deriving metadata from generated proposals.

### Option C: Declarative Manifest

Move strategy metadata into a declarative manifest used to generate or validate
both Python and Rust metadata.

This is the boldest option. It reduces drift most strongly, but adds build and
maintenance complexity that may not be justified for five built-in strategies.

## Recommendation

Start with Option A. If DE strategy count grows or metadata drift becomes a
real maintenance problem, move to Option B. Option C should wait until EvoCore
has enough strategies or code generation infrastructure to make a manifest
worth the weight.

## Testing

Add tests that assert:

- every Python strategy name is accepted by Rust;
- unknown strategy names are rejected by Rust;
- Python and Rust minimum population requirements match;
- jDE strategy calls require valid adaptive state where the Rust kernel expects
  it;
- proposal metadata contains the expected strategy name and slot fields for each
  strategy.

These tests should be direct Rust-kernel tests or adapter tests. They should not
reintroduce Python strategy math as a production fallback.

## Compatibility

This is a contract and test hardening slice. Public optimizer APIs, seeded
sequences, checkpoint payloads, and docs should remain stable unless a real
metadata mismatch is discovered and corrected.
