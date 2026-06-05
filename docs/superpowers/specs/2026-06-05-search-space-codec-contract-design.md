# Search-Space Codec Contract Design

**Date:** 2026-06-05
**Status:** Draft for review
**Scope:** Promote EvoCore's Python gene encode, decode, and repair semantics
from optimizer-local helpers into neutral search-space helpers.

## Summary

EvoCore already has a strong optimizer architecture: GA, CMA-ES, and DE keep
public lifecycle behavior in Python while Rust provides deterministic kernels
and helpers where useful. The current cleanup target is narrower than an
optimizer rewrite. The duplicated behavior is the search-space boundary:
turning Python gene values into encoded numeric vectors, repairing encoded or
mutated values, and decoding values back into user-facing Python types.

Today that behavior is spread across `OperatorCodec`, GA bounds policy
application, CMA-ES bounds-and-round logic, and DE's Rust-kernel decode path.
Those helpers encode similar rules for floats, ints, and bools, but they live in
optimizer-specific modules. This design makes the semantics a first-class
search-space contract.

## Goals

- Add neutral Python helpers in `evocore/search_space/codec.py` for:
  - repairing a single gene value;
  - repairing a full vector;
  - encoding decoded Python gene values into Rust/operator numeric values;
  - decoding encoded numeric values into Python gene values;
  - decoding encoded populations or solutions where useful.
- Keep `OperatorCodec` importable and API-compatible.
- Make `OperatorCodec` delegate encode/decode behavior to the neutral helpers.
- Make DE use shared helpers instead of a private DE decode helper.
- Make `apply_bounds_policy(..., BoundsPolicy.clamp())` delegate to the shared
  repair-vector helper.
- Make CMA-ES use shared repair semantics for user-facing decoded candidates
  while preserving continuous samples for Rust CMA-ES state updates.
- Preserve deterministic seed and checkpoint behavior.
- Add focused tests for helper behavior and optimizer integration.

## Non-Goals

- Do not change public optimizer APIs.
- Do not move optimizer lifecycle behavior into Rust.
- Do not expose new Rust helpers for Python per-gene repair.
- Do not redesign GA operators, CMA-ES state updates, DE strategy math, or
  checkpoint envelopes in this slice.
- Do not change candidate IDs, batch IDs, event schemas, or telemetry schemas.
- Do not support new gene kinds.

## Proposed Python API

Add module-level helpers to `evocore/search_space/codec.py`:

```python
def repair_gene_value(value: object, gene: Gene) -> GeneValue: ...

def repair_gene_values(
    gene_space: GeneSpace,
    values: Sequence[object],
) -> list[GeneValue]: ...

def encode_gene_values(
    gene_space: GeneSpace,
    values: Sequence[GeneValue],
) -> list[float]: ...

def decode_gene_values(
    gene_space: GeneSpace,
    encoded: Sequence[float],
) -> list[GeneValue]: ...
```

`GeneValue` is the existing domain value type. The helpers should be public
within the package domain, but they do not need top-level `evocore` convenience
exports in this slice. `evocore.search_space` may re-export them if that matches
the existing package entrance style.

## Semantics

The shared semantics should match current optimizer behavior:

- `float` genes clamp to `[lower, upper]` and return `float`.
- `int` genes round to the nearest integer, clamp to `[lower, upper]`, and
  return `int`.
- `bool` genes use threshold semantics for numeric inputs:
  - values greater than or equal to `0.5` decode or repair to `True`;
  - values below `0.5` decode or repair to `False`;
  - boolean inputs remain boolean.
- Fixed genes still validate through normal `GeneSpace` validation.
- Vector helpers raise `ValueError` on length mismatch, matching existing
  optimizer helper style.
- `repair_gene_values(...)` validates the repaired vector through
  `gene_space.validate_genes(...)` before returning.
- `decode_gene_values(...)` decodes encoded numeric values and validates the
  decoded vector.
- `encode_gene_values(...)` validates decoded input before encoding.

The helpers should intentionally repair before validation when accepting
encoded or mutated numeric values. They should validate before encoding when
accepting user-facing decoded values.

## Integration

`OperatorCodec` should remain the compatibility facade for operator code. Its
`encode_values`, `decode_values`, `decode_solution`, and `decode_population`
methods should delegate to the new module-level helpers. Operator-specific
normalization, compatibility, crossover, mutation, and selection behavior should
stay where it is.

`apply_bounds_policy(...)` should keep its public signature. For
`BoundsPolicy.clamp()`, it should delegate to `repair_gene_values(...)`. Future
non-clamp policies can remain explicit branches.

DE should remove its private `_decode_de_values(...)` helper and call
`decode_gene_values(...)` for Rust-returned proposal genes. DE population
encoding should call `encode_gene_values(...)` instead of manually converting
bools and numerics.

CMA-ES should preserve continuous samples for Rust state `tell(...)`. Only the
user-facing candidate genes produced from continuous samples should go through
shared repair/decode semantics.

## Testing

Add unit tests for the shared helpers covering:

- float clamp;
- int round then clamp;
- bool threshold;
- vector length mismatch;
- validation after repair;
- validation before encoding;
- population encode/decode behavior if population helpers are added.

Update existing tests so behavior is asserted through the shared contract rather
than optimizer-local helpers:

- operator bounds-policy tests;
- DE ask/tell Rust-kernel decoding tests;
- CMA-ES bounds/round tests;
- `OperatorCodec` encode/decode tests.

Property tests should remain focused: generated out-of-bounds numeric vectors
should repair into values accepted by `GeneSpace.validate_genes(...)`.

## Compatibility

This is intended as a behavior-preserving refactor. Existing imports of
`OperatorCodec` and `apply_bounds_policy` remain valid. Public optimizer
constructors, checkpoint payloads, seeded sequence behavior, and Rust extension
signatures should not change.

If tests reveal a current mismatch between DE, GA, CMA-ES, and `OperatorCodec`,
the plan should document the chosen canonical behavior before implementation.
