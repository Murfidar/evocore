# Rust Gene Codec Parity Design

**Date:** 2026-06-05
**Status:** Draft for review
**Scope:** Consolidate Rust-internal gene kind parsing and encoded repair so
Rust kernels share the same search-space semantics without exposing new tiny
PyO3 helpers.

## Summary

The DE Rust migration moved Differential Evolution trial proposal math into
`src/de.rs`. That was the right architectural move: Rust now owns deterministic
proposal math while Python owns optimizer lifecycle. The migration also exposed
Rust-side duplication. DE has local encoded repair and gene kind parsing, while
the reproduction kernel has similar repair logic in `src/reproduce.rs`.

This design creates a shared Rust-internal gene codec module used by Rust
kernels. It keeps the Rust helper private to the extension implementation and
uses tests to enforce parity with the Python search-space codec contract.

## Goals

- Add one internal Rust module for gene kind parsing and encoded value repair.
- Use the shared module from DE trial generation.
- Use the shared module from reproduction/operator kernels where equivalent
  repair behavior already exists.
- Preserve current Rust extension public signatures.
- Preserve deterministic seed behavior.
- Add Rust and Python integration tests that catch drift between Python-facing
  repair semantics and Rust-internal encoded repair semantics.

## Non-Goals

- Do not expose `repair_gene_value` or similar as a PyO3 public function.
- Do not make Python call Rust for per-gene repair.
- Do not move Python `Gene`, `GeneSpace`, `Candidate`, lifecycle, telemetry, or
  checkpoint behavior into Rust.
- Do not redesign Rust DE strategy generation.
- Do not add new gene kinds or categorical/permutation handling.

## Proposed Rust Module

Create a focused internal module:

```text
src/gene_codec.rs
```

The module should own:

```rust
pub(crate) enum EncodedGeneKind {
    Float,
    Int,
    Bool,
}

pub(crate) fn parse_gene_kind(kind: &str) -> PyResult<EncodedGeneKind>;

pub(crate) fn parse_gene_kinds(kinds: &[String]) -> PyResult<Vec<EncodedGeneKind>>;

pub(crate) fn repair_encoded_value(
    value: f64,
    bounds: (f64, f64),
    kind: EncodedGeneKind,
) -> f64;
```

Names can follow local Rust style, but the responsibilities should stay this
small. The module should not know about optimizer lifecycle or strategy names.

## Semantics

Rust repair should match the Python search-space codec contract for encoded
values:

- float: clamp to `[lower, upper]`;
- int: round to nearest integer, then clamp to `[lower, upper]`;
- bool: threshold to `1.0` when value is greater than or equal to `0.5`, else
  `0.0`.

Rust kernels operate on encoded `f64` values. Rust should therefore return
encoded repaired values, not Python bools or ints. Python remains responsible
for decoding user-facing values.

## Integration

`src/de.rs` should use the shared module for:

- parsing `gene_kinds`;
- repairing generated trial values;
- preserving fixed-gene values in encoded form where applicable.

`src/reproduce.rs` should use the shared module where its current
`clamp_and_round(...)` behavior matches the shared semantics.

`src/lib.rs` should use the shared parsing helper if it currently has local
gene kind parsing that is equivalent.

The implementation should be careful to avoid import cycles or exposing private
types through PyO3 signatures.

## Testing

Rust tests should cover `repair_encoded_value(...)` directly if the module has
unit tests, or indirectly through public extension functions if direct tests are
not ergonomic.

Python tests should cover cross-boundary parity through existing Rust kernels:

- DE generated trial genes should always be encoded-valid for float, int, and
  bool genes.
- Rust operator/reproduction outputs should repair ints and bools the same way
  the Python codec does.
- Unknown kind strings should produce clear errors in all Rust entry points
  using the shared parser.

The parity tests should assert behavior, not private function names, so the
Rust module can stay internal.

## Compatibility

This is an internal refactor. Public Python APIs, `_core.pyi` signatures,
checkpoint payloads, docs, and seeded sequences should remain stable unless a
pre-existing inconsistency is discovered. If a seeded sequence changes because a
kernel changes when repair occurs, that must be documented before implementation
continues.
