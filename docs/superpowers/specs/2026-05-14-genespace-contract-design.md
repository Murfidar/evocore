# GeneSpace Contract Design

**Date:** 2026-05-14
**Status:** Draft approved for specification
**Scope:** Stabilize the flat `GeneDef` and `GeneSpace` public contract before adding new gene kinds or optimizer capabilities

## Summary

EvoCore should make `GeneSpace` the canonical owner of flat search-space identity,
validation, and reproducibility export. The slice keeps the current flat gene model:
`float`, `int`, `bool`, named and unnamed spaces, optional per-gene `sigma`, and fixed
numeric genes.

The goal is not to add categorical genes, permutations, conditional genes, island models,
or multi-variable orchestration. The goal is to make the existing search-space contract
stable, explicit, JSON-safe, and shared by optimizers, result metadata, and compatibility
helpers without duplicate schema construction.

## Goals

- Make `GeneSpace` the single source of truth for its stable signature, hash, dictionary
  export, JSON export, and decoded-gene validation.
- Keep the current flat gene kinds: `float`, `int`, and `bool`.
- Preserve named and unnamed gene-space behavior.
- Preserve fixed numeric genes as full-genome members.
- Include `schema_version` and derived fixed-gene metadata in the canonical signature.
- Update result reproducibility metadata to consume the canonical `GeneSpace` signature
  and hash.
- Keep existing helper functions importable while making them delegate to `GeneSpace`.
- Avoid duplicate code paths for gene-space signature payloads.

## Non-Goals

- Do not add categorical, permutation, conditional, dependent, vector, object, or
  multi-variable gene kinds.
- Do not add `from_dict()` or `from_json()` loaders.
- Do not redesign objective, budget, lifecycle, result, history, telemetry, checkpoint,
  island, or operator contracts except where they directly consume the gene-space
  signature.
- Do not move Rust-boundary encoding and decoding into `GeneSpace` in this slice.
- Do not add custom operator or custom gene-domain extension APIs.

## Public API

`GeneSpace` gains the stable public methods:

```python
space.signature()
space.hash()
space.to_dict()
space.to_json(indent=None)
space.validate_genes(values)
```

`signature()` and `to_dict()` return the same canonical payload. `hash()` returns the
canonical JSON hash of `signature()`. `to_json()` uses the existing deterministic JSON
export helper. `validate_genes(...)` validates decoded Python gene values and raises
`ConfigurationError` on failure.

Existing helpers remain available:

```python
gene_space_signature(space)
gene_space_hash(signature)
```

`gene_space_signature(space)` delegates to `space.signature()`. `gene_space_hash(signature)`
continues to hash an already-built signature payload. The helper does not construct or own
the schema.

## Canonical Signature

The canonical payload is:

```python
{
    "schema_version": 1,
    "genes": [
        {
            "name": "x",
            "kind": "float",
            "low": -5.0,
            "high": 5.0,
            "sigma": None,
            "is_fixed": False,
        }
    ],
    "has_names": True,
    "length": 1,
}
```

Rules:

- Gene order is part of the signature.
- `schema_version` is included now so future gene-space expansions can be versioned.
- `is_fixed` is included for every gene as derived metadata.
- Numeric genes use their configured `low`, `high`, and `sigma` values.
- Bool genes keep `low=None`, `high=None`, `sigma=None`, and `is_fixed=False`.
- `has_names` remains part of the payload because unnamed uniform spaces are public
  behavior.
- `length` remains part of the payload for quick reproducibility checks.

Because the canonical signature expands the current result metadata payload,
`gene_space_hash` values will change in this feature-branch stabilization step. That is
intentional: the stable boundary becomes the `GeneSpace`-owned schema.

## Result Metadata Integration

Result reproducibility metadata should consume `GeneSpace` directly:

```python
result.reproducibility.gene_space_signature == space.signature()
result.reproducibility.gene_space_hash == space.hash()
```

Engines should no longer reconstruct gene-space signatures through a separate schema in
result or stats code. Result metadata remains a consumer of the gene-space contract, not
an owner of a parallel payload shape.

## Validation

`GeneSpace.validate_genes(values)` is a pure validator. It returns `None` on success and
raises `ConfigurationError` on failure. It must not normalize, coerce, clamp, round, or
mutate values.

Validation rules:

- Length must match `space.length`.
- Float genes accept `int` or `float`, but reject `bool`, `nan`, `inf`, and values outside
  inclusive bounds.
- Int genes accept `int`, but reject `bool`, floats, non-integers, and values outside
  inclusive bounds.
- Bool genes accept only `bool`.
- Fixed numeric genes accept only their fixed value.

Error messages should identify the gene name, index, expected condition, and received
value or type where practical:

```text
GeneSpace expected 4 genes, got 3.
Gene 'period' at index 2 expects int, got float.
Gene 'x' at index 0 must be within [-5.0, 5.0], got 9.2.
```

## Encoding Boundary

`OperatorSet` keeps owning Rust-boundary encode and decode behavior. This keeps backend
representation concerns out of `GeneSpace`.

`OperatorSet.encode_genes(...)` should use `GeneSpace.validate_genes(...)` before encoding
decoded values. That gives all optimizers one public validity rule while preserving the
existing PyO3 float-vector boundary.

## Compatibility

The public compatibility rules are:

- `GeneSpace.signature()` owns the schema.
- `GeneSpace.to_dict()` returns the same payload as `signature()`.
- `GeneSpace.hash()` hashes `signature()`.
- `gene_space_signature(space)` delegates to `space.signature()`.
- `gene_space_hash(signature)` hashes an already-built signature.
- Result reproducibility metadata uses `space.signature()` and `space.hash()`.
- Existing helper imports keep working, but docs prefer the `GeneSpace` methods.

No migration loader is needed because this slice does not introduce `from_dict()` or
`from_json()`.

## Documentation

Documentation should cover only the stabilized flat contract:

- Supported kinds: `float`, `int`, and `bool`.
- Named versus unnamed spaces.
- Fixed numeric genes.
- Optional per-gene `sigma`.
- `signature()`, `hash()`, `to_dict()`, and `to_json()`.
- `validate_genes()`.
- Reproducibility metadata consuming the same canonical signature.

The changelog should note that gene-space signatures and hashes now include
`schema_version` and per-gene `is_fixed` metadata.

## Testing

Required unit tests:

- `space.signature() == space.to_dict()`.
- `space.hash() == gene_space_hash(space.signature())`.
- `space.to_json()` is deterministic.
- `gene_space_signature(space) == space.signature()`.
- Fixed numeric genes include `"is_fixed": True`.
- Variable numeric and bool genes include `"is_fixed": False`.
- Valid decoded values pass `validate_genes(...)`.
- Invalid length, type, non-finite float, out-of-bounds value, bool-as-numeric, and
  fixed-value mismatch raise `ConfigurationError`.
- `OperatorSet.encode_genes(...)` rejects invalid decoded genes through the shared
  `GeneSpace` validator.

Required result metadata tests:

- `RunResult.reproducibility.gene_space_signature == space.signature()`.
- `RunResult.reproducibility.gene_space_hash == space.hash()`.

Required property tests:

- Generated valid flat spaces produce JSON-safe signatures.
- Equivalent spaces produce equivalent signatures and hashes.
- `to_json()` output round-trips through JSON parsing.

## Verification

The implementation should run the smallest reliable targeted tests first:

```powershell
python -m pytest tests/unit/test_gene_space.py tests/unit/test_stats.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py -v
python -m pytest tests/property/test_gene_space_properties.py -v
python -m ruff format --check
python -m ruff check
```

If implementation touches Rust or the PyO3 boundary beyond Python-side validation, also
run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

## Acceptance Criteria

- `GeneSpace` owns stable signature, hash, dict export, JSON export, and decoded-gene
  validation.
- There is only one implementation of the gene-space signature payload.
- Existing helper functions remain importable and delegate to the canonical implementation.
- Result reproducibility metadata uses the canonical gene-space signature and hash.
- The canonical signature includes `schema_version` and per-gene `is_fixed`.
- Validation is strict, pure, and uses clear `ConfigurationError` messages.
- The slice does not add new gene kinds or broader optimizer capabilities.
