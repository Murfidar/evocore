# Gene Spaces

`GeneSpace` defines the flat search-space schema shared by EvoCore optimizers.

Supported gene kinds are:

- `float`: bounded continuous values.
- `int`: bounded integer values.
- `bool`: unbounded binary values represented as Python `bool`.

```python
from evocore import Gene, GeneSpace

space = GeneSpace(
    [
        Gene("period", "int", 2, 50, sigma=0.05),
        Gene("threshold", "float", 0.0, 1.0),
        Gene("enabled", "bool"),
        Gene("fixed_mode", "int", 2, 2),
    ]
)
```

Numeric bounds are inclusive. Equal numeric bounds define a fixed gene that remains part
of the full genome and named parameter mapping.

`GeneSpace.uniform(low, high, length)` creates an unnamed float space:

```python
space = GeneSpace.uniform(-5.0, 5.0, 10)
assert space.has_names is False
```

Named spaces expose `params` on decoded candidates and individuals:

```python
params = space.params_for([10, 0.25, True, 2])
assert params == {
    "period": 10,
    "threshold": 0.25,
    "enabled": True,
    "fixed_mode": 2,
}
```

`GeneticAlgorithmOptimizer` and `DifferentialEvolutionOptimizer` support flat
spaces that mix `float`, `int`, and `bool` genes. This lets users model real
boolean switches directly instead of encoding them as integer genes.

```python
from evocore import Gene, GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace(
    [
        Gene("threshold", "float", 0.0, 1.0),
        Gene("period", "int", 2, 50),
        Gene("enabled", "bool"),
    ]
)

optimizer = GeneticAlgorithmOptimizer(space)
```

## Validation

`validate_genes(...)` checks decoded Python values without coercing, clamping, or mutating
them:

```python
space.validate_genes([10, 0.25, True, 2])
```

Invalid values raise `ConfigurationError`. Float genes reject booleans, non-finite values,
and out-of-bounds values. Int genes reject booleans, floats, and out-of-bounds values.
Bool genes accept only Python `bool`.

## Repair And Codec Helpers

Search-space codec helpers provide the same repair semantics used by the built-in
optimizers when crossing between Python values and Rust/operator float vectors:

```python
from evocore.search_space import decode_gene_values, encode_gene_values, repair_gene_values

encoded = encode_gene_values(space, [10, 0.25, True, 2])
decoded = decode_gene_values(space, [10.2, 0.25, 0.9, 2.0])
repaired = repair_gene_values(space, [10.2, 0.25, 0.9, 2.0])
```

Float genes are clamped to inclusive bounds, int genes are rounded then clamped
using Python's nearest-even tie behavior, and bool genes accept booleans or
numeric thresholds where values greater than or equal to `0.5` decode to `True`.
Non-finite numeric repair inputs raise `ConfigurationError`. The helpers
validate vector length and the final decoded values against the `GeneSpace`.

## Stable Signature

`GeneSpace` owns its reproducibility signature:

```python
signature = space.signature()
stable_hash = space.hash()
payload = space.to_dict()
json_text = space.to_json(indent=2)
```

`signature()` and `to_dict()` return the same payload. The signature includes
`schema_version`, ordered gene definitions, `has_names`, `length`, and per-gene
`is_fixed` metadata. `OptimizationResult.reproducibility.gene_space_signature` and
`OptimizationResult.reproducibility.gene_space_hash` use the same canonical values.

Per-gene `sigma` values are consumed by GA mutation operators. They override the global
scheduled mutation sigma for that gene; see [Operator Contract](operator-contract.md)
for the full sigma semantics.

This contract is intentionally flat. Categorical, permutation, conditional, and
multi-variable spaces are not part of this slice.

## Gene-Space Hash Versus Optimizer Config Hash

`GeneSpace.hash()` identifies the search-space structure: gene order, names, kinds,
bounds, sigma values, fixed-gene metadata, and naming mode. Optimizers expose a separate
`config_hash()` for algorithm configuration.

Use both hashes when comparing runs:

```python
same_space = left_result.reproducibility.gene_space_hash == right_result.reproducibility.gene_space_hash
same_optimizer = (
    left_result.reproducibility.optimizer_config_hash
    == right_result.reproducibility.optimizer_config_hash
)
```
