# Operator Contract

EvoCore genetic algorithms expose a public operator contract for crossover,
mutation, selection, and bounds enforcement. The existing string API remains valid:

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = GeneticAlgorithmOptimizer(
    space,
    crossover="sbx",
    mutation="gaussian",
    selection="tournament",
)
```

Typed operator specs make the same setup explicit:

```python
from evocore import (
    BoundsPolicy,
    CrossoverOperator,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    MutationOperator,
    SelectionOperator,
)

space = GeneSpace.uniform(-5.0, 5.0, 4)
optimizer = GeneticAlgorithmOptimizer(
    space,
    crossover=CrossoverOperator.sbx(eta=2.0, probability=0.9),
    mutation=MutationOperator.gaussian(
        probability=0.1,
        individual_probability=1.0,
        sigma=0.2,
    ),
    selection=SelectionOperator.tournament(size=3),
    bounds_policy=BoundsPolicy.clamp(),
)
```

## Compatibility

| Operator | Type | Supported GA spaces |
| --- | --- | --- |
| `sbx` | crossover | numeric-only `float`/`int` spaces |
| `blx` | crossover | numeric-only `float`/`int` spaces |
| `uniform` | crossover | numeric-only, bool-only, and mixed `float`/`int`/`bool` spaces |
| `one_point` | crossover | bool-only spaces |
| `two_point` | crossover | bool-only spaces |
| `gaussian` | mutation | numeric-only and mixed `float`/`int`/`bool` spaces |
| `uniform` | mutation | numeric-only and mixed `float`/`int`/`bool` spaces |
| `bit_flip` | mutation | bool-only and mixed `float`/`int`/`bool` spaces |
| `tournament` | selection | any GA-supported space |
| `roulette` | selection | any GA-supported space |
| `rank` | selection | any GA-supported space |

Numeric spaces may mix `float` and `int` genes. Binary spaces contain only `bool`
genes. Mixed GA spaces contain at least one numeric gene and at least one `bool`
gene.

When GA operator arguments are omitted, EvoCore resolves defaults by space profile:

| Space profile | Default crossover | Default mutation |
| --- | --- | --- |
| Numeric-only `float`/`int` | `sbx` | `gaussian` |
| Bool-only `bool` | `uniform` | `bit_flip` |
| Mixed `float`/`int` plus `bool` | `uniform` | `gaussian` |

In mixed spaces, `gaussian` and `uniform` mutation mutate numeric genes using
their numeric behavior and flip bool genes with `mutation_prob`. `bit_flip`
mutation flips bool genes with `mutation_prob` and leaves numeric genes unchanged.

`sbx`, `blx`, `one_point`, and `two_point` remain incompatible with mixed spaces.
Choose `uniform` crossover for mixed GA spaces.

## Bounds Policy

`BoundsPolicy.clamp()` is the v1 bounds policy:

- Float genes clamp to inclusive bounds.
- Int genes round, then clamp to inclusive bounds.
- Bool genes threshold to `False` or `True`.
- Fixed numeric genes remain fixed.

## Sigma Semantics

`mutation_sigma` is a global fraction of each numeric gene span. A `Gene(..., sigma=...)`
value overrides the global scheduled sigma for that gene. Per-gene sigma overrides do
not decay with `mutation_sigma_schedule`.

## Custom Operators

Custom operators are structured objects, not bare callables. They declare a name,
operator type, supported gene kinds, compatibility validation, and the execution method.

```python
from evocore.optimizers.operators import custom_mutation_operator


class ShiftMutation:
    name = "shift"
    operator_type = "mutation"
    supported_gene_kinds = frozenset({"float", "int"})

    def config_signature(self):
        return {"name": self.name, "amount": 0.1}

    def validate_compatibility(self, gene_space):
        return None

    def mutate(self, values, context):
        return [float(value) + 0.1 for value in values]


mutation = custom_mutation_operator(ShiftMutation())
```

Custom operators execute in Python and are recorded in reproducibility metadata as
partial reproducibility runtime hooks.
