# Operator Contract Design

**Date:** 2026-05-18
**Status:** Draft approved for specification
**Scope:** Stabilize EvoCore's public GA operator API, compatibility validation, bounds enforcement, mutation sigma semantics, and custom operator extension path

## Summary

EvoCore should make variation, mutation, selection, and bounds behavior a first-class
public contract before adding more optimizer families. The current implementation already
has the core behavior split between `OperatorCodec`, GA reproduction helpers, and Rust
dispatch. This design promotes that behavior into a stable public API with explicit names,
validation, signatures, and extension points.

The contract should preserve today's simple constructor strings:

```python
GeneticAlgorithmOptimizer(space, crossover="sbx", mutation="gaussian", selection="tournament")
```

It should also add typed operator objects for users who want explicit configuration,
discoverability, and custom extensions:

```python
GeneticAlgorithmOptimizer(
    space,
    crossover=CrossoverOperator.sbx(eta=2.0, probability=0.9),
    mutation=MutationOperator.gaussian(probability=0.1, sigma=0.2),
    selection=SelectionOperator.tournament(size=3),
    bounds_policy=BoundsPolicy.clamp(),
)
```

The public contract should be conservative. Numeric GA operators support flat `float` and
`int` genes. Binary GA operators support flat `bool` genes. Mixed numeric spaces remain
valid. Mixed `bool` plus numeric spaces remain invalid until EvoCore deliberately designs
segmented or typed-subspace operators.

## Product Direction

The operator API should reinforce EvoCore's existing direction: reproducible, inspectable
optimization over explicit `GeneSpace` schemas. Users should be able to answer:

- Which operators did this run use?
- Which parameters were part of the operator identity?
- Was this operator compatible with the gene space?
- Did bounds enforcement happen, and how?
- Are custom operators represented honestly in reproducibility metadata?

The operator layer should be public now, but it should not overfit future gene kinds. This
slice should leave room for categorical, permutation, conditional, and segmented spaces
without designing them prematurely.

## Goals

- Add a public operator contract API for GA crossover, mutation, selection, and bounds.
- Preserve existing string constructor arguments as stable aliases.
- Normalize strings and typed specs into canonical operator signatures.
- Make operator compatibility declarative and testable by gene kind.
- Clarify `uniform` ambiguity by recording operator type and gene-domain metadata in
  signatures.
- Make bounds enforcement a named public policy.
- Stabilize global and per-gene mutation sigma semantics.
- Add a constrained custom operator protocol instead of accepting arbitrary loose
  callables.
- Integrate operator signatures into optimizer config hashing.
- Represent custom operator reproducibility honestly.
- Move operator name and compatibility rules out of `OperatorCodec` so it can focus on
  Rust-boundary encoding and decoding.

## Non-Goals

- Do not add categorical, permutation, conditional, graph, object, or segmented gene kinds.
- Do not support mixed `bool` plus numeric GA reproduction in this slice.
- Do not add global plugin loading or dynamic operator discovery.
- Do not add Rust-side custom operator registration.
- Do not replace the GA constructor with a single config object.
- Do not redesign objective evaluation, budget policy, lifecycle records, or result
  history.
- Do not change deterministic seed derivation except where custom operator context needs
  to expose existing deterministic inputs.
- Do not remove existing Rust operator functions or Python stubs.

## Public Module

Add a focused public module:

```text
evocore/optimizers/operators.py
```

The root package should re-export the public contract names:

```python
from evocore import BoundsPolicy, CrossoverOperator, MutationOperator, SelectionOperator
```

The GA package may also re-export them for local discoverability:

```python
from evocore.optimizers.ga import CrossoverOperator, MutationOperator, SelectionOperator
```

`evocore.search_space.OperatorCodec` should remain importable for compatibility, but it
should no longer own public operator name sets. It should consume already-normalized
operator specs or a validated compatibility summary.

## Public API Shape

The public operator objects should be immutable dataclasses or small immutable value
objects with factory methods:

```python
@dataclass(frozen=True)
class CrossoverOperator:
    name: str
    parameters: Mapping[str, Any]
    supported_gene_kinds: frozenset[GeneKind]
    domain: str

    @classmethod
    def sbx(cls, *, eta: float = 2.0, probability: float = 0.9) -> CrossoverOperator: ...
    @classmethod
    def blx(cls, *, alpha: float = 0.5, probability: float = 0.9) -> CrossoverOperator: ...
    @classmethod
    def uniform(cls, *, probability: float = 0.9) -> CrossoverOperator: ...
    @classmethod
    def one_point(cls, *, probability: float = 0.9) -> CrossoverOperator: ...
    @classmethod
    def two_point(cls, *, probability: float = 0.9) -> CrossoverOperator: ...

@dataclass(frozen=True)
class MutationOperator:
    name: str
    parameters: Mapping[str, Any]
    supported_gene_kinds: frozenset[GeneKind]
    domain: str

    @classmethod
    def gaussian(
        cls,
        *,
        probability: float = 0.1,
        individual_probability: float = 1.0,
        sigma: float = 0.2,
    ) -> MutationOperator: ...
    @classmethod
    def uniform(
        cls,
        *,
        probability: float = 0.1,
        individual_probability: float = 1.0,
    ) -> MutationOperator: ...
    @classmethod
    def bit_flip(
        cls,
        *,
        probability: float = 0.1,
        individual_probability: float = 1.0,
    ) -> MutationOperator: ...

@dataclass(frozen=True)
class SelectionOperator:
    name: str
    parameters: Mapping[str, Any]

    @classmethod
    def tournament(cls, *, size: int = 3) -> SelectionOperator: ...
    @classmethod
    def roulette(cls) -> SelectionOperator: ...
    @classmethod
    def rank(cls) -> SelectionOperator: ...

@dataclass(frozen=True)
class BoundsPolicy:
    name: str
    parameters: Mapping[str, Any]

    @classmethod
    def clamp(cls) -> BoundsPolicy: ...
```

The exact internal representation may differ, but the public behavior should match this
shape: explicit type, canonical name, stable parameters, compatibility metadata, and a
JSON-safe signature.

## Constructor Compatibility

`GeneticAlgorithmOptimizer` should accept either existing strings or typed operator
objects:

```python
GeneticAlgorithmOptimizer(space, crossover="sbx", mutation="gaussian")
GeneticAlgorithmOptimizer(space, crossover=CrossoverOperator.sbx(eta=3.0))
```

Legacy scalar parameters remain accepted:

```python
crossover_prob
crossover_eta
crossover_alpha
mutation_prob
mutation_individual_prob
mutation_sigma
mutation_sigma_schedule
mutation_sigma_end
selection
tournament_size
```

Conflict rule:

- When a string operator is provided, legacy scalar parameters define that operator's
  parameters, preserving current behavior.
- When a typed operator object is provided, the object owns that component's parameters.
- If a typed operator object is combined with non-default legacy scalar parameters for the
  same component, raise `ConfigurationError` instead of silently picking one source.
- Default scalar values may coexist with typed operator objects because they are part of
  the existing constructor signature.

For example:

```python
GeneticAlgorithmOptimizer(
    space,
    crossover=CrossoverOperator.sbx(eta=3.0),
    crossover_eta=4.0,
)
```

should fail with a message that says `crossover_eta` conflicts with the typed crossover
operator.

This rule keeps the old API stable while making the new API predictable.

## Canonical Names And Aliases

Canonical built-in operator names:

```text
Crossover: sbx, blx, uniform, one_point, two_point
Mutation: gaussian, uniform, bit_flip
Selection: tournament, roulette, rank
Bounds: clamp
```

String aliases:

- `uniform_xo` remains accepted at the Rust boundary but should normalize to public
  `uniform`.
- Existing public strings keep their current behavior.
- New aliases should be added sparingly and always normalize to one canonical name.

The canonical signature should include enough context to avoid ambiguity:

```python
{
    "type": "uniform",
    "operator_type": "crossover",
    "domain": "numeric",
    "parameters": {"probability": 0.9},
}
```

This means `MutationOperator.uniform()` and `CrossoverOperator.uniform()` can share the
human-friendly name `uniform` without sharing a reproducibility identity.

## Compatibility Matrix

The v1 GA compatibility matrix is:

| Operator | Type | Supported Gene Kinds | Domain |
| --- | --- | --- | --- |
| `sbx` | crossover | `float`, `int` | numeric |
| `blx` | crossover | `float`, `int` | numeric |
| `uniform` | crossover | `float`, `int` | numeric |
| `one_point` | crossover | `bool` | binary |
| `two_point` | crossover | `bool` | binary |
| `uniform` | crossover | `bool` | binary |
| `gaussian` | mutation | `float`, `int` | numeric |
| `uniform` | mutation | `float`, `int` | numeric |
| `bit_flip` | mutation | `bool` | binary |
| `tournament` | selection | any supported GA space | score |
| `roulette` | selection | any supported GA space | score |
| `rank` | selection | any supported GA space | score |
| `clamp` | bounds | `float`, `int`, `bool` | repair |

Rules:

- A numeric-only space may contain any mix of `float` and `int` genes.
- A binary-only space may contain only `bool` genes.
- Mixed `bool` plus numeric spaces are rejected for GA v1.
- Selection operators are independent of gene kind and operate on direction-adjusted
  comparison scores.
- Bounds policy is applied after crossover and mutation during reproduction.

Validation errors should include:

- operator type
- operator name
- supported gene kinds
- actual gene-space kinds
- a short remediation hint when possible

Example:

```text
crossover='sbx' supports numeric GeneSpace kinds {'float', 'int'}, got {'bool'}.
Use CrossoverOperator.one_point(), CrossoverOperator.two_point(), or encode booleans as int genes with bounds [0, 1].
```

## Bounds Policy

`BoundsPolicy.clamp()` names the existing enforcement behavior:

- Float genes are clamped to inclusive numeric bounds.
- Int genes are rounded, then clamped to inclusive integer bounds.
- Bool genes are thresholded to binary values.
- Fixed numeric genes are preserved because clamping to equal bounds returns the fixed
  value.

`BoundsPolicy.clamp()` is the only v1 public policy. Naming it now is still useful because
it documents where repair happens and gives future policies a stable slot.

The bounds policy should appear in optimizer configuration:

```python
"bounds_policy": {
    "type": "clamp",
    "parameters": {},
}
```

If implementation keeps the current Rust `clamp_and_round` function, it should be treated
as the Rust implementation of the public `BoundsPolicy.clamp()` contract.

## Mutation Sigma Semantics

EvoCore should stabilize the current per-gene sigma behavior with one explicit
clarification:

- `mutation_sigma` is a global sigma fraction in `[0, 1]`.
- `mutation_sigma_schedule` transforms the global sigma fraction over generations.
- For numeric genes without `Gene.sigma`, absolute sigma is:

```python
scheduled_global_sigma_fraction * (gene.high - gene.low)
```

- `Gene.sigma` is a per-gene sigma fraction in `(0, 1]`.
- When `Gene.sigma` is present, it overrides the global scheduled value for that gene.
- Per-gene sigma overrides do not decay with the global schedule.
- Bool genes do not use sigma.
- Uniform mutation does not use sigma.
- Gaussian mutation on int genes uses absolute sigma, then rounds and clamps.

This matches the current `OperatorCodec.sigma_abs_list(...)` behavior and makes it public:
per-gene sigma is an override, not a multiplier on the global schedule.

## Selection Semantics

Selection operators should stay score-domain operators:

- `tournament` samples contenders with replacement.
- `roulette` derives positive weights from comparison scores while excluding invalid
  scores.
- `rank` assigns rank-based weights over comparison scores.

GA should continue to pass direction-adjusted comparison scores into reproduction. The
operator contract should avoid new public `fitness` terminology in Python-facing APIs and
docs. Internal Rust names can remain unchanged in this slice.

## Config And Reproducibility

All normalized operator specs should feed into `GeneticAlgorithmOptimizer.config()` and
`config_signature()`.

Recommended GA component shape:

```python
"components": {
    "crossover": {
        "type": "sbx",
        "operator_type": "crossover",
        "domain": "numeric",
        "parameters": {"eta": 2.0, "probability": 0.9},
    },
    "mutation": {
        "type": "gaussian",
        "operator_type": "mutation",
        "domain": "numeric",
        "parameters": {
            "probability": 0.1,
            "individual_probability": 1.0,
            "sigma": 0.2,
        },
    },
    "mutation_schedule": {
        "type": "constant",
        "parameters": {"sigma_end": 0.02},
    },
    "selection": {
        "type": "tournament",
        "operator_type": "selection",
        "domain": "score",
        "parameters": {"tournament_size": 3},
    },
    "bounds_policy": {
        "type": "clamp",
        "operator_type": "bounds",
        "domain": "repair",
        "parameters": {},
    },
}
```

Adding `operator_type`, `domain`, and `bounds_policy` changes GA config hashes on the
feature branch. That is acceptable in this stabilization phase because the contract is
becoming more explicit.

`Gene.sigma` values remain part of `GeneSpace.signature()`, not the optimizer config. The
optimizer config records the global mutation sigma and schedule. Comparing two runs still
requires both `gene_space_hash` and `optimizer_config_hash`.

## Custom Operator Protocols

Custom operators should be objects implementing explicit protocols. EvoCore should not
accept bare functions as custom operators in v1.

Recommended protocols:

```python
class CustomCrossoverOperator(Protocol):
    name: str
    operator_type: Literal["crossover"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def crossover(
        self,
        left: Sequence[GeneValue],
        right: Sequence[GeneValue],
        context: CrossoverContext,
    ) -> tuple[Sequence[GeneValue], Sequence[GeneValue]]: ...

class CustomMutationOperator(Protocol):
    name: str
    operator_type: Literal["mutation"]
    supported_gene_kinds: frozenset[GeneKind]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def mutate(
        self,
        values: Sequence[GeneValue],
        context: MutationContext,
    ) -> Sequence[GeneValue]: ...

class CustomSelectionOperator(Protocol):
    name: str
    operator_type: Literal["selection"]

    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
    def select(
        self,
        scores: Sequence[float],
        count: int,
        context: SelectionContext,
    ) -> Sequence[int]: ...
```

The protocols may be implemented as `typing.Protocol` classes plus runtime validation
helpers. A custom operator may also provide `config_signature()` for stable configuration
identity. The public docs should show concrete classes rather than requiring users to
understand protocol internals.

## Custom Operator Context

Custom operator context objects should be immutable and JSON-safe where practical:

```python
@dataclass(frozen=True)
class OperatorContext:
    gene_space: GeneSpace
    generation: int
    seed: int
    individual_index: int | None
    pair_index: int | None
    bounds_policy: BoundsPolicy
```

Specialized contexts may add component-specific fields:

```python
@dataclass(frozen=True)
class MutationContext(OperatorContext):
    probability: float
    mutation_sigma: float
    mutation_sigmas: tuple[float, ...]

@dataclass(frozen=True)
class CrossoverContext(OperatorContext):
    probability: float

@dataclass(frozen=True)
class SelectionContext(OperatorContext):
    tournament_size: int | None = None
```

The context should expose deterministic inputs but not force users to consume Rust's
internal RNG. If a custom operator needs random numbers, docs should recommend deriving a
local deterministic generator from `seed`, `generation`, and index fields.

## Custom Operator Execution

Execution rules:

- Built-in operators continue to use Rust-backed reproduction.
- Custom crossover, mutation, or selection switches reproduction to Python orchestration
  for that generation.
- EvoCore applies `BoundsPolicy.clamp()` after custom crossover and mutation unless the
  custom operator declares a future explicit `enforces_bounds=True` flag.
- EvoCore validates decoded Python values with `GeneSpace.validate_genes()` after bounds
  enforcement.
- Custom selection must return valid population indices of the requested count.
- Custom crossover must return exactly two children with the correct gene length.
- Custom mutation must return one child with the correct gene length.

The first implementation may keep custom execution limited to GA reproduction. CMA-ES does
not consume this operator API in v1.

## Custom Operator Reproducibility

Custom operators participate in config and reproducibility as follows:

- If a custom operator provides `config_signature()`, include that payload in the
  optimizer config component.
- If it does not provide `config_signature()`, include the module-qualified class identity
  in the optimizer config component and mark reproducibility `partial`.
- Never use opaque `repr(...)` output as a stable identity.
- Record the module-qualified class identity in reproducibility metadata.
- Even with `config_signature()`, mark reproducibility `partial` unless a later extension
  adds an explicit way for custom operators to declare fully reproducible behavior.

This is stricter than arbitrary callable support, but it matches EvoCore's reproducibility
direction. Users who want custom behavior must provide structured objects. Users who want
stronger comparison semantics should also provide stable `config_signature()` payloads.

## Implementation Notes

Recommended file changes:

```text
evocore/optimizers/operators.py      # public operator specs, protocols, normalization
evocore/optimizers/ga/config.py      # consume normalized operator signatures
evocore/optimizers/ga/engine.py      # accept typed specs and bounds_policy
evocore/optimizers/ga/reproduction.py# route built-ins to Rust, custom operators to Python
evocore/search_space/codec.py        # retain encoding/decoding and sigma helpers only
evocore/__init__.py                  # top-level re-exports
evocore/_core.pyi                    # update only if Rust-facing signatures change
docs/site/operator-contract.md       # public user docs
```

Normalization should happen once in the GA constructor, before run state is initialized.
The optimizer should retain normalized specs on attributes such as:

```python
self.crossover_operator
self.mutation_operator
self.selection_operator
self.bounds_policy
```

The old string attributes may remain for compatibility, but they should be derived from
the normalized objects. Internal code should move toward the normalized attributes.

## Testing Plan

Unit tests should cover:

- String aliases normalize to the same signatures as typed specs.
- GA explicit defaults still match implicit defaults.
- Operator parameter changes alter `config_hash()`.
- `uniform` crossover and `uniform` mutation have distinct canonical signatures.
- Numeric crossovers reject binary-only spaces.
- Binary crossovers reject numeric-only spaces.
- Numeric mutation rejects binary-only spaces.
- `bit_flip` rejects numeric-only spaces.
- Mixed `float` and `int` spaces remain valid for numeric operators.
- Mixed `bool` and numeric spaces remain rejected.
- Bounds policy preserves valid decoded values after Rust reproduction.
- Fixed numeric genes stay fixed after reproduction.
- Per-gene `Gene.sigma` overrides scheduled global sigma.
- Per-gene sigma does not decay with global schedules.
- Typed operator objects conflict with non-default legacy scalar args for the same
  component.
- Custom operator protocols validate shape, return lengths, return kinds, and indices.
- Custom operator config signatures appear in optimizer config.
- Custom operators are visible in reproducibility metadata.
- Legacy string behavior remains stable for existing GA tests.

Property tests should focus on:

- Bounds policy output validity across random numeric and binary gene spaces.
- Determinism for built-in operators under the same seed and generation.
- Normalized operator signatures round-trip through JSON-safe serialization.

## Documentation Plan

Add `docs/site/operator-contract.md` with:

- Overview of built-in operators.
- Compatibility table.
- String API and typed API examples.
- Bounds policy behavior.
- Mutation sigma semantics.
- Custom operator example.
- Reproducibility notes.

Update:

- `docs/site/ga.md` to link the operator contract and show typed examples.
- `docs/site/gene-space.md` to link per-gene sigma semantics to the operator contract.
- API docs to include the new public operator classes.

## Risks And Decisions

The main API risk is adding typed specs while keeping legacy scalar constructor arguments.
The conflict rule avoids silent ambiguity and gives users a clear migration path.

The main implementation risk is custom Python operators requiring a separate reproduction
path from the Rust built-ins. That is acceptable because custom execution is an extension
path, not the fast path. Built-ins should remain Rust-backed.

The main future-compatibility decision is rejecting mixed `bool` plus numeric spaces for
now. That keeps the v1 operator contract honest. Segmented operators can be designed later
without retrofitting ambiguous behavior into this API.

## Approval Notes

During brainstorming, the approved direction was:

- Public API now.
- Typed built-in operator specs plus constrained custom operator protocols.
- Existing string constructor arguments remain supported.
- Conservative compatibility by gene kind.
- `BoundsPolicy.clamp()` as the named v1 bounds behavior.
- `Gene.sigma` as a per-gene override that does not decay with global schedules.
- Custom operators must be structured objects, not bare callables. Stable
  `config_signature()` payloads are recommended; missing signatures are allowed but make
  reproducibility partial.
