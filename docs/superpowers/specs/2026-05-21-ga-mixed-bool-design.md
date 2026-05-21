# GA Mixed Bool Support Design

**Date:** 2026-05-21
**Status:** Draft approved for specification
**Scope:** Add mixed `float`/`int`/`bool` `GeneSpace` support to `GeneticAlgorithmOptimizer` while keeping CMA-ES bool support out of scope

## Summary

EvoCore should make the existing GA the next practical mixed-variable optimizer by
supporting `GeneSpace` values that combine numeric and bool genes.

The current GA already supports numeric spaces and bool-only spaces, but it rejects
spaces that contain `bool` alongside `float` or `int`. That makes common parameter
tuning problems awkward because users must encode bool switches as integer genes.
This design removes that workaround for GA.

The first slice should not add Differential Evolution. It should extend
`GeneticAlgorithmOptimizer` so users can pass mixed spaces directly, use the default
constructor for the common case, and receive valid decoded Python values throughout
ask/tell, generation-loop runs, telemetry, and checkpoints.

CMA-ES should continue rejecting bool genes. Plain CMA-ES is a continuous
distribution optimizer, and bool-as-thresholded-float is not a good public contract.
A future mixed CMA design can combine continuous CMA state with Bernoulli or
categorical probability state, but that is separate from this GA feature.

## Current Context

Relevant existing behavior:

- `GeneSpace` supports `float`, `int`, and `bool` genes.
- `OperatorCodec` encodes bool as `0.0` or `1.0` across the Rust boundary and
  decodes back to Python `bool`.
- `BoundsPolicy.clamp()` already knows how to clamp numeric genes, round int
  genes, and threshold bool-compatible values.
- GA supports numeric spaces with operators such as `sbx`, `blx`, `uniform`, and
  `gaussian`.
- GA supports bool-only spaces when configured with binary operators such as
  `one_point` and `bit_flip`.
- The operator contract currently rejects mixed bool plus numeric spaces through
  `gene_space_domain(...)`.
- The GA Python reproduction path already applies mutation per gene and is a good
  place to implement typed mixed behavior without forcing a Rust redesign.

The user-facing gap is narrow: a mixed space like this should be valid for GA:

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

## Product Direction

GA should become EvoCore's general-purpose mixed flat-space optimizer. Users
should be able to model true bool switches as bool genes and trust EvoCore to keep
candidate values type-correct.

This is an additive behavior change, not a broad optimizer redesign. It should
reuse the existing GA lifecycle, results, checkpoint envelope, config hashing, and
operator vocabulary as much as possible.

The design should favor automatic behavior for the common case. A user who omits
operator settings for a mixed space should get a sensible typed GA instead of a
configuration error.

## Goals

- Allow `GeneticAlgorithmOptimizer` to accept mixed `float`/`int`/`bool` spaces.
- Make omitted GA operator arguments resolve to compatible defaults for numeric,
  bool-only, and mixed spaces.
- Keep decoded candidate genes and params type-correct in ask/tell and run APIs.
- Preserve existing numeric-only GA behavior and config hashes.
- Preserve existing explicit bool-only GA behavior.
- Keep `sbx` and `blx` numeric-only.
- Keep `one_point` and `two_point` out of mixed spaces in this slice.
- Extend docs and tests so the mixed bool contract is visible and stable.
- Leave CMA-ES bool support unsupported and documented as out of scope.

## Non-Goals

- Do not add Differential Evolution in this slice.
- Do not add bool support to `CMAESOptimizer`.
- Do not encode bool as user-facing integer workarounds in docs.
- Do not add categorical, permutation, conditional, graph, object, or nested
  search spaces.
- Do not redesign `GeneSpace`.
- Do not replace the GA constructor with a config object.
- Do not introduce separate public operator names such as `mixed_uniform` or
  `mixed_gaussian_bitflip`.
- Do not move mixed GA reproduction into Rust unless implementation discovers a
  small safe Rust-side dispatch change is clearly better.

## Public Contract

`GeneticAlgorithmOptimizer` accepts mixed flat spaces with any ordering of
`float`, `int`, and `bool` genes.

When operator arguments are omitted, GA resolves defaults by space profile:

| Space profile | Default crossover | Default mutation |
| --- | --- | --- |
| Numeric-only (`float`, `int`) | `sbx` | `gaussian` |
| Bool-only (`bool`) | `uniform` | `bit_flip` |
| Mixed (`float`/`int` plus `bool`) | `uniform` | `gaussian` |

The numeric-only omitted defaults must continue producing the same effective
operator signatures and config hashes as today.

Explicit operator choices remain respected:

- `uniform` crossover is valid for numeric-only, bool-only, and mixed spaces.
- `gaussian` mutation is valid for numeric-only and mixed spaces. In mixed
  spaces it mutates numeric genes with Gaussian noise and flips bool genes.
- `uniform` mutation is valid for numeric-only and mixed spaces. In mixed spaces
  it resamples numeric genes and flips bool genes.
- `bit_flip` mutation is valid for bool-only and mixed spaces. In mixed spaces it
  only affects bool genes; numeric genes pass through unchanged.
- `sbx` and `blx` reject any space containing bool.
- `one_point` and `two_point` reject mixed spaces in this slice.

If implementation needs to distinguish omitted arguments from explicit
`crossover="sbx"` or `mutation="gaussian"`, it should use an internal sentinel or
equivalent normalization helper. Explicit incompatible operator choices should
still raise `ConfigurationError`.

## Architecture

Keep the feature inside the existing GA and operator modules:

- `evocore.optimizers.operators` owns compatibility normalization and validation.
- `evocore.search_space.codec.OperatorCodec` remains the encode/decode boundary.
- `evocore.optimizers.ga.engine.GeneticAlgorithmOptimizer` resolves omitted
  operator defaults based on the gene-space profile before building operators.
- `evocore.optimizers.ga.reproduction.GeneticAlgorithmReproductionMixin` routes
  mixed spaces through the Python reproduction path.

Do not create a new optimizer package.

Do not treat mixed spaces as a new public gene kind. They are still flat
`GeneSpace` values; reproduction is typed per gene.

Implementation can introduce a private helper such as `gene_space_profile(...)`
with values equivalent to numeric-only, bool-only, and mixed. That helper should
replace the current mixed-space rejection in `gene_space_domain(...)` or sit
alongside it, depending on the smallest clean patch.

The Rust fast path should remain available for existing homogeneous configurations
where it is already safe. Mixed spaces should use Python reproduction first
because Python already has type-aware per-gene hooks and `BoundsPolicy.clamp()`.

## Reproduction Behavior

Initialization remains Rust-backed through `init_population(...)`; it already
receives per-gene kinds and decodes bool values correctly.

For mixed spaces, parent selection remains unchanged.

Uniform crossover:

- Uses existing allele-swap behavior.
- Works across all gene kinds.
- Decodes back to `float`, `int`, and `bool` values through `OperatorCodec`.

Gaussian mutation:

- `float`: add Gaussian noise, then clamp.
- `int`: add Gaussian noise, then round and clamp.
- `bool`: flip with `mutation_prob`.

Uniform mutation:

- `float`: resample uniformly within bounds.
- `int`: resample an integer uniformly within bounds.
- `bool`: flip with `mutation_prob`.

Bit-flip mutation:

- `bool`: flip with `mutation_prob`.
- `float` and `int`: leave unchanged.

Bounds enforcement remains centralized through `apply_bounds_policy(...)` so
offspring are validated before becoming `Solution` objects.

## Ask/Tell, Run, And Checkpoints

Ask/tell semantics should not change.

Mixed candidates returned by `ask(...)` should carry:

- Stable `candidate_id`.
- Stable `batch_id`.
- Decoded `genes` with real Python `bool` values.
- `params` matching named gene spaces.
- Existing event and telemetry behavior.

`tell(...)` should keep the same validation rules around candidate IDs, batch IDs,
duplicate records, confidence handling, cached records, and best-state tracking.

Generation-loop `run(...)` should work for mixed spaces with the existing
objective callable shape and produce normal `OptimizationResult` objects.

Stable checkpoints should preserve mixed values in the same candidate, batch,
event, telemetry, and optimizer-config sections used today. Existing v0.8.0
checkpoint fixtures do not need to change because this feature is additive. New
mixed bool checkpoint coverage can be generated as unit tests rather than as a
new compatibility baseline fixture unless a release plan explicitly asks for it.

## Error Handling

Raise `ConfigurationError` with explicit operator names and gene kinds when a
user selects an incompatible operator.

Required rejection cases:

- `sbx` with any bool gene.
- `blx` with any bool gene.
- `one_point` with mixed bool plus numeric genes.
- `two_point` with mixed bool plus numeric genes.
- `bit_flip` with numeric-only spaces.
- Custom operators whose declared supported gene kinds do not cover the current
  space.

Keep `CMAESOptimizer` errors unchanged for bool spaces.

## Optimizer Config And Reproducibility

Effective operator choices should appear in `OptimizerConfig` exactly like other
GA operators do today.

For numeric-only default GA, the effective config signature should remain stable:

- Crossover remains `sbx`.
- Mutation remains `gaussian`.
- Config hash remains unchanged.

For bool-only default GA, the new effective default is `uniform` crossover and
`bit_flip` mutation.

For mixed default GA, the new effective default is `uniform` crossover and
`gaussian` mutation with typed bool flipping.

Runtime hook and custom-operator reproducibility behavior should not change.

## Documentation

Update user-facing docs:

- `docs/site/gene-space.md`: bool genes can be mixed with numeric genes for GA.
- `docs/site/operator-contract.md`: document typed mixed GA operator behavior and
  the remaining incompatible operator cases.
- `docs/site/ga.md`: add a short mixed bool example with default operators.
- `docs/site/cmaes.md`: clarify that CMA-ES still rejects bool and why.
- `CHANGELOG.md`: note additive GA mixed bool support.

Avoid documenting bool-as-int workarounds as the recommended path.

## Tests

Add focused unit and integration coverage:

- GA accepts a mixed `GeneSpace` with omitted operators.
- Mixed initialization returns valid Python `bool` values, not `0` or `1`.
- Mixed default config resolves to `uniform` crossover and `gaussian` mutation.
- Numeric-only default config hash remains unchanged.
- Bool-only default GA works without explicit operators.
- `uniform` crossover preserves valid mixed gene types.
- `gaussian` mutation can flip bool genes in a mixed space.
- `uniform` mutation can flip bool genes in a mixed space.
- `bit_flip` only changes bool genes in a mixed space.
- `sbx`, `blx`, `one_point`, and `two_point` reject mixed spaces.
- ask/tell accepts mixed candidates and preserves params.
- generation-loop `run(...)` can optimize a simple mixed objective.
- stable checkpoint round trip preserves mixed values.
- CMA-ES still rejects bool.

Property tests can extend existing gene-space and operator contract tests with
mixed spaces once deterministic examples pass.

## Compatibility Notes

This is a public behavior expansion. It should not break existing numeric or
explicit bool-only GA users.

The only intentional change is that some configurations that previously raised
`ConfigurationError` now run:

- Mixed bool plus numeric GA spaces with compatible or omitted operators.
- Bool-only GA spaces with omitted operators.

Incompatible explicit operator choices should continue failing loudly.

## Approved Decisions

- Build this before adding Differential Evolution.
- Include mixed `bool` support in GA, not CMA-ES.
- Make common mixed behavior automatic with omitted operator arguments.
- Keep the design scoped to flat `GeneSpace` values and existing GA lifecycle
  surfaces.
