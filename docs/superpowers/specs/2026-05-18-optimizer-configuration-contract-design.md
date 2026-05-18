# Optimizer Configuration Contract Design

**Date:** 2026-05-18
**Status:** Draft approved for specification
**Scope:** Stabilize exportable, comparable optimizer configuration and hook-aware reproducibility metadata for EvoCore optimizers

## Summary

EvoCore should make optimizer configuration a first-class public contract. Each optimizer
needs an exportable, comparable, JSON-safe configuration signature so users can answer a
simple question: are these two optimizer runs using the same reproducible algorithm setup?

The contract should remain compatible with the current constructor-first API. This slice
does not need to force users to instantiate separate config objects, but it should expose
public methods that return canonical configuration data, compute stable hashes, and
validate compatibility between optimizer, operators, custom components, runtime hooks, and
`GeneSpace`.

The design is hook-aware. EvoCore should not pretend that arbitrary callables are fully
reproducible just because they have a `repr(...)`. Built-in configuration is strict and
fully comparable. Custom algorithmic components may participate in reproducibility when
they implement a small EvoCore export protocol. Runtime hooks are visible in
reproducibility metadata but kept separate from the core optimizer configuration hash.

## Product Direction

EvoCore should feel like a general evolutionary optimization package rather than a set of
engine constructors. A user should be able to inspect an optimizer before or after a run:

```python
optimizer = GeneticAlgorithmOptimizer(space, population_size=64, seed=42)

optimizer.config_signature()
optimizer.config_hash()
optimizer.validate_compatibility()

result = optimizer.run(evaluator)
result.reproducibility.optimizer_config
result.reproducibility.optimizer_config_hash
```

The primary comparison target is the optimizer algorithm configuration, not the whole
execution environment. Progress bars, checkpoint paths, metrics log files, process worker
setup, and evaluator identity should be recorded, but they should not silently change the
core optimizer configuration hash unless they are deliberately modeled as algorithmic
components.

## Goals

- Add stable public optimizer configuration export methods.
- Define which optimizer constructor values are part of reproducible algorithm identity.
- Define stable defaults as public behavior.
- Add a canonical optimizer config hash.
- Represent hook and component identity honestly in reproducibility metadata.
- Distinguish algorithm components, termination hooks, artifact hooks, and environment
  hooks.
- Allow custom algorithmic components to participate in reproducibility through an explicit
  protocol.
- Make compatibility validation explicit and testable.
- Keep GA and CMA-ES optimizer-specific fields while sharing one public export envelope.
- Extend `OptimizationResult.reproducibility` without mixing runtime-only hooks into the
  core config hash.

## Non-Goals

- Do not replace all constructors with config dataclasses in this slice.
- Do not implement custom operators, repair strategies, replacement strategies, or
  categorical/permutation gene support in this slice.
- Do not introduce plugin loading or dynamic component registries.
- Do not add result/config `from_dict()` or `from_json()` loaders.
- Do not make callbacks part of the core optimizer hash by default.
- Do not redesign objective/evaluator semantics, budget policies, checkpoint compatibility,
  island models, or multi-objective results.
- Do not change deterministic seed behavior unless a compatibility bug is found during
  implementation.

## Public API Shape

Each optimizer should expose:

```python
def config(self) -> OptimizerConfig: ...
def config_signature(self) -> dict[str, Any]: ...
def config_hash(self) -> str: ...
def validate_compatibility(self) -> None: ...
```

`config()` returns a frozen public value, likely a dataclass or small domain object, that
can export itself. The first implementation may build it from existing constructor
attributes rather than requiring users to pass config objects directly.

`config_signature()` returns the canonical JSON-safe payload used for comparison and
hashing. It should include a schema version and an optimizer type.

`config_hash()` returns a stable SHA-256 hash over the canonical config signature. It
should use the existing deterministic JSON helper style used for `GeneSpace.hash()`.

`validate_compatibility()` checks all optimizer, operator, gene-space, component, and hook
compatibility rules. Constructors should continue calling validation, but the public method
lets users inspect composed optimizer setups explicitly.

Recommended shared module:

```text
evocore/optimizers/config.py
```

Initial contents:

```python
class ConfigurableComponent(Protocol):
    def config_signature(self) -> dict[str, Any]: ...
    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...

@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: str
    schema_version: int
    parameters: dict[str, Any]
    components: dict[str, Any]

@dataclass(frozen=True)
class RuntimeHookSignature:
    hook_type: str
    identity: str
    config: dict[str, Any]
    reproducibility: str
    notes: tuple[str, ...] = ()
```

The exact class names can adjust during implementation, but the stable public ideas should
remain: canonical config signature, config hash, explicit hook signatures, and explicit
compatibility validation.

## Canonical Config Signature

The canonical optimizer config payload should be JSON-safe and stable:

```python
{
    "schema_version": 1,
    "optimizer_type": "GeneticAlgorithmOptimizer",
    "parameters": {
        "population_size": 64,
        "max_generations": 100,
        "direction": "maximize",
        "seed": 42,
        "parallel": "none",
        "n_workers": None,
    },
    "components": {
        "crossover": {"type": "sbx", "parameters": {"probability": 0.9, "eta": 2.0}},
        "mutation": {
            "type": "gaussian",
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
            "parameters": {"tournament_size": 3},
        },
    },
}
```

Rules:

- Include `schema_version`.
- Include `optimizer_type`, not legacy engine vocabulary.
- Include `seed` and `direction` because they affect reproducible behavior.
- Include stable public defaults even when users do not pass them explicitly.
- Include optimizer-specific parameters under stable keys.
- Represent built-in algorithm components by stable type names and explicit parameters.
- Exclude `gene_space_signature` from `optimizer_config`; result metadata already records
  the gene-space signature and hash separately.
- Exclude evaluator/objective identity from `optimizer_config`; evaluator semantics are a
  separate reproducibility concern.
- Exclude runtime wall-clock observations and output paths from the core config hash.

## Optimizer-Specific Fields

### GeneticAlgorithmOptimizer

Core reproducible fields:

- `population_size`
- `max_generations`
- `seed`
- `direction`
- `crossover`
- `crossover_prob`
- `crossover_eta`
- `crossover_alpha`
- `mutation`
- `mutation_prob`
- `mutation_individual_prob`
- `mutation_sigma`
- `mutation_sigma_schedule`
- `mutation_sigma_end`
- `selection`
- `tournament_size`
- `elitism`
- `max_evaluations`
- `track_diversity`
- `parallel`
- `n_workers`

GA component groups:

- `crossover`
- `mutation`
- `mutation_schedule`
- `selection`
- future `repair`
- future `replacement`

`parallel` and `n_workers` should stay in the exported config because they are public
constructor values and may influence execution behavior. They should not make the
algorithmic component hash unstable through environment-specific data.

`process_initializer` and `process_initargs` are environment hooks, not core config
fields.

### CMAESOptimizer

Core reproducible fields:

- `population_size`
- `initial_mean`
- `initial_sigma`
- `max_generations`
- `seed`
- `direction`
- `parallel`
- `n_workers`
- `track_diversity`

CMA-ES component groups:

- `distribution`
- future `integer_handling`
- future `restart_strategy`
- future `boundary_strategy`

The initial implementation may keep these as parameters rather than separate named
components, but the signature should leave room for componentization without changing the
top-level envelope.

## Component And Hook Classification

Values should be classified into four groups.

### Algorithm Components

Algorithm components affect candidate generation, candidate transformation, selection,
state updates, or replacement. They participate in `optimizer_config_hash`.

Examples:

- initialization strategy
- crossover
- mutation
- mutation schedule
- selection
- repair
- replacement
- CMA distribution strategy
- CMA boundary handling
- future restart strategy

Built-in components are represented by stable name and explicit parameters.

Custom algorithm components must implement:

```python
class ConfigurableComponent(Protocol):
    def config_signature(self) -> dict[str, Any]: ...
    def validate_compatibility(self, gene_space: GeneSpace) -> None: ...
```

Opaque custom algorithm components should not be silently accepted into a fully
reproducible config. The preferred behavior is:

- reject opaque algorithm components at construction when they are passed to algorithmic
  slots that require reproducibility, or
- allow them only when an explicit user opt-in marks reproducibility as partial.

The first implementation can choose the stricter behavior because EvoCore does not yet
expose public custom algorithmic slots.

### Termination Hooks

Termination hooks affect when a run stops but do not define candidate generation or state
updates directly. They must be visible in reproducibility metadata.

Examples:

- `EarlyStopping`
- future target-score stops
- future patience policies
- future wall-clock stop hooks

Termination hooks should not be included in the core optimizer algorithm hash by default.
They should appear in a separate hook list and may influence
`reproducibility_status`.

Built-in termination hooks should export stable configuration when possible:

```python
{
    "hook_type": "termination",
    "identity": "evocore.callbacks.EarlyStopping",
    "config": {"patience": 10, "min_delta": 1e-06},
    "reproducibility": "configured",
}
```

Opaque termination hooks are allowed but should mark the run as partially reproducible
unless they implement a hook signature protocol.

### Artifact Hooks

Artifact hooks produce side effects but do not change optimizer decisions.

Examples:

- `ProgressBar`
- `MetricsLogger`
- `CheckpointCallback`

Artifact hooks should be visible in reproducibility metadata but excluded from the core
optimizer config hash. Paths and output destinations should not change
`optimizer_config_hash`.

Built-in artifact hooks can export their identity and configuration. Path-like values may
be recorded as metadata but should not participate in the core algorithm comparison.

### Environment Hooks

Environment hooks configure process/thread execution or external runtime setup.

Examples:

- `process_initializer`
- `process_initargs`
- worker setup functions
- future executor factories

Environment hooks should be recorded outside `optimizer_config_hash`. Opaque environment
hooks generally make reproducibility partial because their behavior can affect evaluator
state, external data availability, or process-local globals.

## Reproducibility Metadata

`ReproducibilityMetadata` should be extended with config hash and hook-aware status:

```python
@dataclass(frozen=True)
class ReproducibilityMetadata:
    evocore_version: str
    optimizer_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    optimizer_config_hash: str
    reproducibility_status: Literal["full", "partial"]
    reproducibility_notes: tuple[str, ...] = ()
    runtime_hooks: tuple[RuntimeHookSignature, ...] = ()
    extension: dict[str, Any] = field(default_factory=dict)
```

Rules:

- `optimizer_config` is the canonical config signature.
- `optimizer_config_hash` hashes only `optimizer_config`.
- `gene_space_signature` and `gene_space_hash` remain separate.
- `reproducibility_status="full"` means the optimizer config, gene space, and declared
  deterministic components are exportable and comparable.
- `reproducibility_status="partial"` means the run used at least one opaque hook,
  environment dependency, or custom component that is not fully exportable.
- `reproducibility_notes` should explain why a run is partial.
- Runtime, termination, artifact, environment, and evaluator identities should be recorded
  outside `optimizer_config` so config comparison remains meaningful.

`OptimizationResult.to_dict()` should include the new fields through
`reproducibility.to_dict()`. Deterministic JSON export should remain stable when runtime is
not requested.

## Compatibility Validation

Compatibility validation should become explicit and shared where possible.

Validation should cover:

- optimizer supports the configured `GeneSpace` kinds
- optimizer supports fixed genes in the configured positions
- operator supports the `GeneSpace` kinds
- operator parameters are valid
- mutation sigma and per-gene sigma values are compatible
- parallel mode is supported by the optimizer
- custom components declare compatibility
- runtime hooks are classified and representable for metadata
- opaque algorithmic components are rejected or marked partial according to policy

Existing examples:

- `CMAESOptimizer` rejects bool genes.
- `CMAESOptimizer` rejects fixed numeric genes until reconstruction is supported.
- `CMAESOptimizer` rejects `parallel="process"`.
- GA binary spaces require binary crossover and mutation.
- GA numeric spaces require numeric crossover and mutation.
- GA rejects mixed bool and numeric spaces through operator compatibility validation.

The current `OperatorCodec` validation can remain the low-level operator/gene-space check,
but optimizer config validation should call into it through a public compatibility path
rather than making it the only place the rules live.

## Error Policy

Configuration and compatibility errors should raise `ConfigurationError` with messages
that name the incompatible field and suggested fix.

Examples:

```text
GeneticAlgorithmOptimizer binary GeneSpace requires mutation='bit_flip'.
CMAESOptimizer does not support bool genes; use GeneticAlgorithmOptimizer or encode booleans as int genes.
Custom mutation component must implement config_signature() to participate in reproducibility.
EarlyStopping callback is outcome-affecting but does not expose a stable hook signature.
```

Warnings may be appropriate for artifact or environment hooks that are valid but make
reproducibility partial. Algorithmic incompatibilities should raise.

## Stable Defaults

Defaults are public contract. A default value changing in a later release changes config
signatures and should be treated as user-visible behavior.

The config signature should include defaults explicitly. This means these two optimizers
produce identical config signatures:

```python
GeneticAlgorithmOptimizer(space)
GeneticAlgorithmOptimizer(
    space,
    population_size=100,
    max_generations=100,
    crossover="sbx",
    mutation="gaussian",
    selection="tournament",
    seed=0,
)
```

The changelog should call out future default changes, especially for operator names,
operator parameters, seed behavior, direction, termination, and compatibility rules.

## Documentation

Required docs updates:

- `docs/site/ga.md`: show config export and config hash examples.
- `docs/site/cmaes.md`: show config export and config hash examples.
- `docs/site/optimizer-telemetry.md` or a new configuration page: explain
  reproducibility metadata fields.
- `docs/site/gene-space.md`: mention gene-space hash is separate from optimizer config
  hash.
- `docs/site/api.md`: include public config helpers and reproducibility metadata fields.
- `CHANGELOG.md`: note the new optimizer configuration contract.

Docs should explain that config export is for reproducibility comparison, not checkpoint
resume. There is no config loader or run replay promise in this slice.

## Testing

Required unit tests:

- GA `config_signature()` is deterministic.
- GA `config_hash()` is deterministic.
- GA default constructor and explicit default constructor produce equivalent signatures.
- GA algorithm component changes alter the config hash.
- GA artifact hook path changes do not alter the config hash.
- GA termination hook configuration is visible in reproducibility metadata.
- GA opaque runtime hook marks reproducibility partial.
- CMA-ES `config_signature()` is deterministic.
- CMA-ES `config_hash()` is deterministic.
- CMA-ES default constructor and explicit default constructor produce equivalent
  signatures.
- CMA-ES `initial_mean` and `initial_sigma` changes alter the config hash.
- `OptimizationResult.reproducibility.optimizer_config_hash` equals
  `optimizer.config_hash()`.
- `OptimizationResult.to_json()` remains deterministic by default.
- Existing compatibility errors for bool genes, fixed numeric genes, binary operators, and
  process parallelism remain clear.
- Custom component test doubles with `config_signature()` participate in config export.
- Opaque algorithm component test doubles are rejected or mark reproducibility partial
  according to the implementation policy.

Required property tests:

- Generated JSON-safe config signatures round-trip through JSON parsing.
- Equivalent optimizer configs produce equivalent hashes.
- Small parameter changes alter the hash for reproducibility-critical fields.

## Verification

Because this is a public Python API and docs change, implementation should run targeted
tests first:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py tests/unit/test_stats.py -v
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
git diff --check
```

If implementation touches Rust-backed operator validation or PyO3 signatures, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

For this design document itself, a spec review and `git diff --check` are sufficient.

## Rollout

Recommended implementation slices:

1. Add shared config protocol, signature, hash, hook signature, and reproducibility-status
   helpers.
2. Make `GeneticAlgorithmOptimizer` export the new config shape.
3. Make `CMAESOptimizer` export the new config shape.
4. Extend `ReproducibilityMetadata` with `optimizer_config_hash`,
   `reproducibility_status`, `reproducibility_notes`, and runtime hook signatures.
5. Move current compatibility checks behind explicit public `validate_compatibility()`
   methods while preserving constructor behavior.
6. Add docs and changelog coverage.
7. Run targeted verification, then full Python verification after extension rebuild if
   implementation affects runtime behavior.

## Acceptance Criteria

- Each optimizer exposes public config export, config signature, config hash, and
  compatibility validation.
- Built-in optimizer defaults appear explicitly in canonical config signatures.
- Config hashes are stable and compare reproducibility-critical optimizer settings.
- Gene-space signatures and hashes remain separate from optimizer config signatures and
  hashes.
- Algorithm components participate in the optimizer config hash.
- Termination hooks are visible in reproducibility metadata.
- Artifact and environment hooks are recorded outside the core config hash.
- Opaque custom algorithmic components are rejected or explicitly mark reproducibility as
  partial.
- `OptimizationResult.reproducibility` includes config hash and hook-aware status.
- Existing GA and CMA-ES compatibility errors remain clear.
- The slice does not add custom operator behavior, config loaders, replay semantics, or
  broad plugin architecture.

## Deferred Follow-Ups

- Constructor support for passing first-class config objects directly.
- `from_dict()` or `from_json()` config loaders.
- Public custom operator, repair, replacement, and restart strategy APIs.
- Component registries and plugin discovery.
- Objective/evaluator reproducibility contract.
- Checkpoint and result replay semantics.
- Categorical, permutation, conditional, and dependent gene spaces.
- Multi-objective configuration and result schemas.
