# Python Package Domain Restructure Design

## Summary

EvoCore will move from a flat Python package to a domain-oriented package
architecture. This is an intentional breaking public API migration. The refactor
should preserve optimizer behavior, deterministic seed semantics, budget
accounting, checkpoint compatibility unless explicitly changed, and Rust-backed
operator behavior, while replacing bloated modules and awkward public vocabulary
with smaller, clearer domains.

The first implementation target is the Python package under `evocore/`. The
largest pain point is `evocore/ga.py`, currently over 1,200 lines and mixing
result containers, GA configuration, generation-loop execution, checkpoint
resume, ask/tell lifecycle state, policy-driven execution, and multi-run
dispatch. The broader package also exposes flat module names such as
`evocore.cmaes`, `evocore.gene_space`, `evocore.evaluation`, and
`evocore.stats`, which no longer match the conceptual shape of the project.

## Goals

- Split bloated Python modules into domain packages with focused files.
- Fully switch public direct module imports to new domain paths.
- Keep top-level convenience imports from `evocore`, but expose new names only.
- Rename awkward public vocabulary in one coherent breaking change.
- Preserve optimizer behavior and deterministic outputs except for intentional
  serialized result schema changes.
- Update tests, docs, changelog, and root `AGENTS.md` to teach the new layout.
- Make future edits easier by keeping each module small and purpose-specific.

## Non-Goals

- Do not change Rust extension APIs except where Python imports require stub
  maintenance.
- Do not change optimizer algorithms, randomness, selection, mutation,
  covariance updates, scheduling semantics, or budget accounting.
- Do not keep deprecated compatibility shims for old flat module paths.
- Do not perform release version bumps unless this work later becomes explicit
  release preparation.

## Target Package Architecture

```text
evocore/
  __init__.py                  # top-level convenience exports, new names only
  _core.pyi                    # Rust extension type stubs
  core/
    __init__.py
    errors.py                  # EvocoreError, ConfigurationError, FitnessError, warnings
    serialization.py           # JSON-safe export and stable hashing helpers
    parallel.py                # Thread/process evaluation helpers
  search_space/
    __init__.py
    genes.py                   # Gene, GeneSpace, GeneKind, GeneValue
    solutions.py               # Solution, SolutionSet
    codec.py                   # OperatorCodec and Rust boundary encoding
  lifecycle/
    __init__.py
    records.py                 # Candidate, EvaluationRecord, EvaluationContext, ScoreObservation
    policies.py                # BudgetPolicy, EvaluationStage
    scheduler.py               # BudgetScheduler
    protocols.py               # Optimizer, Evaluator
    telemetry.py               # OptimizationTelemetry, UpdateResult, OptimizerStateSummary
    events.py                  # EventRecord, EventHistory, StopReason
  results/
    __init__.py
    generation.py              # GenerationRecord, GenerationHistory
    reproducibility.py         # ReproducibilityMetadata
    run.py                     # OptimizationResult, OptimizationBatchResult
  optimizers/
    __init__.py
    ga/
      __init__.py              # GeneticAlgorithmOptimizer
      engine.py
      ask_tell.py
      generation_loop.py
      checkpointing.py
      reproduction.py
    cmaes/
      __init__.py              # CMAESOptimizer
      engine.py
      ask_tell.py
      mixed.py                 # IntegerMarginDistribution, CategoricalDistributionState
  callbacks/
    __init__.py                # Callback hooks and built-in callbacks
  surrogates/
    __init__.py                # InverseDistanceAdvisor, SurrogateScore
```

## Public Import Shape

Top-level convenience imports remain supported:

```python
from evocore import GeneticAlgorithmOptimizer, CMAESOptimizer
from evocore import Gene, GeneSpace, EvaluationRecord, OptimizationResult
from evocore import BudgetPolicy, EvaluationStage, BudgetScheduler
```

Direct module imports use domain paths:

```python
from evocore.optimizers.ga import GeneticAlgorithmOptimizer
from evocore.optimizers.cmaes import CMAESOptimizer
from evocore.search_space import Gene, GeneSpace, Solution, SolutionSet
from evocore.lifecycle import EvaluationRecord, BudgetPolicy, BudgetScheduler
from evocore.results import OptimizationResult, OptimizationBatchResult
```

Old flat module paths are removed from tests and docs and are not preserved as
compatibility shims:

```text
evocore.ga
evocore.cmaes
evocore.gene_space
evocore.individual
evocore.evaluation
evocore.policies
evocore.scheduler
evocore.stats
evocore.parallel
evocore.callbacks as a single file module
evocore.advisors
evocore.mixed_cma
```

## Naming Migration

Public names migrate as follows:

```text
GAEngine -> GeneticAlgorithmOptimizer
CMAESEngine -> CMAESOptimizer
RunResult -> OptimizationResult
MultiRunResult -> OptimizationBatchResult
GeneDef -> Gene
Individual -> Solution
Population -> SolutionSet
Rung -> EvaluationStage
MultiFidelityPolicy -> BudgetPolicy
EvaluationScheduler -> BudgetScheduler
TellResult -> UpdateResult
EngineStateSummary -> OptimizerStateSummary
Logbook -> GenerationHistory
LogEntry -> GenerationRecord
InverseDistanceSurrogateAdvisor -> InverseDistanceAdvisor
AdvisorScore -> SurrogateScore
IntegerMargin -> IntegerMarginDistribution
CategoricalState -> CategoricalDistributionState
OperatorSet -> OperatorCodec
CandidateScore -> ScoreObservation
```

Public result and export vocabulary moves from `fitness` to `score`:

```text
best_individual -> best_solution
final_population -> final_solutions
elite_history -> elite_solutions
diversity_history -> diversity_by_generation
logbook -> generations
history -> events
best_fitness -> best_score
mean_fitness -> mean_score
std_fitness -> std_score
fitness_summary -> score_summary
```

GA internals may still use fitness terminology where it is algorithmically
natural or where Rust functions require that vocabulary. Public classes, docs,
result exports, and top-level APIs should use score terminology.

## Domain Responsibilities

`evocore.search_space` owns user-facing problem representation. `GeneSpace`
validates definitions, maps decoded values to named parameters, and computes
stable signatures. `OperatorCodec` handles Rust boundary encoding and decoding.
`Solution` and `SolutionSet` represent decoded candidate values and summary
helpers.

`evocore.lifecycle` owns shared ask/tell contracts. It contains candidates,
evaluation records, evaluation stages, budget policies, schedulers, structural
protocols, telemetry, update summaries, optimizer state summaries, and lifecycle
events. This package is shared by all optimizers.

`evocore.results` owns completed-run reporting. It contains generation summaries,
event histories, reproducibility metadata, one-run result envelopes, and
multi-run result envelopes. Serialized exports should bump `schema_version` from
`1` to `2` because field names change.

`evocore.optimizers` owns algorithm implementations. The GA package separates
configuration/public class behavior from ask/tell state, generation-loop
execution, checkpointing, and reproduction. The CMA-ES package separates public
class behavior, ask/tell state, and mixed-variable helpers.

`evocore.core` owns foundational utilities: errors and warnings, serialization,
stable hashing, package version lookup, and parallel execution helpers.

`evocore.callbacks` owns lifecycle callbacks. It should become a package so that
callback types can grow without producing another oversized flat module.

`evocore.surrogates` owns advisor/surrogate ranking helpers. The current
inverse-distance baseline moves here and becomes `InverseDistanceAdvisor`.

## Error Handling And Compatibility

This is a breaking migration. Old public names should fail clearly instead of
silently aliasing to new names. Error messages and docs should reference the new
public names only, for example:

```text
GeneticAlgorithmOptimizer uses max_generations, not generations.
BudgetPolicy.single_full() uses max_evaluations=..., not budget=....
```

Serialized result exports change to schema version `2`. The export shape should
use `score`, `solutions`, `generations`, and `events`. Runtime timing remains
opt-in through `include_runtime=True`.

Checkpoint payload compatibility should be preserved unless implementation
finds a concrete blocker. If checkpoint compatibility cannot be preserved across
the `Individual` to `Solution` rename, the implementation plan must call this
out explicitly and add a targeted migration or documented breaking note.

## Documentation And AGENTS.md

User docs in `docs/site/` should switch to the new import paths and new public
names. MkDocs API references should point at domain modules rather than flat
modules. `CHANGELOG.md` must describe the breaking import-path and vocabulary
migration.

Root `AGENTS.md` should include the target package architecture and guardrails
for future chats:

- new optimizer work goes under `evocore/optimizers`;
- shared ask/tell contracts go under `evocore/lifecycle`;
- search-space and solution containers go under `evocore/search_space`;
- completed-run reporting goes under `evocore/results`;
- foundational utilities go under `evocore/core`;
- old flat module paths should not be recreated.

## Testing Strategy

The verification target is same optimizer behavior, new public surface.

Import contract tests should assert new top-level exports exist, new domain
imports work, and tests/docs no longer use old flat paths.

Behavior regression tests should keep current GA and CMA deterministic behavior
coverage while updating names and imports. Seed reproducibility, ask/tell
validation, budget accounting, callbacks, checkpoint resume, run-multiple
sorting, parallel rejection, and CMA state behavior should remain semantically
identical.

Export schema tests should validate schema version `2` and score-oriented keys:
`best.score`, `score_summary`, `generations`, `events`, `best_solution`, and
`final_solutions`.

Docs and API tests should build the MkDocs site with updated references and
examples.

## Verification Commands

Run the standard checks relevant to a public package restructure:

```bash
python -m ruff format --check
python -m ruff check
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
python -m pytest tests/property/ -v
python -m pytest tests/benchmarks/ -v
python -m mkdocs build
```

Rust checks should also pass because the Python package still depends on the
Rust extension:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Implementation Notes

Use an incremental implementation plan even though the migration is conceptually
big-bang. Move and rename in slices that keep tests informative:

1. Create domain packages and move foundational types.
2. Move result envelopes and update CMA to import results from `evocore.results`.
3. Split GA internals into focused modules under `evocore.optimizers.ga`.
4. Split CMA internals under `evocore.optimizers.cmaes`.
5. Update top-level exports, tests, docs, and changelog.
6. Remove old flat modules after all imports are migrated.

Avoid broad algorithm edits during the move. If a behavior change is required to
complete the restructure, isolate it in the implementation plan with a failing
test first.
