# Changelog

All notable changes to evocore are documented here.

This project follows semantic versioning after the v0.5.0 late-beta baseline.

## [Unreleased]

### Added

- Added external-state integration APIs for GA, DE, and CMA-ES, including warm
  starts, candidate injection, read-only population snapshots, top-k candidate
  access, cached evaluation record helpers, and expensive external evaluation
  recipes.

## [1.0.0] - 2026-06-05

### Added

- Added neutral search-space codec helpers for gene repair and Rust/operator
  boundary encoding: `repair_gene_value(...)`, `repair_gene_values(...)`,
  `encode_gene_values(...)`, and `decode_gene_values(...)`.
- Added practical examples documentation for common EvoCore workflows including
  continuous tuning, mixed-variable configuration, binary selection, budgeted
  evaluation, ask/tell queues, CMA-ES, multi-run comparison, and result export.

### Changed

- Centralized GA, CMA-ES, and Differential Evolution ask/tell helper logic for
  gene repair, event records, telemetry counting, and synchronous evaluator
  record validation without changing optimizer seed, checkpoint, or public
  lifecycle semantics.

## [0.10.0] - 2026-06-05

### Added

- Added `DifferentialEvolutionOptimizer` with mixed bool/numeric gene support,
  ask/tell replacement decisions, stable ask/tell checkpoints, and synchronous
  evaluator-driven runs.
- Added `DifferentialEvolutionOptimizer.run_multiple(...)` and policy-driven
  `DifferentialEvolutionOptimizer.run(evaluator, policy=...)` with delayed
  target-slot replacement until final state-eligible budget stages.
- Added built-in Differential Evolution strategies `best1bin`, `rand2bin`, and
  `current-to-best1bin` with strategy-aware validation and reproducibility
  metadata.
- Added `strategy="jde-rand1bin"` for simple jDE-style Differential Evolution
  adaptation with checkpointed per-slot mutation and crossover parameters.
- Added committed Differential Evolution v0.9.0 golden checkpoint fixtures with
  manifest hashes and deterministic continuation coverage.
- Documented Differential Evolution as a stable ask/tell checkpoint surface,
  including reproducibility guarantees, target replacement decisions, and
  current feature limitations.
- Added `AcceptanceDecision` and `UpdateResult.state_accepted_count` to
  distinguish ask/tell record acceptance from optimizer state acceptance.

### Changed

- Migrated Differential Evolution trial proposal generation for built-in
  strategies to the Rust extension while keeping Python ask/tell, replacement,
  checkpoint, event, telemetry, and policy semantics unchanged. Seeded DE runs
  remain deterministic within the same EvoCore version, but exact trial
  sequences may differ from the previous Python-generated strategy path.

## [0.9.0] - 2026-05-21

### Added

- `GeneticAlgorithmOptimizer` now accepts mixed flat `float`/`int`/`bool`
  `GeneSpace` values with profile-aware default operators, typed bool mutation,
  ask/tell, run, and checkpoint coverage. CMA-ES continues to reject bool genes.

## [0.8.1] - 2026-05-21

### Added

- Checkpoint golden fixtures documenting EvoCore 0.8.0 as the stable JSON
  checkpoint compatibility baseline for GA generation-loop, GA ask/tell, and
  CMA-ES ask/tell resume.

## [0.8.0] - 2026-05-21

### Added

- Structural `Optimizer` and `Evaluator` protocols for the clean-break ask/tell
  lifecycle.
- `EvaluationContext`, `UpdateResult`, and shared `OptimizerStateSummary` records for
  evaluator calls, `tell(...)` summaries, and stable state inspection.
- Public `batch_id` fields on vNext `Candidate` and `EvaluationRecord` so async
  evaluators can group results by ask batch.
- Stable `OptimizationResult`, `OptimizationBatchResult`, `GenerationHistory`, and `OptimizationTelemetry` export
  helpers with deterministic JSON output by default.
- Append-only `EventRecord` and `EventHistory` APIs for ask/tell audit rows and
  generation-level observations.
- `ReproducibilityMetadata` on run results with version, optimizer, seed, direction,
  gene-space signature/hash, and serializable optimizer configuration.
- `GeneSpace` now owns stable `signature()`, `hash()`, `to_dict()`, `to_json()`,
  and `validate_genes(...)` helpers for the flat search-space contract.
- Schema-aware `GeneSpace.value_signature(...)` and `GeneSpace.value_hash(...)`
  helpers for stable search-point identity.
- Lifecycle conversion helpers for explicit `Candidate` to `Solution` and `Solution` to
  `Candidate` transitions.
- Public optimizer configuration signatures and hashes for `GeneticAlgorithmOptimizer`
  and `CMAESOptimizer`, with hook-aware reproducibility metadata.
- Public GA operator contract specs for crossover, mutation, selection, bounds policy,
  compatibility validation, sigma semantics, and custom operator extension.
- Stable JSON checkpoint envelope helpers and GA generation-loop checkpoint/resume
  support with optimizer, gene-space, config, seed, direction, and seed-derivation
  validation.
- Added stable GA ask/tell checkpoints with pending-batch and partial-tell
  resume support.
- Added stable CMA-ES ask/tell checkpoints with Rust state snapshot resume,
  pending-batch, and partial-tell support.
- Rust-backed `PyCMAESState.to_dict()` and `PyCMAESState.from_dict(...)`
  snapshots for deterministic CMA-ES state continuation primitives.

### Changed

- Budget and termination vocabulary now uses `max_generations` and
  `max_evaluations` consistently. Legacy generation and policy-budget names, along
  with the old `RunResult` stop booleans, were removed in favor of
  `OptimizationResult.stop_reason`.
- Whole-package Python imports now use domain packages: `evocore.search_space`,
  `evocore.lifecycle`, `evocore.results`, `evocore.optimizers`, `evocore.core`, and
  `evocore.surrogates`.
- Public optimizer names are now `GeneticAlgorithmOptimizer` and `CMAESOptimizer`.
  Result fields use `best_solution`, `best_score`, `final_solutions`, `generations`,
  `events`, `elite_solutions`, and `diversity_by_generation`.
- Public search-space names are now `Gene`, `Solution`, `SolutionSet`, and
  `OperatorCodec`.
- Compatibility aliases for `Solution.genes`, `Solution.fitness`,
  `Solution.fitness_valid`, `SolutionSet.mean_fitness()`,
  `SolutionSet.std_fitness()`, `OperatorCodec.encode_genes()`,
  `OperatorCodec.decode_genes()`, and `OperatorCodec.decode_individual()` were
  removed. Use `values`, `score`, `score_valid`, `mean_score()`, `std_score()`,
  `encode_values()`, `decode_values()`, and `decode_solution()`.
- Evaluator context and record fields now use `stage`; telemetry exports now use
  `promoted_by_stage`, `eliminated_by_stage`, and `cost_by_stage`.
- `callbacks`, `surrogates`, `lifecycle.events`, `lifecycle.telemetry`, and
  `results.reproducibility` now own their implementations in focused modules instead
  of using implementation-heavy `__init__.py` files or re-export shims.
- `evocore.optimizers.ga` now splits ask/tell, generation-loop execution,
  checkpoint resume, multi-run handling, and reproduction into separate modules.
  `evocore.optimizers.cmaes` now keeps CMA-ES ask/tell state handling in its own
  module.
- `GeneticAlgorithmOptimizer` and `CMAESOptimizer` now expose `direction` and preserve raw scores while
  using direction-aware comparisons for best-candidate tracking.
- Policy-driven evaluators now receive `EvaluationContext` instead of a bare rung.
- `GeneticAlgorithmOptimizer` and `CMAESOptimizer` ask/tell flows now treat partial trusted batches as a
  first-class API, with strict duplicate and batch-mismatch validation.
- `GeneticAlgorithmOptimizer.run(...)` now fails fast when a synchronous evaluator omits assigned
  candidates instead of stalling the policy loop.
- `BudgetPolicy` now requires exactly one `trusted_full` stage and it must be the
  final stage.
- `ProcessParallel` now reuses a persistent process pool across repeated `evaluate(...)`
  calls until closed.
- Cached evaluation records remain eligible for optimizer state updates but no longer
  consume fresh full-evaluation budget; they are counted through
  `OptimizationTelemetry.candidates_cached` and `UpdateResult.cached_count`.
- Objective records now reject non-finite scores uniformly, and `rejected` records must
  use `score=None` with diagnostics in metrics or metadata.
- Runtime timing in result exports now lives under `runtime` and is included only when
  callers pass `include_runtime=True`.
- Run reproducibility metadata now uses the canonical `GeneSpace` signature and hash,
  including `schema_version` and per-gene `is_fixed` metadata.
- Run reproducibility metadata now separates optimizer config hash, gene-space hash,
  reproducibility status, notes, and runtime hook signatures.
- Ask/tell event history and telemetry now use GeneSpace-backed candidate hashes in
  optimizer internals while preserving the zero-argument `Candidate.candidate_hash()`
  compatibility fallback.
- `CheckpointCallback` now supports `format="stable"` for JSON checkpoint files
  while keeping the legacy pickle population format as the checkpoint v1 default.

### Fixed

- Scheduler promotion now ranks candidates by the completed stage score, so surrogate or
  later-stage observations cannot distort promotion from an earlier stage.
- `exploration_fraction` now adds deterministic tail exploration candidates during
  scheduler promotion.
- `OptimizationTelemetry.unique_candidate_hashes` is populated from proposed candidate
  genomes.
- Rust-backed Rayon evaluation now honors `n_threads` and rejects non-positive thread
  counts.
- Numeric gene bounds now reject `nan` and infinite values.
- The inverse-distance surrogate advisor now normalizes feature distances by gene-space
  bounds when provided.
- GA and CMA best-candidate tracking now ignores partial and surrogate observations when
  selecting the optimizer state best candidate.
- `CMAESOptimizer(direction="minimize").run(...)` now optimizes and reports the lowest raw
  score instead of treating larger values as better.
- `GeneticAlgorithmOptimizer.run_multiple(...)` now chooses the best child run using the engine direction.
- Positional `EvaluationRecord(..., metrics, batch_id)` construction again preserves the
  supplied batch ID after adding record metadata.

## [0.7.0] - 2026-05-09

### Breaking

- Reoriented EvoCore around vNext expensive black-box optimization rather than DEAP parity.
- Replaced GA execution with ask/tell and policy-driven multi-fidelity semantics.
- Added vNext CMA ask/tell semantics for trusted-record distribution updates.

### Added

- Candidate, rung, evaluation record, and optimizer telemetry APIs.
- Multi-fidelity policy and scheduler primitives.
- Deterministic Rust candidate ID and confidence-aware ranking helpers.
- Baseline surrogate advisor and audit-aware promotion support.
- Mixed-variable CMA foundation types for integer margins and categorical state.
- vNext docs and examples for budget-aware optimization.

## [0.6.1] - 2026-05-07

### Added

- Numeric `GeneSpace` now accepts `crossover="uniform"` so mixed float, integer,
  and categorical-by-integer chromosomes can use DEAP-style allele swaps.
- `GAEngine(mutation_individual_prob=...)` adds an optional per-offspring mutation
  gate. The default remains `1.0`, preserving existing EvoCore behavior while
  allowing DEAP-parity pipelines to model both outer and per-gene mutation
  probabilities.
- Root-cause and benchmark notes for the Trading-Algo-Scalper-Gold GA migration
  snapshot.

### Changed

- Tournament selection now samples aspirants with replacement, matching DEAP
  `selTournament` semantics and reducing unintended selection pressure.
- GA initialization caps the initial Rust population size when
  `max_evaluations < population_size`, reducing small-budget setup overhead.

### Fixed

- Trading-Algo benchmark parity against DEAP for crossover, mutation gating, RNG
  isolation, and standard snapshot CLI defaults.

## [0.6.0] - 2026-05-06

### Added

- Fixed numeric gene support across GA full-genome workflows, including equal-bound
  `GeneDef` values, fixed/variable gene metadata, initialization, mutation, and
  reproduction.
- Exact GA evaluation budget caps through `max_evaluations`.
- GA stop diagnostics on `RunResult`, including evaluation-budget metadata and a final
  stop status.
- Production CI gates for linting, Rust tests, Python tests, and platform smoke checks.
- PEP 561 typing marker and PyO3 extension stubs.
- MkDocs API documentation.
- Cross-platform release artifact builds.
- Manual PyPI publish workflow.
- Runtime version export and package logging.
- Targeted Hypothesis property tests.

### Changed

- Rust-backed GA initialization and reproduction preserve fixed numeric genes instead of
  treating equal bounds as invalid ranges.
- GA multi-run child engines propagate evaluation budget limits.

### Fixed

- `CMAESEngine` now rejects fixed numeric gene spaces with an explicit configuration
  error until fixed-coordinate CMA-ES support is implemented.

## [0.5.0] - 2026-05-05

### Added

- Rust/PyO3 package scaffold with maturin integration.
- Float, integer, binary, and mixed gene spaces.
- Python `Individual`, `Population`, `GeneDef`, and `GeneSpace` ergonomics.
- Rust operator implementations for crossover, mutation, selection, reproduction, and deterministic seed derivation.
- GA engine with callbacks, checkpoint/resume support, diversity tracking, and multi-run execution.
- CMA-ES engine backed by Rust covariance state and nalgebra.
- Exception and warning hierarchy for configuration, fitness, convergence, parallel, and checkpoint failures.
- Thread and process parallel evaluation modes where supported.
- Unit, integration, benchmark smoke, and example coverage for the v5 implementation.

### Changed

- Migrated PyO3 from 0.21.2 to 0.28.3 for CPython 3.14 compatibility.

### Notes

- `PyCMAESState` remains unsendable.
- `CMAESEngine` rejects `parallel="process"` because its Rust state is not picklable.
