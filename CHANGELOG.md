# Changelog

All notable changes to evocore are documented here.

This project follows semantic versioning after the v0.5.0 late-beta baseline.

## [Unreleased]

### Added

- Structural `Optimizer` and `Evaluator` protocols for the clean-break ask/tell
  lifecycle.
- `EvaluationContext`, `TellResult`, and shared `EngineStateSummary` records for
  evaluator calls, `tell(...)` summaries, and stable state inspection.
- Public `batch_id` fields on vNext `Candidate` and `EvaluationRecord` so async
  evaluators can group results by ask batch.

### Changed

- `GAEngine` and `CMAESEngine` now expose `direction` and preserve raw scores while
  using direction-aware comparisons for best-candidate tracking.
- Policy-driven evaluators now receive `EvaluationContext` instead of a bare rung.
- `GAEngine` and `CMAESEngine` ask/tell flows now treat partial trusted batches as a
  first-class API, with strict duplicate and batch-mismatch validation.
- `GAEngine.run(...)` now fails fast when a synchronous evaluator omits assigned
  candidates instead of stalling the policy loop.
- `MultiFidelityPolicy` now requires exactly one `trusted_full` rung and it must be the
  final rung.
- `ProcessParallel` now reuses a persistent process pool across repeated `evaluate(...)`
  calls until closed.
- Cached evaluation records are now eligible for optimizer state updates and full-budget
  accounting while remaining separately counted in `TellResult.cached_count`.

### Fixed

- Scheduler promotion now ranks candidates by the completed rung score, so surrogate or
  later-rung observations cannot distort promotion from an earlier rung.
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
- `CMAESEngine(direction="minimize").run(...)` now optimizes and reports the lowest raw
  fitness instead of treating larger values as better.
- `GAEngine.run_multiple(...)` now chooses the best child run using the engine direction.
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
- GA stop diagnostics on `RunResult`, including `max_evaluations`, `stop_reason`, and
  `budget_reached`.
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
