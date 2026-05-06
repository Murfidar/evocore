# Changelog

All notable changes to evocore are documented here.

This project follows semantic versioning after the v0.5.0 late-beta baseline.

## [Unreleased]

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
