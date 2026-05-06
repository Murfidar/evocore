# Changelog

All notable changes to evocore are documented here.

This project follows semantic versioning after the v0.5.0 late-beta baseline.

## [Unreleased]

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
