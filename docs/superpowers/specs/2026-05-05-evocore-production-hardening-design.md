# Evocore v0.6.0 Production Hardening Design

## Summary

Evocore v0.6.0 will be a production hardening release around the existing late-beta v5 core. The release will add CI, quality gates, typed packaging, baseline public documentation, cross-platform wheel builds, a manual PyPI publish flow, runtime logging/version polish, and targeted property-based tests.

This is not a v1.0 API freeze. The current architecture remains intact: Python owns user ergonomics and orchestration, while Rust/PyO3 owns hot-path computation through `evocore._core`.

## Goals

- Give maintainers fast, meaningful pull request feedback.
- Make local quality checks match CI expectations.
- Package enough type information for downstream users and type checkers.
- Publish user-facing API documentation without exposing internal planning docs in the main navigation.
- Build inspectable release artifacts for Linux, macOS, and Windows.
- Require a human approval step before PyPI publication.
- Add small runtime polish that helps production users observe runs without taking over their logging configuration.
- Add property-based tests where generated inputs are most likely to reveal boundary bugs.

## Non-Goals

- Do not rewrite GA, CMA-ES, selection, reproduction, or operator algorithms.
- Do not declare a stable v1.0 public API.
- Do not add free-threaded Python support; `PyCMAESState` remains unsendable.
- Do not replace warnings with logging.
- Do not add slow full-run property tests for GA or CMA-ES in this release.

## Implementation Slices

The work should be implemented as reviewable slices:

1. Quality gates: GitHub Actions, pre-commit, ruff, clippy, and pull request template.
2. Typing and docs: `py.typed`, `_core.pyi`, baseline public docstrings, and MkDocs.
3. Distribution and release: cibuildwheel, changelog, GitHub Pages docs, tag artifact builds, and manual PyPI publish flow.
4. Runtime and tests: package logger, `__version__`, and targeted Hypothesis invariants.

## CI And Quality Gates

CI will use a hybrid gated model.

For normal pushes and pull requests, add a lint job on Ubuntu that:

- Installs Rust stable and Python 3.14.
- Runs `ruff format --check`.
- Runs `ruff check --select ALL` with targeted ignores in `pyproject.toml`.
- Runs `cargo fmt --check`.
- Runs `cargo clippy --all-targets -- -D warnings`.

For normal pushes and pull requests, add an Ubuntu test matrix over Python 3.11, 3.12, 3.13, and 3.14 that:

- Runs `cargo test`.
- Runs `maturin develop --release`.
- Runs `pytest tests/unit/ tests/integration/ -v`.

For normal pushes and pull requests, add Windows and macOS smoke jobs that:

- Use Python 3.14 when runner/toolchain support is clean, otherwise Python 3.13.
- Run `cargo test`.
- Run `maturin develop --release`.
- Run `pytest tests/unit/ tests/integration/ -v`.

Pre-commit will mirror the local quality gates:

- `ruff format`
- `ruff check`
- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`

Ruff should be strict but pragmatic. The project should use `ruff check --select ALL` and configure targeted ignores for rules that conflict with practical test style, callback ergonomics, examples, or compatibility needs. Broad ignores should live in `pyproject.toml`; narrow exceptions may be inline only when the exception is genuinely local.

The pull request template will require contributors to confirm:

- Tests pass.
- Docs are updated when behavior or public API changes.
- `CHANGELOG.md` is updated for user-visible changes.
- Type stubs are updated when public API or `_core` exports change.
- Release/checkpoint impact has been considered for seed, checkpoint, or serialization changes.

## Typing And Public API Documentation

Typing becomes part of the package contract for v0.6.0.

Add `evocore/py.typed` so wheels advertise inline typing. Add `evocore/_core.pyi` for Rust extension exports, including:

- `FloatIndividual`, `IntegerIndividual`, `BinaryIndividual`, and `PyCMAESState` where public through `_core`.
- Seed constants and `py_derive_seed`.
- Operator functions.
- Selection functions.
- Population initialization and reproduction helpers.
- Parallel evaluation helpers.

Stubs should be practical rather than over-modeled. Use precise container types such as `list[float]`, `tuple[list[float], list[float]]`, and `Sequence[float]` where they communicate the real contract. Use `Any` only where PyO3 behavior or callback boundaries make precision misleading.

The package configuration must include `py.typed` and `_core.pyi` in built wheels.

Docstrings should follow a complete-baseline standard:

- Prioritize top-level user APIs: `GAEngine`, `CMAESEngine`, `GeneSpace`, `GeneDef`, `Individual`, `Population`, `OperatorSet`, callbacks, result classes, logbook classes, exceptions, and public helper functions.
- Use Google-style docstrings where details matter: arguments, returns, raised exceptions, warnings, deterministic behavior, parallelism constraints, checkpoint behavior, and callback semantics.
- Avoid repetitive prose for obvious dataclass fields or trivial properties.

API docs will use MkDocs and mkdocstrings. Add:

- `mkdocs.yml`
- install page
- quickstart page
- GA page
- CMA-ES page
- parallelism page
- checkpointing and callbacks page
- API reference
- release process page

The existing `docs/superpowers/*` planning archive should remain in the repository, but it should not appear in the public MkDocs navigation.

## Distribution And Release Flow

Wheel distribution will use cibuildwheel in GitHub Actions.

Tag builds for tags such as `v0.6.0` should produce:

- Linux manylinux wheels.
- macOS x86_64 and arm64 wheels.
- Windows wheels.
- A source distribution when the maturin workflow supports it cleanly.
- GitHub Actions artifacts that maintainers can download and inspect.

PyPI publishing will be manual approval, not automatic tag publishing. Prefer PyPI Trusted Publishing through a GitHub Actions environment that requires approval. If Trusted Publishing is not configured yet, use a scoped PyPI API token stored as a GitHub secret as a fallback.

Add `CHANGELOG.md` with:

- A retroactive `0.5.0` entry summarizing the seven completed v5 implementation parts.
- An unreleased `0.6.0` section for production hardening work.
- Semantic versioning guidance for future changes.

### Manual Approval Publish Runbook

1. Update version numbers in `pyproject.toml` and `Cargo.toml`.
2. Update `CHANGELOG.md`.
3. Run local verification:
   - `cargo test`
   - `maturin develop --release`
   - `pytest tests/unit/ tests/integration/ -v`
   - `ruff format --check`
   - `ruff check --select ALL`
   - `cargo fmt --check`
   - `cargo clippy --all-targets -- -D warnings`
4. Commit the release preparation changes.
5. Create and push a version tag such as `v0.6.0`.
6. Wait for the release workflow to build wheels and source artifacts.
7. Download and inspect the artifacts.
8. Create or approve GitHub Release notes from `CHANGELOG.md`.
9. Trigger the manual PyPI publish workflow through GitHub Actions.
10. Approve the publish environment when prompted.
11. Confirm Trusted Publishing or PyPI token configuration.
12. Verify the PyPI project page and wheel availability.
13. Install from PyPI in a clean environment and run a smoke import plus a small optimization.

## Runtime Polish

Expose `__version__` from `evocore.__init__`:

- Prefer `importlib.metadata.version("evocore")`.
- Fall back to the local package version during editable or development builds.
- Include `__version__` in `__all__`.

Add a package logger using `logging.getLogger("evocore")`:

- Use module-level child loggers in `ga.py` and `cmaes.py`.
- Log useful generation progress at debug or info level.
- Log warnings for non-finite fitness handling and parallel/process worker setup.
- Do not configure global handlers.
- Do not replace the current warning classes or warning behavior.

The existing exception and warning hierarchy remains part of the public contract:

- `EvocoreError`
- `ConfigurationError`
- `FitnessError`
- `ConvergenceError`
- `ParallelError`
- `CheckpointError`
- `FitnessWarning`
- `ConfigurationWarning`

## Property-Based Tests

Add Hypothesis as a test dependency and add targeted property tests.

Gene space properties:

- Generated valid float, integer, and boolean gene definitions preserve bounds and kind.
- Named gene definitions restore `Individual.params` consistently.
- Invalid bounds and malformed definitions continue to raise configuration errors.

Individual properties:

- Gene values keep expected Python-side typed ergonomics after wrapper conversion.
- `params` matches gene names when names exist.
- Fitness and metrics assignment behave predictably.

Operator properties:

- Mutations and crossovers return values within expected bounds.
- Binary operators return values compatible with 0/1 genes.
- Deterministic operations repeat exactly for the same master seed, generation, individual index, and operation parameters.

Do not add broad GA or CMA-ES run-level property tests in v0.6.0. Those should be added later as focused regression tests if they catch specific classes of failures.

## Acceptance Criteria

- Pull requests run lint, Ubuntu Python-version tests, and Windows/macOS smoke tests.
- Local pre-commit hooks match the main CI quality checks.
- Ruff and clippy fail on meaningful violations.
- `evocore` wheels include `py.typed` and `_core.pyi`.
- Public user-facing APIs have baseline Google-style docstrings.
- MkDocs builds local public documentation without exposing internal planning docs in navigation.
- Tag builds produce inspectable cross-platform wheel artifacts.
- PyPI publication requires manual workflow dispatch and environment approval.
- `CHANGELOG.md` documents v0.5.0 retroactively and tracks v0.6.0 hardening.
- `evocore.__version__` works in installed and development contexts.
- `logging.getLogger("evocore")` can observe GA/CMA-ES progress without configuring handlers globally.
- Hypothesis tests cover targeted `GeneSpace`, `Individual`, and operator invariants.

## Risks And Mitigations

- Python 3.14 runner support may lag on hosted CI images. Mitigate by using Python 3.13 for Windows/macOS smoke jobs when needed while keeping Ubuntu on the full 3.11-3.14 matrix.
- `ruff check --select ALL` can create noisy failures. Mitigate with explicit, documented targeted ignores in `pyproject.toml`.
- Cross-platform wheel builds may expose maturin or PyO3 platform differences. Mitigate by separating PR smoke tests from tag artifact builds and inspecting artifacts before publishing.
- PyPI Trusted Publishing setup may not be available immediately. Mitigate with a documented scoped-token fallback while keeping manual approval.
- Property tests can become slow or flaky if they exercise full optimization loops. Mitigate by limiting v0.6.0 property tests to targeted invariants.

## Open Follow-Ups After v0.6.0

- Decide what criteria would qualify evocore for a v1.0 API freeze.
- Expand docs with tutorials and migration guides based on user feedback.
- Add benchmark regression tracking after wheel/release infrastructure is stable.
- Revisit free-threaded Python support when PyO3 and CPython support are mature enough for this architecture.
