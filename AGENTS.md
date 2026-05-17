# AGENTS.md

## Instruction Priority

Follow these instructions for source, config, test, build, dependency, script,
generated-code, migration, infrastructure, and CI changes.

Priority order:

1. Direct user instructions in the current prompt.
2. More specific nested `AGENTS.md` files.
3. This root `AGENTS.md`.
4. Existing project conventions.

If the user explicitly skips a workflow step, obey the user and mention the
skipped step in the final response.

## Project Shape

EvoCore is a Python package backed by Rust/PyO3 and built with `maturin`.

Important surfaces:

- Python public API in `evocore/`.
- Rust extension code in `src/`.
- Type marker and stubs in `evocore/py.typed` and `evocore/_core.pyi`.
- Unit, integration, property, and benchmark tests in `tests/`.
- User docs in `docs/site/` with MkDocs config in `mkdocs.yml`.
- Release workflow and PyPI publication in `.github/workflows/`.

## Target Package Architecture

EvoCore uses a domain-oriented Python package layout. New source changes should
follow this architecture and avoid recreating old flat
modules such as `evocore.ga`, `evocore.cmaes`, `evocore.gene_space`,
`evocore.evaluation`, `evocore.policies`, `evocore.scheduler`, or
`evocore.stats`.

Target tree:

```text
evocore/
  __init__.py                  # top-level convenience exports, new names only
  _core.pyi                    # Rust extension type stubs
  core/
    errors.py                  # EvocoreError, ConfigurationError, FitnessError, warnings
    serialization.py           # JSON-safe export and stable hashing helpers
    parallel.py                # Thread/process evaluation helpers
  search_space/
    genes.py                   # Gene, GeneSpace, GeneKind, GeneValue
    solutions.py               # Solution, SolutionSet
    codec.py                   # OperatorCodec and Rust boundary encoding
  lifecycle/
    records.py                 # Candidate, EvaluationRecord, EvaluationContext, ScoreObservation
    policies.py                # BudgetPolicy, EvaluationStage
    scheduler.py               # BudgetScheduler
    protocols.py               # Optimizer, Evaluator
    telemetry.py               # OptimizationTelemetry, UpdateResult, OptimizerStateSummary
    events.py                  # EventRecord, EventHistory, StopReason
  results/
    generation.py              # GenerationRecord, GenerationHistory
    reproducibility.py         # ReproducibilityMetadata
    run.py                     # OptimizationResult, OptimizationBatchResult
  optimizers/
    ga/
      engine.py                # GeneticAlgorithmOptimizer public class
      ask_tell.py              # GA ask/tell lifecycle helpers
      generation_loop.py       # GA generation-loop execution
      checkpointing.py         # GA checkpoint resume helpers
      reproduction.py          # GA reproduction helpers
    cmaes/
      engine.py                # CMAESOptimizer public class
      ask_tell.py              # CMA-ES ask/tell lifecycle helpers
      mixed.py                 # IntegerMarginDistribution, CategoricalDistributionState
  callbacks/
    __init__.py                # Callback hooks and built-in callbacks
  surrogates/
    __init__.py                # InverseDistanceAdvisor, SurrogateScore
```

Public convenience imports from `evocore` should remain available, but should use
the domain vocabulary. Prefer names such as `GeneticAlgorithmOptimizer`,
`CMAESOptimizer`, `Gene`, `Solution`, `OptimizationResult`, `BudgetPolicy`,
`EvaluationStage`, `UpdateResult`, and `OptimizerStateSummary`. Use `stage` for
candidate/evaluation context fields, `events` for append-only event logs, and
`optimizer_type` for optimizer identity. Do not introduce new public APIs with the
old `Engine`, `RunResult`, `MultiRunResult`, `Rung`, `TellResult`, `Individual`,
`Population`, `history`, `engine_type`, or public `fitness` naming unless a
compatibility requirement is explicitly approved.

## Branch Workflow

Before editing code or project setup:

1. Check `git status --short --branch`.
2. Treat `main` as protected.
3. If on `main`, create a task branch before editing.
4. If already on a task branch, continue there unless the user asks for a new branch.
5. Inspect uncommitted changes and avoid staging, overwriting, or reverting unrelated
   user work.

Default branch naming:

```text
<type>/<short-kebab-description>
```

Common types: `feature`, `bugfix`, `hotfix`, `release`, `docs`, `refactor`, `test`.

## Commit Workflow

After verification passes, commit task-related files only.

Use Conventional Commits, matching the existing history:

```text
<type>(optional-scope): <summary>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`,
`chore`, `style`.

## Pull Request Workflow

After committing:

1. Push the task branch.
2. Open a draft PR unless the user asks otherwise.
3. Use `.github/pull_request_template.md`.
4. Fill all relevant sections and remove placeholder comments.
5. Include verification commands and results.
6. Do not mark ready for review or merge unless the user explicitly asks.

## Verification

Run the commands relevant to the touched surface before committing. Prefer the
smallest reliable set for the change, then broaden when public behavior, packaging,
or cross-language contracts change.

Python formatting and linting:

```bash
python -m ruff format --check
python -m ruff check
```

Rust formatting, linting, and tests:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Python extension and tests:

```bash
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

Property tests:

```bash
python -m pytest tests/property/ -v
```

Release or packaging changes:

```bash
python -m maturin build --release --out target/wheels
```

If verification fails, stop. Do not commit, push, or open a PR. Report the failing
command, relevant error summary, and likely files involved.

## Documentation, Changelog, and Versioning

Update docs and `CHANGELOG.md` when a change affects public APIs, behavior,
configuration, packaging, release workflow, docs workflow, dependency constraints,
performance characteristics, reproducibility, checkpoint compatibility, seed
semantics, or user-visible warnings/errors.

Update versions together in `pyproject.toml` and `Cargo.toml` only for release
preparation. Follow `docs/site/release.md` for release work.

Update `evocore/_core.pyi` whenever public Rust-backed exports or signatures change.

## Coding Conventions

- Follow the existing Python and Rust architecture and naming.
- Prefer local helpers and established patterns over new abstractions.
- Keep changes scoped to the task.
- Preserve deterministic seed behavior and checkpoint compatibility unless the task
  explicitly changes them.
- Add or update tests for behavior changes.
- Keep examples and docs aligned with public API changes.
- Do not commit generated build outputs from `target/`, `.pytest_cache/`, `.ruff_cache/`,
  `.hypothesis/`, or virtual environments.
