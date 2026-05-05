# Evocore v0.6.0 Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-grade CI, local quality gates, typing, docs, release artifacts, manual PyPI publishing, runtime observability, and targeted property tests for evocore v0.6.0.

**Architecture:** Keep the v5 two-layer architecture unchanged: Python remains the public orchestration layer and Rust/PyO3 remains the hot-path extension. The plan adds infrastructure around the existing APIs, then adds small runtime hooks and tests that do not change algorithm behavior.

**Tech Stack:** Rust stable, PyO3 0.28.3, maturin, pytest, ruff, pre-commit, GitHub Actions, cibuildwheel, MkDocs, mkdocstrings, Hypothesis.

---

## File Structure

- Create `.github/workflows/ci.yml`: pull request and push quality/test gates.
- Create `.github/workflows/docs.yml`: GitHub Pages documentation build and deploy.
- Create `.github/workflows/release.yml`: tag-triggered wheel and source artifact build.
- Create `.github/workflows/publish.yml`: manually approved PyPI publishing from GitHub Release artifacts.
- Create `.github/pull_request_template.md`: contributor checklist for tests, docs, changelog, stubs, and release impact.
- Create `.pre-commit-config.yaml`: local hooks mirroring ruff, cargo fmt, and clippy.
- Modify `pyproject.toml`: project metadata, optional dependencies, ruff config, maturin include files, cibuildwheel config.
- Create `CHANGELOG.md`: retroactive v0.5.0 plus v0.6.0 unreleased section and semver guidance.
- Create `evocore/py.typed`: PEP 561 marker.
- Create `evocore/_core.pyi`: type stubs for PyO3 exports in `src/lib.rs`, `src/individual.rs`, `src/cmaes.rs`, and `src/parallel.rs`.
- Modify `evocore/__init__.py`: expose `__version__`.
- Modify `evocore/ga.py`: add module logger and progress/worker/non-finite-fitness log records.
- Modify `evocore/cmaes.py`: add module logger and progress/worker/non-finite-fitness log records.
- Modify public Python modules for baseline docstrings: `evocore/gene_space.py`, `evocore/individual.py`, `evocore/operators.py`, `evocore/callbacks.py`, `evocore/stats.py`, `evocore/parallel.py`, `evocore/ga.py`, `evocore/cmaes.py`.
- Create `docs/site/*.md`: public MkDocs pages isolated from `docs/superpowers/*`.
- Create `mkdocs.yml`: public docs config using `docs/site` as `docs_dir`.
- Create `tests/unit/test_runtime_observability.py`: version and logging tests.
- Create `tests/unit/test_type_package.py`: PEP 561 and stub packaging tests.
- Create `tests/property/__init__.py`: property test package marker.
- Create `tests/property/test_gene_space_properties.py`: Hypothesis tests for `GeneSpace`, `Individual`, and wrapper conversion invariants.
- Create `tests/property/test_operator_properties.py`: Hypothesis tests for Rust operator bounds and determinism.

---

### Task 1: Configure Project Metadata And Local Quality Gates

**Files:**
- Modify: `pyproject.toml`
- Create: `.pre-commit-config.yaml`
- Create: `.github/pull_request_template.md`

- [ ] **Step 1: Update `pyproject.toml` metadata, dependencies, ruff, maturin, and cibuildwheel config**

Keep the existing `[build-system]`, `[project]`, `[tool.maturin]`, and `[tool.pytest.ini_options]` sections, then edit them to contain these values and add the new sections below.

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "evocore"
version = "0.5.0"
requires-python = ">=3.11"
description = "Rust-native Genetic Algorithms and CMA-ES for Python"
readme = "README.md"
license = { file = "LICENSE" }
authors = [{ name = "Murfidar" }]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.optional-dependencies]
dev = [
    "hypothesis>=6.100",
    "maturin>=1.5,<2.0",
    "numpy>=1.26",
    "pre-commit>=4.0",
    "pytest>=8.0",
    "ruff>=0.8",
]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.27",
]

[project.urls]
Homepage = "https://github.com/Murfidar/evocore"
Repository = "https://github.com/Murfidar/evocore"
Issues = "https://github.com/Murfidar/evocore/issues"

[tool.maturin]
python-source = "."
module-name = "evocore._core"
features = ["pyo3/extension-module"]
include = ["evocore/py.typed", "evocore/_core.pyi"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "bench_*.py"]

[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"

[tool.ruff.lint]
select = ["ALL"]
ignore = [
    "ANN002",
    "ANN003",
    "ANN204",
    "COM812",
    "D100",
    "D104",
    "D105",
    "D107",
    "ISC001",
    "PLR0913",
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"examples/**/*.py" = ["ANN", "D", "INP001", "T201"]
"tests/**/*.py" = ["ANN", "D", "PLR2004", "S101"]
"tests/benchmarks/**/*.py" = ["T201"]

[tool.cibuildwheel]
build = "cp311-* cp312-* cp313-* cp314-*"
skip = "pp* *-musllinux_*"
test-command = "python -c \"import evocore; from evocore import GAEngine, GeneSpace; assert isinstance(evocore.__version__, str); GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1, seed=1).run(lambda ind: -sum(x*x for x in ind.genes))\""
```

- [ ] **Step 2: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format
        name: ruff format
        entry: ruff format
        language: system
        types_or: [python, pyi]
      - id: ruff-check
        name: ruff check
        entry: ruff check --select ALL
        language: system
        types_or: [python, pyi]
      - id: cargo-fmt
        name: cargo fmt --check
        entry: cargo fmt --check
        language: system
        pass_filenames: false
      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --all-targets -- -D warnings
        language: system
        pass_filenames: false
```

- [ ] **Step 3: Create `.github/pull_request_template.md`**

```markdown
## Summary

Describe the user-visible behavior, infrastructure, or documentation change.

## Checklist

- [ ] Tests pass locally or in CI.
- [ ] Documentation is updated when behavior or public API changed.
- [ ] `CHANGELOG.md` is updated for user-visible changes.
- [ ] Type stubs are updated when public API or `evocore._core` exports changed.
- [ ] Seed, checkpoint, or serialization impact has been considered.

## Verification

List the commands you ran and the relevant result.
```

- [ ] **Step 4: Validate the local hook config**

Run:

```bash
python -m pip install -e ".[dev]"
python -m pre_commit validate-config
```

Expected: `pre-commit` exits with status 0 and prints no config errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml .github/pull_request_template.md
git commit -m "chore: add local quality gates"
```

---

### Task 2: Add Pull Request And Push CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

env:
  CARGO_TERM_COLOR: always

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: Install Python tooling
        run: python -m pip install -e ".[dev]"
      - name: Ruff format
        run: ruff format --check
      - name: Ruff lint
        run: ruff check --select ALL
      - name: Cargo format
        run: cargo fmt --check
      - name: Cargo clippy
        run: cargo clippy --all-targets -- -D warnings

  test-ubuntu:
    name: Python ${{ matrix.python-version }} on Ubuntu
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - name: Install Python tooling
        run: python -m pip install -e ".[dev]"
      - name: Rust tests
        run: cargo test
      - name: Build extension
        run: maturin develop --release
      - name: Python tests
        run: pytest tests/unit/ tests/integration/ -v

  platform-smoke:
    name: Smoke on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest]
        python-version: ["3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - name: Install Python tooling
        run: python -m pip install -e ".[dev]"
      - name: Rust tests
        run: cargo test
      - name: Build extension
        run: maturin develop --release
      - name: Python tests
        run: pytest tests/unit/ tests/integration/ -v
```

- [ ] **Step 2: Validate workflow file presence and triggers**

Run:

```bash
python -c "from pathlib import Path; text = Path('.github/workflows/ci.yml').read_text(); assert 'pull_request' in text and '3.14' in text and 'windows-latest' in text and 'macos-latest' in text"
git diff --check
```

Expected: both commands exit with status 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add hybrid test and lint workflow"
```

---

### Task 3: Add Runtime Version And Logging

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `evocore/ga.py`
- Modify: `evocore/cmaes.py`
- Create: `tests/unit/test_runtime_observability.py`

- [ ] **Step 1: Write failing runtime observability tests**

Create `tests/unit/test_runtime_observability.py`.

```python
import logging

import evocore
from evocore import CMAESEngine, GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def non_finite_once(ind):
    return float("nan")


def test_version_export_is_string():
    assert isinstance(evocore.__version__, str)
    assert evocore.__version__


def test_version_export_is_in_all():
    assert "__version__" in evocore.__all__


def test_ga_logs_generation_progress(caplog):
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=1, seed=7)

    with caplog.at_level(logging.INFO, logger="evocore"):
        engine.run(sphere)

    messages = [record.getMessage() for record in caplog.records if record.name == "evocore.ga"]
    assert any("GA generation=0" in message for message in messages)


def test_ga_logs_non_finite_fitness_warning(caplog):
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1, seed=7)

    with caplog.at_level(logging.WARNING, logger="evocore"):
        engine.run(non_finite_once)

    messages = [record.getMessage() for record in caplog.records if record.name == "evocore.ga"]
    assert any("non-finite fitness" in message for message in messages)


def test_cmaes_logs_generation_progress(caplog):
    engine = CMAESEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=1, seed=7)

    with caplog.at_level(logging.INFO, logger="evocore"):
        engine.run(sphere)

    messages = [record.getMessage() for record in caplog.records if record.name == "evocore.cmaes"]
    assert any("CMA-ES generation=0" in message for message in messages)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/unit/test_runtime_observability.py -v
```

Expected: failure mentioning missing `evocore.__version__` and missing log records.

- [ ] **Step 3: Add `__version__` to `evocore/__init__.py`**

Add this near the top of `evocore/__init__.py`, after the module docstring and before public imports.

```python
from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("evocore")
except _metadata.PackageNotFoundError:
    __version__ = "0.5.0"
```

Add `"__version__"` as the first item in `__all__`.

- [ ] **Step 4: Add logging to `evocore/ga.py`**

Add the import and logger near the imports.

```python
import logging

logger = logging.getLogger(__name__)
```

Add these log calls in `_evaluate_all` before each parallel execution branch.

```python
logger.debug(
    "GA process evaluation generation=%s n_workers=%s pending=%s",
    gen,
    self.n_workers,
    len(pending),
)
```

```python
logger.debug(
    "GA thread evaluation generation=%s n_workers=%s pending=%s",
    gen,
    self.n_workers,
    len(pending),
)
```

Add this warning log immediately before the existing `warnings.warn(...)` in the `nan_count` block.

```python
logger.warning(
    "GA generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
    gen,
    nan_count,
)
```

Add this info log after `logbook.append(...)` in `_run_from_population`.

```python
logger.info(
    "GA generation=%s best_fitness=%s mean_fitness=%s nan_fitness_count=%s cached_count=%s",
    gen,
    float(pop_obj.best(1)[0].fitness),
    pop_obj.mean_fitness(),
    nan_count,
    len(elites),
)
```

Add this debug log after `child_seeds` is computed in `run_multiple`.

```python
logger.debug("GA run_multiple n_runs=%s child_seeds=%s", n_runs, child_seeds)
```

- [ ] **Step 5: Add logging to `evocore/cmaes.py`**

Add the import and logger near the imports.

```python
import logging

logger = logging.getLogger(__name__)
```

Add this debug log before threaded evaluation in `_evaluate_all`.

```python
logger.debug(
    "CMA-ES thread evaluation generation=%s n_workers=%s population=%s",
    gen,
    self.n_workers,
    len(individuals),
)
```

Add this warning log immediately before the existing `warnings.warn(...)` in the `nan_count` block.

```python
logger.warning(
    "CMA-ES generation=%s saw %s non-finite fitness values; assigned fitness=-inf",
    gen,
    nan_count,
)
```

Add this info log after `logbook.append(...)` in `run`.

```python
logger.info(
    "CMA-ES generation=%s best_fitness=%s mean_fitness=%s nan_fitness_count=%s",
    gen,
    float(best.fitness),
    final_population.mean_fitness(),
    nan_count,
)
```

- [ ] **Step 6: Run the runtime observability tests**

Run:

```bash
pytest tests/unit/test_runtime_observability.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Run package import tests**

Run:

```bash
pytest tests/unit/test_package_init.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add evocore/__init__.py evocore/ga.py evocore/cmaes.py tests/unit/test_runtime_observability.py
git commit -m "feat: add version export and runtime logging"
```

---

### Task 4: Add PEP 561 Typing And PyO3 Stubs

**Files:**
- Create: `evocore/py.typed`
- Create: `evocore/_core.pyi`
- Create: `tests/unit/test_type_package.py`

- [ ] **Step 1: Write failing package typing tests**

Create `tests/unit/test_type_package.py`.

```python
from pathlib import Path

import evocore


def test_py_typed_marker_is_packaged_next_to_module():
    package_dir = Path(evocore.__file__).parent

    assert (package_dir / "py.typed").is_file()


def test_core_stub_is_packaged_next_to_module():
    package_dir = Path(evocore.__file__).parent

    assert (package_dir / "_core.pyi").is_file()


def test_core_stub_mentions_exported_symbols():
    stub = (Path(evocore.__file__).parent / "_core.pyi").read_text(encoding="utf-8")

    for symbol in [
        "class FloatIndividual",
        "class IntegerIndividual",
        "class BinaryIndividual",
        "class PyCMAESState",
        "def py_derive_seed",
        "def reproduce_population",
        "def evaluate_parallel_rayon",
    ]:
        assert symbol in stub
```

- [ ] **Step 2: Run the typing package tests and verify they fail**

Run:

```bash
pytest tests/unit/test_type_package.py -v
```

Expected: failure because `evocore/py.typed` and `evocore/_core.pyi` do not exist yet.

- [ ] **Step 3: Create `evocore/py.typed`**

Create an empty file at `evocore/py.typed`.

- [ ] **Step 4: Create `evocore/_core.pyi`**

```python
from collections.abc import Callable, Sequence

OP_INIT: int
OP_CROSSOVER: int
OP_MUTATION: int
OP_SELECTION: int
OP_CMAES_ASK: int
OP_MULTI_RUN: int
OP_CROSSOVER_PROB: int


class FloatIndividual:
    genes: list[float]
    fitness: float | None

    def __init__(self, genes: Sequence[float], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class IntegerIndividual:
    genes: list[int]
    fitness: float | None

    def __init__(self, genes: Sequence[int], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class BinaryIndividual:
    genes: list[bool]
    fitness: float | None

    def __init__(self, genes: Sequence[bool], fitness: float | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


class PyCMAESState:
    generation: int
    sigma: float
    mean: list[float]
    eigendecomp_interval: int

    def __init__(
        self,
        mean: Sequence[float],
        sigma: float,
        lambda_: int,
        bounds: Sequence[tuple[float, float]],
    ) -> None: ...
    def ask(self, master_seed: int, generation: int) -> list[list[float]]: ...
    def tell(self, samples: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None: ...
    def __repr__(self) -> str: ...


def py_derive_seed(master_seed: int, generation: int, individual_idx: int, op: int) -> int: ...
def blend_crossover(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def simulated_binary_crossover(
    a: Sequence[float],
    b: Sequence[float],
    eta: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def gaussian_mutation(
    genes: Sequence[float],
    sigma: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def uniform_mutation(
    genes: Sequence[float],
    low: float,
    high: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def int_simulated_binary_crossover(
    a: Sequence[float],
    b: Sequence[float],
    eta: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def int_gaussian_mutation(
    genes: Sequence[float],
    sigma: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def int_uniform_mutation(
    genes: Sequence[float],
    low: float,
    high: float,
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def one_point_crossover(
    a: Sequence[float],
    b: Sequence[float],
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def two_point_crossover(
    a: Sequence[float],
    b: Sequence[float],
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def uniform_crossover(
    a: Sequence[float],
    b: Sequence[float],
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> tuple[list[float], list[float]]: ...
def bit_flip_mutation(
    genes: Sequence[float],
    prob: float,
    master_seed: int,
    generation: int,
    individual_idx: int,
) -> list[float]: ...
def tournament_selection(
    fitnesses: Sequence[float],
    k: int,
    tournament_size: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def roulette_selection(
    fitnesses: Sequence[float],
    k: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def rank_selection(
    fitnesses: Sequence[float],
    k: int,
    master_seed: int,
    generation: int,
) -> list[int]: ...
def init_population(
    gene_bounds: Sequence[tuple[float, float]],
    gene_kinds_str: Sequence[str],
    population_size: int,
    master_seed: int,
) -> list[list[float]]: ...
def reproduce_population(
    population: Sequence[Sequence[float]],
    fitnesses: Sequence[float],
    crossover_type: str,
    crossover_prob: float,
    crossover_eta: float,
    crossover_alpha: float,
    mutation_type: str,
    mutation_prob: float,
    mutation_sigmas: Sequence[float],
    gene_bounds: Sequence[tuple[float, float]],
    gene_kinds: Sequence[str],
    selection_type: str,
    tournament_size: int,
    population_size: int,
    master_seed: int,
    generation: int,
) -> list[list[float]]: ...
def evaluate_sequential(
    genes_list: Sequence[Sequence[float]],
    fitness_fn: Callable[[list[float]], float],
) -> list[float]: ...
def evaluate_parallel_rayon(
    genes_list: Sequence[Sequence[float]],
    fitness_fn: Callable[[list[float]], float],
    n_threads: int,
) -> list[float]: ...
```

- [ ] **Step 5: Run typing package tests**

Run:

```bash
pytest tests/unit/test_type_package.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Verify built extension packaging still works**

Run:

```bash
maturin develop --release
python -c "from pathlib import Path; import evocore; package_dir = Path(evocore.__file__).parent; assert (package_dir / 'py.typed').is_file(); assert (package_dir / '_core.pyi').is_file()"
```

Expected: both commands exit with status 0.

- [ ] **Step 7: Commit**

```bash
git add evocore/py.typed evocore/_core.pyi tests/unit/test_type_package.py
git commit -m "feat: add typed package marker and core stubs"
```

---

### Task 5: Add Baseline Public Docs And MkDocs

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/site/index.md`
- Create: `docs/site/install.md`
- Create: `docs/site/quickstart.md`
- Create: `docs/site/ga.md`
- Create: `docs/site/cmaes.md`
- Create: `docs/site/parallelism.md`
- Create: `docs/site/callbacks-checkpointing.md`
- Create: `docs/site/api.md`
- Create: `docs/site/release.md`
- Modify: `evocore/gene_space.py`
- Modify: `evocore/individual.py`
- Modify: `evocore/operators.py`
- Modify: `evocore/callbacks.py`
- Modify: `evocore/stats.py`
- Modify: `evocore/parallel.py`
- Modify: `evocore/ga.py`
- Modify: `evocore/cmaes.py`

- [ ] **Step 1: Create `mkdocs.yml`**

```yaml
site_name: evocore
site_description: Rust-native Genetic Algorithms and CMA-ES for Python
repo_url: https://github.com/Murfidar/evocore
docs_dir: docs/site
site_dir: site
theme:
  name: material
nav:
  - Home: index.md
  - Install: install.md
  - Quickstart: quickstart.md
  - Genetic Algorithms: ga.md
  - CMA-ES: cmaes.md
  - Parallelism: parallelism.md
  - Callbacks And Checkpointing: callbacks-checkpointing.md
  - API Reference: api.md
  - Release Process: release.md
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: false
            show_root_heading: true
markdown_extensions:
  - admonition
  - pymdownx.superfences
```

- [ ] **Step 2: Create public docs pages**

Create `docs/site/index.md`.

```markdown
# evocore

evocore is a Rust-native Python optimization library for Genetic Algorithms and CMA-ES.

Python owns the ergonomic API. Rust owns the hot paths exposed through `evocore._core`.

## Current Scope

- Genetic Algorithms over float, integer, binary, and mixed numeric gene spaces.
- CMA-ES over float and integer gene spaces.
- Deterministic reproducibility from explicit seed derivation.
- Optional thread and process parallelism for supported engines.
- Callbacks, checkpoints, logbooks, and metrics.
```

Create `docs/site/install.md`.

````markdown
# Install

## Development Install

```bash
python -m pip install -e ".[dev]"
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

## Runtime Import

```python
import evocore

print(evocore.__version__)
```
````

Create `docs/site/quickstart.md`.

````markdown
# Quickstart

```python
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=100, generations=100, seed=42)
result = engine.run(sphere)

print(result.best_fitness)
print(result.best_individual.genes)
```
````

Create `docs/site/ga.md`.

```markdown
# Genetic Algorithms

`GAEngine` runs deterministic genetic algorithm optimization over a `GeneSpace`.

Use `parallel="none"` for fast Python fitness functions, `parallel="thread"` when the fitness function releases the GIL, and `parallel="process"` for pickle-safe module-level fitness functions.

::: evocore.ga.GAEngine
    options:
      members:
        - run
        - run_multiple
        - resume

::: evocore.ga.RunResult

::: evocore.ga.MultiRunResult
```

Create `docs/site/cmaes.md`.

```markdown
# CMA-ES

`CMAESEngine` provides covariance matrix adaptation backed by Rust and nalgebra.

CMA-ES supports `parallel="none"` and `parallel="thread"`. It rejects `parallel="process"` because the Rust covariance state is not picklable.

::: evocore.cmaes.CMAESEngine
    options:
      members:
        - run
```

Create `docs/site/parallelism.md`.

```markdown
# Parallelism

evocore supports three evaluation modes for `GAEngine`.

- `parallel="none"`: simplest mode and usually best for cheap fitness functions.
- `parallel="thread"`: useful when the fitness function releases the GIL.
- `parallel="process"`: useful for CPU-bound Python fitness functions that are pickle-safe.

Process mode requires module-level functions. Lambdas, nested functions, and closures are rejected because they cannot be pickled reliably.

`CMAESEngine` supports only `parallel="none"` and `parallel="thread"`.
```

Create `docs/site/callbacks-checkpointing.md`.

```markdown
# Callbacks And Checkpointing

Callbacks observe or influence optimization runs.

::: evocore.callbacks.Callback

::: evocore.callbacks.EarlyStopping

::: evocore.callbacks.ProgressBar

::: evocore.callbacks.CheckpointCallback

::: evocore.callbacks.MetricsLogger
```

Create `docs/site/api.md`.

```markdown
# API Reference

::: evocore.gene_space

::: evocore.individual

::: evocore.operators

::: evocore.stats

::: evocore.parallel

::: evocore.exceptions
```

Create `docs/site/release.md`.

```markdown
# Release Process

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
6. Wait for release artifacts to build.
7. Download and inspect the artifacts.
8. Approve or edit the draft GitHub Release.
9. Trigger the manual PyPI publish workflow with the tag name.
10. Approve the `pypi` environment.
11. Verify the PyPI page and install in a clean environment.
```

- [ ] **Step 3: Add baseline docstrings to public Python APIs**

Use Google-style docstrings for these public classes and methods. Keep existing behavior unchanged.

For `evocore/gene_space.py`, add:

```python
"""Gene space definitions for evocore optimizers."""
```

```python
class GeneDef:
    """Describe one named optimization gene.

    Args:
        name: Unique gene name used for parameter dictionaries.
        kind: Gene kind: `"float"`, `"int"`, or `"bool"`.
        low: Inclusive lower bound for float and integer genes.
        high: Inclusive upper bound for float and integer genes.
        sigma: Optional mutation sigma fraction in `(0, 1]`.

    Raises:
        ConfigurationError: If the name, kind, bounds, or sigma are invalid.
    """
```

```python
class GeneSpace:
    """Collection of gene definitions used by optimization engines."""
```

```python
def uniform(cls, low: float, high: float, length: int) -> "GeneSpace":
    """Create an unnamed float gene space with shared bounds.

    Args:
        low: Lower bound for each gene.
        high: Upper bound for each gene.
        length: Number of float genes.

    Returns:
        A `GeneSpace` with `length` float genes.

    Raises:
        ConfigurationError: If `length <= 0` or `low >= high`.
    """
```

```python
def params_for(self, genes: Sequence[float | int | bool]) -> dict[str, float | int | bool] | None:
    """Map genes to parameter names when the space has names.

    Args:
        genes: Decoded Python gene values.

    Returns:
        A name-to-value dictionary, or `None` for unnamed spaces.

    Raises:
        ConfigurationError: If the number of values does not match the gene space length.
    """
```

For the other public modules, add the following concise docstrings:

```python
"""Python-side individual and population containers."""
```

```python
class Individual:
    """Decoded optimization candidate with optional fitness and metadata."""
```

```python
class Population:
    """Sequence wrapper with fitness summary helpers for individuals."""
```

```python
"""Operator encoding and validation helpers."""
```

```python
class OperatorSet:
    """Validate operator choices and translate genes across the PyO3 boundary."""
```

```python
"""Callbacks for observing, stopping, checkpointing, and recording runs."""
```

```python
class Callback:
    """Base class for optimization run callbacks."""
```

```python
class EarlyStopping(Callback):
    """Stop a run after fitness stops improving for a configured patience."""
```

```python
class ProgressBar(Callback):
    """Display optimization progress with tqdm when tqdm is installed."""
```

```python
class CheckpointCallback(Callback):
    """Write pickle checkpoints at a fixed generation interval."""
```

```python
class MetricsLogger(Callback):
    """Append per-generation metrics as JSON Lines records."""
```

```python
"""Run logbook data structures and reporting helpers."""
```

```python
class LogEntry:
    """Per-generation statistics captured by an optimization engine."""
```

```python
class Logbook:
    """Ordered collection of `LogEntry` records with export helpers."""
```

```python
"""Python parallel evaluation helpers and pickle validation."""
```

```python
def ensure_picklable(obj, *, context: str) -> None:
    """Raise a configuration error if an object cannot be pickled for process mode."""
```

```python
class ThreadParallel:
    """ThreadPoolExecutor-backed evaluator for Python-side individuals."""
```

```python
class ProcessParallel:
    """Spawn-based ProcessPoolExecutor evaluator for pickle-safe fitness functions."""
```

For `evocore/ga.py`, add:

```python
"""Genetic algorithm engine and run result containers."""
```

```python
class RunResult:
    """Result returned by a single optimization run."""
```

```python
class MultiRunResult:
    """Aggregated results returned by `GAEngine.run_multiple`."""
```

```python
class GAEngine:
    """Run deterministic genetic algorithm optimization over a gene space.

    Args:
        gene_space: Gene definitions for individuals.
        population_size: Number of individuals per generation.
        generations: Maximum number of generations to run.
        crossover: Crossover operator name.
        crossover_prob: Probability of applying crossover.
        crossover_eta: Eta parameter for simulated binary crossover.
        crossover_alpha: Alpha parameter for blend crossover.
        mutation: Mutation operator name.
        mutation_prob: Per-gene mutation probability.
        mutation_sigma: Global mutation sigma fraction.
        mutation_sigma_schedule: Sigma schedule name.
        mutation_sigma_end: Final sigma fraction for decay schedules.
        selection: Selection operator name.
        tournament_size: Number of candidates per tournament.
        elitism: Number of best individuals copied into each generation.
        parallel: Evaluation mode: `"none"`, `"thread"`, or `"process"`.
        n_workers: Worker count for parallel modes.
        process_initializer: Optional initializer for process workers.
        process_initargs: Arguments passed to the process initializer.
        seed: Master seed for deterministic reproducibility.
        track_diversity: Whether to record per-gene diversity.
        callbacks: Optional callbacks invoked during the run.

    Raises:
        ConfigurationError: If engine configuration is invalid.
    """
```

Add these method docstrings inside `GAEngine`.

```python
def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
    """Run one GA optimization.

    Args:
        fitness_fn: Callable receiving an `Individual` and returning either a fitness
            float or `(fitness, metrics_dict)`.

    Returns:
        Run result containing the best individual, final population, logbook, and timing.

    Raises:
        FitnessError: If the fitness function raises or returns an invalid value.
        ConfigurationError: If process mode receives a non-picklable fitness function.
    """
```

```python
def run_multiple(
    self,
    fitness_fn: Callable,
    n_runs: int = 10,
    aggregate: str = "best",
    run_parallel: bool = False,
) -> MultiRunResult:
    """Run multiple deterministic child runs from derived seeds.

    Args:
        fitness_fn: Fitness callable passed to each child run.
        n_runs: Number of child runs.
        aggregate: Aggregation mode. `"best"` and `"all"` are accepted.
        run_parallel: Whether to execute child runs in spawned processes.

    Returns:
        Multi-run result sorted by descending best fitness.

    Raises:
        ConfigurationError: If `n_runs`, `aggregate`, or pickle constraints are invalid.
    """
```

```python
def resume(self, fitness_fn: Callable, checkpoint: str) -> RunResult:
    """Resume a GA run from a checkpoint file.

    Args:
        fitness_fn: Fitness callable used for remaining generations.
        checkpoint: Path to a checkpoint written by `CheckpointCallback`.

    Returns:
        Run result for the resumed optimization.

    Raises:
        CheckpointError: If the file is missing, corrupt, incompatible, or has a
            seed that differs from the engine seed.
    """
```

For `evocore/cmaes.py`, add:

```python
"""CMA-ES engine backed by Rust covariance state."""
```

```python
class CMAESEngine:
    """Run covariance matrix adaptation evolution strategy optimization.

    Args:
        gene_space: Float or integer gene definitions.
        population_size: Number of sampled candidates per generation.
        initial_mean: Optional encoded initial mean.
        initial_sigma: Initial sigma fraction relative to gene bounds.
        generations: Maximum number of generations to run.
        parallel: Evaluation mode: `"none"` or `"thread"`.
        n_workers: Worker count for thread mode.
        callbacks: Optional callbacks invoked during the run.
        seed: Master seed for deterministic sampling.
        track_diversity: Whether to record per-gene diversity.

    Raises:
        ConfigurationError: If configuration is invalid or process parallelism is requested.
    """
```

Add this method docstring inside `CMAESEngine`.

```python
def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
    """Run one CMA-ES optimization.

    Args:
        fitness_fn: Callable receiving an `Individual` and returning either a fitness
            float or `(fitness, metrics_dict)`.

    Returns:
        Run result containing the best individual, final population, logbook, and timing.

    Raises:
        FitnessError: If the fitness function raises or returns an invalid value.

    Warns:
        FitnessWarning: When NaN or Inf fitness values are assigned `-inf`.
    """
```

- [ ] **Step 4: Build docs**

Run:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

Expected: MkDocs exits with status 0 and writes the site to `site/`.

- [ ] **Step 5: Run ruff docstring checks through the configured lint profile**

Run:

```bash
ruff check --select ALL evocore docs/site
```

Expected: ruff exits with status 0.

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml docs/site evocore/gene_space.py evocore/individual.py evocore/operators.py evocore/callbacks.py evocore/stats.py evocore/parallel.py evocore/ga.py evocore/cmaes.py
git commit -m "docs: add public API documentation"
```

---

### Task 6: Add Docs Deployment, Release Artifacts, Changelog, And Manual Publish

**Files:**
- Create: `.github/workflows/docs.yml`
- Create: `.github/workflows/release.yml`
- Create: `.github/workflows/publish.yml`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create `.github/workflows/docs.yml`**

```yaml
name: Docs

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - name: Install docs dependencies
        run: python -m pip install -e ".[docs]"
      - name: Build docs
        run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Create `.github/workflows/release.yml`**

```yaml
name: Build Release Artifacts

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

permissions:
  contents: write

env:
  CARGO_TERM_COLOR: always
  CIBW_BUILD: "cp311-* cp312-* cp313-* cp314-*"
  CIBW_SKIP: "pp* *-musllinux_*"
  CIBW_TEST_COMMAND: "python -c \"import evocore; assert isinstance(evocore.__version__, str)\""

jobs:
  sdist:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - uses: dtolnay/rust-toolchain@stable
      - name: Install maturin
        run: python -m pip install "maturin>=1.5,<2.0"
      - name: Build source distribution
        run: maturin sdist --out dist
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*

  wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    env:
      CIBW_ARCHS_MACOS: "x86_64 arm64"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
      - uses: dtolnay/rust-toolchain@stable
      - name: Install cibuildwheel
        run: python -m pip install cibuildwheel
      - name: Build wheels
        run: python -m cibuildwheel --output-dir dist
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: dist/*

  draft-release:
    needs: [sdist, wheels]
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          path: dist
          merge-multiple: true
      - name: List artifacts
        run: ls -la dist
      - name: Create draft GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          draft: true
          files: dist/*
```

- [ ] **Step 3: Create `.github/workflows/publish.yml`**

```yaml
name: Publish To PyPI

on:
  workflow_dispatch:
    inputs:
      tag:
        description: Git tag to publish, for example v0.6.0
        required: true
        type: string
      confirm:
        description: Type publish to confirm PyPI publication
        required: true
        type: string

permissions:
  contents: read
  id-token: write

jobs:
  publish:
    if: inputs.confirm == 'publish'
    environment:
      name: pypi
      url: https://pypi.org/project/evocore/
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Download release artifacts
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p dist
          gh release download "${{ inputs.tag }}" --dir dist
          ls -la dist
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: dist
```

- [ ] **Step 4: Create `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to evocore are documented here.

This project follows semantic versioning after the v0.5.0 late-beta baseline.

## [Unreleased]

### Added

- Production CI gates for linting, Rust tests, Python tests, and platform smoke checks.
- PEP 561 typing marker and PyO3 extension stubs.
- MkDocs API documentation.
- Cross-platform release artifact builds.
- Manual PyPI publish workflow.
- Runtime version export and package logging.
- Targeted Hypothesis property tests.

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
```

- [ ] **Step 5: Validate workflow and changelog content**

Run:

```bash
python -c "from pathlib import Path; assert 'workflow_dispatch' in Path('.github/workflows/publish.yml').read_text(); assert '0.5.0' in Path('CHANGELOG.md').read_text(); assert 'draft: true' in Path('.github/workflows/release.yml').read_text()"
git diff --check
```

Expected: both commands exit with status 0.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/docs.yml .github/workflows/release.yml .github/workflows/publish.yml CHANGELOG.md
git commit -m "ci: add docs release and publish workflows"
```

---

### Task 7: Add Targeted Property-Based Tests

**Files:**
- Create: `tests/property/__init__.py`
- Create: `tests/property/test_gene_space_properties.py`
- Create: `tests/property/test_operator_properties.py`

- [ ] **Step 1: Create `tests/property/__init__.py`**

Create an empty file at `tests/property/__init__.py`.

- [ ] **Step 2: Create `tests/property/test_gene_space_properties.py`**

```python
from hypothesis import given, strategies as st

from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual
from evocore.operators import OperatorSet


GENE_NAME_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@st.composite
def valid_numeric_gene_specs(draw):
    kind = draw(st.sampled_from(["float", "int"]))
    name = draw(st.text(alphabet=GENE_NAME_CHARS, min_size=1, max_size=12))
    if kind == "int":
        low = draw(st.integers(min_value=-1000, max_value=999))
        high = draw(st.integers(min_value=low + 1, max_value=low + 1000))
    else:
        low = draw(
            st.floats(
                min_value=-1000.0,
                max_value=999.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        span = draw(
            st.floats(
                min_value=1e-6,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        high = low + span
    sigma = draw(
        st.none()
        | st.floats(
            min_value=1e-6,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return GeneDef(name, kind, low, high, sigma=sigma)


@given(valid_numeric_gene_specs())
def test_numeric_gene_def_preserves_bounds_and_kind(gene):
    assert gene.kind in {"float", "int"}
    assert gene.low < gene.high
    if gene.sigma is not None:
        assert 0.0 < gene.sigma <= 1.0


@given(st.integers(min_value=1, max_value=25))
def test_uniform_space_has_requested_length(length):
    space = GeneSpace.uniform(-5.0, 5.0, length)

    assert space.length == length
    assert space.has_names is False
    assert space.params_for([0.0] * length) is None
    assert space.rust_bounds == [(-5.0, 5.0)] * length


@given(st.lists(st.sampled_from(["float", "int", "bool"]), min_size=1, max_size=10))
def test_named_params_match_gene_order(kinds):
    genes = []
    values = []
    for index, kind in enumerate(kinds):
        name = f"gene_{index}"
        if kind == "float":
            genes.append(GeneDef(name, "float", -10.0, 10.0))
            values.append(float(index) / 10.0)
        elif kind == "int":
            genes.append(GeneDef(name, "int", -10, 10))
            values.append(index - 5)
        else:
            genes.append(GeneDef(name, "bool"))
            values.append(index % 2 == 0)

    space = GeneSpace(genes)

    assert space.params_for(values) == dict(zip(space.names, values))


@given(st.lists(st.integers(min_value=-20, max_value=20), min_size=1, max_size=10))
def test_individual_clone_preserves_genes_and_metadata(values):
    ind = Individual(
        list(values),
        fitness=1.25,
        fitness_valid=True,
        metadata={"params": {"x": 1}},
    )

    cloned = ind.clone()

    assert cloned.genes == ind.genes
    assert cloned.fitness == ind.fitness
    assert cloned.fitness_valid is True
    assert cloned.metadata == ind.metadata
    assert cloned is not ind


@given(
    st.integers(min_value=0, max_value=100),
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_operator_decode_restores_named_params(period, threshold):
    space = GeneSpace(
        [
            GeneDef("period", "int", 0, 100),
            GeneDef("threshold", "float", -1.0, 1.0),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")

    ind = ops.decode_individual([float(period), threshold])

    assert ind.genes == [period, threshold]
    assert ind.params == {"period": period, "threshold": threshold}
```

- [ ] **Step 3: Create `tests/property/test_operator_properties.py`**

```python
from hypothesis import given, strategies as st

from evocore._core import (
    bit_flip_mutation,
    gaussian_mutation,
    int_uniform_mutation,
    one_point_crossover,
    py_derive_seed,
    two_point_crossover,
    uniform_crossover,
    uniform_mutation,
)


bounded_float = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

float_lists = st.lists(
    bounded_float,
    min_size=1,
    max_size=25,
)

binary_lists = st.lists(st.sampled_from([0.0, 1.0]), min_size=2, max_size=30)


@given(float_lists, st.integers(min_value=0, max_value=2**32 - 1))
def test_gaussian_mutation_is_deterministic(genes, seed):
    left = gaussian_mutation(genes, 0.5, 0.75, seed, 3, 4)
    right = gaussian_mutation(genes, 0.5, 0.75, seed, 3, 4)

    assert left == right


@given(
    float_lists,
    st.floats(min_value=-50.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_uniform_mutation_respects_bounds(genes, low, span):
    high = low + span

    mutated = uniform_mutation(genes, low, high, 1.0, 42, 0, 0)

    assert all(low <= value < high for value in mutated)


@given(
    st.lists(bounded_float, min_size=1, max_size=25),
    st.integers(min_value=-100, max_value=99),
    st.integers(min_value=1, max_value=100),
)
def test_int_uniform_mutation_outputs_integer_values(genes, low, span):
    high = low + span

    mutated = int_uniform_mutation(genes, float(low), float(high), 1.0, 42, 0, 0)

    assert all(float(low) <= value <= float(high) for value in mutated)
    assert all(value == int(value) for value in mutated)


@given(binary_lists)
def test_bit_flip_mutation_returns_binary_values(genes):
    mutated = bit_flip_mutation(genes, 0.5, 42, 0, 0)

    assert all(value in {0.0, 1.0} for value in mutated)


@given(binary_lists, binary_lists)
def test_binary_crossovers_return_binary_values(left, right):
    size = min(len(left), len(right))
    left = left[:size]
    right = right[:size]

    for first, second in [
        one_point_crossover(left, right, 42, 0, 0),
        two_point_crossover(left, right, 42, 0, 0),
        uniform_crossover(left, right, 0.5, 42, 0, 0),
    ]:
        assert len(first) == size
        assert len(second) == size
        assert all(value in {0.0, 1.0} for value in first)
        assert all(value in {0.0, 1.0} for value in second)


@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=10),
)
def test_derive_seed_is_stable(master_seed, generation, individual_idx, op):
    assert py_derive_seed(master_seed, generation, individual_idx, op) == py_derive_seed(
        master_seed,
        generation,
        individual_idx,
        op,
    )
```

- [ ] **Step 4: Run property tests**

Run:

```bash
pytest tests/property -v
```

Expected: all property tests pass.

- [ ] **Step 5: Run focused existing tests for affected areas**

Run:

```bash
pytest tests/unit/test_gene_space.py tests/unit/test_individual.py tests/unit/test_operators.py tests/unit/test_operators_rust.py tests/unit/test_rng_reproducibility.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/property
git commit -m "test: add targeted property invariants"
```

---

### Task 8: Final Verification And Cleanup

**Files:**
- Verify all files changed by Tasks 1-7.

- [ ] **Step 1: Run formatting and linting**

Run:

```bash
ruff format --check
ruff check --select ALL
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Expected: all commands exit with status 0.

- [ ] **Step 2: Run Rust and Python tests**

Run:

```bash
cargo test
maturin develop --release
pytest tests/unit/ tests/integration/ tests/property/ -v
```

Expected: Rust tests, extension build, unit tests, integration tests, and property tests pass.

- [ ] **Step 3: Build public docs**

Run:

```bash
mkdocs build --strict
```

Expected: docs build exits with status 0 and writes to `site/`.

- [ ] **Step 4: Run local pre-commit against all files**

Run:

```bash
pre-commit run --all-files
```

Expected: all hooks pass. If `ruff format` changes files, inspect the diff, rerun this command, and commit the formatted files.

- [ ] **Step 5: Inspect final repository state**

Run:

```bash
git status --short
git log --oneline -n 8
```

Expected: `git status --short` prints nothing after the final commit. The log shows the focused commits from this plan.
