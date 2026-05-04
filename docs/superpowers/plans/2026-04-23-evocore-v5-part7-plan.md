# evocore v5 - Part 7: CMAESEngine, Integration Tests, Examples, Benchmarks, README

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the public Python `CMAESEngine`, wire final top-level exports, then complete the project with integration tests, examples, benchmarks, and README documentation.

**Architecture:** `CMAESEngine` is a Python orchestrator around Rust `PyCMAESState`. Rust owns continuous CMA-ES state, deterministic `ask(master_seed, generation)`, mirror-folding, eigendecomposition caching, and `tell()`. Python owns validation, `Individual` decoding, fitness evaluation, callbacks, logbook, `parallel="process"` rejection, and the required two-array loop: continuous samples go to `tell()`, rounded/clamped samples go to the user's fitness function.

**Tech Stack:** Python 3.11+, pytest, PyO3 extension from Parts 1-4, Python API from Parts 5-6, optional DEAP for benchmarks

**Prerequisite:** Parts 1-6 complete and all Rust/Python unit tests green.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evocore/cmaes.py` | Create | `CMAESEngine` wrapper around `_core.PyCMAESState` |
| `evocore/__init__.py` | Modify | Export `CMAESEngine` |
| `tests/unit/test_cmaes_engine.py` | Create | Construction guard, run loop, integer two-array behavior |
| `tests/integration/test_sphere_function.py` | Create | GA float convergence |
| `tests/integration/test_rastrigin.py` | Create | GA multimodal convergence smoke |
| `tests/integration/test_binary_onemax.py` | Create | GA binary convergence |
| `tests/integration/test_mixed_gene_space.py` | Create | GA mixed float/int typing |
| `tests/integration/test_cmaes_rosenbrock.py` | Create | CMA-ES Rosenbrock convergence smoke |
| `examples/sphere_optimization.py` | Create | Minimal GA float example |
| `examples/onemax_binary.py` | Create | Binary GA example |
| `examples/mixed_gene_space.py` | Create | Named mixed GeneSpace example |
| `examples/cmaes_rosenbrock.py` | Create | CMA-ES example |
| `tests/benchmarks/bench_ga_vs_deap.py` | Create | Optional DEAP comparison |
| `tests/benchmarks/bench_parallel_scaling.py` | Create | Parallel scaling timing script |
| `README.md` | Create | Install, quickstart, API, parallelism, reproducibility, benchmarks |

---

## Task 1: `CMAESEngine` Validation and Helpers

**Files:**
- Create: `evocore/cmaes.py`
- Create: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Write failing construction/helper tests**

```python
import pytest
from evocore import CMAESEngine, ConfigurationError, GeneDef, GeneSpace


def test_cmaes_requires_gene_space():
    with pytest.raises(ConfigurationError, match="gene_space required"):
        CMAESEngine(gene_space=None)


def test_cmaes_rejects_bool_genes():
    with pytest.raises(ConfigurationError, match="bool"):
        CMAESEngine(GeneSpace([GeneDef("flag", "bool")]))


def test_cmaes_process_parallel_raises_at_construction():
    with pytest.raises(ConfigurationError) as exc:
        CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), parallel="process")
    assert "parallel='process'" in str(exc.value)
    assert "not picklable" in str(exc.value)
    assert "parallel='thread'" in str(exc.value)


def test_apply_bounds_and_round_for_int_genes():
    space = GeneSpace([GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)])
    engine = CMAESEngine(space, population_size=6, generations=1, seed=42)
    assert engine._apply_bounds_and_round([20.8, 1.5]) == [20.0, 1.0]
    assert engine._decode_individual([10.2, 0.25]).genes == [10, 0.25]
```

- [ ] **Step 2: Implement constructor and helpers**

```python
from __future__ import annotations

import math
import time
import warnings
from typing import Callable, Sequence

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.exceptions import ConfigurationError, FitnessError, FitnessWarning
from evocore.ga import RunResult
from evocore.gene_space import GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ThreadParallel
from evocore.stats import LogEntry, Logbook


class CMAESEngine:
    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 50,
        initial_mean: list[float] | None = None,
        initial_sigma: float = 0.3,
        generations: int = 300,
        parallel: str = "none",
        n_workers: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        seed: int = 0,
        track_diversity: bool = False,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for CMAESEngine. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if "bool" in gene_space.kinds:
            raise ConfigurationError("CMAESEngine does not support bool genes; use float/int genes only.")
        if parallel == "process":
            raise ConfigurationError(
                "CMAESEngine does not support parallel='process'.\n"
                "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable.\n"
                "  Fix: use parallel='thread' if your fitness function releases the GIL, or parallel='none'.\n"
                "  Note: parallel='process' is supported by GAEngine, not CMAESEngine."
            )
        if parallel not in ("none", "thread"):
            raise ConfigurationError("CMAESEngine parallel must be 'none' or 'thread'.")
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if generations < 0:
            raise ConfigurationError("generations must be >= 0.")
        if not (initial_sigma > 0.0):
            raise ConfigurationError("initial_sigma must be > 0.")
        if initial_mean is not None and len(initial_mean) != gene_space.length:
            raise ConfigurationError("initial_mean length must match gene_space.length.")

        self.gene_space = gene_space
        self.population_size = population_size
        self.initial_mean = initial_mean
        self.initial_sigma = initial_sigma
        self.generations = generations
        self.parallel = parallel
        self.n_workers = n_workers
        self.callbacks = list(callbacks or [])
        self.seed = int(seed)
        self.track_diversity = track_diversity
        self.operators = OperatorSet(gene_space, "sbx", "gaussian")
        self._fitness_warning_emitted = False

    @property
    def _bounds_list(self) -> list[tuple[float, float]]:
        return self.operators.gene_bounds

    def _initial_mean_encoded(self) -> list[float]:
        if self.initial_mean is not None:
            return [float(x) for x in self.initial_mean]
        return _core.init_population(self._bounds_list, self.operators.gene_kinds, 1, self.seed)[0]

    def _sigma_abs(self) -> float:
        spans = [hi - lo for lo, hi in self._bounds_list]
        return self.initial_sigma * (sum(spans) / len(spans))

    def _apply_bounds_and_round(self, genes_f64: Sequence[float]) -> list[float]:
        result: list[float] = []
        for value, gene, (lo, hi) in zip(genes_f64, self.gene_space.genes, self._bounds_list):
            x = max(lo, min(hi, float(value)))
            if gene.kind == "int":
                x = float(round(x))
                x = max(lo, min(hi, x))
            result.append(x)
        return result

    def _decode_individual(self, genes_f64: Sequence[float], fitness: float | None = None) -> Individual:
        return self.operators.decode_individual(genes_f64, fitness=fitness, fitness_valid=fitness is not None)
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_cmaes_engine.py -k "requires_gene_space or reject or process_parallel or apply_bounds" -v`

Expected: pass.

```bash
git add evocore/cmaes.py tests/unit/test_cmaes_engine.py
git commit -m "feat(python): CMAESEngine validation and helpers"
```

---

## Task 2: Implement `CMAESEngine.run()`

**Files:**
- Modify: `evocore/cmaes.py`
- Modify: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Add failing run-loop tests**

```python
from evocore import CMAESEngine, GeneDef, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_cmaes_run_returns_result():
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), population_size=10, generations=5, seed=42)
    result = engine.run(sphere)
    assert result.best_fitness <= 0.0
    assert len(result.logbook) == 5
    assert len(result.final_population) == 10
    assert result.seed == 42


def test_cmaes_thread_parallel_allowed():
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), population_size=10, generations=2, parallel="thread")
    assert engine.run(sphere).best_fitness <= 0.0


def test_cmaes_integer_fitness_receives_ints():
    seen_types = []
    space = GeneSpace([GeneDef("period", "int", 5, 20), GeneDef("x", "float", -1.0, 1.0)])

    def fitness(ind):
        seen_types.append(type(ind.genes[0]))
        return -abs(ind.genes[0] - 10) - ind.genes[1] ** 2

    CMAESEngine(space, population_size=12, generations=3, seed=42).run(fitness)
    assert seen_types
    assert all(t is int for t in seen_types)
```

- [ ] **Step 2: Add evaluation helpers and run loop**

Add these methods inside `CMAESEngine`:

```python
    def _normalise_fitness_result(self, result, ind: Individual, gen: int, idx: int) -> tuple[float, int]:
        metrics = {}
        if isinstance(result, tuple):
            if len(result) != 2 or not isinstance(result[1], dict):
                raise FitnessError("fitness_fn tuple return must be (float, dict).")
            result, metrics = result
        try:
            fitness = float(result)
        except (TypeError, ValueError) as exc:
            raise FitnessError(f"fitness_fn must return a float, got {type(result)!r} at generation {gen}, index {idx}.") from exc
        ind.metadata["metrics"] = metrics
        if not math.isfinite(fitness):
            ind.metadata["raw_fitness"] = fitness
            ind.fitness = float("-inf")
            ind.fitness_valid = True
            return float("-inf"), 1
        ind.fitness = fitness
        ind.fitness_valid = True
        return fitness, 0

    def _evaluate_all(self, individuals: Sequence[Individual], fitness_fn: Callable, gen: int) -> tuple[list[float], int]:
        if self.parallel == "thread":
            raw_results = ThreadParallel(self.n_workers).evaluate(individuals, fitness_fn)
        else:
            raw_results = []
            for idx, ind in enumerate(individuals):
                try:
                    raw_results.append(fitness_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"fitness_fn raised {type(exc).__name__} for individual at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc
        fitnesses: list[float] = []
        nan_count = 0
        for idx, (ind, raw) in enumerate(zip(individuals, raw_results)):
            fitness, bad = self._normalise_fitness_result(raw, ind, gen, idx)
            fitnesses.append(fitness)
            nan_count += bad
        if nan_count and not self._fitness_warning_emitted:
            warnings.warn(
                f"{nan_count} individuals in generation {gen} returned NaN or Inf fitness. "
                "They have been assigned fitness=-inf for selection.",
                FitnessWarning,
                stacklevel=2,
            )
            self._fitness_warning_emitted = True
        return fitnesses, nan_count

    def _bind_callbacks(self) -> None:
        for cb in self.callbacks:
            cb.bind_context(seed=self.seed, generations=self.generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)

    def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
        self._fitness_warning_emitted = False
        self._bind_callbacks()
        start = time.perf_counter()
        state = _core.PyCMAESState(
            self._initial_mean_encoded(),
            self._sigma_abs(),
            self.population_size,
            self._bounds_list,
        )
        logbook = Logbook()
        elite_history: list[Individual] = []
        diversity_history: list[list[float]] = []
        final_population = Population([])
        n_evaluations = 0
        stopped_early = False

        for gen in range(self.generations):
            for cb in self.callbacks:
                cb.on_generation_start(gen, final_population)
            if self._callbacks_should_stop():
                stopped_early = True
                break

            gen_start = time.perf_counter()
            samples_continuous = state.ask(self.seed, gen)
            samples_discrete = [self._apply_bounds_and_round(s) for s in samples_continuous]
            individuals = [self._decode_individual(s) for s in samples_discrete]
            fitnesses, nan_count = self._evaluate_all(individuals, fitness_fn, gen)
            n_evaluations += len(individuals)
            state.tell(samples_continuous, fitnesses)

            final_population = Population(individuals)
            info = GenerationInfo(gen, nan_count, 0)
            best = final_population.best(1)[0]
            diversity = final_population.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            elite_history.append(best.clone())
            logbook.append(LogEntry(
                gen=gen,
                best_fitness=float(best.fitness),
                mean_fitness=final_population.mean_fitness(),
                std_fitness=final_population.std_fitness(),
                wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
                n_evaluations=len(individuals),
                nan_fitness_count=nan_count,
                cached_count=0,
                diversity=diversity,
                custom=best.metadata.get("metrics", {}),
            ))
            for cb in self.callbacks:
                cb.on_generation_end(gen, final_population, info)
            if self._callbacks_should_stop():
                stopped_early = True
                break

        best = final_population.best(1)[0] if len(final_population) else Individual([], fitness=float("-inf"))
        result = RunResult(
            best_individual=best.clone(),
            best_fitness=float(best.fitness),
            final_population=final_population,
            logbook=logbook,
            wall_time_seconds=time.perf_counter() - start,
            n_evaluations=n_evaluations,
            elite_history=elite_history,
            diversity_history=diversity_history,
            seed=self.seed,
            stopped_early=stopped_early,
        )
        for cb in self.callbacks:
            cb.on_run_end(result)
        return result
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_cmaes_engine.py -v`

Expected: all tests pass.

```bash
git add evocore/cmaes.py tests/unit/test_cmaes_engine.py
git commit -m "feat(python): CMAESEngine run loop"
```

---

## Task 3: Top-Level Export

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Add import test**

```python
def test_cmaes_export_accessible_from_top_level():
    from evocore import CMAESEngine
    assert CMAESEngine is not None
```

- [ ] **Step 2: Export class**

Add to `evocore/__init__.py`:

```python
from evocore.cmaes import CMAESEngine
```

Add `"CMAESEngine"` to `__all__`.

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_package_init.py tests/unit/test_cmaes_engine.py -v`

Expected: pass.

```bash
git add evocore/__init__.py tests/unit/test_package_init.py
git commit -m "feat(python): export CMAESEngine"
```

---

## Task 4: Integration Tests

**Files:**
- Create: all files under `tests/integration/`

- [ ] **Step 1: Add GA sphere integration**

```python
# tests/integration/test_sphere_function.py
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_ga_sphere_converges_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=80, generations=80, seed=42)
    result = engine.run(sphere)
    assert result.best_fitness > -2.0
```

- [ ] **Step 2: Add GA Rastrigin integration**

```python
# tests/integration/test_rastrigin.py
import math
from evocore import GAEngine, GeneSpace


def rastrigin(ind):
    n = len(ind.genes)
    value = 10 * n + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in ind.genes)
    return -value


def test_ga_rastrigin_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.12, 5.12, 6), population_size=100, generations=120, seed=7)
    result = engine.run(rastrigin)
    assert result.best_fitness > -40.0
```

- [ ] **Step 3: Add binary OneMax integration**

```python
# tests/integration/test_binary_onemax.py
from evocore import GAEngine, GeneDef, GeneSpace


def onemax(ind):
    return sum(1 for value in ind.genes if value)


def test_binary_onemax_smoke():
    space = GeneSpace([GeneDef(f"bit_{i}", "bool") for i in range(50)])
    engine = GAEngine(space, population_size=80, generations=80, crossover="one_point", mutation="bit_flip", seed=42)
    result = engine.run(onemax)
    assert result.best_fitness >= 40
```

- [ ] **Step 4: Add mixed float/int integration**

```python
# tests/integration/test_mixed_gene_space.py
from evocore import GAEngine, GeneDef, GeneSpace


def mixed_target(ind):
    p = ind.params
    return -((p["period"] - 20) ** 2) - ((p["threshold"] - 0.3) ** 2)


def test_mixed_gene_space_keeps_ints_typed():
    space = GeneSpace([
        GeneDef("period", "int", 5, 50, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
    ])
    engine = GAEngine(space, population_size=60, generations=50, seed=42)
    result = engine.run(mixed_target)
    assert isinstance(result.best_individual.genes[0], int)
    assert result.best_fitness > -10.0
```

- [ ] **Step 5: Add CMA-ES Rosenbrock integration**

```python
# tests/integration/test_cmaes_rosenbrock.py
from evocore import CMAESEngine, GeneSpace


def rosenbrock(ind):
    xs = ind.genes
    value = sum(100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1 - xs[i]) ** 2 for i in range(len(xs) - 1))
    return -value


def test_cmaes_rosenbrock_smoke():
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, generations=80, seed=42)
    result = engine.run(rosenbrock)
    assert result.best_fitness > -50.0
```

- [ ] **Step 6: Run integration tests and commit**

Run: `pytest tests/integration/ -v`

Expected: all integration smoke tests pass. If thresholds are flaky, increase generations before weakening assertions.

```bash
git add tests/integration/
git commit -m "test(integration): GA and CMA-ES convergence smoke tests"
```

---

## Task 5: Examples

**Files:**
- Create: `examples/sphere_optimization.py`
- Create: `examples/onemax_binary.py`
- Create: `examples/mixed_gene_space.py`
- Create: `examples/cmaes_rosenbrock.py`

- [ ] **Step 1: Add example scripts**

```python
# examples/sphere_optimization.py
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=80, generations=80, seed=42)
result = engine.run(sphere)
print(result.best_fitness, result.best_individual.genes)
```

```python
# examples/onemax_binary.py
from evocore import GAEngine, GeneDef, GeneSpace


def onemax(ind):
    return sum(ind.genes)


space = GeneSpace([GeneDef(f"bit_{i}", "bool") for i in range(50)])
engine = GAEngine(space, population_size=80, generations=80, crossover="one_point", mutation="bit_flip", seed=42)
result = engine.run(onemax)
print(result.best_fitness, result.best_individual.genes)
```

```python
# examples/mixed_gene_space.py
from evocore import GAEngine, GeneDef, GeneSpace


def objective(ind):
    p = ind.params
    return -abs(p["period"] - 21) - abs(p["threshold"] - 0.35)


space = GeneSpace([
    GeneDef("period", "int", 5, 50, sigma=0.05),
    GeneDef("threshold", "float", 0.0, 1.0),
])
result = GAEngine(space, population_size=60, generations=50, seed=7).run(objective)
print(result.best_fitness, result.best_individual.params)
```

```python
# examples/cmaes_rosenbrock.py
from evocore import CMAESEngine, GeneSpace


def rosenbrock(ind):
    xs = ind.genes
    return -sum(100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1 - xs[i]) ** 2 for i in range(len(xs) - 1))


engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, generations=80, seed=42)
result = engine.run(rosenbrock)
print(result.best_fitness, result.best_individual.genes)
```

- [ ] **Step 2: Run examples and commit**

Run:

```bash
python examples/sphere_optimization.py
python examples/onemax_binary.py
python examples/mixed_gene_space.py
python examples/cmaes_rosenbrock.py
```

Expected: each script prints a best fitness and best genes without exceptions.

```bash
git add examples/
git commit -m "docs(examples): add GA and CMA-ES examples"
```

---

## Task 6: Benchmarks

**Files:**
- Create: `tests/benchmarks/bench_ga_vs_deap.py`
- Create: `tests/benchmarks/bench_parallel_scaling.py`

- [ ] **Step 1: Add benchmark scripts**

```python
# tests/benchmarks/bench_ga_vs_deap.py
import time
import pytest
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_evocore_ga_wall_time_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 20), population_size=300, generations=40, seed=42)
    t0 = time.perf_counter()
    result = engine.run(sphere)
    elapsed = time.perf_counter() - t0
    assert result.best_fitness <= 0.0
    print(f"evocore elapsed={elapsed:.3f}s")


def test_deap_comparison_optional():
    pytest.importorskip("deap")
```

```python
# tests/benchmarks/bench_parallel_scaling.py
import time
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_run_multiple_parallel_scaling_smoke():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=60, generations=20, seed=42)
    t0 = time.perf_counter()
    seq = engine.run_multiple(sphere, n_runs=2, run_parallel=False)
    seq_elapsed = time.perf_counter() - t0
    assert seq.n_runs == 2
    print(f"sequential multi-run elapsed={seq_elapsed:.3f}s")
```

- [ ] **Step 2: Run benchmark smoke tests and commit**

Run: `pytest tests/benchmarks/ -v -s`

Expected: benchmark smoke tests pass. DEAP-specific comparison is skipped if DEAP is not installed.

```bash
git add tests/benchmarks/
git commit -m "test(benchmarks): add GA benchmark smoke scripts"
```

---

## Task 7: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# evocore

Rust-native Genetic Algorithms and CMA-ES for Python.

## Install for Development

```bash
pip install maturin pytest
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

## Genetic Algorithm Quickstart

```python
from evocore import GAEngine, GeneSpace

def sphere(ind):
    return -sum(x * x for x in ind.genes)

engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 10), population_size=100, generations=100, seed=42)
result = engine.run(sphere)
print(result.best_fitness, result.best_individual.genes)
```

## Named Mixed Gene Space

```python
from evocore import GAEngine, GeneDef, GeneSpace

space = GeneSpace([
    GeneDef("period", "int", 5, 200, sigma=0.05),
    GeneDef("threshold", "float", 0.0, 1.0),
])

def objective(ind):
    return -abs(ind.params["period"] - 21) - abs(ind.params["threshold"] - 0.35)

result = GAEngine(space, seed=42).run(objective)
```

## CMA-ES

```python
from evocore import CMAESEngine, GeneSpace

engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 4), population_size=30, generations=80, seed=42)
result = engine.run(lambda ind: -sum(x * x for x in ind.genes))
```

## Reproducibility

All randomness derives from `derive_seed(master_seed, generation, individual_idx, op)`.
There is no global RNG state. Re-running the same engine with the same seed gives the same result,
and thread worker count does not change the generated populations.

## Parallelism

- `parallel="none"`: simplest and best for fast fitness functions.
- `parallel="thread"`: useful when the fitness function releases the GIL.
- `parallel="process"`: available on `GAEngine`; requires a module-level picklable fitness function.
- `CMAESEngine` rejects `parallel="process"` because its Rust covariance state is not picklable.

## Fitness Function Protocol

Fitness functions receive `Individual` and return either `float` or `(float, metrics_dict)`.
NaN and Inf are treated as `-inf` for selection and emit `FitnessWarning` once per run.
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: README quickstart and architecture notes"
```

---

## Task 8: Full Part 7 Verification

- [ ] **Step 1: Full Rust test suite**

Run: `cargo test`

Expected: all Rust tests pass.

- [ ] **Step 2: Rebuild extension**

Run: `maturin develop --release`

Expected: no errors.

- [ ] **Step 3: Full Python test suite**

Run: `pytest tests/unit/ tests/integration/ -v`

Expected: all unit and integration tests pass.

- [ ] **Step 4: Benchmark smoke tests**

Run: `pytest tests/benchmarks/ -v -s`

Expected: benchmark smoke tests pass or optional DEAP test skips.

- [ ] **Step 5: Example smoke tests**

Run:

```bash
python examples/sphere_optimization.py
python examples/onemax_binary.py
python examples/mixed_gene_space.py
python examples/cmaes_rosenbrock.py
```

Expected: all scripts finish without exceptions.

- [ ] **Step 6: Final public API smoke test**

```bash
python - << 'EOF'
from evocore import (
    GAEngine,
    CMAESEngine,
    GeneDef,
    GeneSpace,
    Individual,
    Population,
    RunResult,
    MultiRunResult,
    FitnessWarning,
    ConfigurationError,
)
assert GAEngine and CMAESEngine and GeneSpace and GeneDef
print("evocore v5 public API ok")
EOF
```

Expected: `evocore v5 public API ok`

- [ ] **Step 7: Final commit and tag**

```bash
git add .
git commit -m "chore: Part 7 complete - CMA-ES, tests, examples, benchmarks, README"
git tag v5-plan-complete
```

---

## Part 7 Exit Criteria Checklist

- [ ] `CMAESEngine` importable from top-level `evocore`
- [ ] `CMAESEngine(parallel="process")` raises `ConfigurationError` at construction
- [ ] CMA-ES bool genes are rejected
- [ ] CMA-ES integer genes are rounded/clamped only for the fitness function
- [ ] Continuous samples from `state.ask()` are passed to `state.tell()`
- [ ] `CMAESEngine.run()` returns the same `RunResult` shape as `GAEngine.run()`
- [ ] Unit tests cover CMA-ES run loop, process guard, and integer typing
- [ ] Integration tests cover sphere, rastrigin, binary OneMax, mixed GeneSpace, and CMA-ES Rosenbrock
- [ ] Examples run without exceptions
- [ ] Benchmark smoke scripts run or skip optional dependencies cleanly
- [ ] README documents install, quickstart, mixed spaces, CMA-ES, reproducibility, parallelism, and fitness protocol
