# evocore v5 - Part 6: GAEngine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete public `GAEngine`, `RunResult`, `MultiRunResult`, module-level `_run_child_engine`, resume support, and RNG reproducibility tests.

**Architecture:** `GAEngine` orchestrates deterministic Rust population initialization and one-call-per-generation Rust reproduction. Python owns evaluation because the public fitness protocol receives `Individual` and may return either `float` or `(float, metrics_dict)`. No Python random state is introduced. Child seeds for `run_multiple()` are derived with `_core.py_derive_seed(master, 0, run_idx, OP_MULTI_RUN)`. Process mode always uses spawn and performs picklability probes at `run()`/`run_multiple()` time.

**Tech Stack:** Python 3.11+, pytest, `evocore._core` from Parts 1-4, Python API foundation from Part 5

**Prerequisite:** Parts 1-5 complete and all unit tests green.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `evocore/ga.py` | Create | `GAEngine`, `RunResult`, `MultiRunResult`, `_run_child_engine`, resume |
| `evocore/__init__.py` | Modify | Export `GAEngine`, `RunResult`, `MultiRunResult` |
| `tests/unit/test_ga_engine.py` | Create | GA configuration, run loop, caching, callbacks, warnings, result objects |
| `tests/unit/test_rng_reproducibility.py` | Create | v3/v5 deterministic RNG invariants |
| `tests/unit/test_parallel.py` | Extend | `_run_child_engine` picklability, process mode picklability probe, run_multiple parallel |

---

## Engine Loop Contract

For `generations=N`, the logbook has exactly `N` entries. The initial population is evaluated before the loop. Each loop iteration:

1. Fires `on_generation_start(gen, current_population)`.
2. Selects Python elites from the current evaluated population.
3. Calls `_core.reproduce_population(...)` once to create `population_size - elitism` offspring.
4. Prepends cloned elites, marks offspring invalid, and evaluates invalid individuals only.
5. Builds `GenerationInfo(gen, nan_fitness_count, cached_count)`.
6. Appends a `LogEntry`.
7. Fires `on_generation_end(gen, new_population, info)`.

`fitness_valid` is never present in Rust data. Python elites carry `fitness_valid=True`; offspring start with `False`.

---

## Task 1: Results and Basic Imports

**Files:**
- Create: `evocore/ga.py`
- Create: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Write failing tests for result dataclasses**

```python
from evocore.ga import MultiRunResult, RunResult
from evocore.individual import Individual, Population
from evocore.stats import Logbook


def make_result(seed: int, fitness: float) -> RunResult:
    ind = Individual([fitness], fitness=fitness, fitness_valid=True)
    return RunResult(
        best_individual=ind,
        best_fitness=fitness,
        final_population=Population([ind]),
        logbook=Logbook(),
        wall_time_seconds=0.01,
        n_evaluations=1,
        elite_history=[ind],
        diversity_history=[],
        seed=seed,
        stopped_early=False,
    )


def test_multi_run_best_n_and_summary():
    r1 = make_result(1, 1.0)
    r2 = make_result(2, 3.0)
    r3 = make_result(3, 2.0)
    multi = MultiRunResult(best=r2, all_runs=[r2, r3, r1], n_runs=3, wall_time_seconds=0.03)
    assert multi.best_n(2) == [r2, r3]
    assert multi.fitness_summary() == {"mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0}
```

- [ ] **Step 2: Implement result dataclasses**

Add this to the top of `evocore/ga.py`:

```python
from __future__ import annotations

import copy
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Sequence

from evocore import _core
from evocore.callbacks import Callback, GenerationInfo
from evocore.exceptions import CheckpointError, ConfigurationError, FitnessError, FitnessWarning
from evocore.gene_space import GeneDef, GeneSpace
from evocore.individual import Individual, Population
from evocore.operators import OperatorSet
from evocore.parallel import ProcessParallel, ThreadParallel, ensure_picklable
from evocore.stats import LogEntry, Logbook


@dataclass
class RunResult:
    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
    n_evaluations: int
    elite_history: list[Individual]
    diversity_history: list[list[float]]
    seed: int
    stopped_early: bool


@dataclass
class MultiRunResult:
    best: RunResult
    all_runs: list[RunResult]
    n_runs: int
    wall_time_seconds: float

    def best_n(self, n: int) -> list[RunResult]:
        return self.all_runs[:n]

    def fitness_summary(self) -> dict[str, float]:
        values = [r.best_fitness for r in self.all_runs]
        return {
            "mean": mean(values) if values else float("nan"),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else float("nan"),
            "max": max(values) if values else float("nan"),
        }
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_ga_engine.py::test_multi_run_best_n_and_summary -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GA result dataclasses"
```

---

## Task 2: GAEngine Configuration and Validation

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing tests**

```python
import warnings
import warnings
import pytest
from evocore import ConfigurationError, ConfigurationWarning, GAEngine, GeneDef, GeneSpace


def test_ga_engine_requires_gene_space():
    with pytest.raises(ConfigurationError, match="gene_space required"):
        GAEngine(gene_space=None)


def test_invalid_parallel_mode_rejected():
    with pytest.raises(ConfigurationError, match="parallel"):
        GAEngine(gene_space=GeneSpace.uniform(-1.0, 1.0, 2), parallel="gpu")


def test_binary_space_default_operators_work():
    engine = GAEngine(
        gene_space=GeneSpace([GeneDef("a", "bool"), GeneDef("b", "bool")]),
        crossover="one_point",
        mutation="bit_flip",
    )
    assert engine.population_size == 100


def test_large_int_without_sigma_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        GAEngine(
            gene_space=GeneSpace([GeneDef("ema_slow", "int", 10, 500)]),
            population_size=10,
            generations=2,
            mutation_sigma=0.2,
        )
    warnings_of_type = [w for w in caught if issubclass(w.category, ConfigurationWarning)]
    assert len(warnings_of_type) == 1
    assert "ema_slow" in str(warnings_of_type[0].message)
```

- [ ] **Step 2: Implement `GAEngine.__init__` and validation**

```python
class GAEngine:
    def __init__(
        self,
        gene_space: GeneSpace,
        population_size: int = 100,
        generations: int = 100,
        crossover: str = "sbx",
        crossover_prob: float = 0.9,
        crossover_eta: float = 2.0,
        crossover_alpha: float = 0.5,
        mutation: str = "gaussian",
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.2,
        mutation_sigma_schedule: str = "constant",
        mutation_sigma_end: float = 0.02,
        selection: str = "tournament",
        tournament_size: int = 3,
        elitism: int = 1,
        parallel: str = "none",
        n_workers: int | None = None,
        process_initializer=None,
        process_initargs=(),
        seed: int = 0,
        track_diversity: bool = False,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for GAEngine. Pass GeneSpace.uniform(-5.0, 5.0, length)."
            )
        if population_size < 2:
            raise ConfigurationError("population_size must be at least 2.")
        if generations < 0:
            raise ConfigurationError("generations must be >= 0.")
        if elitism < 0 or elitism >= population_size:
            raise ConfigurationError("elitism must satisfy 0 <= elitism < population_size.")
        if parallel not in ("none", "thread", "process"):
            raise ConfigurationError("parallel must be one of 'none', 'thread', or 'process'.")
        if selection not in ("tournament", "roulette", "rank"):
            raise ConfigurationError("selection must be 'tournament', 'roulette', or 'rank'.")
        if mutation_sigma_schedule not in ("constant", "linear_decay", "cosine_decay"):
            raise ConfigurationError("mutation_sigma_schedule must be 'constant', 'linear_decay', or 'cosine_decay'.")

        self.gene_space = gene_space
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.crossover_alpha = crossover_alpha
        self.mutation = mutation
        self.mutation_prob = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.mutation_sigma_schedule = mutation_sigma_schedule
        self.mutation_sigma_end = mutation_sigma_end
        self.selection = selection
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.parallel = parallel
        self.n_workers = n_workers
        self.process_initializer = process_initializer
        self.process_initargs = process_initargs
        self.seed = int(seed)
        self.track_diversity = track_diversity
        self.callbacks = list(callbacks or [])
        self.operators = OperatorSet(gene_space, crossover, mutation)
        self._fitness_warning_emitted = False

        for gene in gene_space.genes:
            if gene.kind == "int" and gene.sigma is None and (gene.high - gene.low) > 100:
                sigma_abs = mutation_sigma * (gene.high - gene.low)
                warnings.warn(
                    f'GeneDef("{gene.name}", "int", {gene.low}, {gene.high}) has range '
                    f"{gene.high - gene.low} and no per-gene sigma. With mutation_sigma={mutation_sigma}, "
                    f"sigma_abs={sigma_abs:g} may prevent fine-tuning. Consider sigma=0.03.",
                    category=ConfigurationWarning,
                    stacklevel=2,
                )
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_ga_engine.py -k "requires_gene_space or invalid_parallel or binary_space or large_int" -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GAEngine configuration validation"
```

---

## Task 3: Fitness Evaluation Helpers

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing tests**

```python
import pytest
from evocore import FitnessError, FitnessWarning, GAEngine, GeneSpace
from evocore.individual import Individual


def test_tuple_fitness_stores_metrics():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    ind = Individual([0.5, 0.25])
    fitness, nan_count = engine._evaluate_all([ind], lambda x: (1.5, {"sharpe": 2.0}), gen=0)
    assert fitness == [1.5]
    assert nan_count == 0
    assert ind.metadata["metrics"] == {"sharpe": 2.0}
    assert ind.fitness_valid is True


def test_nan_fitness_warns_once_and_sanitizes():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    ind = Individual([0.0, 0.0])
    with pytest.warns(FitnessWarning):
        fitnesses, nan_count = engine._evaluate_all([ind], lambda x: float("nan"), gen=0)
    assert fitnesses == [float("-inf")]
    assert nan_count == 1
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        engine._evaluate_all([Individual([0.0, 0.0])], lambda x: float("nan"), gen=1)
    assert len(second) == 0


def test_fitness_exception_wrapped():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=1)
    with pytest.raises(FitnessError, match="ZeroDivisionError"):
        engine._evaluate_all([Individual([0.0, 0.0])], lambda x: 1 / 0, gen=0)
```

- [ ] **Step 2: Implement evaluation helpers**

Add these methods inside `GAEngine`:

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
        pending = [ind for ind in individuals if not ind.fitness_valid]
        if self.parallel == "process":
            ensure_picklable(fitness_fn, context="parallel='process'")
            raw_results = ProcessParallel(
                self.n_workers,
                initializer=self.process_initializer,
                initargs=self.process_initargs,
            ).evaluate(pending, fitness_fn)
        elif self.parallel == "thread":
            raw_results = ThreadParallel(self.n_workers).evaluate(pending, fitness_fn)
        else:
            raw_results = []
            for idx, ind in enumerate(pending):
                try:
                    raw_results.append(fitness_fn(ind))
                except Exception as exc:
                    raise FitnessError(
                        f"fitness_fn raised {type(exc).__name__} for individual at generation {gen}, index {idx}. "
                        f"Original error: {exc}"
                    ) from exc

        nan_count = 0
        pending_iter = iter(pending)
        for raw_idx, raw in enumerate(raw_results):
            ind = next(pending_iter)
            fitness, n_bad = self._normalise_fitness_result(raw, ind, gen, raw_idx)
            nan_count += n_bad
        if nan_count and not self._fitness_warning_emitted:
            warnings.warn(
                f"{nan_count} individuals in generation {gen} returned NaN or Inf fitness. "
                "They have been assigned fitness=-inf for selection.",
                FitnessWarning,
                stacklevel=2,
            )
            self._fitness_warning_emitted = True
        return [float(ind.fitness) if ind.fitness is not None else float("-inf") for ind in individuals], nan_count
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_ga_engine.py -k "fitness or nan" -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GA fitness evaluation protocol"
```

---

## Task 4: Implement `GAEngine.run()`

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing tests**

```python
from evocore import Callback, GAEngine, GenerationInfo, GeneDef, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_ga_run_returns_result_with_logbook_length():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 3), population_size=20, generations=5, seed=42)
    result = engine.run(sphere)
    assert result.best_fitness <= 0.0
    assert len(result.final_population) == 20
    assert len(result.logbook) == 5
    assert result.seed == 42
    assert result.n_evaluations > 0


def test_on_generation_end_receives_generation_info():
    received = []

    class Capture(Callback):
        def on_generation_end(self, gen, pop, info):
            received.append(info)

    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=3, callbacks=[Capture()])
    engine.run(sphere)
    assert len(received) == 3
    assert all(isinstance(info, GenerationInfo) for info in received)


def test_track_diversity_false_and_true():
    off = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, track_diversity=False)
    on = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, track_diversity=True)
    assert off.run(sphere).diversity_history == []
    assert len(on.run(sphere).diversity_history) == 2


def test_elitism_caches_best_individual():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=12, generations=3, elitism=2, seed=7)
    result = engine.run(sphere)
    assert any(entry.cached_count == 2 for entry in result.logbook)
```

- [ ] **Step 2: Implement initialization, sigma schedule, logging, and run loop**

Add these methods inside `GAEngine`:

```python
    def _compute_sigma_fraction(self, gen: int) -> float:
        if self.generations <= 1 or self.mutation_sigma_schedule == "constant":
            return self.mutation_sigma
        t = gen / max(1, self.generations - 1)
        if self.mutation_sigma_schedule == "linear_decay":
            return self.mutation_sigma + t * (self.mutation_sigma_end - self.mutation_sigma)
        if self.mutation_sigma_schedule == "cosine_decay":
            c = 0.5 * (1.0 + math.cos(math.pi * t))
            return self.mutation_sigma_end + c * (self.mutation_sigma - self.mutation_sigma_end)
        raise ConfigurationError("unknown mutation_sigma_schedule")

    def _initial_population(self) -> list[Individual]:
        encoded = _core.init_population(
            self.operators.gene_bounds,
            self.operators.gene_kinds,
            self.population_size,
            self.seed,
        )
        return self.operators.decode_population(encoded)

    def _clone_elites(self, population: list[Individual]) -> list[Individual]:
        if self.elitism == 0:
            return []
        return [ind.clone() for ind in Population(population).best(self.elitism)]

    def _bind_callbacks(self) -> None:
        for cb in self.callbacks:
            cb.bind_context(seed=self.seed, generations=self.generations)

    def _callbacks_should_stop(self) -> bool:
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)

    def _log_entry(
        self,
        gen: int,
        pop: Population,
        gen_start: float,
        n_evaluations: int,
        info: GenerationInfo,
        diversity: list[float],
    ) -> LogEntry:
        best = pop.best(1)[0]
        return LogEntry(
            gen=gen,
            best_fitness=float(best.fitness),
            mean_fitness=pop.mean_fitness(),
            std_fitness=pop.std_fitness(),
            wall_time_ms=(time.perf_counter() - gen_start) * 1000.0,
            n_evaluations=n_evaluations,
            nan_fitness_count=info.nan_fitness_count,
            cached_count=info.cached_count,
            diversity=diversity,
            custom=best.metadata.get("metrics", {}),
        )

    def run(self, fitness_fn: Callable[[Individual], float | tuple[float, dict]]) -> RunResult:
        if self.parallel == "process":
            ensure_picklable(fitness_fn, context="parallel='process'")
        self._fitness_warning_emitted = False
        self._bind_callbacks()
        start = time.perf_counter()
        logbook = Logbook()
        population = self._initial_population()
        fitnesses, nan_count = self._evaluate_all(population, fitness_fn, gen=-1)
        n_evaluations = len(population)
        elite_history: list[Individual] = []
        diversity_history: list[list[float]] = []
        stopped_early = False

        for gen in range(self.generations):
            gen_start = time.perf_counter()
            current_pop = Population(population)
            for cb in self.callbacks:
                cb.on_generation_start(gen, current_pop)
            if self._callbacks_should_stop():
                stopped_early = True
                break

            elites = self._clone_elites(population)
            for elite in elites:
                elite.fitness_valid = True
            offspring_count = self.population_size - len(elites)
            offspring_encoded = []
            if offspring_count:
                sigma_list = self.operators.sigma_abs_list(self._compute_sigma_fraction(gen))
                offspring_encoded = _core.reproduce_population(
                    self.operators.encode_population(population),
                    fitnesses,
                    self.crossover,
                    self.crossover_prob,
                    self.crossover_eta,
                    self.crossover_alpha,
                    self.mutation,
                    self.mutation_prob,
                    sigma_list,
                    self.operators.gene_bounds,
                    self.operators.gene_kinds,
                    self.selection,
                    self.tournament_size,
                    offspring_count,
                    self.seed,
                    gen,
                )
            offspring = self.operators.decode_population(offspring_encoded)
            population = elites + offspring
            eval_before = n_evaluations
            fitnesses, nan_count = self._evaluate_all(population, fitness_fn, gen=gen)
            n_evaluations += sum(1 for ind in offspring if ind.fitness_valid)
            pop_obj = Population(population)
            info = GenerationInfo(gen, nan_count, len(elites))
            diversity = pop_obj.diversity() if self.track_diversity else []
            if self.track_diversity:
                diversity_history.append(diversity)
            elite_history.append(pop_obj.best(1)[0].clone())
            logbook.append(self._log_entry(gen, pop_obj, gen_start, n_evaluations - eval_before, info, diversity))
            for cb in self.callbacks:
                cb.on_generation_end(gen, pop_obj, info)
            if self._callbacks_should_stop():
                stopped_early = True
                break

        final_pop = Population(population)
        best = final_pop.best(1)[0]
        result = RunResult(
            best_individual=best.clone(),
            best_fitness=float(best.fitness),
            final_population=final_pop,
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

Run: `pytest tests/unit/test_ga_engine.py -k "ga_run or generation_info or diversity or elitism" -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GAEngine run loop with Rust reproduction"
```

---

## Task 5: `run_multiple()` and `_run_child_engine`

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_parallel.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing tests**

```python
import pickle
import pytest
from evocore import ConfigurationError, GAEngine, GeneSpace
from evocore.ga import _run_child_engine


def module_sphere(ind):
    return -sum(x * x for x in ind.genes)


def test_run_child_engine_is_picklable():
    pickle.dumps(_run_child_engine)
    assert ".<locals>." not in _run_child_engine.__qualname__
    assert "." not in _run_child_engine.__qualname__


def test_run_multiple_sequential_returns_sorted_runs():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)
    multi = engine.run_multiple(module_sphere, n_runs=3, run_parallel=False)
    assert multi.n_runs == 3
    assert len(multi.all_runs) == 3
    assert multi.all_runs == sorted(multi.all_runs, key=lambda r: r.best_fitness, reverse=True)
    assert multi.wall_time_seconds > 0.0


def test_run_multiple_parallel_rejects_lambda():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=10, generations=2, seed=42)
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        engine.run_multiple(lambda ind: 1.0, n_runs=2, run_parallel=True)
```

- [ ] **Step 2: Implement module-level helper and `run_multiple()`**

Place `_run_child_engine` at module level, outside `GAEngine`:

```python
def _run_child_engine(engine: "GAEngine", seed: int, fitness_fn: Callable) -> RunResult:
    return engine._copy_with_seed(seed).run(fitness_fn)
```

Add these methods to `GAEngine`:

```python
    def _copy_with_seed(self, seed: int) -> "GAEngine":
        return GAEngine(
            gene_space=self.gene_space,
            population_size=self.population_size,
            generations=self.generations,
            crossover=self.crossover,
            crossover_prob=self.crossover_prob,
            crossover_eta=self.crossover_eta,
            crossover_alpha=self.crossover_alpha,
            mutation=self.mutation,
            mutation_prob=self.mutation_prob,
            mutation_sigma=self.mutation_sigma,
            mutation_sigma_schedule=self.mutation_sigma_schedule,
            mutation_sigma_end=self.mutation_sigma_end,
            selection=self.selection,
            tournament_size=self.tournament_size,
            elitism=self.elitism,
            parallel=self.parallel,
            n_workers=self.n_workers,
            process_initializer=self.process_initializer,
            process_initargs=self.process_initargs,
            seed=int(seed),
            track_diversity=self.track_diversity,
            callbacks=copy.deepcopy(self.callbacks),
        )

    def run_multiple(
        self,
        fitness_fn: Callable,
        n_runs: int = 10,
        aggregate: str = "best",
        run_parallel: bool = False,
    ) -> MultiRunResult:
        if n_runs <= 0:
            raise ConfigurationError("n_runs must be positive.")
        if aggregate not in ("best", "all"):
            raise ConfigurationError("aggregate must be 'best' or 'all'.")
        child_seeds = [
            int(_core.py_derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
            for run_idx in range(n_runs)
        ]
        t0 = time.perf_counter()
        if run_parallel:
            ensure_picklable(fitness_fn, context="run_multiple(run_parallel=True)")
            ensure_picklable(self, context="run_multiple(run_parallel=True) engine")
            import concurrent.futures
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=min(n_runs, self.n_workers or os.cpu_count() or 1),
                mp_context=ctx,
            )
            try:
                futures = [
                    pool.submit(_run_child_engine, self, seed, fitness_fn)
                    for seed in child_seeds
                ]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            finally:
                pool.shutdown(cancel_futures=True, wait=False)
        else:
            results = [self._copy_with_seed(seed).run(fitness_fn) for seed in child_seeds]
        elapsed = time.perf_counter() - t0
        results.sort(key=lambda r: r.best_fitness, reverse=True)
        return MultiRunResult(best=results[0], all_runs=results, n_runs=n_runs, wall_time_seconds=elapsed)
```

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_parallel.py tests/unit/test_ga_engine.py -k "run_multiple or child_engine" -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_parallel.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GAEngine run_multiple with child seed derivation"
```

---

## Task 6: Resume from Checkpoint

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing test**

```python
import pickle
import pytest
from evocore import CheckpointError, GAEngine, GeneSpace
from evocore.individual import Individual, Population


def test_resume_missing_checkpoint_lists_available(tmp_path):
    (tmp_path / "checkpoint_gen_1.pkl").write_bytes(b"bad")
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=2)
    with pytest.raises(CheckpointError, match="Available checkpoints"):
        engine.resume(lambda ind: 1.0, str(tmp_path / "checkpoint_gen_9.pkl"))
```

- [ ] **Step 2: Implement resume loader**

```python
    def resume(self, fitness_fn: Callable, checkpoint: str) -> RunResult:
        if not os.path.exists(checkpoint):
            directory = os.path.dirname(checkpoint) or "."
            available = sorted(name for name in os.listdir(directory) if name.startswith("checkpoint_gen_"))
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} not found. Available checkpoints: {', '.join(available) or 'none'}"
            )
        try:
            with open(checkpoint, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            raise CheckpointError(f"checkpoint file {checkpoint!r} is corrupt or incompatible: {exc}") from exc
        population = payload.get("population")
        if not isinstance(population, list) or not all(isinstance(ind, Individual) for ind in population):
            raise CheckpointError("checkpoint payload must contain a list[Individual] under key 'population'.")
        saved_generation = int(payload.get("generation", -1))
        saved_seed = payload.get("seed")
        if saved_seed is not None and int(saved_seed) != self.seed:
            raise CheckpointError(f"checkpoint seed {saved_seed} does not match engine seed {self.seed}.")
        resumed = self._copy_with_seed(self.seed)
        original_generations = resumed.generations
        resumed.generations = max(0, original_generations - saved_generation - 1)
        return resumed._run_from_population(population, fitness_fn, start_generation=saved_generation + 1)
```

Refactor `run()` so it calls a private `_run_from_population(population, fitness_fn, start_generation=0)` and uses `range(start_generation, self.generations)`. The public `run()` creates `_initial_population()` and calls that helper with `start_generation=0`.

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_ga_engine.py -k resume -v`

Expected: pass.

```bash
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat(python): GAEngine resume checkpoint loader"
```

---

## Task 7: RNG Reproducibility Tests

**Files:**
- Create: `tests/unit/test_rng_reproducibility.py`

- [ ] **Step 1: Write invariant tests**

```python
import numpy as np
from evocore import GAEngine, GeneSpace


def sphere(ind):
    return -sum(x * x for x in ind.genes)


def numpy_sphere(ind):
    arr = np.array(ind.genes)
    return float(-np.sum(arr * arr))


def test_run_twice_same_engine_identical_results():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, generations=8, seed=42)
    r1 = engine.run(sphere)
    r2 = engine.run(sphere)
    assert r1.best_fitness == r2.best_fitness
    assert r1.best_individual.genes == r2.best_individual.genes


def test_sequential_and_thread_parallel_identical():
    seq = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, generations=8, parallel="none", seed=99)
    thr = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=30, generations=8, parallel="thread", n_workers=4, seed=99)
    r_seq = seq.run(numpy_sphere)
    r_thr = thr.run(numpy_sphere)
    assert r_seq.best_fitness == r_thr.best_fitness
    assert r_seq.best_individual.genes == r_thr.best_individual.genes


def test_n_workers_does_not_affect_results():
    results = []
    for n_workers in [1, 2, 4]:
        engine = GAEngine(
            GeneSpace.uniform(-5.0, 5.0, 5),
            population_size=30,
            generations=8,
            parallel="thread",
            n_workers=n_workers,
            seed=123,
        )
        results.append(engine.run(numpy_sphere).best_individual.genes)
    assert results[0] == results[1] == results[2]


def test_different_seeds_diverge():
    e1 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=1)
    e2 = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=2)
    assert e1.run(sphere).best_individual.genes != e2.run(sphere).best_individual.genes


def test_multi_run_child_seeds_are_independent():
    engine = GAEngine(GeneSpace.uniform(-5.0, 5.0, 5), population_size=20, generations=4, seed=42)
    multi = engine.run_multiple(sphere, n_runs=5)
    assert len({tuple(r.best_individual.genes) for r in multi.all_runs}) > 1
```

- [ ] **Step 2: Run tests and commit**

Run: `pytest tests/unit/test_rng_reproducibility.py -v`

Expected: all invariants pass.

```bash
git add tests/unit/test_rng_reproducibility.py
git commit -m "test(python): GA RNG reproducibility invariants"
```

---

## Task 8: Update Top-Level Exports

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Extend import tests**

```python
def test_ga_exports_accessible_from_top_level():
    from evocore import GAEngine, RunResult, MultiRunResult
    assert GAEngine is not None
    assert RunResult is not None
    assert MultiRunResult is not None
```

- [ ] **Step 2: Add exports**

Add to `evocore/__init__.py`:

```python
from evocore.ga import GAEngine, RunResult, MultiRunResult
```

Also add `"GAEngine"`, `"RunResult"`, and `"MultiRunResult"` to `__all__`.

- [ ] **Step 3: Run tests and commit**

Run: `pytest tests/unit/test_package_init.py tests/unit/test_ga_engine.py tests/unit/test_rng_reproducibility.py -v`

Expected: pass.

```bash
git add evocore/__init__.py tests/unit/test_package_init.py
git commit -m "feat(python): export GAEngine public API"
```

---

## Task 9: Full Part 6 Verification

- [ ] **Step 1: Run all Rust tests**

Run: `cargo test`

Expected: Parts 1-4 Rust tests pass.

- [ ] **Step 2: Rebuild extension**

Run: `maturin develop --release`

Expected: no compile errors.

- [ ] **Step 3: Run all Python unit tests**

Run: `pytest tests/unit/ -v`

Expected: all unit tests from Parts 1-6 pass.

- [ ] **Step 4: End-to-end GA smoke test**

```bash
python - << 'EOF'
from evocore import GAEngine, GeneDef, GeneSpace

def sphere(ind):
    return -sum(x * x for x in ind.genes)

engine = GAEngine(
    gene_space=GeneSpace([
        GeneDef("x0", "float", -5.0, 5.0),
        GeneDef("x1", "float", -5.0, 5.0),
    ]),
    population_size=50,
    generations=20,
    seed=42,
    track_diversity=True,
)
result = engine.run(sphere)
assert result.best_fitness <= 0.0
assert len(result.logbook) == 20
assert len(result.diversity_history) == 20
multi = engine.run_multiple(sphere, n_runs=3)
assert multi.n_runs == 3
print("Part 6 complete - GAEngine ok")
EOF
```

Expected: `Part 6 complete - GAEngine ok`

- [ ] **Step 5: Final commit and tag**

```bash
git add .
git commit -m "chore: Part 6 complete - GAEngine and reproducibility tests"
git tag part6-complete
```

---

## Part 6 Exit Criteria Checklist

- [ ] `GAEngine.run()` works for float, int, and binary spaces supported by `OperatorSet`
- [ ] `GAEngine.run()` calls `_core.reproduce_population()` once per generation
- [ ] `fitness_valid` is Python-only and used to skip elite re-evaluation
- [ ] NaN/Inf fitness values become `-inf` for selection and emit `FitnessWarning` once per run
- [ ] Tuple fitness returns store metrics in `Individual.metadata["metrics"]` and `LogEntry.custom`
- [ ] `GenerationInfo` is passed to `on_generation_end(gen, pop, info)`
- [ ] `run()` is idempotent for the same engine and seed
- [ ] Thread worker count does not change GA results
- [ ] `_run_child_engine` is module-level and picklable
- [ ] `run_multiple(run_parallel=True)` uses spawn, try/finally teardown, and elapsed wall-clock timing
- [ ] `MultiRunResult.wall_time_seconds` is actual elapsed time, not sum of child run times
- [ ] `GAEngine(parallel="process").run()` probes fitness picklability at run time
- [ ] Top-level `evocore` exports `GAEngine`, `RunResult`, and `MultiRunResult`
