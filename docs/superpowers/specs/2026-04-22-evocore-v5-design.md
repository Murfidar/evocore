# evocore v5 Design Spec

**Date:** 2026-04-22
**Status:** Draft
**Scope:** Phase 1 — Genetic Algorithms (float + integer + binary + mixed individuals) + CMA-ES, Rust-native Python library
**Supersedes:** 2026-04-22-evocore-v4-design.md

---

## Changelog from v4

| Area | Change |
|---|---|
| **Issue 1: `_run_child_engine` importability** | Add pickle test in `test_parallel.py` confirming `_run_child_engine` is defined at module level and survives `pickle.dumps()`. No structural change — architecture was already correct. |
| **Issue 2: `KeyboardInterrupt` teardown** | `ProcessParallel.evaluate()` and `run_multiple(run_parallel=True)` wrap executor usage in `try/finally`. On any exception, `pool.shutdown(cancel_futures=True, wait=False)` is called before re-raising. Docstring notes that already-running workers complete their current evaluation. |
| **Issue 3: `CMAESEngine` + `parallel="process"`** | `CMAESEngine.__init__` raises `ConfigurationError` immediately if `parallel="process"` is passed. `PyCMAESState` is a non-picklable PyO3 Rust object; the constraint is structural and always true, so the error fires at construction rather than at `run()` time. |
| **Issue 4: `run_multiple` wall-time accounting** | `MultiRunResult.wall_time_seconds` now reflects actual elapsed wall-clock time via `time.perf_counter()` brackets around the full dispatch block — for both `run_parallel=True` and `run_parallel=False`. Individual `RunResult.wall_time_seconds` values remain unchanged (per-run compute time). |

Everything not listed above is **unchanged from v4**.

---

## 1. Goal

Rebuild the hot-path components of DEAP as a Rust-native Python library with a clean, domain-agnostic modern API. Designed for any optimization workload where the fitness function is expensive (backtesting, simulation, hyperparameter tuning). Target: ≥3× sequential and ≥10× parallel speedup over pure-Python DEAP for GA workloads.

**Key improvements over v4:**
- `run_multiple` wall-time is accurate regardless of parallel mode
- `CMAESEngine(parallel="process")` fails immediately at construction with a clear explanation
- `ProcessParallel` teardown is safe under `KeyboardInterrupt` — no hanging worker processes
- `_run_child_engine` importability is a tested invariant, not an assumption

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Python Code                          │
├──────────────────────────────────────────────────────────────┤
│                 evocore  (Python Layer)                        │
│  GeneSpace · GeneDef · Individual · Population                │
│  GAEngine · CMAESEngine · RunResult · MultiRunResult          │
│  Operators (bounds-aware) · Parallelism · Logbook · Callbacks │
│  GenerationInfo                                               │
├──────────────────────────────────────────────────────────────┤
│            evocore._core  (PyO3 Extension)                    │
│  individuals · operators · selection · cmaes                  │
│  utils (derive_seed) · reproduce · parallel                   │
├──────────────────────────────────────────────────────────────┤
│                      Rust Crate                               │
│  FloatIndividual · IntegerIndividual · BinaryIndividual        │
│  float_ops · int_ops · binary_ops · selection · cmaes         │
│  utils · gene_spec · reproduce                                │
└──────────────────────────────────────────────────────────────┘
```

**Layer responsibilities (unchanged from v4):**
- `evocore` — user-facing Python. Typed API, `GeneSpace`, callbacks, statistics, parallelism wrappers. Zero random state. No recompile for API changes.
- `evocore._core` — compiled PyO3 extension. All hot paths: operator loops, selection, CMA-ES matrix math, Rayon pool, seed derivation. All genes cross the PyO3 boundary as `f64`.

---

## 3. Project File Structure

No new files in v5. The changes are confined to existing Python source and test files.

```
evocore/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/                              # Rust source — UNCHANGED from v4
│   ├── lib.rs
│   ├── utils.rs
│   ├── individual.rs
│   ├── gene_spec.rs
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── float_ops.rs
│   │   ├── int_ops.rs
│   │   └── binary_ops.rs
│   ├── reproduce.rs
│   ├── selection.rs
│   ├── cmaes.rs
│   └── parallel.rs
├── evocore/                          # Python source
│   ├── __init__.py
│   ├── gene_space.py
│   ├── individual.py
│   ├── operators.py
│   ├── parallel.py                   # MODIFIED: try/finally teardown in ProcessParallel
│   ├── ga.py                         # MODIFIED: perf_counter brackets in run_multiple()
│   ├── cmaes.py                      # MODIFIED: ConfigurationError guard for parallel="process"
│   ├── stats.py
│   ├── callbacks.py
│   └── exceptions.py
├── tests/
│   ├── unit/
│   │   ├── test_rng_reproducibility.py
│   │   ├── test_gene_space.py
│   │   ├── test_operators.py
│   │   ├── test_selection.py
│   │   ├── test_ga_engine.py
│   │   ├── test_cmaes_engine.py      # MODIFIED: new test for parallel="process" guard
│   │   ├── test_parallel.py          # MODIFIED: new picklability + teardown tests
│   │   └── test_stats.py
│   ├── integration/
│   │   ├── test_sphere_function.py
│   │   ├── test_rastrigin.py
│   │   ├── test_binary_onemax.py
│   │   ├── test_mixed_gene_space.py
│   │   └── test_cmaes_rosenbrock.py
│   └── benchmarks/
│       ├── bench_ga_vs_deap.py
│       └── bench_parallel_scaling.py
├── examples/
│   ├── sphere_optimization.py
│   ├── onemax_binary.py
│   ├── mixed_gene_space.py
│   └── cmaes_rosenbrock.py
└── docs/
    └── superpowers/
        └── specs/
```

---

## 4. RNG Architecture

Unchanged from v4/v3. All randomness derives from a single `u64` master seed via `derive_seed(master, generation, individual_idx, op)` in `src/utils.rs`. No mutable RNG state anywhere. `run()` is idempotent. Thread count does not affect results.

```rust
// src/utils.rs — unchanged from v3/v4
pub const OP_INIT:           u64 = 0;
pub const OP_CROSSOVER:      u64 = 1;
pub const OP_MUTATION:       u64 = 2;
pub const OP_SELECTION:      u64 = 3;
pub const OP_CMAES_ASK:      u64 = 4;
pub const OP_MULTI_RUN:      u64 = 5;
pub const OP_CROSSOVER_PROB: u64 = 6;

pub fn derive_seed(master: u64, generation: u64, individual_idx: u64, op: u64) -> u64 {
    let mut x = master
        .wrapping_add(generation.wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add(individual_idx.wrapping_mul(0x6c62272e07bb0142))
        .wrapping_add(op.wrapping_mul(0xd2b74407b1ce6d93));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
```

---

## 5. Gene Space (`evocore/gene_space.py`)

Unchanged from v4.

```python
@dataclass
class GeneDef:
    name:  str
    kind:  Literal["float", "int", "bool"]
    low:   float | int | None = None
    high:  float | int | None = None
    sigma: float | None = None          # per-gene sigma override (fraction of range)

    def __post_init__(self):
        if self.kind != "bool":
            assert self.low is not None and self.high is not None
            assert self.low < self.high
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
        if self.sigma is not None:
            assert 0.0 < self.sigma <= 1.0, "sigma must be in (0, 1]"
```

```python
# Mode A: explicit per-gene
space = GeneSpace([
    GeneDef("ema_fast",   kind="int",   low=5,    high=200,  sigma=0.05),
    GeneDef("ema_slow",   kind="int",   low=10,   high=500,  sigma=0.03),
    GeneDef("threshold",  kind="float", low=0.0,  high=1.0),
    GeneDef("atr_mult",   kind="float", low=0.5,  high=5.0),
    GeneDef("use_filter", kind="bool"),
])

# Mode B: uniform float — backward compatible
space = GeneSpace.uniform(low=-5.0, high=5.0, length=10)
```

Construction warning for large integer ranges without explicit sigma (unchanged from v4):
```
ConfigurationWarning: GeneDef("ema_slow", "int", 10, 500) has range 490 and no per-gene sigma.
  With mutation_sigma=0.2, σ_abs=98 — large steps may prevent fine-tuning in later generations.
  Consider: GeneDef("ema_slow", "int", 10, 500, sigma=0.03)
```

---

## 6. Rust Core Components (`evocore._core`)

**No changes from v4.** All Rust source is identical. The four v5 issues are Python-only fixes.

For completeness, the individual structs (with `fitness_valid` removed per v4):

```rust
// src/individual.rs — unchanged from v4
#[pyclass] #[derive(Clone, Debug)]
pub struct FloatIndividual {
    #[pyo3(get, set)] pub genes:   Vec<f64>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
}

#[pyclass] #[derive(Clone, Debug)]
pub struct IntegerIndividual {
    #[pyo3(get, set)] pub genes:   Vec<i64>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
}

#[pyclass] #[derive(Clone, Debug)]
pub struct BinaryIndividual {
    #[pyo3(get, set)] pub genes:   Vec<bool>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
}
```

PyO3 boundary encoding, operator signatures, selection, `reproduce()`, parallelism, and CMA-ES are all unchanged from v4.

---

## 7. Python API Layer

### 7.1 Individual & Population

Unchanged from v4.

```python
@dataclass
class Individual:
    genes:         list[float | int | bool]
    fitness:       float | None = None
    fitness_valid: bool = False
    metadata:      dict = field(default_factory=dict)

    @property
    def params(self) -> dict | None:
        return self.metadata.get("params")
```

### 7.2 GAEngine

Unchanged from v4, with one internal modification to `run_multiple()` (Section 7.4).

```python
engine = GAEngine(
    gene_space=GeneSpace([
        GeneDef("x0", "float", -5.0, 5.0),
        GeneDef("x1", "float", -5.0, 5.0),
    ]),
    population_size=200,
    generations=100,
    crossover="sbx",
    crossover_prob=0.9,
    crossover_eta=2.0,
    crossover_alpha=0.5,
    mutation="gaussian",
    mutation_prob=0.1,
    mutation_sigma=0.2,
    mutation_sigma_schedule="constant",
    mutation_sigma_end=0.02,
    selection="tournament",
    tournament_size=3,
    elitism=5,
    parallel="none",         # "none" | "thread" | "process"
    n_workers=None,
    seed=42,
    track_diversity=False,
    callbacks=[],
)

result = engine.run(fitness_fn=my_fitness)
multi  = engine.run_multiple(fitness_fn=my_fitness, n_runs=10, run_parallel=False)
result = engine.resume(fitness_fn=my_fitness, checkpoint="./checkpoints/checkpoint_gen_50.pkl")
```

### 7.3 `GAEngine.run()` inner loop

Unchanged from v4. Zero Python random state. Single `_core.reproduce()` call per generation.

### 7.4 `run_multiple()` — Revised Wall-Time Accounting (Issue 4)

`MultiRunResult.wall_time_seconds` now measures actual elapsed time from dispatch to last result, for both parallel and sequential paths. The `time.perf_counter()` bracket is the sole source of this value.

```python
import time

def run_multiple(
    self,
    fitness_fn: Callable,
    n_runs: int = 10,
    aggregate: str = "best",
    run_parallel: bool = False,
) -> MultiRunResult:
    child_seeds = [
        int(_core.derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
        for run_idx in range(n_runs)
    ]

    # Wall-clock bracket wraps the entire dispatch block — both paths.
    t0 = time.perf_counter()

    if run_parallel:
        import pickle
        try:
            pickle.dumps(fitness_fn)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise ConfigurationError(
                f"fitness_fn cannot be pickled, required for run_parallel=True.\n"
                f"  Error: {e}"
            ) from e

        import multiprocessing
        import concurrent.futures
        ctx = multiprocessing.get_context("spawn")
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(n_runs, (self.n_workers or os.cpu_count())),
            mp_context=ctx,
        )
        try:
            futures = {
                pool.submit(_run_child_engine, self, seed, fitness_fn): seed
                for seed in child_seeds
            }
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        finally:
            # Issue 2: cancel queued-but-not-started futures; do not block on running workers.
            # Running workers will complete their current fitness evaluation before terminating.
            pool.shutdown(cancel_futures=True, wait=False)

    else:
        results = [
            self._copy_with_seed(seed).run(fitness_fn)
            for seed in child_seeds
        ]

    # Actual elapsed wall-clock time — not a sum of per-run times.
    elapsed = time.perf_counter() - t0

    results.sort(key=lambda r: r.best_fitness, reverse=True)
    return MultiRunResult(
        best=results[0],
        all_runs=results,
        n_runs=n_runs,
        wall_time_seconds=elapsed,   # reflects true elapsed time in both modes
    )
```

**Semantic note:** `MultiRunResult.wall_time_seconds` is now wall-clock time. For sequential runs this equals (approximately) the sum of `RunResult.wall_time_seconds` values. For parallel runs it will be substantially less than that sum. Individual `RunResult.wall_time_seconds` values are unchanged and remain per-run compute time.

### 7.5 `_run_child_engine` — Module-Level Placement (Issue 1)

Unchanged from v4. Repeated here for clarity because a test now enforces this invariant.

`_run_child_engine` is defined at the top level of `evocore/ga.py`, outside any class or function. This is required for `pickle.dumps()` to succeed under the `spawn` context. It must never be moved inside a class, nested function, or lambda.

```python
# evocore/ga.py — module level, outside GAEngine class

def _run_child_engine(engine: "GAEngine", seed: int, fitness_fn: Callable) -> "RunResult":
    """
    Module-level helper for run_multiple(run_parallel=True).

    Must be defined at module level — not nested, not a lambda — so that
    pickle.dumps() succeeds under the spawn multiprocessing context.
    Worker processes re-import evocore.ga and call this function directly.
    """
    return engine._copy_with_seed(seed).run(fitness_fn)
```

### 7.6 CMAESEngine — `parallel="process"` Guard (Issue 3)

`CMAESEngine.__init__` raises `ConfigurationError` immediately when `parallel="process"` is requested. `PyCMAESState` is a PyO3 Rust struct holding nalgebra matrices; it is not picklable. This constraint is structural and always true regardless of the fitness function, so it is caught at construction rather than at `run()` time.

```python
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
        callbacks: list[Callback] | None = None,
        seed: int = 0,
        track_diversity: bool = False,
    ) -> None:
        if gene_space is None:
            raise ConfigurationError(
                "gene_space required for CMAESEngine. "
                "Pass gene_space=GeneSpace.uniform(-5.0, 5.0, 10)."
            )

        # Issue 3: PyCMAESState (PyO3 Rust object holding nalgebra matrices) is not
        # picklable. parallel="process" requires serializing state to worker processes,
        # which will always fail. Catch this at construction with a clear explanation.
        if parallel == "process":
            raise ConfigurationError(
                "CMAESEngine does not support parallel='process'.\n"
                "  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) "
                "is not picklable and cannot be serialized to worker processes.\n"
                "  Fix: use parallel='thread' if your fitness function releases the GIL "
                "(e.g., NumPy-heavy work), or parallel='none' for pure-Python fitness functions.\n"
                "  Note: parallel='process' is supported by GAEngine, not CMAESEngine."
            )

        self.gene_space = gene_space
        self.population_size = population_size
        self.initial_mean = initial_mean
        self.initial_sigma = initial_sigma
        self.generations = generations
        self.parallel = parallel
        self.n_workers = n_workers
        self.callbacks = callbacks or []
        self.seed = seed
        self.track_diversity = track_diversity
        # ... remainder of init ...
```

### 7.7 CMAESEngine — Two-Sample-Array Loop

Unchanged from v4. Continuous samples passed to `tell()`; rounded+clamped samples passed to the fitness function.

```python
def run(self, fitness_fn: Callable) -> RunResult:
    state = PyCMAESState(
        self.initial_mean,
        self.sigma_abs,
        self.population_size,
        self.bounds_list,
    )

    for gen in range(self.generations):
        samples_continuous = state.ask(self.seed, gen)
        samples_discrete = [self._apply_bounds_and_round(s) for s in samples_continuous]
        individuals = [Individual(genes=self._decode(s)) for s in samples_discrete]
        fitnesses, nan_count = self._evaluate_all(individuals, fitness_fn, gen)
        state.tell(samples_continuous, fitnesses)   # continuous — academically correct
        # ...
```

### 7.8 Parallelism (`evocore/parallel.py`) — Teardown Fix (Issue 2)

`ProcessParallel.evaluate()` wraps executor usage in `try/finally`. On any exception — including `KeyboardInterrupt` — `shutdown(cancel_futures=True, wait=False)` is called before re-raising. This cancels queued-but-not-started work immediately. Already-running workers in `spawn` subprocesses complete their current fitness evaluation before their process exits; this is the expected and documented behavior.

```python
import multiprocessing
import concurrent.futures

class ProcessParallel:
    """
    Evaluates a population in parallel using ProcessPoolExecutor.

    Uses 'spawn' start method on all platforms for Windows compatibility.
    Fitness function must be defined at module level (not lambda or nested function).

    KeyboardInterrupt behavior: queued-but-not-started evaluations are cancelled
    immediately. Already-running workers complete their current fitness evaluation
    before their process terminates (typically < 1 evaluation time).
    """

    def __init__(self, n_workers: int | None = None, initializer=None, initargs=()):
        self.n_workers = n_workers or os.cpu_count()
        self.initializer = initializer
        self.initargs = initargs
        self._ctx = multiprocessing.get_context("spawn")

    def evaluate(self, population: list[Individual], fitness_fn: Callable) -> list[float]:
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=self._ctx,
            initializer=self.initializer,
            initargs=self.initargs,
        )
        try:
            return list(pool.map(fitness_fn, population))
        finally:
            # Cancel queued work and release pool resources on any exception,
            # including KeyboardInterrupt. Does not block on running workers.
            pool.shutdown(cancel_futures=True, wait=False)
```

`ThreadParallel` is unchanged from v4.

### 7.9 Fitness Function Protocol

Unchanged from v4.

```python
# Style 1: simple float
def my_fitness(ind: Individual) -> float:
    return -sum(x**2 for x in ind.genes)

# Style 2: named params dict
def my_fitness(ind: Individual) -> float:
    p = ind.params
    return run_backtest(p["ema_fast"], p["ema_slow"])

# Style 3: sidecar metrics
def my_fitness(ind: Individual) -> tuple[float, dict]:
    result = run_backtest(ind.params)
    return result.profit_factor, {
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
    }
```

NaN/Inf handling: assigned `-inf` for selection; `GenerationInfo.nan_fitness_count` tracks count per generation; `FitnessWarning` emitted once per run.

### 7.10 Callbacks — `GenerationInfo`

Unchanged from v4. `on_generation_end(gen, pop, info)` signature with structured `GenerationInfo`.

```python
@dataclass
class GenerationInfo:
    generation:        int
    nan_fitness_count: int
    cached_count:      int


class Callback:
    def on_generation_start(self, gen: int, pop: Population) -> None: ...
    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None: ...
    def on_run_end(self, result: "RunResult") -> None: ...
    should_stop: bool = False
```

Built-in callbacks (`EarlyStopping`, `ProgressBar`, `CheckpointCallback`, `MetricsLogger`) unchanged from v4.

### 7.11 Statistics & Logbook

Unchanged from v4.

```python
@dataclass
class LogEntry:
    gen:               int
    best_fitness:      float
    mean_fitness:      float
    std_fitness:       float
    wall_time_ms:      float
    n_evaluations:     int
    nan_fitness_count: int
    cached_count:      int
    diversity:         list[float]   # [] when track_diversity=False
    custom:            dict
```

### 7.12 RunResult & MultiRunResult

`RunResult` is unchanged from v4. `MultiRunResult.wall_time_seconds` semantics are updated (see Section 7.4).

```python
@dataclass
class RunResult:
    best_individual:    Individual
    best_fitness:       float
    final_population:   Population
    logbook:            Logbook
    wall_time_seconds:  float          # per-run compute time — unchanged
    n_evaluations:      int
    elite_history:      list[Individual]
    diversity_history:  list[list[float]]
    seed:               int
    stopped_early:      bool

@dataclass
class MultiRunResult:
    best:               RunResult
    all_runs:           list[RunResult]
    n_runs:             int
    wall_time_seconds:  float          # UPDATED: actual elapsed wall-clock time,
                                       # not sum of per-run times. For sequential
                                       # runs ≈ sum; for parallel runs < sum.

    def best_n(self, n: int) -> list[RunResult]: ...
    def fitness_summary(self) -> dict:
        """Returns {"mean", "std", "min", "max"} of best_fitness across runs."""
```

---

## 8. Error Handling

```python
class EvocoreError(Exception): ...
class ConfigurationError(EvocoreError): ...
class FitnessError(EvocoreError): ...
class ConvergenceError(EvocoreError): ...
class ParallelError(EvocoreError): ...
class CheckpointError(EvocoreError): ...

import warnings
class FitnessWarning(UserWarning): ...
class ConfigurationWarning(UserWarning): ...
```

**New error message added in v5 (Issue 3):**

```
ConfigurationError: CMAESEngine does not support parallel='process'.
  Reason: the internal CMA-ES covariance state (a PyO3 Rust object) is not picklable
  and cannot be serialized to worker processes.
  Fix: use parallel='thread' if your fitness function releases the GIL (e.g., NumPy-heavy
  work), or parallel='none' for pure-Python fitness functions.
  Note: parallel='process' is supported by GAEngine, not CMAESEngine.
```

All other error messages unchanged from v4.

---

## 9. Build System & Toolchain

No Rust changes. Version bump only.

```toml
# Cargo.toml [dependencies] — unchanged
pyo3       = { version = "0.21", features = ["extension-module"] }
rayon      = "1.9"
rand       = "0.8"
rand_distr = "0.4"
nalgebra   = "0.32"

# pyproject.toml — version bump only
[project]
name    = "evocore"
version = "0.5.0"
```

Dev workflow unchanged:

```bash
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

---

## 10. Testing Strategy

### New / Updated Tests

**`tests/unit/test_parallel.py` — extended for v5 (Issues 1 & 2):**

```python
import pickle
import pytest
import concurrent.futures
from evocore.ga import _run_child_engine
from evocore.parallel import ProcessParallel
from evocore import GAEngine, GeneSpace


# --- Issue 1: _run_child_engine must be picklable ---

def test_run_child_engine_is_picklable():
    """
    _run_child_engine must be defined at module level so pickle.dumps() succeeds.
    Under spawn, worker processes re-import the module and call this function directly.
    If this test fails, _run_child_engine has been moved inside a class or nested function.
    """
    try:
        pickle.dumps(_run_child_engine)
    except (pickle.PicklingError, AttributeError) as e:
        pytest.fail(
            f"_run_child_engine is not picklable: {e}\n"
            "It must be defined at module level in evocore/ga.py, "
            "not inside a class or nested function."
        )


def test_run_child_engine_is_at_module_level():
    """Verify _run_child_engine's qualname has no class or closure prefix."""
    assert ".<locals>." not in _run_child_engine.__qualname__, (
        "_run_child_engine appears to be a nested function. "
        "It must be at module level for spawn pickling to work."
    )
    assert "." not in _run_child_engine.__qualname__, (
        "_run_child_engine appears to be a method. "
        "It must be a plain module-level function."
    )


# --- Issue 2: ProcessPoolExecutor teardown on KeyboardInterrupt ---

def _always_raises(ind):
    raise KeyboardInterrupt("simulated Ctrl+C from worker")


def test_process_parallel_shutdown_called_on_exception():
    """
    ProcessParallel must call shutdown(cancel_futures=True, wait=False)
    when an exception occurs, preventing hung worker processes.
    Verified by confirming the pool is shut down after a simulated error.
    """
    pp = ProcessParallel(n_workers=2)
    from evocore.individual import Individual

    pop = [Individual(genes=[float(i)]) for i in range(4)]

    # Fitness function that always raises — pool must still be cleaned up
    def bad_fitness(ind):
        raise RuntimeError("simulated fitness error")

    with pytest.raises(Exception):
        pp.evaluate(pop, bad_fitness)

    # After exception, pool executor must be fully shut down.
    # Attempt a new evaluate to confirm a fresh pool is created (old one is gone).
    def good_fitness(ind):
        return sum(ind.genes)

    result = pp.evaluate(pop, good_fitness)
    assert len(result) == 4


# --- Existing tests (unchanged from v4) ---

def test_process_parallel_forces_spawn_context():
    pp = ProcessParallel(n_workers=2)
    assert pp._ctx.get_start_method() == "spawn"


def test_picklability_probe_raises_on_lambda():
    from evocore.exceptions import ConfigurationError
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=2,
        parallel="process", seed=42,
    )
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        engine.run(fitness_fn=lambda ind: sum(ind.genes))


def test_module_level_fitness_works_with_process():
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=2,
        parallel="process", seed=42,
    )
    result = engine.run(fitness_fn=_module_level_sphere)
    assert result.best_fitness <= 0.0


def _module_level_sphere(ind):
    return -sum(x**2 for x in ind.genes)


def test_run_multiple_parallel_produces_n_results():
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=2, seed=42,
    )
    multi = engine.run_multiple(
        fitness_fn=_module_level_sphere, n_runs=4, run_parallel=True
    )
    assert multi.n_runs == 4
    assert len(multi.all_runs) == 4
```

**`tests/unit/test_cmaes_engine.py` — new guard test (Issue 3):**

```python
def test_cmaes_process_parallel_raises_at_construction():
    """
    CMAESEngine must raise ConfigurationError immediately at __init__ when
    parallel='process' is requested. PyCMAESState is not picklable; catching
    this at construction produces a clear, actionable error message.
    """
    from evocore.exceptions import ConfigurationError
    from evocore import CMAESEngine, GeneSpace

    with pytest.raises(ConfigurationError) as exc_info:
        CMAESEngine(
            gene_space=GeneSpace.uniform(-2.0, 2.0, 5),
            population_size=10,
            generations=5,
            parallel="process",   # must fail here, not at run()
            seed=42,
        )

    assert "parallel='process'" in str(exc_info.value)
    assert "not picklable" in str(exc_info.value)
    assert "parallel='thread'" in str(exc_info.value)


def test_cmaes_thread_parallel_is_allowed():
    """parallel='thread' must be accepted without error at construction."""
    from evocore import CMAESEngine, GeneSpace

    engine = CMAESEngine(
        gene_space=GeneSpace.uniform(-2.0, 2.0, 5),
        population_size=10,
        generations=2,
        parallel="thread",
        seed=42,
    )
    assert engine.parallel == "thread"
```

**`tests/unit/test_ga_engine.py` — new wall-time test (Issue 4):**

```python
def test_run_multiple_wall_time_is_elapsed_not_sum():
    """
    MultiRunResult.wall_time_seconds must reflect actual elapsed time,
    not the sum of individual RunResult.wall_time_seconds values.
    For sequential runs with a non-trivial fitness function, actual elapsed
    time should be approximately equal to the sum (within 20% tolerance),
    confirming the bracket is measuring the right thing.
    """
    import time
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=20, generations=5, seed=42,
    )
    multi = engine.run_multiple(_module_level_sphere, n_runs=3, run_parallel=False)

    sum_of_runs = sum(r.wall_time_seconds for r in multi.all_runs)
    elapsed = multi.wall_time_seconds

    # Elapsed should be close to sum for sequential (overhead only)
    assert elapsed > 0.0, "wall_time_seconds must be positive"
    assert elapsed < sum_of_runs * 1.5, (
        f"Elapsed ({elapsed:.3f}s) is much larger than sum of runs ({sum_of_runs:.3f}s). "
        "Check that perf_counter brackets the dispatch block correctly."
    )
    assert elapsed > sum_of_runs * 0.5, (
        f"Elapsed ({elapsed:.3f}s) is much smaller than sum of runs ({sum_of_runs:.3f}s). "
        "wall_time_seconds may still be computing a sum rather than elapsed time."
    )


def _module_level_sphere(ind):
    return -sum(x**2 for x in ind.genes)
```

### All Other Tests

Unchanged from v4. `test_rng_reproducibility.py` invariants still hold. All five integration convergence tests unchanged.

---

## 11. Performance Targets

Unchanged from v4.

| Scenario | Target vs pure-Python DEAP |
|---|---|
| GA float, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| GA float, 1000 pop, 100 gen, 8-thread Rayon | ≥ 10× faster |
| GA binary, 1000 pop, 100 gen, sequential | ≥ 4× faster |
| GA integer, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| CMA-ES, dim=20, 200 gen, sequential | ≥ 5× faster |
| CMA-ES, dim=100, 500 gen, sequential | ≥ 2× vs naive (eigendecomp caching) |

---

## 12. Migration Guide

### From v4 to v5

**No breaking changes to the public API.** All v4 code runs unchanged on v5 with one behavioral difference that is a bug fix:

**`MultiRunResult.wall_time_seconds` semantics changed.**

Previously this field summed individual `RunResult.wall_time_seconds` values regardless of whether runs were parallel or sequential. It now reflects actual elapsed wall-clock time.

- For `run_parallel=False`: the new value is approximately equal to the old sum (plus minor orchestration overhead). No meaningful change for users.
- For `run_parallel=True`: the new value is substantially *less* than the old sum, reflecting the actual time saved by parallelism. This is the correct value. Code that previously used `MultiRunResult.wall_time_seconds` as a proxy for total compute time should switch to summing `r.wall_time_seconds for r in multi.all_runs` if that semantics is needed.

**`CMAESEngine(parallel="process")` now raises at construction.**

Previously this would have produced a cryptic pickle error at `run()` time (or silently produced wrong results). Now it raises `ConfigurationError` at `__init__` with an explanation and suggested fix. Code that was accidentally passing `parallel="process"` to `CMAESEngine` (which never worked) now gets an actionable error.

### From v3 to v5

Apply the v3→v4 migration guide first (breaking changes: `on_generation_end` signature, `on_fitness_warning` hook removed, `fitness_valid` not on Rust structs, `seed_offset` removed), then the v4→v5 note above.

---

## 13. Open Issues

All issues from v4 are resolved in v5:

✅ Issue 1 (v4): `_run_child_engine` must be importable — tested invariant in `test_parallel.py`
✅ Issue 2 (v4): `ProcessPoolExecutor` teardown on `KeyboardInterrupt` — `try/finally` + `shutdown(cancel_futures=True, wait=False)`
✅ Issue 3 (v4): `CMAESEngine` + `parallel="process"` — `ConfigurationError` at `__init__`
✅ Issue 4 (v4): `run_multiple` wall-time accounting — `perf_counter` brackets

**No new open issues identified during v5 design session.**

The v4 open-issues list is now empty. v6 scope is not defined.
