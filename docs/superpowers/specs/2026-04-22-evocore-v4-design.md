# evocore v4 Design Spec

**Date:** 2026-04-22
**Status:** Draft
**Scope:** Phase 1 — Genetic Algorithms (float + integer + binary + mixed individuals) + CMA-ES, Rust-native Python library
**Supersedes:** 2026-04-21-evocore-v3-design.md

---

## Changelog from v3

| Area | Change |
|---|---|
| **Issue 1: Windows `spawn`/`fork`** | `ProcessParallel` forces `spawn` context on all platforms. `GAEngine` validates fitness function picklability at construction when `parallel="process"`. |
| **Issue 2: Mixed-gene PyO3 boundary** | All genes encoded as `f64` at the PyO3 boundary (`int` as `f64`, `bool` as `0.0`/`1.0`). `OperatorSet` handles encoding/decoding in Python. |
| **Issue 3: CMA-ES integer rounding** | `CMAESEngine.run()` maintains two sample arrays per generation — continuous for `tell()`, rounded+clamped for the fitness function. |
| **Issue 4: `fitness_valid` placement** | `fitness_valid` removed from all three Rust individual structs. Lives exclusively on Python `Individual`. Engine tracks elite indices as `set[int]` in Python. |
| **Issue 5: `run_multiple` parallelism** | New `run_parallel: bool = False` opt-in parameter. When `True`, dispatches child engines to a `ProcessPoolExecutor`. |
| **Issue 6: Integer sigma scaling** | `GeneDef` gains optional `sigma: float | None = None` for per-gene sigma override. Construction warning emitted for large-range int genes without explicit sigma. |
| **Issue 7: Windows Rayon stack** | `lib.rs` module init calls `rayon::ThreadPoolBuilder::new().stack_size(8 * 1024 * 1024).build_global().ok()` universally. |
| **Issue 8: `diversity_history` memory** | `track_diversity: bool = False` added to `GAEngine` and `CMAESEngine`. `RunResult.diversity_history` is `[]` when `False`. |
| **Issue 9: `on_fitness_warning` hook** | Hook removed. `GenerationInfo` dataclass introduced. `on_generation_end` signature updated to `(gen, pop, info)` — **breaking change**. |
| **Issue 10: UTF-8 encoding** | `MetricsLogger` uses `encoding="utf-8"` on all file opens. (Covered jointly with Issue 1.) |

Everything not listed above is **unchanged from v3**.

---

## 1. Goal

Rebuild the hot-path components of DEAP as a Rust-native Python library with a clean, domain-agnostic modern API. Designed for any optimization workload where the fitness function is expensive (backtesting, simulation, hyperparameter tuning). Target: ≥3× sequential and ≥10× parallel speedup over pure-Python DEAP for GA workloads.

**Key improvements over v3:**
- Cross-platform correctness: `parallel="process"` behaves identically on Linux, macOS, and Windows
- Fitness function picklability validated at construction, not at runtime
- CMA-ES integer gene handling is academically correct (continuous samples passed to `tell`)
- `fitness_valid` is an engine concern, not a Rust data concern — Rust structs are pure data again
- `run_multiple` is parallelizable via opt-in `run_parallel=True`
- Per-gene sigma overrides for fine-grained mutation control
- `on_generation_end` receives structured `GenerationInfo` instead of raw warning counts mid-generation
- `diversity_history` is opt-in — no memory cost by default

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

**Layer responsibilities:**
- `evocore` — user-facing Python. Typed API, `GeneSpace`, callbacks, statistics, parallelism wrappers. Zero random state. No recompile for API changes.
- `evocore._core` — compiled PyO3 extension. All hot paths: operator loops, selection, CMA-ES matrix math, Rayon pool, seed derivation. All genes cross the PyO3 boundary as `f64`.

---

## 3. Project File Structure

```
evocore/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/                              # Rust source
│   ├── lib.rs                        # PyO3 module root + Rayon pool init
│   ├── utils.rs                      # derive_seed(), OP_* constants
│   ├── individual.rs                 # FloatIndividual, IntegerIndividual, BinaryIndividual
│   │                                 # (fitness_valid field REMOVED in v4)
│   ├── gene_spec.rs                  # GeneKind enum, per-gene spec
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── float_ops.rs              # BLX-α, SBX, Gaussian, Uniform mutation
│   │   ├── int_ops.rs                # Integer-aware operators (rounding in Rust)
│   │   └── binary_ops.rs             # one-point, two-point, uniform XO, bit-flip
│   ├── reproduce.rs                  # Single-call reproduce() per generation
│   ├── selection.rs                  # tournament, roulette, rank (NaN-safe)
│   ├── cmaes.rs                      # CMAESState, ask/tell, eigen cache, mirror folding
│   └── parallel.rs                   # Rayon pool management
├── evocore/                          # Python source
│   ├── __init__.py
│   ├── gene_space.py                 # GeneDef (+ sigma field), GeneSpace
│   ├── individual.py                 # Individual (fitness_valid here), Population
│   ├── operators.py                  # Bounds-aware operator layer
│   ├── parallel.py                   # ThreadParallel, ProcessParallel (spawn-forced)
│   ├── ga.py                         # GAEngine, RunResult, MultiRunResult
│   ├── cmaes.py                      # CMAESEngine (two-sample-array loop)
│   ├── stats.py                      # LogEntry, Logbook
│   ├── callbacks.py                  # Callback base + built-ins (GenerationInfo signature)
│   └── exceptions.py                 # EvocoreError hierarchy
├── tests/
│   ├── unit/
│   │   ├── test_rng_reproducibility.py
│   │   ├── test_gene_space.py
│   │   ├── test_operators.py
│   │   ├── test_selection.py
│   │   ├── test_ga_engine.py
│   │   ├── test_cmaes_engine.py
│   │   ├── test_parallel.py         # spawn consistency, picklability probe
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

Unchanged from v3. All randomness derives from a single `u64` master seed via `derive_seed(master, generation, individual_idx, op)` in `src/utils.rs`. No mutable RNG state anywhere. `run()` is idempotent. Thread count does not affect results.

```rust
// src/utils.rs — unchanged from v3
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

`GeneDef` gains an optional `sigma` field for per-gene mutation sigma override.

```python
@dataclass
class GeneDef:
    name:  str
    kind:  Literal["float", "int", "bool"]
    low:   float | int | None = None    # None only for bool
    high:  float | int | None = None    # None only for bool
    sigma: float | None = None          # NEW: per-gene sigma override (fraction of range)
                                        # Overrides GAEngine.mutation_sigma for this gene.
                                        # None = use engine-level mutation_sigma.

    def __post_init__(self):
        if self.kind != "bool":
            assert self.low is not None and self.high is not None
            assert self.low < self.high
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
        if self.sigma is not None:
            assert 0.0 < self.sigma <= 1.0, "sigma must be in (0, 1]"
```

`GeneSpace` unchanged otherwise:

```python
# Mode A: explicit per-gene (recommended)
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

**Construction warning for large integer ranges:** If any `GeneDef` has `kind="int"`, `(high - low) > 100`, and `sigma is None`, `GAEngine.__init__` emits:

```
ConfigurationWarning: GeneDef("ema_slow", "int", 10, 500) has range 490 and no per-gene sigma.
  With mutation_sigma=0.2, σ_abs=98 — large steps may prevent fine-tuning in later generations.
  Consider: GeneDef("ema_slow", "int", 10, 500, sigma=0.03)
```

---

## 6. Rust Core Components (`evocore._core`)

### 6.1 Individual Types — `fitness_valid` Removed

`fitness_valid` is removed from all three Rust structs. The Rust layer is pure typed data. Engine-level caching is tracked in Python (Section 7.2).

```rust
// src/individual.rs — fitness_valid field removed
#[pyclass]
#[derive(Clone, Debug)]
pub struct FloatIndividual {
    #[pyo3(get, set)] pub genes:   Vec<f64>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
    // fitness_valid REMOVED — engine concern, not data concern
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct IntegerIndividual {
    #[pyo3(get, set)] pub genes:   Vec<i64>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BinaryIndividual {
    #[pyo3(get, set)] pub genes:   Vec<bool>,
    #[pyo3(get, set)] pub fitness: Option<f64>,
}
```

### 6.2 PyO3 Boundary Encoding (Issue 2 Resolution)

All genes cross the PyO3 boundary as `Vec<Vec<f64>>`:

| Gene kind | Encoding | Decoding |
|---|---|---|
| `float` | identity | identity |
| `int` | `x as f64` | `x.round() as i64` |
| `bool` | `false → 0.0`, `true → 1.0` | `x >= 0.5 → true` |

Encoding and decoding live in `evocore/operators.py` (`OperatorSet._encode_population` / `_decode_population`). Rust operators never see types other than `f64`. This is the canonical representation for all `reproduce()`, `evaluate_sequential()`, and `evaluate_parallel_rayon()` calls.

### 6.3 Genetic Operators

Unchanged from v3. All operators receive `(master_seed, generation, individual_idx)` and call `derive_seed` internally.

**Float operators (`float_ops.rs`):** `blend_crossover`, `simulated_binary_crossover`, `gaussian_mutation`, `uniform_mutation`

**Integer operators (`int_ops.rs`):** `int_simulated_binary_crossover`, `int_gaussian_mutation`, `int_uniform_mutation` — all operate on `f64` (encoded integers), round results before returning.

**Binary operators (`binary_ops.rs`):** `one_point_crossover`, `two_point_crossover`, `uniform_crossover`, `bit_flip_mutation` — operate on `f64` (0.0/1.0 encoded booleans), threshold at 0.5 on decode.

### 6.4 Selection Algorithms

Unchanged from v3. NaN/Inf handling: NaN and -Inf → rank 0 (worst); +Inf clamped to `f64::MAX`. Warnings flow through `GenerationInfo.nan_fitness_count` (Section 7.8).

```rust
tournament_selection(fitnesses, k, tournament_size, master_seed, generation) → Vec<usize>
roulette_selection(fitnesses, k, master_seed, generation)                    → Vec<usize>
rank_selection(fitnesses, k, master_seed, generation)                        → Vec<usize>
```

### 6.5 `src/reproduce.rs`

Unchanged from v3. Single Rust call per generation: selection → crossover prob check → crossover → mutation → clamp → round. Elitism is handled in Python before calling `reproduce()` (Section 7.2).

### 6.6 Parallelism (`parallel.rs`)

```rust
evaluate_sequential(genes_list, fitness_fn)          → PyResult<Vec<f64>>
evaluate_parallel_rayon(genes_list, fitness_fn, n)   → PyResult<Vec<f64>>
```

Both accept `Vec<Vec<f64>>` (universally encoded). Unchanged from v3.

### 6.7 CMA-ES Engine (`cmaes.rs`)

Unchanged from v3. `ask()` takes `(master_seed, generation)`, returns continuous samples. `tell()` takes continuous samples and fitnesses. No RNG stored on struct. Eigendecomp caching via `RefCell`. Mirror-folding boundary correction.

```rust
// PyO3 wrapper — unchanged from v3
fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>>
fn tell(&mut self, samples: Vec<Vec<f64>>, fitnesses: Vec<f64>)
```

### 6.8 `lib.rs` — Rayon Stack Size Init (Issue 7)

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Rayon thread pool with 8MB stack on all platforms.
    // Matches Linux/macOS default. Prevents stack overflow in nalgebra
    // symmetric_eigen() for large CMA-ES covariance matrices on Windows.
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();  // silently ignores if pool already initialized

    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;
    m.add_class::<PyCMAESState>()?;
    // ... all other registrations ...
    Ok(())
}
```

---

## 7. Python API Layer

### 7.1 Individual & Population

`fitness_valid` moves from Rust to Python `Individual`:

```python
@dataclass
class Individual:
    genes:         list[float | int | bool]
    fitness:       float | None = None
    fitness_valid: bool = False            # owned here in v4, not in Rust
    metadata:      dict = field(default_factory=dict)

    @property
    def params(self) -> dict | None:
        """Named gene dict if GeneSpace.has_names, else None."""
        return self.metadata.get("params")
```

`Population` unchanged from v3.

### 7.2 GAEngine — Elite Index Tracking

Python tracks which individuals are elites this generation as a `set[int]`, not via a field on `Individual`. This replaces the `fitness_valid` propagation from v3.

```python
class GAEngine:
    def run(self, fitness_fn: Callable) -> RunResult:
        ...
        elite_indices: set[int] = set()   # indices of elites in current population

        for gen in range(self.generations):
            # Determine which individuals to skip re-evaluation
            skip_indices = elite_indices if self.elitism > 0 else set()

            # Evaluate only non-elite individuals
            fitnesses = self._evaluate_with_cache(
                population, fitnesses, fitness_fn, skip_indices
            )

            # Run reproduction (Rust call)
            new_population = _core.reproduce(
                population, fitnesses,
                self._reproduction_config(current_sigma),
                self.seed, gen,
            )

            # Identify new elite indices (top-k by fitness, carried unchanged)
            elite_indices = self._get_elite_indices(fitnesses, self.elitism)
            # Insert elite genes at front of new_population
            for rank, idx in enumerate(elite_indices):
                new_population[rank] = population[idx]

            population = new_population
            ...

    def _evaluate_with_cache(
        self,
        population: list[list[float]],
        prev_fitnesses: list[float],
        fitness_fn: Callable,
        skip_indices: set[int],
    ) -> list[float]:
        """Evaluate non-skipped individuals; carry forward fitnesses for skipped ones."""
        fitnesses = list(prev_fitnesses)
        to_evaluate = [i for i in range(len(population)) if i not in skip_indices]
        for i in to_evaluate:
            ind = self._decode_individual(population[i])
            val = fitness_fn(ind)
            fitnesses[i] = self._validate_fitness(val, i, gen)
        return fitnesses
```

No `fitness_valid` flag propagates through any data structure. The engine is the sole authority on which individuals need evaluation.

### 7.3 GAEngine Constructor — Picklability Probe (Issue 1)

```python
class GAEngine:
    def __init__(self, ..., parallel: str = "none", ...):
        ...
        if parallel == "process":
            # Validate at construction, not at first run()
            import pickle
            try:
                pickle.dumps(None)  # warm-up; actual probe deferred to run()
            except Exception:
                pass
            # Store for probe at run() time (fitness_fn not available yet at __init__)
            self._validate_picklable = True
        else:
            self._validate_picklable = False

    def run(self, fitness_fn: Callable) -> RunResult:
        if self._validate_picklable:
            import pickle
            try:
                pickle.dumps(fitness_fn)
            except (pickle.PicklingError, AttributeError, TypeError) as e:
                raise ConfigurationError(
                    f"fitness_fn cannot be pickled, which is required for parallel='process'.\n"
                    f"  Error: {e}\n"
                    f"  Fix: define fitness_fn at module level (not as a lambda or nested function).\n"
                    f"  Alternatively, use parallel='thread' if your fitness function releases the GIL."
                ) from e
        ...
```

> **Note:** The probe runs at `run()` time, not `__init__()`, because `fitness_fn` is passed to `run()`, not the constructor. The error surface is still before any pool is allocated.

### 7.4 `run_multiple` — Parallel Opt-in (Issue 5)

```python
def run_multiple(
    self,
    fitness_fn: Callable,
    n_runs: int = 10,
    aggregate: str = "best",
    run_parallel: bool = False,     # NEW in v4
) -> MultiRunResult:
    child_seeds = [
        int(_core.derive_seed(self.seed, 0, run_idx, _core.OP_MULTI_RUN))
        for run_idx in range(n_runs)
    ]

    if run_parallel:
        # Each child engine runs in its own process.
        # fitness_fn must be picklable — same constraint as parallel="process".
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
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(n_runs, (self.n_workers or os.cpu_count())),
            mp_context=ctx,
        ) as pool:
            futures = [
                pool.submit(_run_child_engine, self, seed, fitness_fn)
                for seed in child_seeds
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
    else:
        results = [
            self._copy_with_seed(seed).run(fitness_fn)
            for seed in child_seeds
        ]

    results.sort(key=lambda r: r.best_fitness, reverse=True)
    return MultiRunResult(
        best=results[0],
        all_runs=results,
        n_runs=n_runs,
        wall_time_seconds=sum(r.wall_time_seconds for r in results),
    )


def _run_child_engine(engine: "GAEngine", seed: int, fitness_fn: Callable) -> "RunResult":
    """Module-level function so it is picklable for spawn context."""
    return engine._copy_with_seed(seed).run(fitness_fn)
```

### 7.5 CMAESEngine — Two-Sample-Array Loop (Issue 3)

```python
class CMAESEngine:
    def run(self, fitness_fn: Callable) -> RunResult:
        state = PyCMAESState(
            self.initial_mean,
            self.sigma_abs,
            self.population_size,
            self.bounds_list,
        )

        for gen in range(self.generations):
            # Continuous samples — used for tell() (mathematically correct)
            samples_continuous = state.ask(self.seed, gen)

            # Rounded + clamped samples — used for fitness evaluation
            samples_discrete = [
                self._apply_bounds_and_round(s) for s in samples_continuous
            ]

            individuals = [Individual(genes=self._decode(s)) for s in samples_discrete]
            fitnesses, nan_count = self._evaluate_all(individuals, fitness_fn, gen)

            # Covariance update uses CONTINUOUS samples (not rounded)
            state.tell(samples_continuous, fitnesses)

            ...

    def _apply_bounds_and_round(self, genes_f64: list[float]) -> list[float]:
        """Clamp to bounds; round integer genes (still f64 for Rust boundary)."""
        result = []
        for i, x in enumerate(genes_f64):
            lo, hi = self._bounds_list[i]
            x = max(lo, min(hi, x))
            if self._gene_kinds[i] == "int":
                x = float(round(x))
            result.append(x)
        return result
```

### 7.6 Operator Layer (`evocore/operators.py`)

`OperatorSet` now reads per-gene sigma from `GeneSpace` when computing mutation sigma for each gene:

```python
class OperatorSet:
    def _sigma_for_gene(self, gene_idx: int, global_sigma_abs: float) -> float:
        """Per-gene sigma override takes precedence over engine-level sigma."""
        gene_def = self._gene_space.genes[gene_idx]
        if gene_def.sigma is not None:
            lo, hi = gene_def.low, gene_def.high
            return gene_def.sigma * (hi - lo)   # convert fraction to absolute
        return global_sigma_abs
```

### 7.7 Parallelism (`evocore/parallel.py`) — Spawn-Forced (Issue 1)

```python
import multiprocessing
import concurrent.futures

class ProcessParallel:
    """
    Uses ProcessPoolExecutor with an explicit 'spawn' start method on all platforms.

    Why spawn everywhere:
    - On Linux, the default is 'fork'. Fork is fast but hides Windows-only bugs:
      lambda functions and local closures fail silently on Windows with fork's default.
    - By forcing 'spawn' universally, Linux CI catches Windows compatibility issues.
    - Cost: ~300ms pool startup (amortized across generations — negligible for real workloads).

    Fitness function must be defined at module level (not lambda, not nested function).
    Validated at engine.run() time with a pickle probe.
    """

    def __init__(self, n_workers: int | None = None, initializer=None, initargs=()):
        self.n_workers = n_workers or os.cpu_count()
        self.initializer = initializer
        self.initargs = initargs
        self._ctx = multiprocessing.get_context("spawn")

    def evaluate(self, population: list[Individual], fitness_fn: Callable) -> list[float]:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=self._ctx,
            initializer=self.initializer,
            initargs=self.initargs,
        ) as pool:
            return list(pool.map(fitness_fn, population))
```

`ThreadParallel` unchanged from v3 (uses Rayon via PyO3).

### 7.8 Callbacks — `GenerationInfo` (Issue 9)

`on_fitness_warning` hook removed. `on_generation_end` now receives a `GenerationInfo` dataclass:

```python
@dataclass
class GenerationInfo:
    generation:       int
    nan_fitness_count: int    # individuals with NaN/Inf fitness this generation
    cached_count:     int     # elites skipped (fitness carried from prior generation)


class Callback:
    """Base callback — all methods are no-ops."""

    def on_generation_start(self, gen: int, pop: Population) -> None:
        pass

    # BREAKING CHANGE from v3: signature now includes `info: GenerationInfo`
    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None:
        pass

    def on_run_end(self, result: "RunResult") -> None:
        pass

    should_stop: bool = False
```

**Built-in callbacks updated:**

```python
class EarlyStopping(Callback):
    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None:
        current_best = pop.best(1)
        if not current_best:
            return
        fitness = current_best[0].fitness
        if fitness - self._best > self.min_delta:
            self._best = fitness
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
            if self._no_improve_count >= self.patience:
                self.should_stop = True


class ProgressBar(Callback):
    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None:
        if self._bar is not None:
            best = pop.best(1)
            fitness = best[0].fitness if best else float("nan")
            postfix = {"best": f"{fitness:.4f}"}
            if info.nan_fitness_count:
                postfix["nan"] = info.nan_fitness_count
            self._bar.set_postfix(**postfix)
            self._bar.update(1)


class CheckpointCallback(Callback):
    """Saves population pickle every `every` generations. Pickle is binary — no encoding needed."""
    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None:
        if gen % self.every == 0:
            filename = os.path.join(self.path, f"checkpoint_gen_{gen}.pkl")
            with open(filename, "wb") as f:  # binary — no encoding param
                pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, f)


class MetricsLogger(Callback):
    """Logs per-generation metrics to a JSON Lines file."""
    def __init__(self, path: str = "./metrics.jsonl") -> None:
        self.path = path

    def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None:
        import json
        best = pop.best(1)
        record = {
            "generation": gen,
            "best_fitness": best[0].fitness if best else None,
            "nan_fitness_count": info.nan_fitness_count,
            "cached_count": info.cached_count,
        }
        # encoding="utf-8" required: gene names or metric keys may contain non-ASCII
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
```

### 7.9 Statistics & Logbook

`LogEntry` gains `nan_fitness_count` and `cached_count` sourced from `GenerationInfo`. Unchanged otherwise.

```python
@dataclass
class LogEntry:
    gen:               int
    best_fitness:      float
    mean_fitness:      float
    std_fitness:       float
    wall_time_ms:      float
    n_evaluations:     int           # fitness calls this generation (excludes cached)
    nan_fitness_count: int           # NEW: from GenerationInfo
    cached_count:      int           # NEW: from GenerationInfo
    diversity:         list[float]   # per-gene std dev; [] if track_diversity=False
    custom:            dict          # sidecar metrics from fitness_fn tuple return
```

### 7.10 `diversity_history` Opt-in (Issue 8)

```python
engine = GAEngine(
    ...,
    track_diversity=False,   # Default False. Set True to populate RunResult.diversity_history.
)
```

`RunResult.diversity_history` is `[]` when `track_diversity=False`. No per-generation `diversity` computation is performed. `LogEntry.diversity` is also `[]` in this case.

### 7.11 RunResult & MultiRunResult

```python
@dataclass
class RunResult:
    best_individual:    Individual
    best_fitness:       float
    final_population:   Population
    logbook:            Logbook
    wall_time_seconds:  float
    n_evaluations:      int
    elite_history:      list[Individual]
    diversity_history:  list[list[float]]   # [] when track_diversity=False
    seed:               int
    stopped_early:      bool

@dataclass
class MultiRunResult:
    best:               RunResult
    all_runs:           list[RunResult]
    n_runs:             int
    wall_time_seconds:  float

    def best_n(self, n: int) -> list[RunResult]: ...
    def fitness_summary(self) -> dict:
        """Returns {"mean", "std", "min", "max"} of best_fitness across runs."""
```

### 7.12 Fitness Function Protocol

Unchanged from v3 — three styles supported:

```python
# Style 1: simple float
def my_fitness(ind: Individual) -> float:
    return -sum(x**2 for x in ind.genes)

# Style 2: named params dict
def my_fitness(ind: Individual) -> float:
    p = ind.params
    return run_backtest(p["ema_fast"], p["ema_slow"])

# Style 3: sidecar metrics (float, dict)
def my_fitness(ind: Individual) -> tuple[float, dict]:
    result = run_backtest(ind.params)
    return result.profit_factor, {
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
    }
```

**NaN/Inf handling:** Individuals returning `float('nan')`, `float('inf')`, or `float('-inf')` are assigned `-inf` for selection. `GenerationInfo.nan_fitness_count` tracks the count per generation. `FitnessWarning` is emitted once per run (not once per generation) to avoid log spam.

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
class ConfigurationWarning(UserWarning): ...    # NEW: emitted for large-range int genes
```

**New example messages (v4 additions):**

```
ConfigurationError: fitness_fn cannot be pickled, which is required for parallel='process'.
  Error: Can't pickle <function <lambda> at 0x...>: attribute lookup failed
  Fix: define fitness_fn at module level (not as a lambda or nested function).
  Alternatively, use parallel='thread' if your fitness function releases the GIL.

ConfigurationWarning: GeneDef("ema_slow", "int", 10, 500) has range 490 and no per-gene sigma.
  With mutation_sigma=0.2, σ_abs=98 — large steps may prevent fine-tuning in later generations.
  Consider: GeneDef("ema_slow", "int", 10, 500, sigma=0.03)
```

---

## 9. Build System & Toolchain

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
version = "0.4.0"
```

Dev workflow unchanged:

```bash
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

---

## 10. Testing Strategy

### New / Updated Unit Tests

**`test_parallel.py` — extended for v4:**

```python
def test_process_parallel_forces_spawn_context():
    """ProcessParallel must use spawn context on all platforms."""
    from evocore.parallel import ProcessParallel
    import multiprocessing
    pp = ProcessParallel(n_workers=2)
    assert pp._ctx.get_start_method() == "spawn"


def test_picklability_probe_raises_on_lambda():
    """GAEngine.run() must raise ConfigurationError for lambda with parallel='process'."""
    from evocore import GAEngine, GeneSpace, GeneDef
    from evocore.exceptions import ConfigurationError

    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=2,
        parallel="process", seed=42,
    )
    with pytest.raises(ConfigurationError, match="cannot be pickled"):
        engine.run(fitness_fn=lambda ind: sum(ind.genes))


def test_module_level_fitness_works_with_process():
    """Module-level fitness function must work with parallel='process'."""
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

**`test_cmaes_engine.py` — integer rounding correctness:**

```python
def test_cmaes_integer_genes_tell_receives_continuous():
    """
    tell() must receive continuous (unrounded) samples.
    Verify by checking that samples passed to tell differ from rounded values
    when integer genes exist.
    """
    from evocore._core import PyCMAESState

    bounds = [(-5.0, 5.0)] * 3
    state = PyCMAESState([0.0] * 3, 0.5, 10, bounds)
    samples_continuous = state.ask(42, 0)

    # At least some values should be non-integer
    all_integer = all(x == round(x) for s in samples_continuous for x in s)
    assert not all_integer, "Continuous samples should not all be integers"
```

**`test_gene_space.py` — sigma override:**

```python
def test_gene_def_sigma_override():
    gd = GeneDef("ema_slow", "int", 10, 500, sigma=0.03)
    assert gd.sigma == 0.03

def test_gene_def_sigma_none_by_default():
    gd = GeneDef("x", "float", -1.0, 1.0)
    assert gd.sigma is None

def test_gene_def_sigma_must_be_in_range():
    with pytest.raises(AssertionError):
        GeneDef("x", "float", -1.0, 1.0, sigma=1.5)


def test_large_int_range_without_sigma_emits_warning():
    import warnings
    from evocore.exceptions import ConfigurationWarning
    from evocore import GAEngine, GeneSpace, GeneDef

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GAEngine(
            gene_space=GeneSpace([GeneDef("ema_slow", "int", 10, 500)]),
            population_size=10, generations=2, seed=42,
        )
    config_warnings = [x for x in w if issubclass(x.category, ConfigurationWarning)]
    assert len(config_warnings) == 1
    assert "ema_slow" in str(config_warnings[0].message)
```

**`test_ga_engine.py` — `GenerationInfo` callback signature:**

```python
def test_on_generation_end_receives_generation_info():
    received_infos = []

    class InfoCapture(Callback):
        def on_generation_end(self, gen, pop, info):
            received_infos.append(info)

    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=3, seed=42,
        callbacks=[InfoCapture()],
    )
    engine.run(lambda ind: sum(ind.genes))
    assert len(received_infos) == 3
    for info in received_infos:
        assert isinstance(info.generation, int)
        assert isinstance(info.nan_fitness_count, int)
        assert isinstance(info.cached_count, int)


def test_track_diversity_false_produces_empty_history():
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=3, seed=42,
        track_diversity=False,
    )
    result = engine.run(lambda ind: sum(ind.genes))
    assert result.diversity_history == []


def test_track_diversity_true_produces_history():
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-1.0, 1.0, 3),
        population_size=10, generations=3, seed=42,
        track_diversity=True,
    )
    result = engine.run(lambda ind: sum(ind.genes))
    assert len(result.diversity_history) == 3
    assert len(result.diversity_history[0]) == 3  # one value per gene
```

**`test_rng_reproducibility.py`** — unchanged from v3. All four invariants still hold.

### Integration Tests

Unchanged from v3. All five convergence tests pass with the new architecture.

---

## 11. Performance Targets

Unchanged from v3. The `run_multiple(run_parallel=True)` addition is expected to reduce multi-run wall time proportionally to `n_workers` for sufficiently expensive fitness functions.

| Scenario | Target vs pure-Python DEAP |
|---|---|
| GA float, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| GA float, 1000 pop, 100 gen, 8-thread Rayon | ≥ 10× faster |
| GA binary, 1000 pop, 100 gen, sequential | ≥ 4× faster |
| GA integer, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| CMA-ES, dim=20, 200 gen, sequential | ≥ 5× faster |
| CMA-ES, dim=100, 500 gen, sequential | ≥ 2× vs naive (eigendecomp caching) |

---

## 12. Migration Guide from v3

### Breaking Changes

**1. `Callback.on_generation_end` signature** — the most impactful change.

```python
# v3
def on_generation_end(self, gen: int, pop: Population) -> None: ...

# v4
def on_generation_end(self, gen: int, pop: Population, info: GenerationInfo) -> None: ...
```

Any custom `Callback` subclass must add the `info` parameter. The built-in callbacks (`EarlyStopping`, `ProgressBar`, `CheckpointCallback`, `MetricsLogger`) are updated in v4. Third-party callbacks will raise `TypeError` until updated.

**Migration:** Add `info: GenerationInfo` as the third parameter. If you don't need it, accept `**kwargs`:

```python
# Minimal migration — accepts but ignores info
def on_generation_end(self, gen, pop, info):
    ...your existing code...
```

**2. `on_fitness_warning` callback hook removed.**

```python
# v3 — no longer called
def on_fitness_warning(self, count: int, generation: int) -> None: ...

# v4 — use GenerationInfo instead
def on_generation_end(self, gen, pop, info):
    if info.nan_fitness_count > 0:
        print(f"Warning: {info.nan_fitness_count} NaN fitness values at gen {gen}")
```

**3. `fitness_valid` removed from Rust individual structs.**

If you accessed `_core.FloatIndividual.fitness_valid` directly (undocumented internal usage), that attribute no longer exists on the Rust type. Use `Individual.fitness_valid` on the Python dataclass.

**4. `run_multiple` no longer accepts `seed_offset`** (removed in v3, confirmed gone in v4).

---

### Non-Breaking Changes (Additive)

| Feature | Action needed |
|---|---|
| `GeneDef(sigma=...)` | Optional; existing code unaffected |
| `GAEngine(track_diversity=False)` | Default is `False`; no change needed |
| `GAEngine(parallel="process")` | Now validated at `run()` time; lambdas now raise `ConfigurationError` instead of silent failure |
| `run_multiple(run_parallel=False)` | Default is `False`; existing calls identical |
| `LogEntry.nan_fitness_count` | New field; existing code that reads `LogEntry` by position may need updating |
| `LogEntry.cached_count` | Same as above |

---

## 13. Open Issues

The following issues from the v3 open list are **resolved** in v4:

✅ Issue 1: Windows `spawn`/`fork`
✅ Issue 2: Mixed-gene Rust evaluation path
✅ Issue 3: CMA-ES `tell()` with rounded integer genes
✅ Issue 4: `fitness_valid` layer placement
✅ Issue 5: `run_multiple` parallelism
✅ Issue 6: `mutation_sigma` for large integer ranges
✅ Issue 7: Windows Rayon stack size
✅ Issue 8: `diversity_history` memory cost
✅ Issue 9: `on_fitness_warning` hook inconsistency
✅ Issue 10: `MetricsLogger` UTF-8 encoding

**New issues discovered during v4 design session:**

1. **`_run_child_engine` must be importable** — `run_multiple(run_parallel=True)` uses `pool.submit(_run_child_engine, ...)`. Under spawn, worker processes re-import the module. `_run_child_engine` must be defined at module level in `evocore/ga.py`, not as a nested function or closure. This is already the design; worth a test.

2. **`ProcessPoolExecutor` teardown on `KeyboardInterrupt`** — if the user Ctrl+C during `run_multiple(run_parallel=True)`, worker processes may hang. The `with` block handles normal teardown but not signals. Deferred to v5; documented as a known limitation.

3. **`CMAESEngine` + `parallel="process"`** — the `PyCMAESState` Rust object is not picklable (it holds Rust-side nalgebra matrices). This means `CMAESEngine` cannot use `parallel="process"` for inner evaluation. Should be validated at construction with a clear `ConfigurationError`. Deferred to v5.

4. **`run_multiple` wall-time accounting** — `MultiRunResult.wall_time_seconds` sums sequential run times even when `run_parallel=True`. Should use actual wall-clock time of the parallel dispatch. Deferred to v5.
