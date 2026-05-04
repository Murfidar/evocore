# evocore v2 Design Spec

**Date:** 2026-04-05
**Status:** Draft — Revised
**Scope:** Phase 1 — Genetic Algorithms (float + integer + binary + mixed individuals) + CMA-ES, Rust-native Python library
**Supersedes:** 2026-04-04-evocore-design.md

---

## 1. Goal

Rebuild the hot-path components of DEAP as a Rust-native Python library with a clean, domain-agnostic modern API. Designed for any optimization workload where the fitness function is expensive (backtesting, simulation, hyperparameter tuning). Target: ≥3× sequential and ≥10× parallel speedup over pure-Python DEAP for GA workloads.

**Key improvements over v1:**
- Per-gene type and bounds (`GeneSpace`)
- Integer gene type (native, not float-with-rounding)
- Named gene parameters (`ind.params` dict)
- Elitism
- Fitness caching for unchanged individuals
- NaN/Inf fitness handling
- Operator bounds clamping (engine-owned)
- Global seeded RNG (Rust-owned, deterministic)
- Adaptive mutation schedule
- Resume from checkpoint
- Multi-run API
- Custom sidecar metrics from fitness function
- Parallel mode: `"thread"` (Rayon, requires GIL release) or `"process"` (multiprocessing, works with pure-Python fitness)
- CMA-ES eigendecomposition caching + mirror-folding boundary correction

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
├──────────────────────────────────────────────────────────────┤
│            evocore._core  (PyO3 Extension)                    │
│  individuals · operators · selection · cmaes                  │
│  rng (global seeded) · fitness_cache · parallel               │
├──────────────────────────────────────────────────────────────┤
│                      Rust Crate                               │
│  FloatIndividual · IntegerIndividual · BinaryIndividual       │
│  float_ops · int_ops · binary_ops · selection · cmaes        │
│  StdRng pool · GeneSpec · fitness validity flags              │
└──────────────────────────────────────────────────────────────┘
```

**Layer responsibilities (unchanged from v1):**
- `evocore` — user-facing Python. Clean typed API, `GeneSpace`, callbacks, statistics, parallelism wrappers. No recompile needed when iterating on API.
- `evocore._core` — compiled PyO3 extension. All hot paths: operator loops, selection, CMA-ES matrix math, Rayon pool. The Rust layer owns all RNG state.

**New in v2:**
- `evocore/gene_space.py` — `GeneDef`, `GeneSpace`: per-gene type + bounds + name. Decouples gene specification from engine configuration.
- `evocore/operators.py` — now a real bounds-aware operator layer: wraps `_core` operators, applies clamping, applies integer rounding. The engine never calls `_core` operators directly.
- `evocore/parallel.py` — `ThreadParallel` (Rayon via PyO3) and `ProcessParallel` (`concurrent.futures.ProcessPoolExecutor`). Engine uses either transparently.

---

## 3. Project File Structure

```
evocore/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/                              # Rust source
│   ├── lib.rs                        # PyO3 module root
│   ├── individual.rs                 # FloatIndividual, IntegerIndividual, BinaryIndividual
│   ├── gene_spec.rs                  # GeneKind enum, per-gene spec (used for clamping in Rust)
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── float_ops.rs              # BLX-α, SBX, Gaussian, Uniform mutation
│   │   ├── int_ops.rs                # Integer-aware SBX, Gaussian mutation with rounding
│   │   └── binary_ops.rs             # one-point, two-point, uniform XO, bit-flip
│   ├── selection.rs                  # tournament, roulette, rank
│   ├── cmaes.rs                      # CMAESState, ask/tell, eigen cache, mirror folding
│   ├── rng.rs                        # Global seeded StdRng pool
│   └── parallel.rs                   # Rayon pool management
├── evocore/                          # Python source
│   ├── __init__.py
│   ├── gene_space.py                 # GeneDef, GeneSpace
│   ├── individual.py                 # Individual, Population
│   ├── operators.py                  # Bounds-aware operator layer (wraps _core)
│   ├── parallel.py                   # ThreadParallel, ProcessParallel
│   ├── ga.py                         # GAEngine, RunResult, MultiRunResult
│   ├── cmaes.py                      # CMAESEngine
│   ├── stats.py                      # LogEntry, Logbook
│   ├── callbacks.py                  # Callback base + built-ins
│   └── exceptions.py                 # EvocoreError hierarchy
├── tests/
│   ├── unit/
│   │   ├── test_gene_space.py
│   │   ├── test_operators.py
│   │   ├── test_selection.py
│   │   ├── test_ga_engine.py
│   │   ├── test_cmaes_engine.py
│   │   ├── test_parallel.py
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

## 4. Gene Space (`evocore/gene_space.py`)

This is the central new concept in v2. It replaces the flat `gene_bounds=(-5.0, 5.0)` parameter.

```python
from evocore import GeneDef, GeneSpace

# Mode A: explicit per-gene (recommended)
space = GeneSpace([
    GeneDef("ema_fast",   kind="int",   low=5,    high=200),
    GeneDef("ema_slow",   kind="int",   low=10,   high=500),
    GeneDef("threshold",  kind="float", low=0.0,  high=1.0),
    GeneDef("atr_mult",   kind="float", low=0.5,  high=5.0),
    GeneDef("use_filter", kind="bool"),             # no bounds needed
])

# Mode B: uniform float — backward compatible, no names
space = GeneSpace.uniform(low=-5.0, high=5.0, length=10)

# Introspection
space.length        # → 5
space.names         # → ["ema_fast", "ema_slow", "threshold", "atr_mult", "use_filter"]
space.bounds        # → [(5, 200), (10, 500), (0.0, 1.0), (0.5, 5.0), None]
space.kinds         # → ["int", "int", "float", "float", "bool"]
space.has_names     # → True (False for Mode B)
```

```python
@dataclass
class GeneDef:
    name: str
    kind: Literal["float", "int", "bool"]
    low: float | int | None = None   # None only for bool
    high: float | int | None = None  # None only for bool

    def __post_init__(self):
        if self.kind != "bool":
            assert self.low is not None and self.high is not None
            assert self.low < self.high
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
```

`GeneSpace` is passed to both `GAEngine` and `CMAESEngine` (CMA-ES only supports float/int genes — booleans raise `ConfigurationError` if included in a CMA-ES space).

---

## 5. Rust Core Components (`evocore._core`)

### 5.1 Individual Types

Three concrete types, all `Clone + Send + Sync`:

```rust
FloatIndividual   { genes: Vec<f64>,  fitness: Option<f64>, fitness_valid: bool }
IntegerIndividual { genes: Vec<i64>,  fitness: Option<f64>, fitness_valid: bool }
BinaryIndividual  { genes: Vec<bool>, fitness: Option<f64>, fitness_valid: bool }
```

`fitness_valid` is a new field. The engine sets it to `false` after any operator application and to `true` after a successful fitness evaluation. Individuals with `fitness_valid = true` bypass the fitness function entirely. This is how elitism and operator-skip caching work.

All three types expose `__repr__`, `__len__`, `genes`, `fitness`, `fitness_valid` as Python attributes.

### 5.2 RNG Management (`rng.rs`)

**Critical v2 change.** In v1, every Rust operator received a `seed: u64` from Python. This was fragile — reproducibility depended on the Python RNG state sequence.

In v2, the Rust module owns a global seeded RNG pool:

```rust
// One RNG per Rayon thread, seeded deterministically from the master seed
pub struct RngPool {
    master_seed: u64,
    thread_rngs: Vec<StdRng>,  // one per thread
}
```

Initialization:
```python
from evocore._core import init_rng
init_rng(seed=42)  # called once by GAEngine/CMAESEngine constructor
```

All Rust operators draw from this pool. No seeds are passed from Python. Reproducibility is guaranteed for any fixed `seed` regardless of Python-side call ordering.

### 5.3 Genetic Operators

Operators in Rust take genes `by reference`, return new gene vectors, and do **not** clamp to bounds — clamping is the engine's responsibility (see Section 6.3). This keeps operators mathematically pure and reusable.

**Float operators (`float_ops.rs`):**

| Function | Description | Parameters |
|---|---|---|
| `blend_crossover(a, b, alpha)` | BLX-α | alpha: f64 |
| `simulated_binary_crossover(a, b, eta)` | SBX | eta: f64 |
| `gaussian_mutation(genes, sigma, prob)` | Per-gene Gaussian noise | sigma, prob: f64 |
| `uniform_mutation(genes, low, high, prob)` | Per-gene uniform reset | low, high, prob: f64 |

`mu` removed from `gaussian_mutation` — mutation adds noise around the current value (`mu=0` is the only sensible default; biased mutation is a separate operator if needed).

**Integer operators (`int_ops.rs`):**

| Function | Description | Parameters |
|---|---|---|
| `int_simulated_binary_crossover(a, b, eta)` | SBX with `round()` | eta: f64 |
| `int_gaussian_mutation(genes, sigma, prob)` | Gaussian + `round()` | sigma, prob: f64 |
| `int_uniform_mutation(genes, low, high, prob)` | Uniform integer draw | low, high: i64, prob: f64 |

Integer operators round to the nearest integer after the real-valued operation. `int_uniform_mutation` draws uniformly from `[low, high]` (inclusive) via `rand::Rng::gen_range`.

**Binary operators (`binary_ops.rs`):** Unchanged from v1 — one-point, two-point, uniform XO, bit-flip.

### 5.4 Selection Algorithms (`selection.rs`)

Unchanged signatures from v1:
```rust
tournament_selection(fitnesses, k, tournament_size) → Vec<usize>
roulette_selection(fitnesses, k)                    → Vec<usize>
rank_selection(fitnesses, k)                        → Vec<usize>
```

**New in v2:** All three handle NaN/Inf fitness values safely. NaN and -Inf individuals are assigned rank 0 (worst) in all selection methods. A warning is emitted via the callback system when NaN fitness is encountered. +Inf fitness is clamped to `f64::MAX` with a warning.

### 5.5 Parallelism (`parallel.rs`)

Rayon-based batch evaluation. Used only for `parallel="thread"` mode (fitness functions that release the GIL). The `parallel="process"` mode is handled in Python via `concurrent.futures.ProcessPoolExecutor` (Section 6.5).

```rust
evaluate_sequential(genes_list, fitness_fn)         → PyResult<Vec<f64>>
evaluate_parallel_rayon(genes_list, fitness_fn, n)  → PyResult<Vec<f64>>
```

### 5.6 CMA-ES Engine (`cmaes.rs`)

**v2 changes:**

**Eigendecomposition caching.** The decomposition is recomputed every `update_interval` generations, not every generation. The interval is computed per the Hansen reference implementation:

```rust
eigendecomp_interval = max(1, floor(1.0 / (10.0 * n as f64 * (c1 + cmu))))
```

For `n=10`: interval ≈ 1 (recompute nearly every generation — correct).
For `n=50`: interval ≈ 5 (meaningful saving).
For `n=100`: interval ≈ 20 (significant saving at O(n³) cost).

```rust
CMAESState {
    // ... all v1 fields ...
    eigendecomp_cache: EigenCache,   // cached D, B matrices
    eigendecomp_age: usize,          // generations since last decomposition
    eigendecomp_interval: usize,     // computed at construction
}
```

**Mirror-folding boundary correction.** When a sample falls outside `[low, high]` for any gene, the excess is folded back from the violated boundary rather than clipped:

```
mirror_fold(x, low, high):
    range = high - low
    x = x - low
    x = x mod (2 * range)
    if x > range: x = 2 * range - x
    return x + low
```

This preserves the distributional shape near boundaries instead of collapsing probability mass at the boundary (the clipping bias). Applied inside `ask()` after sampling.

**Updated `CMAESState` struct:**
```rust
CMAESState {
    mean: DVector<f64>,
    sigma: f64,
    cov: DMatrix<f64>,
    pc: DVector<f64>,
    ps: DVector<f64>,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64, cs: f64, c1: f64, cmu: f64, damps: f64, chiN: f64,
    eigendecomp_cache: EigenCache,
    eigendecomp_age: usize,
    eigendecomp_interval: usize,
    bounds: Vec<(f64, f64)>,   // per-gene bounds for mirror folding
    generation: usize,
}
```

Python-facing interface is unchanged — `ask()` and `tell()` as in v1.

---

## 6. Python API Layer

### 6.1 Individual & Population

```python
@dataclass
class Individual:
    genes: list[float | int | bool]
    fitness: float | None = None
    fitness_valid: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def params(self) -> dict | None:
        """Named gene dict, populated if GeneSpace has names. None otherwise."""
        return self.metadata.get("params")

    def __repr__(self) -> str: ...
```

`params` is populated automatically by the engine when `GeneSpace.has_names` is `True`. The fitness function can use either `ind.genes[i]` (index) or `ind.params["name"]` (named). Both always work when names are set.

```python
class Population:
    def __init__(self, individuals: list[Individual]): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, idx: int) -> Individual: ...
    def best(self, n: int = 1) -> list[Individual]: ...
    def mean_fitness(self) -> float: ...
    def std_fitness(self) -> float: ...

    def diversity(self) -> list[float]:
        """Per-gene standard deviation across the population.
        Returns list of length gene_length. High values = diverse, near 0 = converged."""
        ...

    def to_dataframe(self) -> "pd.DataFrame":
        """Requires pandas. Columns: gene_0..gene_n (or named), fitness, fitness_valid."""
        ...
```

### 6.2 GAEngine

```python
engine = GAEngine(
    gene_space=GeneSpace([                   # Required. Replaces gene_bounds + gene_length.
        GeneDef("x0", "float", -5.0, 5.0),
        GeneDef("x1", "float", -5.0, 5.0),
    ]),
    population_size=200,
    generations=100,

    # Crossover
    crossover="sbx",                         # "blx" | "sbx" | "one_point" | "two_point" | "uniform"
    crossover_prob=0.9,
    crossover_eta=2.0,                       # SBX/BLX distribution parameter (was hardcoded in v1)
    crossover_alpha=0.5,                     # BLX-α alpha parameter

    # Mutation
    mutation="gaussian",                     # "gaussian" | "uniform" | "bit_flip"
    mutation_prob=0.1,
    mutation_sigma=0.2,                      # Initial sigma (fraction of gene range, applied per-gene)
    mutation_sigma_schedule="constant",      # "constant" | "linear_decay" | "cosine_decay"
    mutation_sigma_end=0.02,                 # Final sigma for decay schedules (ignored if constant)

    # Selection
    selection="tournament",                  # "tournament" | "roulette" | "rank"
    tournament_size=3,

    # Elitism
    elitism=5,                               # Top k individuals carried unchanged. Default: 1.

    # Parallelism
    parallel="none",                         # "none" | "thread" | "process"
    n_workers=None,                          # None = os.cpu_count()

    # Reproducibility
    seed=42,

    # Callbacks
    callbacks=[],
)

result = engine.run(fitness_fn=my_fitness)         # → RunResult
multi  = engine.run_multiple(                       # → MultiRunResult
    fitness_fn=my_fitness,
    n_runs=10,
    aggregate="best",                              # "best" | "all"
    seed_offset=1000,                              # seeds: 42, 1042, 2042, ...
)
result = engine.resume(                            # → RunResult (continue from checkpoint)
    fitness_fn=my_fitness,
    checkpoint="./checkpoints/checkpoint_gen_50.pkl",
)
```

**`gene_space` is the only required parameter.** All others have sensible defaults. `gene_bounds` (v1) is still accepted as a deprecated alias for `GeneSpace.uniform(...)` and raises a `DeprecationWarning`.

**Crossover operator selection rules** (engine validates at construction):

| `crossover` | Allowed for |
|---|---|
| `"sbx"`, `"blx"` | float + int genes only (ConfigurationError if binary genes present) |
| `"one_point"`, `"two_point"`, `"uniform"` | binary individuals only (ConfigurationError if float/int genes present) |

Mixed `GeneSpace` (float + int genes) always uses `sbx` or `blx`. If the space contains bool genes alongside float/int, a `ConfigurationError` is raised — use `BinaryIndividual` separately or encode booleans as int `{0, 1}`.

**Mutation sigma interpretation:** `mutation_sigma` is expressed as a fraction of each gene's range (i.e., `sigma_abs = mutation_sigma * (high - low)` per gene). This makes the parameter scale-invariant across different gene ranges — a sigma of `0.1` means "10% of the gene's range" regardless of whether that range is `[5, 200]` or `[-1.0, 1.0]`.

### 6.3 Operator Layer (`evocore/operators.py`)

The operator layer sits between `GAEngine` and `_core`. The engine calls the operator layer; it never calls `_core` operators directly. This layer owns three responsibilities:

**1. Dispatch.** Maps string operator names to the correct `_core` function for the individual's gene types. For a mixed float+int `GeneSpace`, it applies float operators to float genes and integer operators to integer genes in a single pass.

**2. Bounds clamping.** After every crossover and mutation call, clamps each gene to its `[low, high]` bounds. This is the canonical location for clamping — not inside individual operators (which stay mathematically pure) and not inside the engine (which stays strategy-agnostic).

**3. Integer rounding.** After clamping, rounds any gene with `kind="int"` to the nearest integer. Order: crossover → clamp → round. This ensures `int_simulated_binary_crossover` produces exact integers.

```python
class OperatorSet:
    """Constructed by GAEngine from engine config + GeneSpace."""
    def __init__(self, gene_space: GeneSpace, crossover: str, mutation: str, ...): ...

    def apply_crossover(self, a: list, b: list) -> tuple[list, list]:
        """Returns two new gene lists, clamped and rounded."""
        ...

    def apply_mutation(self, genes: list, sigma: float) -> list:
        """Returns new gene list, clamped and rounded. sigma passed explicitly for schedule."""
        ...
```

### 6.4 Fitness Function Protocol

```python
# Minimal — list-indexed genes, returns float
def my_fitness(ind: Individual) -> float:
    return -sum(x**2 for x in ind.genes)

# Named params dict — if gene_space has names
def my_fitness(ind: Individual) -> float:
    p = ind.params   # dict: {"ema_fast": 20, "ema_slow": 50, ...}
    return run_backtest(p["ema_fast"], p["ema_slow"])

# With sidecar metrics — return (fitness, metrics_dict)
# metrics_dict is stored in individual.metadata["metrics"]
# and forwarded to LogEntry.custom for that generation's best individual
def my_fitness(ind: Individual) -> tuple[float, dict]:
    result = run_backtest(ind.params)
    return result.profit_factor, {
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "n_trades": result.n_trades,
    }
```

No decorators, no registration. The engine auto-detects tuple returns via `isinstance(result, tuple)`.

**NaN/Inf handling:** If the fitness function returns `float('nan')`, `float('inf')`, or `float('-inf')`, the engine:
1. Stores the raw value on the individual (for diagnostic purposes).
2. Treats the individual as if it had fitness `-inf` for selection purposes.
3. Emits a `FitnessWarning` (not an exception) once per run, with the count of affected individuals. Use `warnings.filterwarnings("error", category=FitnessWarning)` to promote to an exception.

### 6.5 Parallelism (`evocore/parallel.py`)

Two modes with identical external interface:

```python
# "thread" mode — Rayon via PyO3. Fast but only truly parallel if fitness_fn
# releases the GIL (e.g., NumPy-heavy work). For pure-Python fitness, use "process".
GAEngine(parallel="thread", n_workers=8)

# "process" mode — concurrent.futures.ProcessPoolExecutor. Bypasses GIL entirely.
# Works with any pure-Python fitness function. Serializes Individual via pickle.
# Startup cost is ~0.5s for pool initialization (amortized over generations).
GAEngine(parallel="process", n_workers=8)
```

**Guidance (documented prominently):**

| Fitness function type | Recommended mode |
|---|---|
| Pure Python (loops, dicts) | `"process"` |
| NumPy-heavy (array math) | `"thread"` |
| Calls a compiled extension that releases GIL | `"thread"` |
| Single-core evaluation < 5ms | `"none"` (overhead exceeds benefit) |

`ProcessParallel` uses `initializer` to pre-load any large shared state (e.g., market data arrays) into each worker process once, avoiding repeated serialization:

```python
def load_data():
    global BARS
    BARS = load_ohlcv("data.parquet")  # loaded once per worker

engine = GAEngine(
    parallel="process",
    n_workers=8,
    process_initializer=load_data,   # called once per worker at pool startup
)
```

### 6.6 RunResult & MultiRunResult

```python
@dataclass
class RunResult:
    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
    n_evaluations: int               # total fitness function calls (excludes cache hits)
    elite_history: list[Individual]  # best individual at each generation (len = actual generations run)
    diversity_history: list[list[float]]  # per-gene std dev per generation (shape: [gens, n_genes])
    seed: int                        # seed used for this run
    stopped_early: bool              # True if EarlyStopping fired before final generation

@dataclass
class MultiRunResult:
    best: RunResult                  # run with highest best_fitness
    all_runs: list[RunResult]        # all n_runs results, sorted by best_fitness descending
    n_runs: int
    wall_time_seconds: float         # total wall time across all runs

    def best_n(self, n: int) -> list[RunResult]: ...
    def fitness_summary(self) -> dict:
        """Returns {"mean", "std", "min", "max"} of best_fitness across runs."""
```

### 6.7 CMAESEngine

```python
engine = CMAESEngine(
    gene_space=GeneSpace([                   # Required
        GeneDef("x0", "float", -2.0, 2.0),
        GeneDef("x1", "float", -2.0, 2.0),
    ]),
    population_size=50,
    initial_mean=None,              # None = random within bounds per gene
    initial_sigma=0.3,              # Step size as fraction of mean gene range
    generations=300,
    parallel="none",                # "none" | "thread" | "process"
    n_workers=None,
    callbacks=[],
    seed=42,
)

result = engine.run(fitness_fn=my_fitness)   # → RunResult
```

`initial_sigma` interpretation mirrors `mutation_sigma` in `GAEngine`: it is a fraction of the gene range, so `0.3` means 30% of `(high - low)`. This makes it scale-invariant across gene definitions.

CMA-ES only supports `kind="float"` and `kind="int"` genes. Integer genes are handled by rounding the sampled floats after mirror-folding, before passing to the fitness function.

### 6.8 Callbacks

```python
class Callback:
    def on_generation_start(self, gen: int, pop: Population) -> None: ...
    def on_generation_end(self, gen: int, pop: Population) -> None: ...
    def on_run_end(self, result: RunResult) -> None: ...

    # Checked by engine after BOTH on_generation_start and on_generation_end
    should_stop: bool = False
```

**Built-in callbacks:**

```python
# Stop if best fitness does not improve by min_delta for patience generations
EarlyStopping(patience=10, min_delta=1e-6)

# tqdm progress bar with live best_fitness display
ProgressBar()

# Save Population pickle every `every` generations
# File: {path}/checkpoint_gen_{n}.pkl
# Contains: {"population": list[Individual], "generation": n, "seed": seed}
CheckpointCallback(path="./checkpoints", every=10)

# Log custom metrics per generation to a JSON Lines file (one object per line)
# Useful for external monitoring (e.g., feeding a dashboard)
MetricsLogger(path="./metrics.jsonl")
```

**`on_fitness_warning(self, count: int, generation: int) -> None`** — new optional hook called when NaN/Inf fitness values are encountered. Default: no-op. Override to log or raise.

### 6.9 Statistics & Logbook

```python
@dataclass
class LogEntry:
    gen: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    wall_time_ms: float
    n_evaluations: int        # fitness calls this generation (0 for cached elites)
    diversity: list[float]    # per-gene std dev (length = n_genes)
    custom: dict              # sidecar metrics from fitness function for this gen's best individual
                              # empty dict if fitness_fn doesn't return tuple

class Logbook:
    def append(self, entry: LogEntry) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __getitem__(self, idx: int) -> LogEntry: ...

    def print(self) -> None:
        # gen | best_fitness | mean_fitness | std_fitness | wall_time_ms | n_evaluations
        # custom columns appended if any entry has custom metrics

    def to_dataframe(self) -> "pd.DataFrame":
        # Optional dep: pandas. Raises ImportError with install instructions.
        # Custom dict keys become additional columns.

    def plot(self, metrics: list[str] | None = None) -> "plt.Figure":
        # Optional dep: matplotlib. Raises ImportError with install instructions.
        # metrics: which columns to plot. Default: ["best_fitness", "mean_fitness"]
        # Custom metrics can be included by name.
```

---

## 7. Error Handling

```python
class EvocoreError(Exception): ...
class ConfigurationError(EvocoreError): ...   # bad engine/gene_space config
class FitnessError(EvocoreError): ...         # fitness_fn raised an exception
class ConvergenceError(EvocoreError): ...     # CMA-ES numerical failure
class ParallelError(EvocoreError): ...        # worker pool failure
class CheckpointError(EvocoreError): ...      # checkpoint file corrupt/missing

import warnings
class FitnessWarning(UserWarning): ...        # NaN/Inf fitness (warning, not exception by default)
```

Rust panics are caught at the PyO3 boundary and re-raised as `ConvergenceError` or `EvocoreError` — no raw Rust panics surface.

**Example messages:**

```
ConfigurationError: GeneSpace contains bool genes alongside float/int genes.
  Use a separate BinaryIndividual population for boolean genes, or encode
  booleans as int genes with low=0, high=1.

ConfigurationError: crossover="sbx" is not compatible with a binary-only GeneSpace.
  Use crossover="one_point", "two_point", or "uniform" for binary individuals.

FitnessError: fitness_fn raised ZeroDivisionError for individual at index 12
  (generation 5). Original error: division by zero.
  Hint: guard against zero-trade or zero-variance results in your fitness function.

FitnessWarning: 8 individuals in generation 14 returned NaN fitness.
  They have been assigned fitness=-inf for selection. Check your fitness function
  for edge cases (empty series, division by zero, no trades generated).

ConvergenceError: CMA-ES covariance matrix became non-positive-definite at
  generation 47. Try reducing initial_sigma (current: 0.5 → suggestion: 0.2).
  Gene ranges: x0=[−2,2], x1=[−2,2].

CheckpointError: checkpoint file './checkpoints/checkpoint_gen_50.pkl' not found.
  Available checkpoints: checkpoint_gen_10.pkl, checkpoint_gen_20.pkl, checkpoint_gen_40.pkl
```

---

## 8. Build System & Toolchain

```toml
# Cargo.toml [dependencies]
pyo3       = { version = "0.21", features = ["extension-module"] }
rayon      = "1.9"
rand       = "0.8"
rand_distr = "0.4"
nalgebra   = "0.32"

# pyproject.toml
[build-system]
requires      = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name            = "evocore"
version         = "0.2.0"
requires-python = ">=3.11"

[tool.maturin]
python-source = "."
module-name   = "evocore._core"
features      = ["pyo3/extension-module"]
```

Dev workflow unchanged from v1:
```bash
maturin develop --release    # compile Rust + install editable
# Python-only changes: no recompile
# Rust changes: re-run maturin develop --release
```

---

## 9. Testing Strategy

### Unit Tests (pytest)

- `test_gene_space.py` — GeneSpace construction (named, uniform, invalid configs), `GeneDef` validation, bounds/kinds introspection
- `test_operators.py` — via Python: per-operator output length; bounds never exceeded after clamping; integer genes always integer-valued; bit-flip correctness; prob=0 unchanged; prob=1 all-mutated
- `test_selection.py` — returns k indices; all in range; NaN fitness sorted to worst; roulette/rank correct count
- `test_ga_engine.py` — float run, binary run, mixed gene_space run; logbook length = generations; elitism preserves best; early stopping fires; ConfigurationError cases; FitnessWarning on NaN return; sidecar metrics stored; resume from checkpoint; multi-run returns correct n_runs; diversity_history shape
- `test_cmaes_engine.py` — run, logbook, ConfigurationError on missing gene_space, mirror-folding keeps all samples within bounds
- `test_parallel.py` — process mode produces same results as none mode (same seed); thread mode produces same results as none mode for NumPy fitness; process_initializer called exactly n_workers times
- `test_stats.py` — append, len, iter, to_dataframe custom columns, plot (mock matplotlib)

### Rust Unit Tests (inline `#[cfg(test)]`)

- Per-operator: output length, gene length invariance, determinism given same RNG state
- Integer operators: all outputs are exact integers
- Selection: NaN handling, correct count
- CMA-ES: ask returns correct size and gene length; tell updates generation; sigma stays positive over 10 iterations; eigendecomp interval computed correctly; mirror-folded samples always within bounds

### Integration Tests (pytest)

- **Sphere** (`Σxᵢ²`, dim=10, bounds=±5): `best_fitness > -0.01` after 200 generations
- **Rastrigin** (multimodal, dim=10, bounds=±5.12): `best_fitness > -10.0` after 300 generations
- **OneMax binary** (50 bits): `best_fitness >= 48` after 100 generations
- **Mixed GeneSpace** (5 int + 5 float genes, sphere): convergence within 200 generations, all int genes integer-valued throughout
- **CMA-ES Rosenbrock** (dim=10, bounds=±2): `best_fitness > -1.0` after 300 generations

### Benchmarks

- `bench_ga_vs_deap.py` — same problem (sphere, pop=1000, gen=100, dim=20), assert evocore ≥ 2× faster wall time
- `bench_parallel_scaling.py` — NumPy fitness, measure thread speedup at 1/2/4/8 workers; pure-Python fitness, measure process speedup at 1/2/4/8 workers

---

## 10. Performance Targets

| Scenario | Target vs pure-Python DEAP |
|---|---|
| GA float, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| GA float, 1000 pop, 100 gen, 8-thread Rayon | ≥ 10× faster |
| GA binary, 1000 pop, 100 gen, sequential | ≥ 4× faster |
| GA integer, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| CMA-ES, dim=20, 200 gen, sequential | ≥ 5× faster |
| CMA-ES, dim=100, 500 gen, sequential | eigendecomp caching provides ≥ 2× vs naive v1 |

---

## 11. Migration from v1

| v1 | v2 equivalent |
|---|---|
| `gene_bounds=(-5.0, 5.0)`, `gene_length=10` | `gene_space=GeneSpace.uniform(-5.0, 5.0, 10)` |
| `gene_bounds=(-5.0, 5.0)` | deprecated alias, raises `DeprecationWarning`, still works |
| `individual_type="float"` | inferred from `GeneSpace` gene kinds |
| `individual_type="binary"` | `GeneSpace` with all `kind="bool"` genes |
| `gaussian_mutation(ind, mu=0, sigma, prob, seed)` | `gaussian_mutation(genes, sigma, prob)`, no seed arg |
| `evaluate_parallel_rayon` | `parallel="thread"` on the engine |
| none | `parallel="process"` for pure-Python fitness |

---

## 12. Out of Scope (Phase 1)

- Genetic Programming (tree individuals)
- Multi-objective optimization (NSGA-II, SPEA2) — `fitness` remains a scalar; tuple return is for sidecar metrics, not Pareto fronts
- Evolution Strategies beyond CMA-ES (BIPOP-CMA-ES — Phase 2 candidate)
- PyPI wheel publishing / CI distribution
- GPU acceleration
- Distributed evaluation
- Walk-forward validation harness (user-implemented via fitness function and callbacks)
