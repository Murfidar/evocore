# evocore Design Spec
**Date:** 2026-04-04
**Status:** Approved
**Scope:** Phase 1 — Genetic Algorithms (float + binary individuals) + CMA-ES, Rust-native Python library

---

## 1. Goal

Rebuild the hot-path components of DEAP (fitness evaluation loop, genetic operators, CMA-ES) as a Rust-native Python library with a clean modern API. Target: ≥3× sequential and ≥10× parallel speedup over pure-Python DEAP for GA workloads.

---

## 2. Architecture

Two-layer design — Rust owns speed, Python owns API ergonomics.

```
┌─────────────────────────────────────────────────────┐
│                  User Python Code                    │
├─────────────────────────────────────────────────────┤
│              evocore  (Python Layer)                 │
│   Individual · Population · GAEngine · CMAESEngine   │
│   Operators · Selection · Statistics · Callbacks     │
├─────────────────────────────────────────────────────┤
│           evocore._core  (PyO3 Extension)            │
│  ga_float · ga_binary · cmaes · operators · rayon    │
├─────────────────────────────────────────────────────┤
│                    Rust Crate                        │
│  individuals · operators · selection · cmaes · rng  │
└─────────────────────────────────────────────────────┘
```

- **`evocore`** — user-facing Python package. Clean typed API, dataclasses, callbacks, statistics. Pure Python — no recompile needed when iterating on API.
- **`evocore._core`** — compiled PyO3 extension. All hot paths live here: evaluation dispatch, crossover/mutation loops, CMA-ES covariance updates, Rayon thread pool management.

---

## 3. Project File Structure

```
evocore/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/                            # Rust source
│   ├── lib.rs                      # PyO3 module root
│   ├── individual.rs               # FloatIndividual, BinaryIndividual
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── float_ops.rs            # BLX-α, SBX, Gaussian, Uniform mutation
│   │   └── binary_ops.rs           # one-point, two-point, uniform XO, bit-flip
│   ├── selection.rs                # tournament, roulette, rank
│   ├── cmaes.rs                    # CMAESState, ask/tell
│   └── parallel.rs                 # Rayon pool management
├── evocore/                        # Python source
│   ├── __init__.py
│   ├── individual.py               # Individual, Population
│   ├── ga.py                       # GAEngine, RunResult
│   ├── cmaes.py                    # CMAESEngine
│   ├── operators.py                # Operator registry/validation
│   ├── stats.py                    # Logbook, Statistics
│   ├── callbacks.py                # Callback base + built-ins
│   └── exceptions.py               # EvocoreError hierarchy
├── tests/
│   ├── unit/
│   │   ├── test_operators.py
│   │   ├── test_selection.py
│   │   ├── test_ga_engine.py
│   │   └── test_cmaes_engine.py
│   ├── integration/
│   │   ├── test_sphere_function.py
│   │   ├── test_rastrigin.py
│   │   └── test_binary_onemax.py
│   └── benchmarks/
│       ├── bench_ga_vs_deap.py
│       └── bench_parallel_scaling.py
├── examples/
│   ├── sphere_optimization.py
│   ├── onemax_binary.py
│   └── cmaes_rosenbrock.py
└── docs/
    └── superpowers/
        └── specs/
```

---

## 4. Rust Core Components (`evocore._core`)

### 4.1 Individual Types

```rust
FloatIndividual  { genes: Vec<f64>, fitness: Option<f64> }
BinaryIndividual { genes: Vec<bool>, fitness: Option<f64> }
```

Both implement `Clone`, `Send`, `Sync` (required for Rayon). Exposed as Python classes with `__repr__`, `__len__`, direct attribute access.

### 4.2 Genetic Operators

**Float operators (`float_ops.rs`):**

| Function | Description | Key Parameters |
|---|---|---|
| `blend_crossover(a, b, alpha)` | BLX-α crossover | alpha: f64 |
| `simulated_binary_crossover(a, b, eta)` | SBX | eta: f64 |
| `gaussian_mutation(ind, mu, sigma, prob)` | Per-gene Gaussian noise | mu, sigma, prob: f64 |
| `uniform_mutation(ind, low, high, prob)` | Per-gene uniform reset | low, high, prob: f64 |

**Binary operators (`binary_ops.rs`):**

| Function | Description | Key Parameters |
|---|---|---|
| `one_point_crossover(a, b)` | Single cut point | — |
| `two_point_crossover(a, b)` | Two cut points | — |
| `uniform_crossover(a, b, prob)` | Per-gene swap | prob: f64 |
| `bit_flip_mutation(ind, prob)` | Per-gene flip | prob: f64 |

All operators take individuals by value and return new individuals — no in-place mutation exposed to Python.

### 4.3 Selection Algorithms (`selection.rs`)

```
tournament_selection(population, k, tournament_size) → Vec<usize>
roulette_selection(population, k)                    → Vec<usize>
rank_selection(population, k)                        → Vec<usize>
```

Returns indices into the population — avoids unnecessary allocation.

### 4.4 Parallelism (`parallel.rs`)

Rayon thread pool initialized once and reused across generations.

```rust
evaluate_parallel(population, fitness_fn, n_threads) → Vec<f64>
evaluate_sequential(population, fitness_fn)          → Vec<f64>
```

`fitness_fn` is a Python callable passed via PyO3's `Py<PyAny>`. True parallelism only when `fitness_fn` releases the GIL (e.g., NumPy-heavy work). For pure-Python fitness functions, sequential mode is recommended and documented.

### 4.5 CMA-ES Engine (`cmaes.rs`)

Internal state maintained between generations:

```rust
CMAESState {
    mean: Vec<f64>,
    sigma: f64,
    cov: Vec<Vec<f64>>,          // covariance matrix (n×n), via nalgebra
    pc: Vec<f64>,                // evolution path (cov)
    ps: Vec<f64>,                // evolution path (sigma)
    eigenvalues: Vec<f64>,       // cached decomposition
    eigenvectors: Vec<Vec<f64>>,
    generation: usize,
}
```

Exposed via ask/tell interface:
```python
state = _core.CMAESState(mean, sigma, population_size)
# mean: list[float] of length gene_length, or None for zero-init
samples = state.ask()           # sample new individuals
state.tell(samples, fitnesses)  # update distribution
```

---

## 5. Python API Layer

### 5.1 Individual & Population

```python
@dataclass
class Individual(Generic[T]):
    genes: list[T]
    fitness: float | None = None
    metadata: dict = field(default_factory=dict)

class Population:
    def __init__(self, individuals: list[Individual]): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def best(self, n: int = 1) -> list[Individual]: ...
    def mean_fitness(self) -> float: ...
    def std_fitness(self) -> float: ...
```

### 5.2 GAEngine

```python
engine = GAEngine(
    individual_type="float",        # "float" | "binary"
    gene_length=20,
    population_size=200,
    generations=100,
    crossover="sbx",                # "blx" | "sbx" | "one_point" | "two_point" | "uniform"
    crossover_prob=0.9,
    mutation="gaussian",            # "gaussian" | "uniform" | "bit_flip"
    mutation_prob=0.1,
    selection="tournament",         # "tournament" | "roulette" | "rank"
    tournament_size=3,
    gene_bounds=(-5.0, 5.0),        # float only, required for float individuals. Single (low, high) tuple applied uniformly to all genes.
    parallel=False,                 # True = Rayon multi-threaded
    n_threads=None,                 # None = all available cores
    callbacks=[],
    seed=42,
)

result = engine.run(fitness_fn=my_fitness)   # → RunResult
```

### 5.3 CMAESEngine

```python
engine = CMAESEngine(
    gene_length=10,
    population_size=50,
    initial_mean=None,              # None = random within bounds
    initial_sigma=0.5,
    gene_bounds=(-5.0, 5.0),
    generations=200,
    parallel=False,
    n_threads=None,
    callbacks=[],
    seed=42,
)

result = engine.run(fitness_fn=my_fitness)   # → RunResult
```

### 5.4 RunResult

```python
@dataclass
class RunResult:
    best_individual: Individual
    best_fitness: float
    final_population: Population
    logbook: Logbook
    wall_time_seconds: float
```

### 5.5 Fitness Function Protocol

```python
# Single-threaded: receives one Individual, returns float
def my_fitness(ind: Individual) -> float:
    return sum(x**2 for x in ind.genes)

# Parallel-safe: release GIL via NumPy
import numpy as np
def my_fitness_np(ind: Individual) -> float:
    arr = np.array(ind.genes)
    return float(np.sum(arr ** 2))
```

No decorators, no registration — just a plain callable.

### 5.6 Callbacks

```python
class Callback:
    def on_generation_start(self, gen: int, pop: Population): ...
    def on_generation_end(self, gen: int, pop: Population): ...
    def on_run_end(self, result: RunResult): ...

# Built-in callbacks
EarlyStopping(patience=10, min_delta=1e-6)
ProgressBar()
CheckpointCallback(path="./checkpoints", every=10)  # saves Population as pickle: checkpoint_gen_{n}.pkl
```

### 5.7 Statistics & Logbook

```python
result.logbook.print()
# gen | best_fitness | mean_fitness | std_fitness | wall_time_ms

result.logbook.to_dataframe()   # → pandas DataFrame (optional dep)
result.logbook.plot()           # → matplotlib figure (optional dep)
```

`pandas` and `matplotlib` are optional — missing imports raise `ImportError` with install instructions.

---

## 6. Error Handling

All errors subclass `EvocoreError` defined in `evocore/exceptions.py`:

```python
class EvocoreError(Exception): ...
class ConfigurationError(EvocoreError): ...
class FitnessError(EvocoreError): ...
class ConvergenceError(EvocoreError): ...
class ParallelError(EvocoreError): ...
```

Rust panics are caught at the PyO3 boundary and re-raised as `EvocoreError` — no raw Rust panics surface to users.

**Example messages:**
```
ConfigurationError: gene_bounds required for individual_type='float'.
  Pass gene_bounds=(-5.0, 5.0) to GAEngine.

FitnessError: fitness_fn must return a float, got <class 'list'> for
  individual at index 3. Check your fitness function return type.

ConvergenceError: CMA-ES covariance matrix became non-positive-definite
  at generation 47. Try reducing initial_sigma (current: 2.0).
```

---

## 7. Build System & Toolchain

```toml
# Cargo.toml [dependencies]
pyo3      = { version = "0.21", features = ["extension-module"] }
rayon     = "1.9"
rand      = "0.8"
rand_distr = "0.4"
nalgebra  = "0.32"

# pyproject.toml [build-system]
requires = ["maturin>=1.5"]
build-backend = "maturin"

[tool.maturin]
python-source = "evocore"
module-name   = "evocore._core"
features      = ["pyo3/extension-module"]
```

**Dev workflow:**
```bash
pip install maturin
maturin develop --release    # compile Rust + install
# Python-only changes: no recompile needed
# Rust changes: re-run maturin develop --release
```

---

## 8. Testing Strategy

### Unit Tests (pytest)
- Test Python API layer in isolation
- Cover: engine config validation, callback firing, logbook population, error messages
- Use small populations (20 individuals) and few generations (5) for speed

### Integration Tests (pytest)
- **Sphere function** (`f(x) = Σx²`, optimum = 0.0): assert `best_fitness > -0.01` after 200 generations
- **Rastrigin function** (multimodal, dim=10, bounds=[-5.12, 5.12]): assert `best_fitness > -10.0` after 300 generations (global optimum = 0.0, acceptable near-optimum threshold)
- **OneMax binary** (maximise 1s in bitstring): assert `best_fitness >= 48/50` after 100 generations
- **CMA-ES Rosenbrock**: assert convergence within 200 generations

### Rust Unit Tests (inline `#[cfg(test)]`)
- Per-operator correctness (bounds preservation, gene length invariance)
- Selection returns correct count k
- CMA-ES ask returns correct population size

### Benchmark Tests
- `bench_ga_vs_deap.py`: same problem, same params, assert evocore ≥ 2× faster wall time
- `bench_parallel_scaling.py`: measure speedup at 1, 2, 4, 8 threads

---

## 9. Performance Targets

| Scenario | Target vs pure-Python DEAP |
|---|---|
| GA float, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| GA float, 1000 pop, 100 gen, 8-thread Rayon | ≥ 10× faster |
| GA binary, 1000 pop, 100 gen, sequential | ≥ 4× faster |
| CMA-ES, dim=20, 200 gen, sequential | ≥ 5× faster |

---

## 10. Out of Scope (Phase 1)

- Genetic Programming (tree individuals)
- Multi-objective optimization (NSGA-II, SPEA2)
- Evolution Strategies beyond CMA-ES
- PyPI wheel publishing / CI distribution
- GPU acceleration
- Distributed evaluation (SCOOP equivalent)
