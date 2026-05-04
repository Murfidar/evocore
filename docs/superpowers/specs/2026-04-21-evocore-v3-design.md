# evocore v3 Design Spec

**Date:** 2026-04-21
**Status:** Draft
**Scope:** Phase 1 — Genetic Algorithms (float + integer + binary + mixed individuals) + CMA-ES, Rust-native Python library
**Supersedes:** 2026-04-05-evocore-v2-design.md

---

## Changelog from v2

| Area | Change |
|---|---|
| **RNG Architecture** | Complete redesign — hierarchical seed derivation replaces global `RngPool`. `src/rng.rs` deleted. `src/utils.rs` added. |
| **Python random state** | `_py_rng` (`random.Random`) deleted from `GAEngine`. Zero random state in Python. |
| **Operator signatures** | All Rust operators receive `(master_seed, generation, individual_idx)` instead of `seed: u64`. |
| **`init_rng()` call** | Deleted from public API — no initialization step needed. |
| **`reproduce()` Rust function** | New single Rust call per generation replaces Python per-pair loop. |
| **`PyCMAESState.ask()`** | Now `ask(master_seed, generation)` — engine passes these; no RNG stored on state. |
| **`run_multiple` seeding** | `seed_offset` parameter removed; child seeds derived via `derive_seed(master, 0, run_idx, OP_MULTI_RUN)`. |
| **`OP_CROSSOVER_PROB`** | Crossover probability check moved to Rust as `OP_CROSSOVER_PROB = 6`. |
| **New test file** | `tests/unit/test_rng_reproducibility.py` — four invariant tests including `n_workers` independence. |

Everything not listed above is **unchanged from v2**.

---

## 1. Goal

Rebuild the hot-path components of DEAP as a Rust-native Python library with a clean, domain-agnostic modern API. Designed for any optimization workload where the fitness function is expensive (backtesting, simulation, hyperparameter tuning). Target: ≥3× sequential and ≥10× parallel speedup over pure-Python DEAP for GA workloads.

**Key improvements over v2:**
- All randomness derived from a single `u64` master seed — no mutable random state anywhere in the system
- Thread-count-independent results: `parallel="thread"` with 1, 2, 4, or 8 workers produces byte-identical output for a fixed seed
- Calling `engine.run()` twice on the same engine produces identical results
- `run_multiple` child seeds derived hierarchically, not via arithmetic offset

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
│  utils (derive_seed) · fitness_cache · parallel               │
├──────────────────────────────────────────────────────────────┤
│                      Rust Crate                               │
│  FloatIndividual · IntegerIndividual · BinaryIndividual        │
│  float_ops · int_ops · binary_ops · selection · cmaes         │
│  utils · GeneSpec · fitness validity flags                    │
└──────────────────────────────────────────────────────────────┘
```

**Layer responsibilities:**
- `evocore` — user-facing Python. Clean typed API, `GeneSpace`, callbacks, statistics, parallelism wrappers. No random state. No recompile needed for API changes.
- `evocore._core` — compiled PyO3 extension. All hot paths: operator loops, selection, CMA-ES matrix math, Rayon pool, seed derivation. Owns all randomness.

---

## 3. Project File Structure

```
evocore/
├── Cargo.toml
├── pyproject.toml
├── README.md
├── src/                              # Rust source
│   ├── lib.rs                        # PyO3 module root
│   ├── utils.rs                      # NEW: derive_seed(), OP_* constants
│   ├── individual.rs                 # FloatIndividual, IntegerIndividual, BinaryIndividual
│   ├── gene_spec.rs                  # GeneKind enum, per-gene spec
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── float_ops.rs              # BLX-α, SBX, Gaussian, Uniform mutation
│   │   ├── int_ops.rs                # Integer-aware SBX, Gaussian mutation with rounding
│   │   └── binary_ops.rs             # one-point, two-point, uniform XO, bit-flip
│   ├── reproduce.rs                  # NEW: single-call reproduce() function
│   ├── selection.rs                  # tournament, roulette, rank
│   ├── cmaes.rs                      # CMAESState, ask/tell, eigen cache, mirror folding
│   └── parallel.rs                   # Rayon pool management
│   # DELETED: rng.rs
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
│   │   ├── test_rng_reproducibility.py   # NEW
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

## 4. RNG Architecture (v3 Core Change)

### 4.1 Design Principle

Every random decision in the library is derived deterministically from four inputs:

```
seed_for_this_call = derive_seed(master_seed, generation, individual_idx, operation_type)
```

The `master_seed` is the only RNG configuration value in the system. It is a plain `u64` stored on the engine struct. There is no global mutable RNG state, no stored `StdRng` objects, and no random state on the Python side.

**Properties this guarantees:**

| Property | How it holds |
|---|---|
| Same seed → same result | `derive_seed` is a pure function; same inputs always produce the same output |
| Thread-count independent | Individual `i` at generation `g` always gets the same seed regardless of Rayon thread count |
| `run()` idempotent | No state is consumed; calling `run()` twice on the same engine gives identical results |
| Parallel modes identical | `parallel="none"` and `parallel="thread"` produce byte-identical output for the same seed |
| Multi-run independent | Each run's seed is derived, not offset; seeds share no subsequences |

### 4.2 `src/utils.rs` — New File

```rust
// Operation type constants — one per random decision category.
// Adding a new operator: assign the next available u64, never reuse.
pub const OP_INIT:           u64 = 0;  // initial population gene values
pub const OP_CROSSOVER:      u64 = 1;  // gene values produced by crossover
pub const OP_MUTATION:       u64 = 2;  // gene values produced by mutation
pub const OP_SELECTION:      u64 = 3;  // selection indices
pub const OP_CMAES_ASK:      u64 = 4;  // CMA-ES sample generation
pub const OP_MULTI_RUN:      u64 = 5;  // child seed derivation for run_multiple
pub const OP_CROSSOVER_PROB: u64 = 6;  // whether to apply crossover to a pair

/// SplitMix64 bijective mixing function.
///
/// Algorithm: used by NumPy SeedSequence, Java SplittableRandom, and numerous
/// game engines. Guarantees avalanche: a 1-bit change in any input flips ~32
/// output bits. Different field positions use different multipliers to prevent
/// commutativity collisions (derive_seed(m,1,2,3) != derive_seed(m,2,1,3)).
///
/// Cost: ~5ns per call — negligible against any fitness evaluation.
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

Exposed to Python as `_core.derive_seed(master, generation, idx, op)` and `_core.OP_*` constants.

### 4.3 Files Deleted vs v2

- `src/rng.rs` — **deleted entirely**
- `RngPool` struct — **deleted**
- `init_rng(seed: u64)` Python export — **deleted**

---

## 5. Gene Space (`evocore/gene_space.py`)

Unchanged from v2.

```python
from evocore import GeneDef, GeneSpace

# Mode A: explicit per-gene (recommended)
space = GeneSpace([
    GeneDef("ema_fast",   kind="int",   low=5,    high=200),
    GeneDef("ema_slow",   kind="int",   low=10,   high=500),
    GeneDef("threshold",  kind="float", low=0.0,  high=1.0),
    GeneDef("atr_mult",   kind="float", low=0.5,  high=5.0),
    GeneDef("use_filter", kind="bool"),
])

# Mode B: uniform float — backward compatible
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
    low: float | int | None = None
    high: float | int | None = None

    def __post_init__(self):
        if self.kind != "bool":
            assert self.low is not None and self.high is not None
            assert self.low < self.high
        if self.kind == "int":
            assert isinstance(self.low, int) and isinstance(self.high, int)
```

---

## 6. Rust Core Components (`evocore._core`)

### 6.1 Individual Types

Three concrete types, all `Clone + Send + Sync`:

```rust
FloatIndividual   { genes: Vec<f64>,  fitness: Option<f64>, fitness_valid: bool }
IntegerIndividual { genes: Vec<i64>,  fitness: Option<f64>, fitness_valid: bool }
BinaryIndividual  { genes: Vec<bool>, fitness: Option<f64>, fitness_valid: bool }
```

`fitness_valid` is set to `false` after any operator application and `true` after successful fitness evaluation. Used by elitism and caching logic in the engine.

### 6.2 Genetic Operators — Revised Signatures

All Rust operators replace the `seed: u64` parameter with `(master_seed: u64, generation: u64, individual_idx: u64)`. They call `derive_seed` internally and construct a stack-local `StdRng`. No RNG state escapes the call.

**Float operators (`float_ops.rs`):**

```rust
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};
use rand::SeedableRng;
use rand::rngs::StdRng;

pub fn blend_crossover(
    a: &[f64], b: &[f64], alpha: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    // ... operator logic ...
}

pub fn simulated_binary_crossover(
    a: &[f64], b: &[f64], eta: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    // ...
}

// Note: mu parameter removed from v1/v2. Mutation adds noise around current value (mu=0).
pub fn gaussian_mutation(
    genes: &[f64], sigma: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    // ...
}

pub fn uniform_mutation(
    genes: &[f64], low: f64, high: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    // ...
}
```

**Integer operators (`int_ops.rs`):** Same `(master_seed, generation, individual_idx)` signature pattern.

**Binary operators (`binary_ops.rs`):** Same pattern.

### 6.3 Selection Algorithms — Revised Signatures

Selection operates on the whole population, so `individual_idx` is always `0`. The seed is unique per generation via `generation`.

```rust
use crate::utils::{derive_seed, OP_SELECTION};

pub fn tournament_selection(
    fitnesses: &[f64], k: usize, tournament_size: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, 0, OP_SELECTION)
    );
    // ...
}

pub fn roulette_selection(
    fitnesses: &[f64], k: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> { /* same pattern */ }

pub fn rank_selection(
    fitnesses: &[f64], k: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> { /* same pattern */ }
```

All three handle NaN/Inf fitness values safely: NaN and -Inf individuals are assigned rank 0 (worst). +Inf is clamped to `f64::MAX` with a warning.

### 6.4 `src/reproduce.rs` — New File

A single Rust function replaces the Python per-pair loop. The engine calls this once per generation.

```rust
use crate::utils::{derive_seed, OP_CROSSOVER_PROB};

pub struct ReproductionConfig {
    pub crossover_type:    CrossoverType,    // Sbx | Blx | OnePoint | TwoPoint | UniformXO
    pub crossover_prob:    f64,
    pub crossover_eta:     f64,              // SBX/BLX distribution parameter
    pub crossover_alpha:   f64,              // BLX-α alpha
    pub mutation_type:     MutationType,     // Gaussian | Uniform | BitFlip
    pub mutation_prob:     f64,
    pub mutation_sigma:    f64,              // abs sigma for this generation (schedule applied in Python)
    pub gene_bounds:       Vec<(f64, f64)>,  // per-gene bounds for clamping; (0,1) for bool genes
    pub gene_kinds:        Vec<GeneKind>,    // Float | Int | Bool per gene
    pub elitism_count:     usize,
    pub population_size:   usize,
}

/// Full reproduction: selection → crossover → mutation → clamping → rounding.
/// Returns new population as Vec<Vec<f64>> (integer genes encoded as f64, bool as 0.0/1.0).
/// Elite individuals are prepended and their fitness_valid flag is preserved.
pub fn reproduce(
    population:    &[Vec<f64>],
    fitnesses:     &[f64],
    config:        &ReproductionConfig,
    master_seed:   u64,
    generation:    u64,
) -> Vec<Vec<f64>> {
    // 1. Select parent indices
    let parent_indices = selection::tournament_selection(
        fitnesses, config.population_size, /* tournament_size from config */,
        master_seed, generation,
    );

    // 2. Preserve elites (top elitism_count by fitness, fitness_valid = true)
    let mut new_pop: Vec<Vec<f64>> = get_elites(population, fitnesses, config.elitism_count);

    // 3. Reproduce pairs
    let pairs_needed = (config.population_size - config.elitism_count + 1) / 2;
    for pair_idx in 0..pairs_needed {
        let i = parent_indices[pair_idx * 2];
        let j = parent_indices[pair_idx * 2 + 1];

        // Crossover probability check — fully in Rust
        let apply_xo = {
            let mut rng = StdRng::seed_from_u64(
                derive_seed(master_seed, generation, pair_idx as u64, OP_CROSSOVER_PROB)
            );
            rng.gen::<f64>() < config.crossover_prob
        };

        let (mut c1, mut c2) = if apply_xo {
            apply_crossover(&population[i], &population[j], config, master_seed, generation, pair_idx as u64)
        } else {
            (population[i].clone(), population[j].clone())
        };

        // Mutation
        c1 = apply_mutation(&c1, config, master_seed, generation, (pair_idx * 2) as u64);
        c2 = apply_mutation(&c2, config, master_seed, generation, (pair_idx * 2 + 1) as u64);

        // Clamp to bounds + round integer genes
        c1 = clamp_and_round(&c1, &config.gene_bounds, &config.gene_kinds);
        c2 = clamp_and_round(&c2, &config.gene_bounds, &config.gene_kinds);

        new_pop.push(c1);
        if new_pop.len() < config.population_size { new_pop.push(c2); }
    }

    new_pop.truncate(config.population_size);
    new_pop
}
```

Exposed to Python as `_core.reproduce(population, fitnesses, config_dict, master_seed, generation)`.

### 6.5 Parallelism (`parallel.rs`)

Used only for `parallel="thread"` mode. Signatures unchanged from v2 — no seed parameters since fitness evaluation is deterministic given the genes:

```rust
evaluate_sequential(genes_list, fitness_fn)          → PyResult<Vec<f64>>
evaluate_parallel_rayon(genes_list, fitness_fn, n)   → PyResult<Vec<f64>>
```

### 6.6 CMA-ES Engine (`cmaes.rs`) — Revised

**v3 change: `ask()` receives generation from engine; no RNG stored on state.**

```rust
pub struct CMAESState {
    pub n:                    usize,
    pub lambda:               usize,
    pub mu:                   usize,
    pub mean:                 DVector<f64>,
    pub sigma:                f64,
    pub cov:                  DMatrix<f64>,
    pub pc:                   DVector<f64>,
    pub ps:                   DVector<f64>,
    pub weights:              Vec<f64>,
    pub mueff:                f64,
    pub cc:                   f64,
    pub cs:                   f64,
    pub c1:                   f64,
    pub cmu:                  f64,
    pub damps:                f64,
    pub chiN:                 f64,
    pub eigendecomp_cache:    std::cell::RefCell<EigenCache>,  // interior mutability for caching
    pub eigendecomp_age:      std::cell::Cell<usize>,
    pub eigendecomp_interval: usize,
    pub bounds:               Vec<(f64, f64)>,                 // per-gene for mirror folding
    pub generation:           usize,                           // tracks eigendecomp_interval only
    // DELETED from v2: rng: StdRng
}

impl CMAESState {
    // Constructor: no seed parameter
    pub fn new(mean: Vec<f64>, sigma: f64, lambda: usize, bounds: Vec<(f64, f64)>) -> Self { ... }

    // ask() is &self — no mutation of RNG state, only eigendecomp cache via RefCell
    // master_seed and generation passed by the engine, not stored here
    pub fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        let (eigenvectors, eigenvalues_sqrt) = self.get_or_update_eigen();

        (0..self.lambda).map(|sample_idx| {
            let mut rng = StdRng::seed_from_u64(
                derive_seed(master_seed, generation, sample_idx as u64, OP_CMAES_ASK)
            );
            let z: DVector<f64> = DVector::from_iterator(
                self.n,
                (0..self.n).map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng))
            );
            let scaled = &eigenvectors * DVector::from_iterator(
                self.n,
                eigenvalues_sqrt.iter().zip(z.iter()).map(|(e, zi)| e * zi)
            );
            let raw = &self.mean + self.sigma * scaled;
            raw.iter().enumerate()
                .map(|(i, &x)| mirror_fold(x, self.bounds[i].0, self.bounds[i].1))
                .collect()
        }).collect()
    }

    // tell() remains &mut self — updates mean, cov, pc, ps, sigma, generation
    pub fn tell(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) {
        // ... covariance update logic ...
        self.generation += 1;
    }
}
```

**Eigendecomposition caching** uses `RefCell<EigenCache>` for interior mutability — `ask()` can cache the result without requiring `&mut self`:

```rust
struct EigenCache {
    eigenvectors:     DMatrix<f64>,
    eigenvalues_sqrt: DVector<f64>,
    valid:            bool,
}

impl CMAESState {
    fn get_or_update_eigen(&self) -> (DMatrix<f64>, DVector<f64>) {
        let age = self.eigendecomp_age.get();
        let mut cache = self.eigendecomp_cache.borrow_mut();
        if !cache.valid || age >= self.eigendecomp_interval {
            let eigen = self.cov.clone().symmetric_eigen();
            cache.eigenvalues_sqrt = eigen.eigenvalues.map(|v| v.max(1e-20).sqrt());
            cache.eigenvectors = eigen.eigenvectors;
            cache.valid = true;
            self.eigendecomp_age.set(0);
        } else {
            self.eigendecomp_age.set(age + 1);
        }
        (cache.eigenvectors.clone(), cache.eigenvalues_sqrt.clone())
    }
}
```

**Mirror-folding boundary correction** (unchanged from v2):

```rust
fn mirror_fold(x: f64, low: f64, high: f64) -> f64 {
    let range = high - low;
    let mut x = x - low;
    x = x % (2.0 * range);
    if x > range { x = 2.0 * range - x; }
    x + low
}
```

**PyO3 wrapper — revised:**

```rust
#[pyclass]
struct PyCMAESState {
    inner: CMAESState,
    // NO rng field — deleted from v2
}

#[pymethods]
impl PyCMAESState {
    #[new]
    fn new(mean: Vec<f64>, sigma: f64, lambda: usize, bounds: Vec<(f64, f64)>) -> Self {
        PyCMAESState { inner: CMAESState::new(mean, sigma, lambda, bounds) }
    }

    // Engine passes its master_seed and its own generation counter
    fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        self.inner.ask(master_seed, generation)
    }

    fn tell(&mut self, samples: Vec<Vec<f64>>, fitnesses: Vec<f64>) {
        self.inner.tell(&samples, &fitnesses);
    }

    #[getter] fn generation(&self) -> usize { self.inner.generation }
    #[getter] fn sigma(&self) -> f64 { self.inner.sigma }
    #[getter] fn mean(&self) -> Vec<f64> { self.inner.mean.iter().cloned().collect() }
}
```

---

## 7. Python API Layer

### 7.1 Individual & Population

Unchanged from v2.

```python
@dataclass
class Individual:
    genes: list[float | int | bool]
    fitness: float | None = None
    fitness_valid: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def params(self) -> dict | None:
        return self.metadata.get("params")
```

### 7.2 GAEngine — Revised

**Zero random state on the Python side.** `_py_rng` is deleted entirely. All randomness — gene initialization, selection, crossover application, crossover values, mutation values — flows through `derive_seed` in Rust.

```python
engine = GAEngine(
    gene_space=GeneSpace([
        GeneDef("x0", "float", -5.0, 5.0),
        GeneDef("x1", "float", -5.0, 5.0),
    ]),
    population_size=200,
    generations=100,

    # Crossover
    crossover="sbx",
    crossover_prob=0.9,
    crossover_eta=2.0,
    crossover_alpha=0.5,

    # Mutation
    mutation="gaussian",
    mutation_prob=0.1,
    mutation_sigma=0.2,
    mutation_sigma_schedule="constant",
    mutation_sigma_end=0.02,

    # Selection
    selection="tournament",
    tournament_size=3,

    # Elitism
    elitism=5,

    # Parallelism
    parallel="none",         # "none" | "thread" | "process"
    n_workers=None,

    # Reproducibility — the only RNG parameter
    seed=42,

    callbacks=[],
)

result = engine.run(fitness_fn=my_fitness)
multi  = engine.run_multiple(fitness_fn=my_fitness, n_runs=10, aggregate="best")
result = engine.resume(fitness_fn=my_fitness, checkpoint="./checkpoints/checkpoint_gen_50.pkl")
```

**Revised `run()` inner loop** — Python is pure orchestration, zero random state:

```python
def run(self, fitness_fn: Callable) -> RunResult:
    start = time.perf_counter()
    logbook = Logbook()

    # Population initialization — Rust derives seed from (master, gen=0, idx, OP_INIT)
    population = _core.init_population(
        self.gene_space.to_rust_spec(),
        self.population_size,
        self.seed,                          # master seed only
    )
    fitnesses = self._evaluate(population, fitness_fn)

    for gen in range(self.generations):
        gen_start = time.perf_counter()
        current_pop = self._to_population(population, fitnesses)

        for cb in self.callbacks:
            cb.on_generation_start(gen, current_pop)

        # Single Rust call: selection + crossover + mutation + clamping + rounding
        current_sigma = self._compute_sigma(gen)
        population = _core.reproduce(
            population,
            fitnesses,
            self._reproduction_config(current_sigma),
            self.seed,                      # master seed — same every call
            gen,                            # generation index — changes each iteration
        )

        fitnesses = self._evaluate(population, fitness_fn)

        # ... logbook, callbacks, early stopping ...

    # ... build RunResult ...
```

### 7.3 `run_multiple` — Revised Seeding

`seed_offset` parameter removed. Child seeds are derived hierarchically:

```python
def run_multiple(self, fitness_fn, n_runs=10, aggregate="best"):
    results = []
    for run_idx in range(n_runs):
        # Independent child seed: different run_idx → different child seed
        # Uses same derive_seed function for consistency
        child_seed = _core.derive_seed(
            self.seed, 0, run_idx, _core.OP_MULTI_RUN
        )
        run_engine = self._copy_with_seed(int(child_seed))
        results.append(run_engine.run(fitness_fn))

    results.sort(key=lambda r: r.best_fitness, reverse=True)
    return MultiRunResult(
        best=results[0],
        all_runs=results,
        n_runs=n_runs,
        wall_time_seconds=sum(r.wall_time_seconds for r in results),
    )
```

### 7.4 CMAESEngine — Revised

```python
def run(self, fitness_fn: Callable) -> RunResult:
    state = PyCMAESState(
        self.initial_mean,
        self.sigma_abs,
        self.population_size,
        self.bounds_list,
        # NO seed parameter — engine passes it per call
    )

    for gen in range(self.generations):
        # Engine owns the generation counter and passes it each call
        samples = state.ask(self.seed, gen)
        # ...
        state.tell(samples, fitnesses)
```

### 7.5 Fitness Function Protocol

Unchanged from v2 — three supported return styles:

```python
# Style 1: simple float
def my_fitness(ind: Individual) -> float:
    return -sum(x**2 for x in ind.genes)

# Style 2: named params
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

NaN/Inf handling, `FitnessWarning`, and tuple auto-detection unchanged from v2.

### 7.6 Parallelism (`evocore/parallel.py`)

Unchanged from v2:

```python
GAEngine(parallel="thread",  n_workers=8)   # Rayon, requires GIL release
GAEngine(parallel="process", n_workers=8)   # ProcessPoolExecutor, pure-Python safe
```

The `process_initializer` pattern is unchanged.

### 7.7 Callbacks

Unchanged from v2. Built-in callbacks: `EarlyStopping`, `ProgressBar`, `CheckpointCallback`, `MetricsLogger`. The `on_fitness_warning` hook signature is unchanged.

### 7.8 RunResult & MultiRunResult

Unchanged from v2.

### 7.9 Statistics & Logbook

Unchanged from v2.

---

## 8. Error Handling

Unchanged from v2.

```python
class EvocoreError(Exception): ...
class ConfigurationError(EvocoreError): ...
class FitnessError(EvocoreError): ...
class ConvergenceError(EvocoreError): ...
class ParallelError(EvocoreError): ...
class CheckpointError(EvocoreError): ...

import warnings
class FitnessWarning(UserWarning): ...
```

---

## 9. Testing Strategy

### New Test File: `tests/unit/test_rng_reproducibility.py`

This file tests the core RNG invariant — the most important correctness property in v3.

```python
"""
RNG Reproducibility Invariants
-------------------------------
These tests encode the core v3 guarantee: all randomness flows from a single
u64 master seed through derive_seed(). Results must be identical across:
  - repeated run() calls on the same engine
  - different n_workers settings in parallel="thread" mode
  - parallel="none" vs parallel="thread"

If any of these tests fail, the RNG architecture has been violated.
"""
import numpy as np
import pytest
from evocore import GAEngine, CMAESEngine, GeneSpace, GeneDef


def numpy_fitness(ind):
    """NumPy fitness — releases GIL, safe for thread parallel mode."""
    return float(-np.sum(np.array(ind.genes) ** 2))

def sphere(ind):
    return -sum(x**2 for x in ind.genes)


def test_run_twice_same_engine_identical_results():
    """
    Calling run() twice on the same engine must give identical results.
    Would FAIL if any random state is consumed and not reset between runs.
    """
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-5.0, 5.0, 10),
        population_size=50, generations=20, seed=42,
    )
    r1 = engine.run(sphere)
    r2 = engine.run(sphere)

    assert r1.best_fitness == r2.best_fitness
    assert r1.best_individual.genes == r2.best_individual.genes
    for e1, e2 in zip(r1.logbook, r2.logbook):
        assert e1.best_fitness == e2.best_fitness


def test_sequential_and_thread_parallel_identical():
    """
    parallel="thread" must produce byte-identical results to parallel="none"
    for a fixed seed when the fitness function releases the GIL.
    """
    engine_seq = GAEngine(
        gene_space=GeneSpace.uniform(-5.0, 5.0, 10),
        population_size=50, generations=10,
        parallel="none", seed=42,
    )
    engine_par = GAEngine(
        gene_space=GeneSpace.uniform(-5.0, 5.0, 10),
        population_size=50, generations=10,
        parallel="thread", n_workers=4, seed=42,
    )

    r_seq = engine_seq.run(numpy_fitness)
    r_par = engine_par.run(numpy_fitness)

    assert r_seq.best_fitness == r_par.best_fitness
    assert r_seq.best_individual.genes == r_par.best_individual.genes
    for es, ep in zip(r_seq.logbook, r_par.logbook):
        assert es.best_fitness == ep.best_fitness


def test_n_workers_does_not_affect_results():
    """
    Thread count must not affect results — the key property Approach C guarantees
    over Approach A (thread-local RNGs) and Approach B (atomic counter).
    """
    results = {}
    for n_workers in [1, 2, 4, 8]:
        engine = GAEngine(
            gene_space=GeneSpace.uniform(-5.0, 5.0, 10),
            population_size=50, generations=10,
            parallel="thread", n_workers=n_workers, seed=99,
        )
        results[n_workers] = engine.run(numpy_fitness)

    ref = results[1].best_fitness
    for n, r in results.items():
        assert r.best_fitness == ref, (
            f"n_workers={n} produced {r.best_fitness}, expected {ref}. "
            "Thread count must not affect results (derive_seed invariant violated)."
        )


def test_different_seeds_diverge():
    """Sanity check: different seeds must produce different gene sequences."""
    e1 = GAEngine(gene_space=GeneSpace.uniform(-5.0, 5.0, 5),
                  population_size=20, generations=5, seed=1)
    e2 = GAEngine(gene_space=GeneSpace.uniform(-5.0, 5.0, 5),
                  population_size=20, generations=5, seed=2)

    r1 = e1.run(sphere)
    r2 = e2.run(sphere)
    assert r1.best_individual.genes != r2.best_individual.genes


def test_cmaes_run_idempotent():
    """CMA-ES run() called twice on same engine gives identical results."""
    engine = CMAESEngine(
        gene_space=GeneSpace.uniform(-2.0, 2.0, 5),
        population_size=10, generations=20, seed=42,
    )
    r1 = engine.run(sphere)
    r2 = engine.run(sphere)

    assert r1.best_fitness == r2.best_fitness
    assert r1.best_individual.genes == r2.best_individual.genes


def test_multi_run_seeds_are_independent():
    """
    run_multiple child seeds must not share RNG subsequences.
    Verify all n_runs produce distinct results (not all same seed).
    """
    engine = GAEngine(
        gene_space=GeneSpace.uniform(-5.0, 5.0, 5),
        population_size=20, generations=5, seed=42,
    )
    multi = engine.run_multiple(sphere, n_runs=5)

    fitness_values = [r.best_fitness for r in multi.all_runs]
    # All 5 runs should differ (probability of collision is astronomically low)
    assert len(set(fitness_values)) > 1, (
        "All run_multiple results are identical — seed derivation may be broken."
    )
```

### All Other Tests

Unchanged from v2 testing strategy. `test_rng_reproducibility.py` is additive.

---

## 10. Build System & Toolchain

```toml
# Cargo.toml [dependencies] — unchanged from v2
pyo3       = { version = "0.21", features = ["extension-module"] }
rayon      = "1.9"
rand       = "0.8"
rand_distr = "0.4"
nalgebra   = "0.32"

# pyproject.toml — version bump only
[project]
name    = "evocore"
version = "0.3.0"
```

Dev workflow unchanged:

```bash
maturin develop --release
pytest tests/unit/ tests/integration/ -v
```

---

## 11. Performance Targets

Unchanged from v2. The `reproduce()` consolidation (single Rust call per generation vs Python per-pair loop) is expected to improve the sequential GA benchmark by an additional 15–25% over v2 due to eliminated Python overhead.

| Scenario | Target vs pure-Python DEAP |
|---|---|
| GA float, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| GA float, 1000 pop, 100 gen, 8-thread Rayon | ≥ 10× faster |
| GA binary, 1000 pop, 100 gen, sequential | ≥ 4× faster |
| GA integer, 1000 pop, 100 gen, sequential | ≥ 3× faster |
| CMA-ES, dim=20, 200 gen, sequential | ≥ 5× faster |
| CMA-ES, dim=100, 500 gen, sequential | ≥ 2× vs naive (eigendecomp caching) |

---

## 12. Migration from v2

| v2 | v3 equivalent |
|---|---|
| `init_rng(seed)` call before engine | **Deleted** — no call needed |
| `PyCMAESState(mean, sigma, lambda, seed)` | `PyCMAESState(mean, sigma, lambda, bounds)` — no seed |
| `state.ask()` | `state.ask(master_seed, generation)` — engine passes both |
| `operator_fn(..., seed=rng.randint(...))` | `operator_fn(..., master_seed, generation, idx)` |
| `engine._py_rng = random.Random(seed)` | **Deleted** — no Python random state |
| `run_multiple(seed_offset=1000)` | `seed_offset` removed; use `run_multiple(n_runs=10)` |

---

## 13. Out of Scope (Phase 1)

Unchanged from v2 — Genetic Programming, multi-objective, BIPOP-CMA-ES, PyPI distribution, GPU, distributed evaluation.

---

## Open Issues (Deferred to v4)

The following design gaps were identified in review and are tracked for the next spec revision:

1. **Windows `spawn` vs `fork` for `parallel="process"`** — lambda fitness functions silently fail on Windows; `ProcessPoolExecutor` behavioral differences need explicit documentation and guard
2. **Mixed-gene Rust evaluation path** — `evaluate_parallel_rayon` takes `Vec<Vec<f64>>`; bool and integer genes need a unified crossing strategy for the PyO3 boundary
3. **CMA-ES `tell()` with rounded integer genes** — unrounded continuous samples should be passed to `tell()`; only rounded values passed to fitness function
4. **`fitness_valid` layer placement** — engine concern currently leaking into Rust struct definition
5. **`run_multiple` parallelism** — n_runs is embarrassingly parallel; currently sequential
6. **`mutation_sigma` for large integer ranges** — `sigma=0.2` on `GeneDef("ema_slow", "int", 10, 500)` produces σ=98; needs per-gene override or integer-specific default
7. **Windows Rayon stack size** — default 1MB Windows thread stack risks segfault for large CMA-ES n with nalgebra eigendecomposition
8. **`diversity_history` memory cost** — should be opt-in flag, not always-on
9. **`on_fitness_warning` callback signature** — fires mid-generation; inconsistent with generation-boundary hook design
10. **`MetricsLogger` UTF-8 encoding** — must specify `encoding="utf-8"` explicitly for Windows compatibility
