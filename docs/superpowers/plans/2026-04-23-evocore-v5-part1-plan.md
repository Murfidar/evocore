# evocore v5 — Part 1: Project Scaffold + Rust Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the complete project scaffold and compile a working `evocore._core` PyO3 extension that exposes the three individual types (`FloatIndividual`, `IntegerIndividual`, `BinaryIndividual`) and the `derive_seed` seed architecture that every downstream Rust module depends on.

**Architecture:** Two-layer design — a Rust PyO3 extension (`evocore._core`) owns all hot paths; a pure-Python package (`evocore`) owns the API. Part 1 establishes the skeleton: build toolchain, directory layout, the deterministic seed derivation function (`src/utils.rs`), the three individual types (`src/individual.rs`), the gene kind enum (`src/gene_spec.rs`), and the module root (`src/lib.rs`) with Rayon stack initialization. The Python side in Part 1 is minimal: the error hierarchy (`exceptions.py`) and a near-empty `__init__.py`.

**Tech Stack:** Rust 1.78+, PyO3 0.21, Rayon 1.9, maturin 1.5+, Python 3.11+, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `Cargo.toml` | Rust crate config + all dependencies |
| `pyproject.toml` | maturin build config + Python package metadata |
| `src/lib.rs` | PyO3 module root — Rayon stack init, class/function registration |
| `src/utils.rs` | `derive_seed()` + all `OP_*` constants |
| `src/individual.rs` | `FloatIndividual`, `IntegerIndividual`, `BinaryIndividual` |
| `src/gene_spec.rs` | `GeneKind` enum |
| `src/operators/mod.rs` | Operator submodule declaration (stub — populated in Part 2) |
| `src/selection.rs` | Stub (populated in Part 3) |
| `src/reproduce.rs` | Stub (populated in Part 3) |
| `src/cmaes.rs` | Stub (populated in Part 4) |
| `src/parallel.rs` | Stub (populated in Part 3) |
| `evocore/__init__.py` | Minimal skeleton — exports added as Parts complete |
| `evocore/exceptions.py` | Full `EvocoreError` hierarchy + `FitnessWarning` + `ConfigurationWarning` |

---

## Task 1: Directory Structure

**Files:**
- Create: full directory tree

- [ ] **Step 1: Create all directories**

```bash
mkdir -p src/operators \
         evocore \
         tests/unit \
         tests/integration \
         tests/benchmarks \
         examples \
         docs/superpowers/specs \
         docs/superpowers/plans

touch tests/__init__.py \
      tests/unit/__init__.py \
      tests/integration/__init__.py \
      tests/benchmarks/__init__.py
```

- [ ] **Step 2: Verify structure**

```bash
find . -type d | sort
```

Expected output includes:
```
./docs/superpowers/specs
./evocore
./examples
./src/operators
./tests/benchmarks
./tests/integration
./tests/unit
```

---

## Task 2: Build Configuration

**Files:**
- Create: `Cargo.toml`
- Create: `pyproject.toml`

- [ ] **Step 1: Write Cargo.toml**

```toml
[package]
name = "evocore"
version = "0.5.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]

[dependencies]
pyo3       = { version = "0.21", features = ["extension-module"] }
rayon      = "1.9"
rand       = "0.8"
rand_distr = "0.4"
nalgebra   = "0.32"

[profile.release]
lto           = true
codegen-units = 1
opt-level     = 3
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[build-system]
requires      = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name            = "evocore"
version         = "0.5.0"
requires-python = ">=3.11"
description     = "Rust-native Genetic Algorithms and CMA-ES for Python"

[tool.maturin]
python-source = "."
module-name   = "evocore._core"
features      = ["pyo3/extension-module"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Install build toolchain**

```bash
pip install maturin pytest
```

---

## Task 3: Rust Stub Files

Create placeholder stubs for modules that will be fleshed out in Parts 2–4. This lets `src/lib.rs` declare all `mod` statements without compile errors now.

**Files:**
- Create: `src/operators/mod.rs`
- Create: `src/selection.rs`
- Create: `src/reproduce.rs`
- Create: `src/cmaes.rs`
- Create: `src/parallel.rs`

- [ ] **Step 1: Write src/operators/mod.rs**

```rust
// Populated in Part 2
pub mod float_ops;
pub mod int_ops;
pub mod binary_ops;
```

- [ ] **Step 2: Write src/operators/float_ops.rs stub**

```rust
// Populated in Part 2
```

- [ ] **Step 3: Write src/operators/int_ops.rs stub**

```rust
// Populated in Part 2
```

- [ ] **Step 4: Write src/operators/binary_ops.rs stub**

```rust
// Populated in Part 2
```

- [ ] **Step 5: Write src/selection.rs stub**

```rust
// Populated in Part 3
```

- [ ] **Step 6: Write src/reproduce.rs stub**

```rust
// Populated in Part 3
```

- [ ] **Step 7: Write src/cmaes.rs stub**

```rust
// Populated in Part 4
```

- [ ] **Step 8: Write src/parallel.rs stub**

```rust
// Populated in Part 3
```

---

## Task 4: `src/utils.rs` — Seed Architecture

The `derive_seed` function is the single most important piece in the entire codebase. Every random decision in the library — gene initialization, crossover, mutation, selection, CMA-ES sampling — derives its RNG seed from this one pure function. Getting it right here means all downstream randomness is deterministic, thread-count-independent, and idempotent.

**Files:**
- Create: `src/utils.rs`

- [ ] **Step 1: Write the failing Rust tests first**

Create `src/utils.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Sanity: same inputs always produce the same output
    #[test]
    fn test_derive_seed_deterministic() {
        let a = derive_seed(42, 1, 0, OP_CROSSOVER);
        let b = derive_seed(42, 1, 0, OP_CROSSOVER);
        assert_eq!(a, b);
    }

    // Different master seeds must produce different outputs
    #[test]
    fn test_derive_seed_different_masters_diverge() {
        let a = derive_seed(1, 0, 0, OP_INIT);
        let b = derive_seed(2, 0, 0, OP_INIT);
        assert_ne!(a, b);
    }

    // Different generations must produce different outputs
    #[test]
    fn test_derive_seed_different_generations_diverge() {
        let a = derive_seed(42, 0, 0, OP_MUTATION);
        let b = derive_seed(42, 1, 0, OP_MUTATION);
        assert_ne!(a, b);
    }

    // Different individual indices must produce different outputs
    #[test]
    fn test_derive_seed_different_indices_diverge() {
        let a = derive_seed(42, 0, 0, OP_MUTATION);
        let b = derive_seed(42, 0, 1, OP_MUTATION);
        assert_ne!(a, b);
    }

    // Different operation types must produce different outputs
    #[test]
    fn test_derive_seed_different_ops_diverge() {
        let a = derive_seed(42, 0, 0, OP_CROSSOVER);
        let b = derive_seed(42, 0, 0, OP_MUTATION);
        assert_ne!(a, b);
    }

    // Commutativity guard: derive_seed(m,1,2,op) != derive_seed(m,2,1,op)
    // The multipliers for generation and individual_idx must differ so that
    // swapping these two arguments changes the output.
    #[test]
    fn test_derive_seed_not_commutative_gen_idx() {
        let a = derive_seed(99, 1, 2, OP_SELECTION);
        let b = derive_seed(99, 2, 1, OP_SELECTION);
        assert_ne!(a, b, "derive_seed must not be commutative across generation and individual_idx");
    }

    // Avalanche property: a 1-bit change in master should flip many output bits.
    // We check that at least 16 of 64 bits differ (expected ~32).
    #[test]
    fn test_derive_seed_avalanche_on_master() {
        let base  = derive_seed(0b0000_0000_0000_0000u64, 5, 3, OP_CROSSOVER);
        let flipped = derive_seed(0b0000_0000_0000_0001u64, 5, 3, OP_CROSSOVER);
        let bits_changed = (base ^ flipped).count_ones();
        assert!(
            bits_changed >= 16,
            "Expected ≥16 bits to flip (avalanche), got {}",
            bits_changed
        );
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL (nothing defined yet)**

```bash
cargo test utils 2>&1 | head -20
```

Expected: `error[E0425]: cannot find function 'derive_seed'`

- [ ] **Step 3: Implement src/utils.rs**

```rust
use pyo3::prelude::*;

// ── Operation-type constants ──────────────────────────────────────────────────
// One constant per random-decision category. Values must never be reused.
// Adding a new operator: append the next u64 — never reuse or reorder existing.
pub const OP_INIT:           u64 = 0;  // initial population gene values
pub const OP_CROSSOVER:      u64 = 1;  // gene values produced by crossover
pub const OP_MUTATION:       u64 = 2;  // gene values produced by mutation
pub const OP_SELECTION:      u64 = 3;  // selection indices
pub const OP_CMAES_ASK:      u64 = 4;  // CMA-ES sample generation
pub const OP_MULTI_RUN:      u64 = 5;  // child seed derivation for run_multiple
pub const OP_CROSSOVER_PROB: u64 = 6;  // whether to apply crossover to a pair

// ── Seed derivation ───────────────────────────────────────────────────────────
/// Derive a deterministic u64 seed from four inputs using SplitMix64 mixing.
///
/// Every random decision in evocore is derived from these four inputs:
///   - `master`         — the user-supplied engine seed (e.g. 42)
///   - `generation`     — the current generation index (0-based)
///   - `individual_idx` — index of the individual being operated on (0 for selection)
///   - `op`             — one of the `OP_*` constants above
///
/// Properties:
///   - Pure function: same inputs → same output, always.
///   - Thread-safe: no shared state. Any thread can call this independently.
///   - Avalanche: a 1-bit change in any input flips ~32 output bits.
///   - Non-commutative: different multipliers per field prevent
///     derive_seed(m,1,2,op) == derive_seed(m,2,1,op).
///
/// Cost: ~5 ns per call — negligible against any fitness evaluation.
pub fn derive_seed(master: u64, generation: u64, individual_idx: u64, op: u64) -> u64 {
    // Mix all four inputs with distinct prime multipliers so that no two
    // fields are interchangeable.
    let mut x = master
        .wrapping_add(generation.wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add(individual_idx.wrapping_mul(0x6c62272e07bb0142))
        .wrapping_add(op.wrapping_mul(0xd2b74407b1ce6d93));

    // SplitMix64 finaliser — two rounds of xor-shift-multiply for full avalanche
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// ── Python exposure ───────────────────────────────────────────────────────────
/// Expose derive_seed to Python for use in test_rng_reproducibility.py
/// and for run_multiple child-seed derivation.
#[pyfunction]
pub fn py_derive_seed(master: u64, generation: u64, individual_idx: u64, op: u64) -> u64 {
    derive_seed(master, generation, individual_idx, op)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        let a = derive_seed(42, 1, 0, OP_CROSSOVER);
        let b = derive_seed(42, 1, 0, OP_CROSSOVER);
        assert_eq!(a, b);
    }

    #[test]
    fn test_derive_seed_different_masters_diverge() {
        let a = derive_seed(1, 0, 0, OP_INIT);
        let b = derive_seed(2, 0, 0, OP_INIT);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_different_generations_diverge() {
        let a = derive_seed(42, 0, 0, OP_MUTATION);
        let b = derive_seed(42, 1, 0, OP_MUTATION);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_different_indices_diverge() {
        let a = derive_seed(42, 0, 0, OP_MUTATION);
        let b = derive_seed(42, 0, 1, OP_MUTATION);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_different_ops_diverge() {
        let a = derive_seed(42, 0, 0, OP_CROSSOVER);
        let b = derive_seed(42, 0, 0, OP_MUTATION);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_not_commutative_gen_idx() {
        let a = derive_seed(99, 1, 2, OP_SELECTION);
        let b = derive_seed(99, 2, 1, OP_SELECTION);
        assert_ne!(a, b);
    }

    #[test]
    fn test_derive_seed_avalanche_on_master() {
        let base    = derive_seed(0u64, 5, 3, OP_CROSSOVER);
        let flipped = derive_seed(1u64, 5, 3, OP_CROSSOVER);
        let bits_changed = (base ^ flipped).count_ones();
        assert!(bits_changed >= 16, "Expected ≥16 bits to flip, got {}", bits_changed);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test utils
```

Expected:
```
test utils::tests::test_derive_seed_avalanche_on_master ... ok
test utils::tests::test_derive_seed_deterministic ... ok
test utils::tests::test_derive_seed_different_generations_diverge ... ok
test utils::tests::test_derive_seed_different_indices_diverge ... ok
test utils::tests::test_derive_seed_different_masters_diverge ... ok
test utils::tests::test_derive_seed_different_ops_diverge ... ok
test utils::tests::test_derive_seed_not_commutative_gen_idx ... ok

test result: ok. 7 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/utils.rs
git commit -m "feat(rust): derive_seed() with SplitMix64 mixing and OP_* constants"
```

---

## Task 5: `src/gene_spec.rs` — GeneKind Enum

**Files:**
- Create: `src/gene_spec.rs`

- [ ] **Step 1: Write the failing tests**

Create `src/gene_spec.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_kind_clone() {
        let k = GeneKind::Float;
        let k2 = k.clone();
        assert!(matches!(k2, GeneKind::Float));
    }

    #[test]
    fn test_gene_kind_all_variants_distinct() {
        let variants = [GeneKind::Float, GeneKind::Int, GeneKind::Bool];
        // Every pair must be distinguishable via pattern matching
        assert!(!matches!(variants[0], GeneKind::Int));
        assert!(!matches!(variants[1], GeneKind::Bool));
        assert!(!matches!(variants[2], GeneKind::Float));
    }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test gene_spec 2>&1 | head -10
```

Expected: `error[E0422]: cannot find struct, variant or union type 'GeneKind'`

- [ ] **Step 3: Implement src/gene_spec.rs**

```rust
/// The kind of value a gene can take.
/// Used by the operator layer to dispatch the correct crossover/mutation function
/// and by the PyO3 boundary layer to encode/decode between Python and Rust types.
#[derive(Clone, Debug, PartialEq)]
pub enum GeneKind {
    /// Continuous floating-point value, stored as f64.
    Float,
    /// Integer value, stored as i64 in Rust; encoded as f64 at the PyO3 boundary.
    Int,
    /// Boolean value; encoded as 0.0 (false) / 1.0 (true) at the PyO3 boundary.
    Bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_kind_clone() {
        let k = GeneKind::Float;
        let k2 = k.clone();
        assert!(matches!(k2, GeneKind::Float));
    }

    #[test]
    fn test_gene_kind_all_variants_distinct() {
        let variants = [GeneKind::Float, GeneKind::Int, GeneKind::Bool];
        assert!(!matches!(variants[0], GeneKind::Int));
        assert!(!matches!(variants[1], GeneKind::Bool));
        assert!(!matches!(variants[2], GeneKind::Float));
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test gene_spec
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/gene_spec.rs
git commit -m "feat(rust): GeneKind enum (Float, Int, Bool)"
```

---

## Task 6: `src/individual.rs` — Three Individual Types

All three types expose `genes`, `fitness`, `__repr__`, and `__len__` to Python. `fitness_valid` is intentionally absent — in v5 it is an engine-level Python concern, not a data concern.

**Files:**
- Create: `src/individual.rs`

- [ ] **Step 1: Write the failing Rust tests**

Create `src/individual.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // FloatIndividual
    #[test]
    fn test_float_individual_len() {
        let ind = FloatIndividual { genes: vec![1.0, 2.0, 3.0], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_float_individual_fitness_none_default() {
        let ind = FloatIndividual { genes: vec![0.0], fitness: None };
        assert!(ind.fitness.is_none());
    }

    #[test]
    fn test_float_individual_clone_preserves_fitness() {
        let ind = FloatIndividual { genes: vec![1.0, 2.0], fitness: Some(3.14) };
        let cloned = ind.clone();
        assert_eq!(cloned.fitness, Some(3.14));
        assert_eq!(cloned.genes, vec![1.0, 2.0]);
    }

    // IntegerIndividual
    #[test]
    fn test_integer_individual_len() {
        let ind = IntegerIndividual { genes: vec![10, 20, 30], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_integer_individual_clone() {
        let ind = IntegerIndividual { genes: vec![5, -3], fitness: Some(1.0) };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![5, -3]);
        assert_eq!(cloned.fitness, Some(1.0));
    }

    // BinaryIndividual
    #[test]
    fn test_binary_individual_len() {
        let ind = BinaryIndividual { genes: vec![true, false, true], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_binary_individual_clone() {
        let ind = BinaryIndividual { genes: vec![true, false], fitness: Some(2.0) };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![true, false]);
    }

    // No fitness_valid field on any struct (v4/v5 requirement)
    // The following must compile — if fitness_valid existed it would be a field access
    #[test]
    fn test_float_individual_has_no_fitness_valid_field() {
        let ind = FloatIndividual { genes: vec![1.0], fitness: None };
        // Accessing ind.fitness_valid here would be a compile error — the test
        // passes by compiling at all. The Python layer owns fitness_valid.
        let _ = ind.fitness;
    }
}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
cargo test individual 2>&1 | head -10
```

Expected: `error[E0422]: cannot find struct, variant or union type 'FloatIndividual'`

- [ ] **Step 3: Implement src/individual.rs**

```rust
use pyo3::prelude::*;

// ── FloatIndividual ───────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct FloatIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<f64>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl FloatIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<f64>, fitness: Option<f64>) -> Self {
        FloatIndividual { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "FloatIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

// ── IntegerIndividual ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct IntegerIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<i64>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl IntegerIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<i64>, fitness: Option<f64>) -> Self {
        IntegerIndividual { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "IntegerIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

// ── BinaryIndividual ──────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone, Debug)]
pub struct BinaryIndividual {
    #[pyo3(get, set)]
    pub genes: Vec<bool>,
    #[pyo3(get, set)]
    pub fitness: Option<f64>,
}

#[pymethods]
impl BinaryIndividual {
    #[new]
    #[pyo3(signature = (genes, fitness=None))]
    pub fn new(genes: Vec<bool>, fitness: Option<f64>) -> Self {
        BinaryIndividual { genes, fitness }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BinaryIndividual(genes={:?}, fitness={:?})",
            self.genes, self.fitness
        )
    }

    pub fn __len__(&self) -> usize {
        self.genes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_individual_len() {
        let ind = FloatIndividual { genes: vec![1.0, 2.0, 3.0], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_float_individual_fitness_none_default() {
        let ind = FloatIndividual { genes: vec![0.0], fitness: None };
        assert!(ind.fitness.is_none());
    }

    #[test]
    fn test_float_individual_clone_preserves_fitness() {
        let ind = FloatIndividual { genes: vec![1.0, 2.0], fitness: Some(3.14) };
        let cloned = ind.clone();
        assert_eq!(cloned.fitness, Some(3.14));
        assert_eq!(cloned.genes, vec![1.0, 2.0]);
    }

    #[test]
    fn test_integer_individual_len() {
        let ind = IntegerIndividual { genes: vec![10, 20, 30], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_integer_individual_clone() {
        let ind = IntegerIndividual { genes: vec![5, -3], fitness: Some(1.0) };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![5, -3]);
        assert_eq!(cloned.fitness, Some(1.0));
    }

    #[test]
    fn test_binary_individual_len() {
        let ind = BinaryIndividual { genes: vec![true, false, true], fitness: None };
        assert_eq!(ind.genes.len(), 3);
    }

    #[test]
    fn test_binary_individual_clone() {
        let ind = BinaryIndividual { genes: vec![true, false], fitness: Some(2.0) };
        let cloned = ind.clone();
        assert_eq!(cloned.genes, vec![true, false]);
    }

    #[test]
    fn test_float_individual_has_no_fitness_valid_field() {
        let ind = FloatIndividual { genes: vec![1.0], fitness: None };
        let _ = ind.fitness;
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test individual
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/individual.rs
git commit -m "feat(rust): FloatIndividual, IntegerIndividual, BinaryIndividual (no fitness_valid)"
```

---

## Task 7: `src/lib.rs` — Module Root

Wire everything together: Rayon 8 MB stack initialization, all module declarations, all class/function registrations.

**Files:**
- Create: `src/lib.rs`

- [ ] **Step 1: Write the failing test**

```bash
# We'll verify the module loads from Python — write the smoke test first
cat > /tmp/test_import.py << 'EOF'
# This must pass after maturin develop
from evocore._core import (
    FloatIndividual,
    IntegerIndividual,
    BinaryIndividual,
    py_derive_seed,
    OP_INIT,
    OP_CROSSOVER,
    OP_MUTATION,
    OP_SELECTION,
    OP_CMAES_ASK,
    OP_MULTI_RUN,
    OP_CROSSOVER_PROB,
)
print("imports ok")
EOF
python /tmp/test_import.py 2>&1 | head -5
```

Expected: `ModuleNotFoundError` (lib.rs not written yet).

- [ ] **Step 2: Implement src/lib.rs**

```rust
use pyo3::prelude::*;

mod gene_spec;
mod individual;
mod operators;
mod selection;
mod reproduce;
mod cmaes;
mod parallel;
pub mod utils;

use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use utils::{
    py_derive_seed,
    OP_CMAES_ASK, OP_CROSSOVER, OP_CROSSOVER_PROB,
    OP_INIT, OP_MULTI_RUN, OP_MUTATION, OP_SELECTION,
};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ── Rayon thread pool — 8 MB stack on all platforms ──────────────────────
    // The default Windows stack is 1 MB, which can overflow during nalgebra's
    // eigendecomposition of large CMA-ES covariance matrices. Setting 8 MB
    // universally matches the Linux/macOS default and prevents silent crashes.
    // .ok() silently ignores the error if the pool is already initialized
    // (e.g., during hot-reload in a Jupyter notebook).
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    // ── Individual types ──────────────────────────────────────────────────────
    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;

    // ── Seed architecture ─────────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(py_derive_seed, m)?)?;

    // OP_* constants — exposed so Python test_rng_reproducibility.py can import them
    m.add("OP_INIT",           OP_INIT)?;
    m.add("OP_CROSSOVER",      OP_CROSSOVER)?;
    m.add("OP_MUTATION",       OP_MUTATION)?;
    m.add("OP_SELECTION",      OP_SELECTION)?;
    m.add("OP_CMAES_ASK",      OP_CMAES_ASK)?;
    m.add("OP_MULTI_RUN",      OP_MULTI_RUN)?;
    m.add("OP_CROSSOVER_PROB", OP_CROSSOVER_PROB)?;

    // Operators, selection, reproduce, cmaes, parallel registered in Parts 2–4

    Ok(())
}
```

- [ ] **Step 3: Compile**

```bash
maturin develop --release
```

Expected: compilation succeeds with no errors or warnings about unused stubs.

- [ ] **Step 4: Run the smoke test**

```bash
python /tmp/test_import.py
```

Expected:
```
imports ok
```

- [ ] **Step 5: Verify individuals from Python**

```bash
python - << 'EOF'
from evocore._core import FloatIndividual, IntegerIndividual, BinaryIndividual

f = FloatIndividual([1.0, 2.0, 3.0])
assert len(f) == 3
assert f.fitness is None
f.fitness = 5.5
assert f.fitness == 5.5
print(f)

i = IntegerIndividual([10, 20, 30], 7.0)
assert len(i) == 3
assert i.fitness == 7.0
print(i)

b = BinaryIndividual([True, False, True])
assert len(b) == 3
assert b.fitness is None
print(b)

print("individual round-trip ok")
EOF
```

Expected:
```
FloatIndividual(genes=[1.0, 2.0, 3.0], fitness=Some(5.5))
IntegerIndividual(genes=[10, 20, 30], fitness=Some(7.0))
BinaryIndividual(genes=[true, false, true], fitness=None)
individual round-trip ok
```

- [ ] **Step 6: Verify derive_seed from Python**

```bash
python - << 'EOF'
from evocore._core import py_derive_seed, OP_INIT, OP_CROSSOVER, OP_MUTATION

# determinism
a = py_derive_seed(42, 1, 0, OP_CROSSOVER)
b = py_derive_seed(42, 1, 0, OP_CROSSOVER)
assert a == b, "derive_seed must be deterministic"

# different inputs diverge
c = py_derive_seed(42, 2, 0, OP_CROSSOVER)
assert a != c, "different generation must produce different seed"

d = py_derive_seed(42, 1, 0, OP_MUTATION)
assert a != d, "different op must produce different seed"

# OP_* constants are the right type and distinct
ops = [OP_INIT, OP_CROSSOVER, OP_MUTATION]
assert len(set(ops)) == 3, "OP_* constants must be distinct"

print("derive_seed Python API ok")
EOF
```

Expected:
```
derive_seed Python API ok
```

- [ ] **Step 7: Run all Rust tests**

```bash
cargo test
```

Expected: All 17 Rust tests pass (7 utils + 2 gene_spec + 8 individual).

- [ ] **Step 8: Commit**

```bash
git add src/lib.rs
git commit -m "feat(rust): lib.rs module root — Rayon 8MB stack, individuals + derive_seed exposed"
```

---

## Task 8: Python Exception Hierarchy

**Files:**
- Create: `evocore/exceptions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_exceptions.py`:

```python
import pytest
import warnings
from evocore.exceptions import (
    EvocoreError,
    ConfigurationError,
    FitnessError,
    ConvergenceError,
    ParallelError,
    CheckpointError,
    FitnessWarning,
    ConfigurationWarning,
)


def test_all_errors_subclass_evocore_error():
    for cls in [ConfigurationError, FitnessError, ConvergenceError,
                ParallelError, CheckpointError]:
        assert issubclass(cls, EvocoreError), f"{cls.__name__} must subclass EvocoreError"


def test_evocore_error_subclasses_exception():
    assert issubclass(EvocoreError, Exception)


def test_fitness_warning_subclasses_user_warning():
    assert issubclass(FitnessWarning, UserWarning)


def test_configuration_warning_subclasses_user_warning():
    assert issubclass(ConfigurationWarning, UserWarning)


def test_errors_are_distinct():
    classes = [ConfigurationError, FitnessError, ConvergenceError,
               ParallelError, CheckpointError]
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert not issubclass(a, b), f"{a.__name__} must not subclass {b.__name__}"
            assert not issubclass(b, a), f"{b.__name__} must not subclass {a.__name__}"


def test_configuration_error_carries_message():
    with pytest.raises(ConfigurationError, match="gene_bounds"):
        raise ConfigurationError("gene_bounds required for individual_type='float'.")


def test_fitness_warning_can_be_promoted_to_error():
    """Users can promote FitnessWarning to an exception via warnings.filterwarnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FitnessWarning)
        with pytest.raises(FitnessWarning):
            warnings.warn("8 individuals returned NaN fitness.", FitnessWarning)


def test_configuration_warning_can_be_promoted_to_error():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConfigurationWarning)
        with pytest.raises(ConfigurationWarning):
            warnings.warn("Large int range without sigma.", ConfigurationWarning)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_exceptions.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'evocore.exceptions'`

- [ ] **Step 3: Implement evocore/exceptions.py**

```python
"""
evocore exception and warning hierarchy.

All exceptions subclass EvocoreError so callers can catch the entire library
with a single `except EvocoreError` if they want broad coverage, while still
being able to handle specific cases individually.

FitnessWarning and ConfigurationWarning are UserWarning subclasses so they
surface in normal Python warning machinery and can be promoted to exceptions
via `warnings.filterwarnings("error", category=FitnessWarning)`.
"""


class EvocoreError(Exception):
    """Base class for all evocore exceptions."""


class ConfigurationError(EvocoreError):
    """Raised when engine or GeneSpace configuration is invalid.

    Examples:
        - gene_space not provided
        - crossover operator incompatible with gene types
        - parallel='process' requested for CMAESEngine
        - fitness_fn is not picklable when parallel='process' is set
    """


class FitnessError(EvocoreError):
    """Raised when the fitness function raises an unexpected exception or returns a wrong type.

    Note: NaN/Inf returns are NOT a FitnessError — they trigger a FitnessWarning
    and are handled gracefully by assigning -inf for selection purposes.
    """


class ConvergenceError(EvocoreError):
    """Raised when a numerical failure makes continuation impossible.

    Example: CMA-ES covariance matrix becomes non-positive-definite.
    """


class ParallelError(EvocoreError):
    """Raised when a parallel worker pool encounters an unrecoverable failure."""


class CheckpointError(EvocoreError):
    """Raised when a checkpoint file is missing, corrupt, or incompatible.

    Example message includes a list of available checkpoint files in the same directory.
    """


# ── Warnings ──────────────────────────────────────────────────────────────────

class FitnessWarning(UserWarning):
    """Emitted (at most once per run) when NaN or Inf fitness values are encountered.

    Affected individuals are assigned fitness = -inf for selection.
    To promote to an exception:
        import warnings
        warnings.filterwarnings("error", category=evocore.FitnessWarning)
    """


class ConfigurationWarning(UserWarning):
    """Emitted when a configuration is valid but likely unintended.

    Example: a large-range integer GeneDef without an explicit per-gene sigma,
    which may produce mutation steps so large that fine-tuning is impossible.
    """
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_exceptions.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evocore/exceptions.py tests/unit/test_exceptions.py
git commit -m "feat(python): EvocoreError hierarchy + FitnessWarning + ConfigurationWarning"
```

---

## Task 9: `evocore/__init__.py` — Minimal Skeleton

**Files:**
- Create: `evocore/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_package_init.py`:

```python
def test_evocore_imports_without_error():
    import evocore  # noqa: F401


def test_exceptions_accessible_from_top_level():
    from evocore import (
        EvocoreError,
        ConfigurationError,
        FitnessError,
        ConvergenceError,
        ParallelError,
        CheckpointError,
        FitnessWarning,
        ConfigurationWarning,
    )
    assert issubclass(ConfigurationError, EvocoreError)


def test_core_extension_accessible():
    from evocore import _core
    assert hasattr(_core, "FloatIndividual")
    assert hasattr(_core, "IntegerIndividual")
    assert hasattr(_core, "BinaryIndividual")
    assert hasattr(_core, "py_derive_seed")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/unit/test_package_init.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'evocore'` (no `__init__.py` yet).

- [ ] **Step 3: Implement evocore/__init__.py**

```python
"""
evocore — Rust-native Genetic Algorithms and CMA-ES for Python.

Public API is populated incrementally as Parts complete:
  Part 1 (this file): exceptions, _core extension
  Part 5: GeneSpace, GeneDef, Individual, Population
  Part 6: GAEngine, RunResult, MultiRunResult
  Part 7: CMAESEngine

Import the _core extension directly only in tests or internal modules.
User-facing code should import from the top-level evocore package.
"""

from evocore._core import (  # noqa: F401  — re-export extension module
    FloatIndividual,
    IntegerIndividual,
    BinaryIndividual,
    py_derive_seed,
    OP_INIT,
    OP_CROSSOVER,
    OP_MUTATION,
    OP_SELECTION,
    OP_CMAES_ASK,
    OP_MULTI_RUN,
    OP_CROSSOVER_PROB,
)

from evocore.exceptions import (  # noqa: F401
    EvocoreError,
    ConfigurationError,
    FitnessError,
    ConvergenceError,
    ParallelError,
    CheckpointError,
    FitnessWarning,
    ConfigurationWarning,
)

# Parts 5–7 will append:
# from evocore.gene_space import GeneDef, GeneSpace
# from evocore.individual import Individual, Population
# from evocore.ga import GAEngine, RunResult, MultiRunResult
# from evocore.cmaes import CMAESEngine

__all__ = [
    # Rust types
    "FloatIndividual",
    "IntegerIndividual",
    "BinaryIndividual",
    # Seed architecture
    "py_derive_seed",
    "OP_INIT", "OP_CROSSOVER", "OP_MUTATION", "OP_SELECTION",
    "OP_CMAES_ASK", "OP_MULTI_RUN", "OP_CROSSOVER_PROB",
    # Exceptions
    "EvocoreError", "ConfigurationError", "FitnessError",
    "ConvergenceError", "ParallelError", "CheckpointError",
    # Warnings
    "FitnessWarning", "ConfigurationWarning",
]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_package_init.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add evocore/__init__.py tests/unit/test_package_init.py
git commit -m "feat(python): evocore/__init__.py skeleton with exceptions + _core re-exports"
```

---

## Task 10: Full Part 1 Verification

Run all tests and do a final end-to-end smoke test to confirm Part 1 is complete.

**Files:** No new files.

- [ ] **Step 1: Run all Rust tests**

```bash
cargo test
```

Expected:
```
test gene_spec::tests::test_gene_kind_all_variants_distinct ... ok
test gene_spec::tests::test_gene_kind_clone ... ok
test individual::tests::test_binary_individual_clone ... ok
test individual::tests::test_binary_individual_len ... ok
test individual::tests::test_float_individual_clone_preserves_fitness ... ok
test individual::tests::test_float_individual_fitness_none_default ... ok
test individual::tests::test_float_individual_has_no_fitness_valid_field ... ok
test individual::tests::test_float_individual_len ... ok
test individual::tests::test_integer_individual_clone ... ok
test individual::tests::test_integer_individual_len ... ok
test utils::tests::test_derive_seed_avalanche_on_master ... ok
test utils::tests::test_derive_seed_deterministic ... ok
test utils::tests::test_derive_seed_different_generations_diverge ... ok
test utils::tests::test_derive_seed_different_indices_diverge ... ok
test utils::tests::test_derive_seed_different_masters_diverge ... ok
test utils::tests::test_derive_seed_different_ops_diverge ... ok
test utils::tests::test_derive_seed_not_commutative_gen_idx ... ok

test result: ok. 17 passed; 0 failed
```

- [ ] **Step 2: Run all Python unit tests**

```bash
pytest tests/unit/ -v
```

Expected:
```
tests/unit/test_exceptions.py::test_all_errors_subclass_evocore_error PASSED
tests/unit/test_exceptions.py::test_evocore_error_subclasses_exception PASSED
tests/unit/test_exceptions.py::test_fitness_warning_subclasses_user_warning PASSED
tests/unit/test_exceptions.py::test_configuration_warning_subclasses_user_warning PASSED
tests/unit/test_exceptions.py::test_errors_are_distinct PASSED
tests/unit/test_exceptions.py::test_configuration_error_carries_message PASSED
tests/unit/test_exceptions.py::test_fitness_warning_can_be_promoted_to_error PASSED
tests/unit/test_exceptions.py::test_configuration_warning_can_be_promoted_to_error PASSED
tests/unit/test_package_init.py::test_evocore_imports_without_error PASSED
tests/unit/test_package_init.py::test_exceptions_accessible_from_top_level PASSED
tests/unit/test_package_init.py::test_core_extension_accessible PASSED

11 passed, 0 failed
```

- [ ] **Step 3: Final end-to-end smoke test**

```bash
python - << 'EOF'
import evocore
from evocore import (
    FloatIndividual, IntegerIndividual, BinaryIndividual,
    py_derive_seed, OP_CROSSOVER, OP_MUTATION,
    ConfigurationError, FitnessWarning,
)

# Individuals work
f = FloatIndividual([1.0, 2.0])
i = IntegerIndividual([5, 10])
b = BinaryIndividual([True, False])
assert len(f) == 2 and len(i) == 2 and len(b) == 2

# derive_seed is deterministic
s1 = py_derive_seed(42, 0, 0, OP_CROSSOVER)
s2 = py_derive_seed(42, 0, 0, OP_CROSSOVER)
assert s1 == s2

# different ops diverge
s3 = py_derive_seed(42, 0, 0, OP_MUTATION)
assert s1 != s3

# exceptions are accessible
try:
    raise ConfigurationError("test")
except ConfigurationError:
    pass

print("Part 1 complete — all checks passed")
EOF
```

Expected:
```
Part 1 complete — all checks passed
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: Part 1 complete — scaffold, derive_seed, individuals, exceptions"
git tag part1-complete
```

---

## Part 1 Exit Criteria Checklist

- [ ] `maturin develop --release` succeeds with zero errors
- [ ] `cargo test` passes 17 Rust tests
- [ ] `pytest tests/unit/` passes 11 Python tests
- [ ] `from evocore._core import FloatIndividual, IntegerIndividual, BinaryIndividual` works
- [ ] `from evocore._core import py_derive_seed, OP_INIT, OP_CROSSOVER` works
- [ ] `derive_seed` is deterministic, avalanche-compliant, and non-commutative across `generation`/`individual_idx`
- [ ] None of the three individual structs has a `fitness_valid` field
- [ ] All stub files (`selection.rs`, `reproduce.rs`, `cmaes.rs`, `parallel.rs`, `operators/*.rs`) compile without errors
- [ ] `evocore` package is importable and exposes exceptions at the top level
