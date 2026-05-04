# evocore v5 — Part 4: Rust CMA-ES Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/cmaes.rs` — the full Covariance Matrix Adaptation Evolution Strategy engine with eigendecomposition caching, mirror-folding boundary correction, and a stateless `ask(master_seed, generation)` interface — then expose `PyCMAESState` to Python via PyO3.

**Architecture:** `CMAESState` holds all mutable algorithmic state (mean, covariance, evolution paths, sigma) but stores **no RNG** — the engine passes `(master_seed, generation)` to `ask()` each call, and `derive_seed(master_seed, generation, sample_idx, OP_CMAES_ASK)` produces a fresh stack-local `StdRng` per sample. Eigendecomposition is cached inside `RefCell<EigenCache>` + `Cell<usize>` so `ask()` can remain `&self` while still lazily recomputing when the cache is stale. Boundary correction uses mirror-folding (not clipping) to preserve the sampling distribution shape near boundaries. The `tell()` method implements the full Hansen CMA-ES rank-1 + rank-μ covariance update with σ-CSA step-size control.

**Tech Stack:** Rust 1.78+, PyO3 0.21, nalgebra 0.32, rand 0.8, rand_distr 0.4, maturin 1.5+, Python 3.11+, pytest

**Prerequisite:** Parts 1–3 complete — `src/utils.rs`, `src/individual.rs`, all operator files, `src/selection.rs`, `src/reproduce.rs`, `src/parallel.rs`, and `src/lib.rs` compile and pass their tests (97 Rust tests, all Python tests green).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/cmaes.rs` | Replace stub | `mirror_fold`, `EigenCache`, `CMAESState`, `PyCMAESState` with `ask` / `tell` |
| `src/lib.rs` | Modify | Register `PyCMAESState` class in `_core` module |
| `tests/unit/test_cmaes_rust.py` | Create | Python smoke tests for `PyCMAESState` — shape, determinism, convergence signal |

---

## CMA-ES Mathematics Reference

The `tell()` update follows the Hansen (2016) reference implementation. Variables use the standard naming:

| Symbol | Meaning |
|---|---|
| `n` | Problem dimension (gene count) |
| `λ` (`lambda`) | Population size (samples per generation) |
| `μ` (`mu`) | Number of recombination parents (`lambda / 2`) |
| `wᵢ` | Recombination weights for top-μ individuals |
| `μ_eff` (`mueff`) | Effective number of recombinants = `1 / Σwᵢ²` |
| `cc` | Time constant for cumulation path `pc` |
| `cs` | Time constant for step-size path `ps` |
| `c1` | Rank-1 learning rate |
| `cμ` (`cmu`) | Rank-μ learning rate |
| `damps` | Step-size damping coefficient |
| `χN` (`chiN`) | Expected norm of a standard N(0,I) sample = `√n · (1 - 1/(4n) + 1/(21n²))` |

**hsig (Heaviside indicator):** Signals whether step-size evolution path `ps` is too large. Prevents large sigma from dominating the rank-1 covariance update during initial phases.

```
hsig = 1   if  ||ps|| / (χN · √(1 - (1-cs)^(2·(gen+1))))  <  1.4 + 2/(n+1)
hsig = 0   otherwise
```

**`tell()` update sequence (applied in this order):**

1. Sort samples by fitness descending.
2. New mean ← `Σᵢ wᵢ · x_{ranked[i]}` for i in 0..mu.
3. `mean_diff` ← `(new_mean - old_mean) / sigma`.
4. Compute `invsqrtC = B · D⁻¹ · Bᵀ` from eigendecomposition of C.
5. `ps` ← `(1-cs)·ps + √(cs·(2-cs)·mueff) · invsqrtC · mean_diff`.
6. Compute `hsig`.
7. `pc` ← `(1-cc)·pc + hsig · √(cc·(2-cc)·mueff) · mean_diff`.
8. Rank-1 term: `c1 · (pc·pcᵀ + (1-hsig)·cc·(2-cc)·C)`.
9. Rank-μ term: `cmu · Σᵢ wᵢ · artmpᵢ·artmpᵢᵀ` where `artmpᵢ = (x_{ranked[i]} - old_mean) / sigma`.
10. `C` ← `(1-c1-cmu)·C` + rank-1 term + rank-μ term.
11. `sigma` ← `sigma · exp((cs/damps) · (||ps||/chiN - 1))`, clamped to `≥ 1e-20`.
12. `generation` += 1.

---

## Task 1: `src/cmaes.rs` — Mirror-Fold, Eigen Cache, CMAESState

This is the largest single file in the project. Work through it test-first: write all tests first, verify they fail, then implement.

**Files:**
- Modify: `src/cmaes.rs`

- [ ] **Step 1: Write the failing Rust tests**

Replace the stub `src/cmaes.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    // ── mirror_fold ───────────────────────────────────────────────────────────

    #[test]
    fn test_mirror_fold_in_bounds_unchanged() {
        // Values already within [low, high] must pass through unchanged.
        assert!((mirror_fold(0.5, 0.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((mirror_fold(-2.0, -5.0, 5.0) - (-2.0)).abs() < 1e-12);
        assert!((mirror_fold(0.0, -1.0, 1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_at_boundaries_unchanged() {
        assert!((mirror_fold(0.0, 0.0, 1.0) - 0.0).abs() < 1e-12);
        assert!((mirror_fold(1.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_slightly_above_high_reflects() {
        // x = 1.1 in [0, 1]: range=1, x-low=1.1, x mod 2 = 1.1, x > 1 → 2-1.1 = 0.9
        let result = mirror_fold(1.1, 0.0, 1.0);
        assert!((result - 0.9).abs() < 1e-12, "expected 0.9, got {}", result);
    }

    #[test]
    fn test_mirror_fold_slightly_below_low_reflects() {
        // x = -0.1 in [0, 1]: x-low = -0.1, x mod 2 = -0.1 (negative)
        // Rust's % is remainder, not modulo — handle negative: -0.1 mod 2.0 in modulo sense = 1.9
        // then 1.9 > 1 → 2 - 1.9 = 0.1
        let result = mirror_fold(-0.1, 0.0, 1.0);
        assert!((result - 0.1).abs() < 1e-12, "expected 0.1, got {}", result);
    }

    #[test]
    fn test_mirror_fold_result_always_in_bounds() {
        // Sample a range of out-of-bounds values and verify all land in [low, high].
        let low = -3.0_f64;
        let high = 3.0_f64;
        for i in -50..=50 {
            let x = i as f64 * 0.7; // spans well outside [−3, 3]
            let result = mirror_fold(x, low, high);
            assert!(
                result >= low - 1e-10 && result <= high + 1e-10,
                "mirror_fold({}, {}, {}) = {} is outside bounds",
                x, low, high, result
            );
        }
    }

    #[test]
    fn test_mirror_fold_far_outside_still_in_bounds() {
        // A value 10× the range outside still folds back correctly.
        let result = mirror_fold(25.0, 0.0, 1.0);
        assert!(result >= 0.0 && result <= 1.0, "got {}", result);
    }

    // ── CMAESState::new ───────────────────────────────────────────────────────

    fn make_state(n: usize, lambda: usize) -> CMAESState {
        let mean = vec![0.0_f64; n];
        let bounds = vec![(-5.0_f64, 5.0); n];
        CMAESState::new(mean, 0.5, lambda, bounds)
    }

    #[test]
    fn test_new_sets_correct_n() {
        let s = make_state(5, 10);
        assert_eq!(s.n, 5);
    }

    #[test]
    fn test_new_sets_correct_lambda() {
        let s = make_state(5, 10);
        assert_eq!(s.lambda, 10);
    }

    #[test]
    fn test_new_mu_is_half_lambda() {
        let s = make_state(5, 10);
        assert_eq!(s.mu, 5);
    }

    #[test]
    fn test_new_sigma_preserved() {
        let mean = vec![0.0_f64; 3];
        let bounds = vec![(-5.0_f64, 5.0); 3];
        let s = CMAESState::new(mean, 0.3, 6, bounds);
        assert!((s.sigma - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_new_covariance_is_identity() {
        let s = make_state(4, 8);
        // Diagonal entries should be 1.0, off-diagonal 0.0
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.cov[(i, j)] - expected).abs() < 1e-12,
                    "cov[{},{}] = {}, expected {}", i, j, s.cov[(i, j)], expected
                );
            }
        }
    }

    #[test]
    fn test_new_evolution_paths_are_zero() {
        let s = make_state(5, 10);
        assert!(s.pc.iter().all(|&x| x.abs() < 1e-12), "pc must be zero");
        assert!(s.ps.iter().all(|&x| x.abs() < 1e-12), "ps must be zero");
    }

    #[test]
    fn test_new_generation_is_zero() {
        let s = make_state(5, 10);
        assert_eq!(s.generation, 0);
    }

    #[test]
    fn test_new_eigendecomp_interval_at_least_one() {
        // The interval formula: max(1, floor(1 / (10 * n * (c1 + cmu))))
        // For any reasonable n, this must be >= 1.
        for n in [3, 5, 10, 20, 50, 100] {
            let lambda = 4 + (3.0 * (n as f64).ln()) as usize;
            let s = make_state(n, lambda);
            assert!(s.eigendecomp_interval >= 1,
                "eigendecomp_interval must be >= 1 for n={}", n);
        }
    }

    #[test]
    fn test_new_eigendecomp_interval_increases_with_n() {
        // For large n, the interval should be larger than for small n.
        // We only assert the direction for a large enough gap.
        let s_small = make_state(5,  4 + (3.0 * 5.0_f64.ln()) as usize);
        let s_large = make_state(200, 4 + (3.0 * 200.0_f64.ln()) as usize);
        assert!(
            s_large.eigendecomp_interval >= s_small.eigendecomp_interval,
            "larger n should have >= eigendecomp_interval; small={}, large={}",
            s_small.eigendecomp_interval, s_large.eigendecomp_interval
        );
    }

    // ── ask ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ask_returns_correct_lambda_samples() {
        let s = make_state(5, 12);
        let samples = s.ask(42, 0);
        assert_eq!(samples.len(), 12);
    }

    #[test]
    fn test_ask_returns_correct_n_genes_per_sample() {
        let s = make_state(7, 10);
        let samples = s.ask(42, 0);
        assert!(samples.iter().all(|samp| samp.len() == 7));
    }

    #[test]
    fn test_ask_deterministic_same_inputs() {
        let s = make_state(5, 10);
        let a = s.ask(42, 3);
        let b = s.ask(42, 3);
        assert_eq!(a, b, "ask() must be deterministic for the same (master_seed, generation)");
    }

    #[test]
    fn test_ask_different_generation_diverges() {
        let s = make_state(5, 10);
        let a = s.ask(42, 0);
        let b = s.ask(42, 1);
        assert_ne!(a, b, "different generation must produce different samples");
    }

    #[test]
    fn test_ask_different_master_seed_diverges() {
        let s = make_state(5, 10);
        let a = s.ask(1, 0);
        let b = s.ask(2, 0);
        assert_ne!(a, b, "different master_seed must produce different samples");
    }

    #[test]
    fn test_ask_samples_within_bounds_after_mirror_folding() {
        // All returned samples must lie within their per-gene bounds.
        let bounds = vec![(-2.0_f64, 2.0), (-1.0, 1.0), (0.0, 5.0), (-10.0, 0.0), (3.0, 7.0)];
        let mean = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        // Use a large sigma to force many samples outside bounds, testing mirror-folding.
        let s = CMAESState::new(mean, 5.0, 20, bounds.clone());
        let samples = s.ask(42, 0);
        for (i, sample) in samples.iter().enumerate() {
            for (j, &g) in sample.iter().enumerate() {
                let (lo, hi) = bounds[j];
                assert!(
                    g >= lo - 1e-10 && g <= hi + 1e-10,
                    "sample[{}][{}]={} outside [{}, {}]", i, j, g, lo, hi
                );
            }
        }
    }

    // ── tell ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_tell_increments_generation() {
        let mut s = make_state(3, 6);
        let samples = s.ask(42, 0);
        let fitnesses: Vec<f64> = samples.iter()
            .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
            .collect();
        s.tell(&samples, &fitnesses);
        assert_eq!(s.generation, 1, "tell() must increment generation");
    }

    #[test]
    fn test_tell_sigma_stays_positive_over_many_generations() {
        let mut s = make_state(4, 8);
        for gen in 0..20 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            assert!(s.sigma > 0.0, "sigma must stay positive after {} generations", gen + 1);
        }
    }

    #[test]
    fn test_tell_generation_matches_number_of_tell_calls() {
        let mut s = make_state(3, 6);
        for gen in 0..5 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        assert_eq!(s.generation, 5);
    }

    #[test]
    fn test_tell_mean_moves_toward_optimum() {
        // After 30 generations on the sphere function with optimum at origin,
        // the mean norm should decrease relative to the initial norm.
        let n = 5;
        let mean_start = vec![3.0_f64; n];
        let bounds = vec![(-10.0_f64, 10.0); n];
        let initial_norm: f64 = mean_start.iter().map(|x| x * x).sum::<f64>().sqrt();

        let mut s = CMAESState::new(mean_start, 1.0, 20, bounds.clone());
        for gen in 0..30 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        let final_norm: f64 = s.mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            final_norm < initial_norm,
            "mean norm should decrease toward optimum: initial={:.4}, final={:.4}",
            initial_norm, final_norm
        );
    }

    #[test]
    fn test_tell_covariance_stays_finite() {
        let mut s = make_state(4, 8);
        for gen in 0..10 {
            let samples = s.ask(99, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            // All covariance entries must be finite (no NaN or Inf)
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        s.cov[(i, j)].is_finite(),
                        "cov[{},{}] is not finite after {} generations", i, j, gen + 1
                    );
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL (nothing defined yet)**

```bash
cargo test cmaes 2>&1 | head -15
```

Expected: `error[E0422]: cannot find struct 'CMAESState'` or `error[E0425]: cannot find function 'mirror_fold'`

- [ ] **Step 3: Implement src/cmaes.rs**

Replace the stub `src/cmaes.rs` with the full implementation and tests. Create the file in three logical sections: the mirror-fold helper, the `CMAESState` struct, and the `PyCMAESState` PyO3 wrapper.

```rust
use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::{Cell, RefCell};

use crate::utils::{derive_seed, OP_CMAES_ASK};

// ── Mirror-folding boundary correction ───────────────────────────────────────

/// Fold an out-of-bounds value back into [low, high] by mirror reflection.
///
/// Unlike clipping, mirror-folding preserves the shape of the sampling
/// distribution near boundaries — probability mass is reflected rather than
/// collapsed at the boundary point.
///
/// Algorithm (handles arbitrary out-of-bounds distance):
///   1. Translate so low → 0.
///   2. Take modulo of the 2×range period.
///   3. If in the upper half, reflect back.
///   4. Translate back.
///
/// Rust's `%` operator is remainder (can return negative), so we normalise
/// negative remainders explicitly.
pub fn mirror_fold(x: f64, low: f64, high: f64) -> f64 {
    let range = high - low;
    if range <= 0.0 {
        return low; // degenerate bounds — return lower bound
    }
    let period = 2.0 * range;
    let mut t = x - low;
    // Normalise to [0, period) using true modulo (handles negative t)
    t = t % period;
    if t < 0.0 {
        t += period;
    }
    // Reflect upper half back into [0, range]
    if t > range {
        t = period - t;
    }
    t + low
}

// ── Eigendecomposition cache ──────────────────────────────────────────────────

/// Cached result of `symmetric_eigen()` on the covariance matrix.
/// Stored inside `RefCell` so `ask()` can lazily refresh it without `&mut self`.
struct EigenCache {
    eigenvectors:     DMatrix<f64>,   // B matrix (columns = eigenvectors)
    eigenvalues_sqrt: DVector<f64>,   // D matrix diagonal = √eigenvalues, clamped to √1e-20
    valid:            bool,
}

impl EigenCache {
    fn invalid(n: usize) -> Self {
        EigenCache {
            eigenvectors:     DMatrix::identity(n, n),
            eigenvalues_sqrt: DVector::from_element(n, 1.0),
            valid:            false,
        }
    }
}

// ── CMAESState ────────────────────────────────────────────────────────────────

/// Internal CMA-ES state. Holds all mutable algorithmic fields.
///
/// # Design
///
/// - No RNG stored here. `ask()` accepts `(master_seed, generation)` and derives
///   a fresh `StdRng` per sample via `derive_seed(master, gen, sample_idx, OP_CMAES_ASK)`.
/// - `ask()` is `&self` — it only mutates the eigendecomp cache (via `RefCell`)
///   and the age counter (via `Cell`). This lets the Python engine call `ask()`
///   and `tell()` in any order without borrow conflicts.
/// - `tell()` is `&mut self` — it updates mean, cov, pc, ps, sigma, generation.
pub struct CMAESState {
    // ── Dimensions ────────────────────────────────────────────────
    pub n:      usize,   // problem dimension
    pub lambda: usize,   // population size (samples per generation)
    pub mu:     usize,   // number of recombination parents

    // ── Core state ────────────────────────────────────────────────
    pub mean:  DVector<f64>,   // distribution mean
    pub sigma: f64,            // overall step size
    pub cov:   DMatrix<f64>,   // covariance matrix (n×n)
    pub pc:    DVector<f64>,   // evolution path for covariance
    pub ps:    DVector<f64>,   // evolution path for sigma (CSA)

    // ── Recombination weights ─────────────────────────────────────
    pub weights: Vec<f64>,
    pub mueff:   f64,   // effective number of recombinants = 1/Σwᵢ²

    // ── Strategy parameters (computed once at construction) ───────
    pub cc:    f64,   // time constant for cumulation path pc
    pub cs:    f64,   // time constant for step-size path ps
    pub c1:    f64,   // rank-1 learning rate
    pub cmu:   f64,   // rank-μ learning rate
    pub damps: f64,   // step-size damping
    pub chiN:  f64,   // E[||N(0,I)||] ≈ √n · (1 - 1/(4n) + 1/(21n²))

    // ── Eigendecomposition cache (interior mutability) ────────────
    eigendecomp_cache:    RefCell<EigenCache>,
    pub eigendecomp_age:      Cell<usize>,
    pub eigendecomp_interval: usize,

    // ── Per-gene bounds for mirror folding ────────────────────────
    pub bounds: Vec<(f64, f64)>,

    // ── Generation counter ────────────────────────────────────────
    pub generation: usize,
}

impl CMAESState {
    /// Construct a new `CMAESState`.
    ///
    /// # Parameters
    /// - `mean`: initial distribution mean (one value per gene)
    /// - `sigma`: initial step size (absolute)
    /// - `lambda`: population size (number of samples per `ask()`)
    /// - `bounds`: per-gene `(low, high)` used in mirror-folding inside `ask()`
    pub fn new(mean: Vec<f64>, sigma: f64, lambda: usize, bounds: Vec<(f64, f64)>) -> Self {
        let n = mean.len();
        assert_eq!(bounds.len(), n, "CMAESState::new: mean and bounds lengths must match");
        let mu = lambda / 2;

        // ── Recombination weights (log-optimal) ───────────────────
        let raw_w: Vec<f64> = (1..=mu)
            .map(|i| (mu as f64 + 0.5).ln() - (i as f64).ln())
            .collect();
        let sum_w: f64 = raw_w.iter().sum();
        let weights: Vec<f64> = raw_w.iter().map(|w| w / sum_w).collect();
        let mueff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // ── Strategy parameters ───────────────────────────────────
        let n_f = n as f64;
        let cc    = (4.0 + mueff / n_f) / (n_f + 4.0 + 2.0 * mueff / n_f);
        let cs    = (mueff + 2.0) / (n_f + mueff + 5.0);
        let c1    = 2.0 / ((n_f + 1.3).powi(2) + mueff);
        let cmu   = f64::min(
            1.0 - c1,
            2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n_f + 2.0).powi(2) + mueff),
        );
        let damps = 1.0
            + 2.0 * f64::max(0.0, ((mueff - 1.0) / (n_f + 1.0)).sqrt() - 1.0)
            + cs;
        let chiN  = n_f.sqrt()
            * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        // ── Eigendecomp interval: max(1, ⌊1 / (10·n·(c1+cmu))⌋) ─
        let raw_interval = 1.0 / (10.0 * n_f * (c1 + cmu));
        let eigendecomp_interval = (raw_interval.floor() as usize).max(1);

        CMAESState {
            n, lambda, mu,
            mean:  DVector::from_vec(mean),
            sigma,
            cov:   DMatrix::identity(n, n),
            pc:    DVector::zeros(n),
            ps:    DVector::zeros(n),
            weights, mueff,
            cc, cs, c1, cmu, damps, chiN,
            eigendecomp_cache:    RefCell::new(EigenCache::invalid(n)),
            eigendecomp_age:      Cell::new(0),
            eigendecomp_interval,
            bounds,
            generation: 0,
        }
    }

    // ── Eigendecomposition cache management ──────────────────────────────────

    /// Return `(eigenvectors, eigenvalues_sqrt)`, recomputing if cache is stale.
    ///
    /// Called from `ask()` which is `&self`. Uses `RefCell` + `Cell` for interior
    /// mutability so `ask()` need not be `&mut self`.
    ///
    /// Cache is invalid when `age >= eigendecomp_interval` or on first call.
    /// Eigenvalues are clamped to `≥ 1e-20` before taking the square root, so
    /// near-singular covariance matrices do not panic.
    fn get_or_update_eigen(&self) -> (DMatrix<f64>, DVector<f64>) {
        let age = self.eigendecomp_age.get();
        let mut cache = self.eigendecomp_cache.borrow_mut();

        if !cache.valid || age >= self.eigendecomp_interval {
            // Enforce symmetry numerically before decomposition
            let sym_cov = (&self.cov + self.cov.transpose()) * 0.5;
            let eigen = sym_cov.symmetric_eigen();
            cache.eigenvalues_sqrt = eigen.eigenvalues.map(|v| v.max(1e-20_f64).sqrt());
            cache.eigenvectors     = eigen.eigenvectors;
            cache.valid            = true;
            self.eigendecomp_age.set(0);
        } else {
            self.eigendecomp_age.set(age + 1);
        }

        (cache.eigenvectors.clone(), cache.eigenvalues_sqrt.clone())
    }

    // ── ask ───────────────────────────────────────────────────────────────────

    /// Sample `lambda` candidate solutions from the current distribution.
    ///
    /// Each sample is generated with an independent `StdRng` seeded via
    /// `derive_seed(master_seed, generation, sample_idx, OP_CMAES_ASK)`.
    /// This guarantees thread-count-independent, idempotent sampling:
    /// calling `ask(42, 5)` twice always returns the same samples.
    ///
    /// All returned values are mirror-folded into their per-gene bounds.
    /// The **caller must pass continuous (unrounded) samples to `tell()`**.
    pub fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        let (eigenvectors, eigenvalues_sqrt) = self.get_or_update_eigen();
        let normal = Normal::new(0.0_f64, 1.0).unwrap();

        (0..self.lambda)
            .map(|sample_idx| {
                let mut rng = StdRng::seed_from_u64(
                    derive_seed(master_seed, generation, sample_idx as u64, OP_CMAES_ASK)
                );

                // Sample z ~ N(0, I)
                let z: DVector<f64> = DVector::from_iterator(
                    self.n,
                    (0..self.n).map(|_| normal.sample(&mut rng)),
                );

                // Transform: y = B · (D ⊙ z),  then  x = mean + sigma · y
                let dz: DVector<f64> = DVector::from_iterator(
                    self.n,
                    eigenvalues_sqrt.iter().zip(z.iter()).map(|(d, zi)| d * zi),
                );
                let y = &eigenvectors * dz;
                let raw = &self.mean + self.sigma * y;

                // Apply mirror-folding per gene
                raw.iter()
                    .enumerate()
                    .map(|(i, &x)| mirror_fold(x, self.bounds[i].0, self.bounds[i].1))
                    .collect()
            })
            .collect()
    }

    // ── tell ──────────────────────────────────────────────────────────────────

    /// Update the distribution parameters from a batch of evaluated samples.
    ///
    /// Implements the Hansen (2016) rank-1 + rank-μ CMA-ES update with
    /// σ-CSA step-size control. Call with the **continuous** (unrounded) samples
    /// returned by `ask()`, not the rounded/discrete values shown to the fitness
    /// function. The `CMAESEngine` Python layer maintains two parallel sample
    /// arrays for this reason.
    ///
    /// `fitnesses` are maximisation values (higher = better). Pass negated sphere
    /// values, profit factors, etc. Internally samples are ranked descending.
    pub fn tell(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) {
        assert_eq!(
            samples.len(), fitnesses.len(),
            "tell(): samples and fitnesses must have the same length"
        );
        let n_f = self.n as f64;

        // 1. Sort indices by fitness descending (highest fitness = best)
        let mut ranked: Vec<usize> = (0..fitnesses.len()).collect();
        ranked.sort_by(|&a, &b| {
            let fa = if fitnesses[a].is_nan() { f64::NEG_INFINITY } else { fitnesses[a] };
            let fb = if fitnesses[b].is_nan() { f64::NEG_INFINITY } else { fitnesses[b] };
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let old_mean = self.mean.clone();

        // 2. Update mean: weighted sum of top-mu individuals
        self.mean = DVector::zeros(self.n);
        for (i, &idx) in ranked[..self.mu].iter().enumerate() {
            let s = DVector::from_vec(samples[idx].clone());
            self.mean += self.weights[i] * s;
        }

        // 3. mean_diff = (new_mean - old_mean) / sigma
        let mean_diff = (&self.mean - &old_mean) / self.sigma;

        // 4. Compute invsqrtC = B · D⁻¹ · Bᵀ from current covariance
        //    (reuses or refreshes cache)
        let (eigenvectors, eigenvalues_sqrt) = self.get_or_update_eigen();
        // D⁻¹ diagonal = 1/eigenvalue_sqrt
        let inv_d: DVector<f64> = eigenvalues_sqrt.map(|v| 1.0 / v.max(1e-20));
        let invsqrtC = &eigenvectors
            * DMatrix::from_diagonal(&inv_d)
            * eigenvectors.transpose();

        // 5. Update ps (step-size evolution path)
        let ps_new = (1.0 - self.cs) * &self.ps
            + (self.cs * (2.0 - self.cs) * self.mueff).sqrt()
              * &invsqrtC * &mean_diff;
        self.ps = ps_new;

        // 6. Compute hsig
        let ps_norm = self.ps.norm();
        let expected_ps_norm = self.chiN
            * (1.0 - (1.0 - self.cs).powi(2 * (self.generation as i32 + 1))).sqrt();
        let hsig = if ps_norm / expected_ps_norm < 1.4 + 2.0 / (n_f + 1.0) {
            1.0_f64
        } else {
            0.0_f64
        };

        // 7. Update pc (cumulation path for covariance)
        self.pc = (1.0 - self.cc) * &self.pc
            + hsig * (self.cc * (2.0 - self.cc) * self.mueff).sqrt() * &mean_diff;

        // 8. Rank-1 update term
        let rank_one = &self.pc * self.pc.transpose()
            + (1.0 - hsig) * self.cc * (2.0 - self.cc) * &self.cov;

        // 9. Rank-μ update term: Σᵢ wᵢ · artmpᵢ · artmpᵢᵀ
        let rank_mu: DMatrix<f64> = ranked[..self.mu]
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let artmp =
                    (DVector::from_vec(samples[idx].clone()) - &old_mean) / self.sigma;
                self.weights[i] * &artmp * artmp.transpose()
            })
            .fold(DMatrix::zeros(self.n, self.n), |acc, m| acc + m);

        // 10. Update covariance matrix
        self.cov = (1.0 - self.c1 - self.cmu) * &self.cov
            + self.c1 * rank_one
            + self.cmu * rank_mu;

        // Invalidate eigendecomp cache — C has changed
        self.eigendecomp_cache.borrow_mut().valid = false;
        self.eigendecomp_age.set(0);

        // 11. Update sigma (σ-CSA)
        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chiN - 1.0)).exp();
        self.sigma = self.sigma.max(1e-20);

        // 12. Increment generation counter (used for eigendecomp interval only)
        self.generation += 1;
    }
}

// ── PyCMAESState — PyO3 wrapper ───────────────────────────────────────────────

/// Python-facing CMA-ES state wrapper.
///
/// The engine (Python) owns the generation counter and passes it to `ask()`.
/// No random state is stored here.
///
/// # Example (Python)
/// ```python
/// from evocore._core import PyCMAESState
/// state = PyCMAESState([0.0]*5, 0.5, 20, [(-5.0, 5.0)]*5)
/// for gen in range(200):
///     samples = state.ask(master_seed=42, generation=gen)
///     fitnesses = [evaluate(s) for s in samples]
///     state.tell(samples, fitnesses)
/// ```
#[pyclass]
pub struct PyCMAESState {
    inner: CMAESState,
}

#[pymethods]
impl PyCMAESState {
    /// Create a new CMA-ES state.
    ///
    /// Parameters
    /// ----------
    /// mean : list[float]
    ///     Initial distribution mean. Length determines problem dimension.
    /// sigma : float
    ///     Initial step size (absolute, not a fraction of range).
    /// lambda_ : int
    ///     Population size (number of samples per ask() call).
    /// bounds : list[tuple[float, float]]
    ///     Per-gene (low, high) bounds used for mirror-folding in ask().
    ///     Must have the same length as `mean`.
    #[new]
    pub fn new(mean: Vec<f64>, sigma: f64, lambda_: usize, bounds: Vec<(f64, f64)>) -> Self {
        PyCMAESState {
            inner: CMAESState::new(mean, sigma, lambda_, bounds),
        }
    }

    /// Sample lambda candidate solutions.
    ///
    /// Parameters
    /// ----------
    /// master_seed : int
    ///     The engine's master seed (u64). Never changes between calls.
    /// generation : int
    ///     Current generation index (0-based). Caller increments this.
    ///
    /// Returns
    /// -------
    /// list[list[float]]
    ///     lambda samples, each a list of n floats within their gene bounds.
    pub fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        self.inner.ask(master_seed, generation)
    }

    /// Update the distribution from evaluated samples.
    ///
    /// Parameters
    /// ----------
    /// samples : list[list[float]]
    ///     The **continuous** (unrounded) samples returned by ask().
    ///     Do NOT pass rounded/discrete values — use the unrounded array
    ///     for tell() and the rounded array for fitness evaluation.
    /// fitnesses : list[float]
    ///     Fitness values for each sample. Higher = better (maximisation).
    ///     NaN fitnesses are treated as -inf (worst possible).
    pub fn tell(&mut self, samples: Vec<Vec<f64>>, fitnesses: Vec<f64>) {
        self.inner.tell(&samples, &fitnesses);
    }

    /// Current generation (number of completed tell() calls).
    #[getter]
    pub fn generation(&self) -> usize {
        self.inner.generation
    }

    /// Current step size sigma.
    #[getter]
    pub fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    /// Current distribution mean.
    #[getter]
    pub fn mean(&self) -> Vec<f64> {
        self.inner.mean.iter().cloned().collect()
    }

    /// Eigendecomposition update interval (computed at construction from n, c1, cmu).
    #[getter]
    pub fn eigendecomp_interval(&self) -> usize {
        self.inner.eigendecomp_interval
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    // ── mirror_fold ───────────────────────────────────────────────────────────

    #[test]
    fn test_mirror_fold_in_bounds_unchanged() {
        assert!((mirror_fold(0.5, 0.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((mirror_fold(-2.0, -5.0, 5.0) - (-2.0)).abs() < 1e-12);
        assert!((mirror_fold(0.0, -1.0, 1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_at_boundaries_unchanged() {
        assert!((mirror_fold(0.0, 0.0, 1.0) - 0.0).abs() < 1e-12);
        assert!((mirror_fold(1.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_slightly_above_high_reflects() {
        let result = mirror_fold(1.1, 0.0, 1.0);
        assert!((result - 0.9).abs() < 1e-12, "expected 0.9, got {}", result);
    }

    #[test]
    fn test_mirror_fold_slightly_below_low_reflects() {
        let result = mirror_fold(-0.1, 0.0, 1.0);
        assert!((result - 0.1).abs() < 1e-12, "expected 0.1, got {}", result);
    }

    #[test]
    fn test_mirror_fold_result_always_in_bounds() {
        let low = -3.0_f64;
        let high = 3.0_f64;
        for i in -50..=50 {
            let x = i as f64 * 0.7;
            let result = mirror_fold(x, low, high);
            assert!(
                result >= low - 1e-10 && result <= high + 1e-10,
                "mirror_fold({}, {}, {}) = {} is outside bounds",
                x, low, high, result
            );
        }
    }

    #[test]
    fn test_mirror_fold_far_outside_still_in_bounds() {
        let result = mirror_fold(25.0, 0.0, 1.0);
        assert!(result >= 0.0 && result <= 1.0, "got {}", result);
    }

    // ── CMAESState::new ───────────────────────────────────────────────────────

    fn make_state(n: usize, lambda: usize) -> CMAESState {
        let mean = vec![0.0_f64; n];
        let bounds = vec![(-5.0_f64, 5.0); n];
        CMAESState::new(mean, 0.5, lambda, bounds)
    }

    #[test]
    fn test_new_sets_correct_n() {
        assert_eq!(make_state(5, 10).n, 5);
    }

    #[test]
    fn test_new_sets_correct_lambda() {
        assert_eq!(make_state(5, 10).lambda, 10);
    }

    #[test]
    fn test_new_mu_is_half_lambda() {
        assert_eq!(make_state(5, 10).mu, 5);
    }

    #[test]
    fn test_new_sigma_preserved() {
        let s = CMAESState::new(vec![0.0; 3], 0.3, 6, vec![(-5.0, 5.0); 3]);
        assert!((s.sigma - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_new_covariance_is_identity() {
        let s = make_state(4, 8);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.cov[(i, j)] - expected).abs() < 1e-12,
                    "cov[{},{}] = {}, expected {}", i, j, s.cov[(i, j)], expected
                );
            }
        }
    }

    #[test]
    fn test_new_evolution_paths_are_zero() {
        let s = make_state(5, 10);
        assert!(s.pc.iter().all(|&x| x.abs() < 1e-12));
        assert!(s.ps.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_new_generation_is_zero() {
        assert_eq!(make_state(5, 10).generation, 0);
    }

    #[test]
    fn test_new_eigendecomp_interval_at_least_one() {
        for n in [3_usize, 5, 10, 20, 50, 100] {
            let lambda = 4 + (3.0 * (n as f64).ln()) as usize;
            let s = make_state(n, lambda);
            assert!(s.eigendecomp_interval >= 1, "n={}", n);
        }
    }

    #[test]
    fn test_new_eigendecomp_interval_increases_with_n() {
        let s_small = make_state(5,  4 + (3.0 * 5.0_f64.ln()) as usize);
        let s_large = make_state(200, 4 + (3.0 * 200.0_f64.ln()) as usize);
        assert!(
            s_large.eigendecomp_interval >= s_small.eigendecomp_interval,
            "small={}, large={}",
            s_small.eigendecomp_interval, s_large.eigendecomp_interval
        );
    }

    // ── ask ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ask_returns_correct_lambda_samples() {
        assert_eq!(make_state(5, 12).ask(42, 0).len(), 12);
    }

    #[test]
    fn test_ask_returns_correct_n_genes_per_sample() {
        let s = make_state(7, 10);
        let samples = s.ask(42, 0);
        assert!(samples.iter().all(|samp| samp.len() == 7));
    }

    #[test]
    fn test_ask_deterministic_same_inputs() {
        let s = make_state(5, 10);
        assert_eq!(s.ask(42, 3), s.ask(42, 3));
    }

    #[test]
    fn test_ask_different_generation_diverges() {
        let s = make_state(5, 10);
        assert_ne!(s.ask(42, 0), s.ask(42, 1));
    }

    #[test]
    fn test_ask_different_master_seed_diverges() {
        let s = make_state(5, 10);
        assert_ne!(s.ask(1, 0), s.ask(2, 0));
    }

    #[test]
    fn test_ask_samples_within_bounds_after_mirror_folding() {
        let bounds = vec![
            (-2.0_f64, 2.0),
            (-1.0, 1.0),
            (0.0, 5.0),
            (-10.0, 0.0),
            (3.0, 7.0),
        ];
        let mean = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        // Large sigma forces many samples outside bounds — mirror-folding must handle them
        let s = CMAESState::new(mean, 5.0, 20, bounds.clone());
        let samples = s.ask(42, 0);
        for (i, sample) in samples.iter().enumerate() {
            for (j, &g) in sample.iter().enumerate() {
                let (lo, hi) = bounds[j];
                assert!(
                    g >= lo - 1e-10 && g <= hi + 1e-10,
                    "sample[{}][{}]={} outside [{}, {}]", i, j, g, lo, hi
                );
            }
        }
    }

    // ── tell ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_tell_increments_generation() {
        let mut s = make_state(3, 6);
        let samples = s.ask(42, 0);
        let fitnesses: Vec<f64> = samples.iter()
            .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
            .collect();
        s.tell(&samples, &fitnesses);
        assert_eq!(s.generation, 1);
    }

    #[test]
    fn test_tell_sigma_stays_positive_over_many_generations() {
        let mut s = make_state(4, 8);
        for gen in 0..20 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            assert!(s.sigma > 0.0, "sigma must stay positive after {} gens", gen + 1);
        }
    }

    #[test]
    fn test_tell_generation_matches_number_of_tell_calls() {
        let mut s = make_state(3, 6);
        for gen in 0..5 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        assert_eq!(s.generation, 5);
    }

    #[test]
    fn test_tell_mean_moves_toward_optimum() {
        let n = 5;
        let mean_start = vec![3.0_f64; n];
        let bounds = vec![(-10.0_f64, 10.0); n];
        let initial_norm: f64 = mean_start.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut s = CMAESState::new(mean_start, 1.0, 20, bounds);
        for gen in 0..30 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        let final_norm: f64 = s.mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            final_norm < initial_norm,
            "mean norm should decrease: initial={:.4}, final={:.4}",
            initial_norm, final_norm
        );
    }

    #[test]
    fn test_tell_covariance_stays_finite() {
        let mut s = make_state(4, 8);
        for gen in 0..10 {
            let samples = s.ask(99, gen as u64);
            let fitnesses: Vec<f64> = samples.iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        s.cov[(i, j)].is_finite(),
                        "cov[{},{}] is NaN/Inf after gen {}", i, j, gen + 1
                    );
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run Rust tests — expect PASS**

```bash
cargo test cmaes
```

Expected:
```
test cmaes::tests::test_ask_deterministic_same_inputs ... ok
test cmaes::tests::test_ask_different_generation_diverges ... ok
test cmaes::tests::test_ask_different_master_seed_diverges ... ok
test cmaes::tests::test_ask_returns_correct_lambda_samples ... ok
test cmaes::tests::test_ask_returns_correct_n_genes_per_sample ... ok
test cmaes::tests::test_ask_samples_within_bounds_after_mirror_folding ... ok
test cmaes::tests::test_mirror_fold_at_boundaries_unchanged ... ok
test cmaes::tests::test_mirror_fold_far_outside_still_in_bounds ... ok
test cmaes::tests::test_mirror_fold_in_bounds_unchanged ... ok
test cmaes::tests::test_mirror_fold_result_always_in_bounds ... ok
test cmaes::tests::test_mirror_fold_slightly_above_high_reflects ... ok
test cmaes::tests::test_mirror_fold_slightly_below_low_reflects ... ok
test cmaes::tests::test_new_covariance_is_identity ... ok
test cmaes::tests::test_new_eigendecomp_interval_at_least_one ... ok
test cmaes::tests::test_new_eigendecomp_interval_increases_with_n ... ok
test cmaes::tests::test_new_evolution_paths_are_zero ... ok
test cmaes::tests::test_new_generation_is_zero ... ok
test cmaes::tests::test_new_mu_is_half_lambda ... ok
test cmaes::tests::test_new_sets_correct_lambda ... ok
test cmaes::tests::test_new_sets_correct_n ... ok
test cmaes::tests::test_new_sigma_preserved ... ok
test cmaes::tests::test_tell_covariance_stays_finite ... ok
test cmaes::tests::test_tell_generation_matches_number_of_tell_calls ... ok
test cmaes::tests::test_tell_increments_generation ... ok
test cmaes::tests::test_tell_mean_moves_toward_optimum ... ok
test cmaes::tests::test_tell_sigma_stays_positive_over_many_generations ... ok

test result: ok. 26 passed; 0 failed
```

- [ ] **Step 5: Run all Rust tests — confirm no regressions from Parts 1–3**

```bash
cargo test
```

Expected: `test result: ok. 123 passed; 0 failed`
(97 from Parts 1–3 + 26 new = 123)

- [ ] **Step 6: Commit**

```bash
git add src/cmaes.rs
git commit -m "feat(rust): CMAESState with RefCell eigen cache, mirror-folding, stateless ask/tell"
```

---

## Task 2: Update `src/lib.rs` — Register PyCMAESState

Add `PyCMAESState` to the `_core` module. This is a one-section change to the module root.

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Add cmaes import and class registration**

Open `src/lib.rs`. Make two changes:

**Change A — add use statement** at the top with the other `use` declarations (after `use parallel::{...}`):

```rust
use cmaes::PyCMAESState;
```

**Change B — register the class** in the `_core` pymodule function, after the `// CMA-ES registered in Part 4` comment (replace that comment):

```rust
    // ── CMA-ES
    m.add_class::<PyCMAESState>()?;
```

The full updated `_core` function body for reference (only the CMA-ES section changes):

```rust
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    // ── Individual types
    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;

    // ── Seed architecture
    m.add_function(wrap_pyfunction!(py_derive_seed, m)?)?;
    m.add("OP_INIT",           OP_INIT)?;
    m.add("OP_CROSSOVER",      OP_CROSSOVER)?;
    m.add("OP_MUTATION",       OP_MUTATION)?;
    m.add("OP_SELECTION",      OP_SELECTION)?;
    m.add("OP_CMAES_ASK",      OP_CMAES_ASK)?;
    m.add("OP_MULTI_RUN",      OP_MULTI_RUN)?;
    m.add("OP_CROSSOVER_PROB", OP_CROSSOVER_PROB)?;

    // ── Float operators
    m.add_function(wrap_pyfunction!(blend_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_mutation, m)?)?;

    // ── Integer operators
    m.add_function(wrap_pyfunction!(int_simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(int_gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(int_uniform_mutation, m)?)?;

    // ── Binary operators
    m.add_function(wrap_pyfunction!(one_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(two_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(bit_flip_mutation, m)?)?;

    // ── Selection
    m.add_function(wrap_pyfunction!(tournament_selection, m)?)?;
    m.add_function(wrap_pyfunction!(roulette_selection, m)?)?;
    m.add_function(wrap_pyfunction!(rank_selection, m)?)?;

    // ── Population initialisation + reproduction
    m.add_function(wrap_pyfunction!(init_population, m)?)?;
    m.add_function(wrap_pyfunction!(reproduce_population, m)?)?;

    // ── Parallel evaluation
    m.add_function(wrap_pyfunction!(evaluate_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_parallel_rayon, m)?)?;

    // ── CMA-ES
    m.add_class::<PyCMAESState>()?;

    Ok(())
}
```

- [ ] **Step 2: Compile**

```bash
maturin develop --release
```

Expected: no errors.

- [ ] **Step 3: Verify PyCMAESState is accessible from Python**

```bash
python - << 'EOF'
from evocore._core import PyCMAESState

state = PyCMAESState([0.0] * 5, 0.5, 10, [(-5.0, 5.0)] * 5)
print(f"generation={state.generation}")
print(f"sigma={state.sigma:.4f}")
print(f"mean={state.mean}")
print(f"eigendecomp_interval={state.eigendecomp_interval}")

samples = state.ask(42, 0)
assert len(samples) == 10, f"expected 10 samples, got {len(samples)}"
assert all(len(s) == 5 for s in samples), "each sample must have 5 genes"
print(f"ask() returned {len(samples)} samples of length {len(samples[0])}")

fitnesses = [-sum(x**2 for x in s) for s in samples]
state.tell(samples, fitnesses)
assert state.generation == 1, f"expected generation=1, got {state.generation}"
assert state.sigma > 0.0, "sigma must stay positive"
print(f"after tell(): generation={state.generation}, sigma={state.sigma:.6f}")

print("PyCMAESState Python API ok")
EOF
```

Expected:
```
generation=0
sigma=0.5000
mean=[0.0, 0.0, 0.0, 0.0, 0.0]
eigendecomp_interval=1
ask() returned 10 samples of length 5
after tell(): generation=1, sigma=...
PyCMAESState Python API ok
```

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "feat(rust): register PyCMAESState in _core module"
```

---

## Task 3: Python Smoke Tests (`tests/unit/test_cmaes_rust.py`)

**Files:**
- Create: `tests/unit/test_cmaes_rust.py`

- [ ] **Step 1: Write the test file**

```python
"""
Smoke tests for PyCMAESState exposed via PyO3.

Focus:
  - Correct shapes (lambda × n) from ask()
  - Determinism: same (master_seed, generation) → identical samples
  - Boundary compliance: all samples within bounds after mirror-folding
  - tell() correctness: generation increments, sigma stays positive
  - RNG independence: different seeds/generations diverge
  - Idempotency: ask() is a pure function — calling twice gives the same result
  - Convergence signal: mean norm decreases on sphere over 30 generations
  - Integer-gene workflow: CMAESEngine uses continuous samples for tell()
    and rounded samples for fitness — test that continuous samples are not all integers
"""
import math
import pytest
from evocore._core import PyCMAESState


def make_state(n: int = 5, lambda_: int = 10, sigma: float = 0.5,
               bounds=None) -> PyCMAESState:
    if bounds is None:
        bounds = [(-5.0, 5.0)] * n
    mean = [(lo + hi) / 2.0 for lo, hi in bounds]
    return PyCMAESState(mean, sigma, lambda_, bounds)


def neg_sphere(genes: list[float]) -> float:
    return -sum(x * x for x in genes)


# ── Construction ──────────────────────────────────────────────────────────────

class TestPyCMAESStateConstruction:

    def test_generation_starts_at_zero(self):
        assert make_state().generation == 0

    def test_sigma_preserved(self):
        s = make_state(sigma=0.3)
        assert abs(s.sigma - 0.3) < 1e-12

    def test_mean_matches_input(self):
        mean_in = [1.0, 2.0, 3.0]
        s = PyCMAESState(mean_in, 0.5, 6, [(-5.0, 5.0)] * 3)
        assert s.mean == mean_in

    def test_eigendecomp_interval_at_least_one(self):
        for n in [3, 5, 10, 20]:
            s = make_state(n=n, lambda_=4 + int(3 * math.log(n)))
            assert s.eigendecomp_interval >= 1, f"n={n}"

    def test_invalid_bounds_length_raises(self):
        """bounds length must match mean length."""
        with pytest.raises(Exception):
            PyCMAESState([0.0] * 5, 0.5, 10, [(-1.0, 1.0)] * 3)  # wrong bounds length


# ── ask() shape and content ───────────────────────────────────────────────────

class TestAskShape:

    def test_returns_correct_lambda_samples(self):
        s = make_state(lambda_=15)
        samples = s.ask(42, 0)
        assert len(samples) == 15

    def test_returns_correct_n_genes_per_sample(self):
        s = make_state(n=7, lambda_=10)
        samples = s.ask(42, 0)
        assert all(len(samp) == 7 for samp in samples)

    def test_all_samples_within_bounds(self):
        bounds = [(-2.0, 2.0), (-1.0, 1.0), (0.0, 5.0), (-10.0, 0.0)]
        s = PyCMAESState([0.0, 0.0, 2.5, -5.0], 5.0, 20, bounds)
        samples = s.ask(42, 0)
        for i, sample in enumerate(samples):
            for j, (g, (lo, hi)) in enumerate(zip(sample, bounds)):
                assert lo - 1e-9 <= g <= hi + 1e-9, (
                    f"sample[{i}][{j}]={g} outside [{lo}, {hi}]"
                )

    def test_samples_are_floats(self):
        samples = make_state().ask(42, 0)
        assert all(isinstance(g, float) for samp in samples for g in samp)


# ── ask() determinism invariant ───────────────────────────────────────────────

class TestAskDeterminism:
    """
    The key v3/v4/v5 RNG guarantee: ask(master_seed, generation) is a pure
    function — same inputs always produce the same samples, regardless of
    how many times tell() has been called in between.
    """

    def test_same_args_same_result(self):
        s = make_state()
        a = s.ask(42, 5)
        b = s.ask(42, 5)
        assert a == b, "ask() must be deterministic for the same (master_seed, generation)"

    def test_different_generation_diverges(self):
        s = make_state()
        a = s.ask(42, 0)
        b = s.ask(42, 1)
        assert a != b, "different generation must produce different samples"

    def test_different_master_seed_diverges(self):
        s = make_state()
        a = s.ask(1, 0)
        b = s.ask(2, 0)
        assert a != b, "different master_seed must produce different samples"

    def test_ask_uses_updated_distribution_after_tell(self):
        """
        ask(master_seed, generation) uses deterministic random draws, but those
        draws are transformed by the current CMA-ES distribution. After tell()
        updates mean/covariance, asking for the same seed/generation may produce
        different samples. That is correct: only RNG state consumption is banned.
        """
        s = make_state(n=3, lambda_=6)

        samples_before = s.ask(42, 0)
        fitnesses = [neg_sphere(samp) for samp in samples_before]
        s.tell(samples_before, fitnesses)

        samples_after = s.ask(42, 0)
        assert samples_before != samples_after, (
            "tell() should update the distribution used by ask(); "
            "deterministic RNG does not mean immutable CMA-ES state"
        )


# ── tell() correctness ────────────────────────────────────────────────────────

class TestTell:

    def test_increments_generation(self):
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(samp) for samp in samples])
        assert s.generation == 1

    def test_generation_tracks_call_count(self):
        s = make_state(n=3, lambda_=6)
        for gen in range(5):
            samples = s.ask(42, gen)
            s.tell(samples, [neg_sphere(samp) for samp in samples])
        assert s.generation == 5

    def test_sigma_stays_positive(self):
        s = make_state(n=4, lambda_=8)
        for gen in range(20):
            samples = s.ask(42, gen)
            s.tell(samples, [neg_sphere(samp) for samp in samples])
            assert s.sigma > 0.0, f"sigma must stay positive at gen {gen + 1}"

    def test_sigma_is_finite(self):
        s = make_state(n=4, lambda_=8)
        for gen in range(20):
            samples = s.ask(42, gen)
            s.tell(samples, [neg_sphere(samp) for samp in samples])
            assert math.isfinite(s.sigma), f"sigma became non-finite at gen {gen + 1}"

    def test_mean_is_a_list_of_floats(self):
        s = make_state(n=5, lambda_=10)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(samp) for samp in samples])
        assert all(isinstance(x, float) for x in s.mean)
        assert len(s.mean) == 5

    def test_nan_fitness_handled_gracefully(self):
        """NaN fitness values must not crash tell()."""
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        fitnesses = [float("nan")] * 3 + [neg_sphere(samp) for samp in samples[3:]]
        s.tell(samples, fitnesses)  # must not raise
        assert s.generation == 1

    def test_mismatched_lengths_raise(self):
        """tell() must raise if samples and fitnesses have different lengths."""
        s = make_state(n=3, lambda_=6)
        samples = s.ask(42, 0)
        with pytest.raises(Exception):
            s.tell(samples, [1.0, 2.0])  # wrong fitness length


# ── Convergence signal ────────────────────────────────────────────────────────

class TestConvergence:

    def test_mean_norm_decreases_on_sphere_30_gens(self):
        """
        After 30 generations of CMA-ES on the sphere function with the optimum
        at the origin, the mean norm must be smaller than the initial norm.
        This is a basic sanity check that the update equations move the mean
        in the right direction.
        """
        n = 5
        bounds = [(-10.0, 10.0)] * n
        mean_start = [3.0] * n
        initial_norm = math.sqrt(sum(x * x for x in mean_start))

        s = PyCMAESState(mean_start, 1.0, 20, bounds)
        for gen in range(30):
            samples = s.ask(42, gen)
            fitnesses = [neg_sphere(samp) for samp in samples]
            s.tell(samples, fitnesses)

        final_norm = math.sqrt(sum(x * x for x in s.mean))
        assert final_norm < initial_norm, (
            f"CMA-ES must decrease mean norm on sphere: "
            f"initial={initial_norm:.4f}, final={final_norm:.4f}"
        )

    def test_run_twice_same_engine_identical_trajectory(self):
        """
        Running two independent PyCMAESState instances with the same mean/sigma/seed
        must produce identical trajectories — tests the idempotency guarantee.
        """
        n = 4
        bounds = [(-5.0, 5.0)] * n
        mean = [2.0] * n

        def run_trajectory(seed: int, n_gens: int) -> list[float]:
            s = PyCMAESState(mean, 0.5, 10, bounds)
            return [
                s.mean[0]  # track first gene of mean
                for gen in range(n_gens)
                for _ in [s.tell(s.ask(seed, gen),
                                 [neg_sphere(samp) for samp in s.ask(seed, gen)])]
            ]

        traj1 = run_trajectory(42, 10)
        traj2 = run_trajectory(42, 10)
        assert traj1 == traj2, "identical seeds must produce identical mean trajectories"


# ── Integer-gene two-array workflow ──────────────────────────────────────────

class TestIntegerGeneWorkflow:
    """
    CMAESEngine passes continuous samples to tell() and rounded samples to
    fitness functions. Verify that:
    1. ask() returns non-integer continuous samples (at least some)
    2. Rounding those samples produces integers
    3. tell() with the continuous (unrounded) samples runs without error
    """

    def test_continuous_samples_are_not_all_integers(self):
        """
        Continuous CMA-ES samples from a float distribution should not all
        happen to be exact integers. If any sample has a non-integer value,
        the two-array pattern is meaningful.
        """
        s = make_state(n=3, lambda_=20, sigma=0.5)
        samples = s.ask(42, 0)
        any_non_integer = any(
            g != round(g)
            for samp in samples
            for g in samp
        )
        assert any_non_integer, (
            "Continuous CMA-ES samples should not all be exact integers. "
            "If they are, the two-array tell/evaluate distinction has no effect."
        )

    def test_tell_with_continuous_samples_succeeds(self):
        """tell() must succeed even when samples have non-integer values."""
        s = make_state(n=3, lambda_=10, sigma=0.5)
        samples_continuous = s.ask(42, 0)
        fitnesses = [neg_sphere(samp) for samp in samples_continuous]
        s.tell(samples_continuous, fitnesses)  # must not raise
        assert s.generation == 1

    def test_rounded_samples_are_integers(self):
        """Rounding continuous samples produces integer-valued floats."""
        s = make_state(n=3, lambda_=10, sigma=0.5)
        samples_continuous = s.ask(42, 0)
        samples_rounded = [[round(g) for g in samp] for samp in samples_continuous]
        for samp in samples_rounded:
            for g in samp:
                assert g == int(g), f"rounded value {g} is not integer-valued"
```

- [ ] **Step 2: Run the tests — expect PASS**

```bash
pytest tests/unit/test_cmaes_rust.py -v
```

Expected: All tests pass. Example output:

```
tests/unit/test_cmaes_rust.py::TestPyCMAESStateConstruction::test_generation_starts_at_zero PASSED
tests/unit/test_cmaes_rust.py::TestPyCMAESStateConstruction::test_sigma_preserved PASSED
tests/unit/test_cmaes_rust.py::TestPyCMAESStateConstruction::test_mean_matches_input PASSED
tests/unit/test_cmaes_rust.py::TestPyCMAESStateConstruction::test_eigendecomp_interval_at_least_one PASSED
tests/unit/test_cmaes_rust.py::TestPyCMAESStateConstruction::test_invalid_bounds_length_raises PASSED
tests/unit/test_cmaes_rust.py::TestAskShape::test_returns_correct_lambda_samples PASSED
... (all pass)
```

- [ ] **Step 3: Run all Python unit tests — confirm no regressions**

```bash
pytest tests/unit/ -v
```

Expected: All prior tests pass plus the new CMA-ES tests.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_cmaes_rust.py
git commit -m "test(python): PyCMAESState smoke tests — shape, determinism, convergence, integer workflow"
```

---

## Task 4: Full Part 4 Verification

- [ ] **Step 1: Run the complete Rust test suite**

```bash
cargo test
```

Expected:
```
test result: ok. 123 passed; 0 failed
```

(97 from Parts 1–3 + 26 new from cmaes.rs)

- [ ] **Step 2: Run the complete Python unit test suite**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: All tests pass, 0 failures, 0 errors.

- [ ] **Step 3: Final end-to-end smoke test — 50-generation CMA-ES mini loop**

```bash
python - << 'EOF'
import math
from evocore._core import PyCMAESState

n = 10
bounds = [(-5.0, 5.0)] * n
mean_start = [3.0] * n
initial_norm = math.sqrt(sum(x*x for x in mean_start))

state = PyCMAESState(mean_start[:], 0.5, 50, bounds)

for gen in range(50):
    # Continuous samples → tell()
    samples_continuous = state.ask(42, gen)
    assert len(samples_continuous) == 50, "wrong lambda"
    assert all(len(s) == n for s in samples_continuous), "wrong gene length"
    assert all(
        b[0] - 1e-9 <= g <= b[1] + 1e-9
        for s in samples_continuous
        for g, b in zip(s, bounds)
    ), "samples outside bounds"

    # Fitness on continuous samples (CMAESEngine also computes discrete for user)
    fitnesses = [-sum(x*x for x in s) for s in samples_continuous]
    state.tell(samples_continuous, fitnesses)

    assert state.sigma > 0, "sigma must stay positive"
    assert all(math.isfinite(x) for x in state.mean), "mean must be finite"

final_norm = math.sqrt(sum(x*x for x in state.mean))
print(f"initial mean norm: {initial_norm:.4f}")
print(f"final   mean norm: {final_norm:.4f}")
print(f"final sigma:        {state.sigma:.6f}")
print(f"eigendecomp_interval: {state.eigendecomp_interval}")

assert final_norm < initial_norm, (
    f"CMA-ES did not converge: initial={initial_norm:.4f}, final={final_norm:.4f}"
)
print("\nPart 4 complete — 50-generation CMA-ES loop passed all assertions")
EOF
```

Expected:
```
initial mean norm: 9.4868
final   mean norm: ... (smaller value)
final sigma:        ...
eigendecomp_interval: 1

Part 4 complete — 50-generation CMA-ES loop passed all assertions
```

- [ ] **Step 4: Final commit and tag**

```bash
git add .
git commit -m "chore: Part 4 complete — PyCMAESState fully implemented and tested"
git tag part4-complete
```

---

## Part 4 Exit Criteria Checklist

- [ ] `cargo test` passes **123 Rust tests** (97 from Parts 1–3 + 26 new)
- [ ] `maturin develop --release` succeeds with no errors or warnings
- [ ] `pytest tests/unit/` passes all tests including `test_cmaes_rust.py`
- [ ] `PyCMAESState` importable from `evocore._core`
- [ ] `PyCMAESState` constructor accepts `(mean, sigma, lambda_, bounds)` — **no seed parameter**
- [ ] `ask(master_seed, generation)` returns `lambda` samples, each with `n` genes, all within bounds
- [ ] `ask()` is deterministic: same `(master_seed, generation)` always returns the same samples
- [ ] `ask()` is deterministic for a fixed CMA-ES state and `(master_seed, generation)`; after `tell()` updates mean/covariance, the same `(master_seed, generation)` may produce different samples
- [ ] Different `(master_seed, generation)` pairs produce different samples
- [ ] All samples from `ask()` lie within their per-gene `bounds` (mirror-folding enforced)
- [ ] `tell()` increments `generation` by 1 per call
- [ ] `sigma` stays positive and finite over at least 20 consecutive `tell()` calls
- [ ] All covariance matrix entries remain finite over at least 10 consecutive `tell()` calls
- [ ] Mean norm decreases on the sphere function over 30 generations (convergence signal)
- [ ] `tell()` accepts continuous (non-rounded) samples without error — two-array workflow verified
- [ ] NaN fitness values passed to `tell()` do not panic (treated as worst)
- [ ] Eigendecomposition cache uses `RefCell` + `Cell` for interior mutability — `ask()` is `&self`
- [ ] No RNG stored on `CMAESState` or `PyCMAESState` — all randomness derived per-call
- [ ] `mirror_fold` passes all 6 property tests including the "always in bounds" loop over 101 values
