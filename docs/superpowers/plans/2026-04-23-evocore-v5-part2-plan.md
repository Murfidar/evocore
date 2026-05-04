# evocore v5 — Part 2: Rust Genetic Operators

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all three Rust genetic operator families (`float_ops`, `int_ops`, `binary_ops`) and expose all 11 operator functions to Python via PyO3, using the v3 deterministic seed signature `(master_seed, generation, individual_idx)` throughout.

**Architecture:** All operators are pure functions — they take gene slices by reference, derive a stack-local `StdRng` via `derive_seed()`, apply the operation, and return new gene vectors. No RNG state escapes the function. All genes are `f64` at the PyO3 boundary — integer genes are encoded as `f64` and rounded by the operator; boolean genes are encoded as `0.0`/`1.0` and flipped numerically. Clamping to bounds is the Python `OperatorSet`'s responsibility (Part 5), not the operator's.

**Tech Stack:** Rust 1.78+, PyO3 0.21, rand 0.8, rand_distr 0.4, maturin 1.5+, pytest

**Prerequisite:** Part 1 complete — `src/utils.rs`, `src/individual.rs`, `src/lib.rs`, stubs for `src/operators/*.rs` all compile and pass their tests.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/operators/float_ops.rs` | Replace stub | BLX-α, SBX crossover; Gaussian, Uniform mutation |
| `src/operators/int_ops.rs` | Replace stub | Integer SBX, Gaussian, Uniform mutation — all operate on f64, round outputs |
| `src/operators/binary_ops.rs` | Replace stub | One-point, two-point, uniform XO; bit-flip mutation — all operate on f64 (0.0/1.0) |
| `src/operators/mod.rs` | Replace stub | `pub mod` declarations (already correct from Part 1 — verify only) |
| `src/lib.rs` | Modify | Add 11 PyO3 wrapper functions + register in `_core` module |

---

## Operator Signature Convention

Every operator in this part follows this exact pattern — no exceptions:

```rust
pub fn some_operator(
    /* operator-specific params */,
    master_seed:    u64,
    generation:     u64,
    individual_idx: u64,
) -> /* return type */ {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use crate::utils::{derive_seed, OP_CROSSOVER}; // or OP_MUTATION

    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    // ... operator logic uses rng ...
}
```

- Crossover operators use `OP_CROSSOVER`.
- Mutation operators use `OP_MUTATION`.
- The `StdRng` is stack-local — it is created, used, and dropped within the function.
- No seed parameter is passed from Python; no RNG state is stored anywhere.

---

## Task 1: Float Operators (`src/operators/float_ops.rs`)

Four functions: `blend_crossover`, `simulated_binary_crossover`, `gaussian_mutation`, `uniform_mutation`.

**Key v3 change:** `gaussian_mutation` has **no `mu` parameter** (removed in v2). It always adds noise centred at 0 — `g += N(0, sigma)`.

**Files:**
- Modify: `src/operators/float_ops.rs`

- [ ] **Step 1: Write failing Rust tests**

Replace the stub `src/operators/float_ops.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ── blend_crossover ──────────────────────────────────────────────────────

    #[test]
    fn test_blend_crossover_output_lengths() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let (c1, c2) = blend_crossover(&a, &b, 0.5, 42, 0, 0);
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
    }

    #[test]
    fn test_blend_crossover_deterministic() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1a, c2a) = blend_crossover(&a, &b, 0.5, 42, 5, 3);
        let (c1b, c2b) = blend_crossover(&a, &b, 0.5, 42, 5, 3);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_blend_crossover_different_generations_diverge() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1a, _) = blend_crossover(&a, &b, 0.5, 42, 0, 0);
        let (c1b, _) = blend_crossover(&a, &b, 0.5, 42, 1, 0);
        assert_ne!(c1a, c1b);
    }

    #[test]
    fn test_blend_crossover_alpha_zero_stays_within_parents() {
        // With alpha=0, offspring must lie within [min(ai,bi), max(ai,bi)]
        let a = vec![1.0_f64, 5.0, 3.0];
        let b = vec![4.0_f64, 2.0, 8.0];
        let (c1, c2) = blend_crossover(&a, &b, 0.0, 42, 0, 0);
        for i in 0..3 {
            let lo = a[i].min(b[i]);
            let hi = a[i].max(b[i]);
            assert!(c1[i] >= lo - 1e-10 && c1[i] <= hi + 1e-10,
                "c1[{}]={} outside [{}, {}]", i, c1[i], lo, hi);
            assert!(c2[i] >= lo - 1e-10 && c2[i] <= hi + 1e-10,
                "c2[{}]={} outside [{}, {}]", i, c2[i], lo, hi);
        }
    }

    // ── simulated_binary_crossover ───────────────────────────────────────────

    #[test]
    fn test_sbx_output_lengths() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1, c2) = simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert_eq!(c1.len(), 2);
        assert_eq!(c2.len(), 2);
    }

    #[test]
    fn test_sbx_deterministic() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let (c1a, c2a) = simulated_binary_crossover(&a, &b, 2.0, 99, 3, 7);
        let (c1b, c2b) = simulated_binary_crossover(&a, &b, 2.0, 99, 3, 7);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_sbx_children_sum_equals_parents_sum() {
        // SBX satisfies: c1[i] + c2[i] == a[i] + b[i] for all i
        let a = vec![1.0_f64, 3.0, 5.0];
        let b = vec![2.0_f64, 6.0, 8.0];
        let (c1, c2) = simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        for i in 0..3 {
            assert!((c1[i] + c2[i] - a[i] - b[i]).abs() < 1e-9,
                "SBX conservation violated at gene {}", i);
        }
    }

    // ── gaussian_mutation ────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_mutation_length_preserved() {
        let genes = vec![1.0_f64; 10];
        let result = gaussian_mutation(&genes, 0.1, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_gaussian_mutation_prob_zero_unchanged() {
        let genes = vec![5.0_f64; 5];
        let result = gaussian_mutation(&genes, 1.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes, "prob=0 must leave genes unchanged");
    }

    #[test]
    fn test_gaussian_mutation_prob_one_changes_genes() {
        // With prob=1, sigma=1, every gene should be perturbed (virtually certain)
        let genes = vec![100.0_f64; 20];
        let result = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 0);
        let changed = result.iter().zip(genes.iter()).filter(|(r, g)| (r - g).abs() > 1e-12).count();
        assert!(changed > 0, "prob=1 should mutate at least one gene");
    }

    #[test]
    fn test_gaussian_mutation_deterministic() {
        let genes = vec![1.0_f64, 2.0, 3.0];
        let r1 = gaussian_mutation(&genes, 0.5, 0.8, 42, 10, 3);
        let r2 = gaussian_mutation(&genes, 0.5, 0.8, 42, 10, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_gaussian_mutation_different_individuals_diverge() {
        let genes = vec![1.0_f64; 5];
        let r1 = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 0);
        let r2 = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 1);
        assert_ne!(r1, r2, "different individual_idx must produce different mutations");
    }

    // ── uniform_mutation ─────────────────────────────────────────────────────

    #[test]
    fn test_uniform_mutation_length_preserved() {
        let genes = vec![0.0_f64; 8];
        let result = uniform_mutation(&genes, -1.0, 1.0, 0.5, 42, 0, 0);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_uniform_mutation_prob_zero_unchanged() {
        let genes = vec![3.0_f64; 5];
        let result = uniform_mutation(&genes, 0.0, 1.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_uniform_mutation_respects_bounds_prob_one() {
        // With prob=1, every gene must be resampled within [low, high]
        let genes = vec![0.0_f64; 100];
        let result = uniform_mutation(&genes, -1.0, 1.0, 1.0, 42, 0, 0);
        for v in &result {
            assert!(*v >= -1.0 && *v < 1.0, "value {} outside [-1.0, 1.0)", v);
        }
    }

    #[test]
    fn test_uniform_mutation_deterministic() {
        let genes = vec![0.0_f64; 5];
        let r1 = uniform_mutation(&genes, -1.0, 1.0, 0.8, 7, 2, 4);
        let r2 = uniform_mutation(&genes, -1.0, 1.0, 0.8, 7, 2, 4);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL (functions not defined yet)**

```bash
cargo test float_ops 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function 'blend_crossover'`

- [ ] **Step 3: Implement src/operators/float_ops.rs**

Replace the file entirely with the implementation + tests:

```rust
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

// ── BLX-α Crossover ──────────────────────────────────────────────────────────

/// BLX-α (Blend) crossover.
///
/// For each gene i, samples two offspring values from the interval
/// [min(ai,bi) - α·|ai-bi|, max(ai,bi) + α·|ai-bi|].
/// α=0 restricts offspring to the parents' range; α=0.5 extends by 50%.
/// Does NOT clamp to gene bounds — that is the OperatorSet's responsibility.
pub fn blend_crossover(
    a: &[f64], b: &[f64], alpha: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "blend_crossover: parent lengths must match");
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    let mut c1 = Vec::with_capacity(a.len());
    let mut c2 = Vec::with_capacity(a.len());
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff  = (ai - bi).abs();
        let lo    = ai.min(bi) - alpha * diff;
        let hi    = ai.max(bi) + alpha * diff;
        // gen_range(lo..hi) is safe when lo < hi; when lo == hi (identical genes)
        // diff == 0 so lo == hi == ai — return the parent value unchanged.
        if lo < hi {
            c1.push(rng.gen_range(lo..hi));
            c2.push(rng.gen_range(lo..hi));
        } else {
            c1.push(ai);
            c2.push(bi);
        }
    }
    (c1, c2)
}

// ── Simulated Binary Crossover (SBX) ─────────────────────────────────────────

/// Simulated Binary Crossover (SBX).
///
/// Mimics single-point crossover on binary strings in the continuous domain.
/// Higher η (distribution index) biases offspring closer to parents.
/// Satisfies c1[i] + c2[i] = a[i] + b[i] for every gene i.
/// Does NOT clamp to gene bounds.
pub fn simulated_binary_crossover(
    a: &[f64], b: &[f64], eta: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "sbx: parent lengths must match");
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    let mut c1 = Vec::with_capacity(a.len());
    let mut c2 = Vec::with_capacity(a.len());
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let u: f64 = rng.gen();
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };
        c1.push(0.5 * ((1.0 + beta) * ai + (1.0 - beta) * bi));
        c2.push(0.5 * ((1.0 - beta) * ai + (1.0 + beta) * bi));
    }
    (c1, c2)
}

// ── Gaussian Mutation ─────────────────────────────────────────────────────────

/// Per-gene Gaussian mutation.
///
/// Each gene is independently perturbed: g += N(0, sigma) with probability `prob`.
/// `mu` is intentionally absent — mutation noise is always centred at 0.
/// A biased mutation (mu ≠ 0) is a different operator; this one is unbiased.
/// Does NOT clamp to gene bounds.
pub fn gaussian_mutation(
    genes: &[f64], sigma: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    let normal = Normal::new(0.0_f64, sigma).expect("sigma must be > 0");
    genes.iter().map(|&g| {
        if rng.gen::<f64>() < prob {
            g + normal.sample(&mut rng)
        } else {
            g
        }
    }).collect()
}

// ── Uniform Mutation ─────────────────────────────────────────────────────────

/// Per-gene uniform mutation.
///
/// Each gene is independently resampled from U(low, high) with probability `prob`.
/// Note: samples from the half-open interval [low, high).
/// Does NOT further clamp — caller is responsible for passing valid bounds.
pub fn uniform_mutation(
    genes: &[f64], low: f64, high: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    genes.iter().map(|&g| {
        if rng.gen::<f64>() < prob {
            rng.gen_range(low..high)
        } else {
            g
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_crossover_output_lengths() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let (c1, c2) = blend_crossover(&a, &b, 0.5, 42, 0, 0);
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
    }

    #[test]
    fn test_blend_crossover_deterministic() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1a, c2a) = blend_crossover(&a, &b, 0.5, 42, 5, 3);
        let (c1b, c2b) = blend_crossover(&a, &b, 0.5, 42, 5, 3);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_blend_crossover_different_generations_diverge() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1a, _) = blend_crossover(&a, &b, 0.5, 42, 0, 0);
        let (c1b, _) = blend_crossover(&a, &b, 0.5, 42, 1, 0);
        assert_ne!(c1a, c1b);
    }

    #[test]
    fn test_blend_crossover_alpha_zero_stays_within_parents() {
        let a = vec![1.0_f64, 5.0, 3.0];
        let b = vec![4.0_f64, 2.0, 8.0];
        let (c1, c2) = blend_crossover(&a, &b, 0.0, 42, 0, 0);
        for i in 0..3 {
            let lo = a[i].min(b[i]);
            let hi = a[i].max(b[i]);
            assert!(c1[i] >= lo - 1e-10 && c1[i] <= hi + 1e-10,
                "c1[{}]={} outside [{}, {}]", i, c1[i], lo, hi);
            assert!(c2[i] >= lo - 1e-10 && c2[i] <= hi + 1e-10,
                "c2[{}]={} outside [{}, {}]", i, c2[i], lo, hi);
        }
    }

    #[test]
    fn test_sbx_output_lengths() {
        let a = vec![1.0_f64, 2.0];
        let b = vec![3.0_f64, 4.0];
        let (c1, c2) = simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert_eq!(c1.len(), 2);
        assert_eq!(c2.len(), 2);
    }

    #[test]
    fn test_sbx_deterministic() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![4.0_f64, 5.0, 6.0];
        let (c1a, c2a) = simulated_binary_crossover(&a, &b, 2.0, 99, 3, 7);
        let (c1b, c2b) = simulated_binary_crossover(&a, &b, 2.0, 99, 3, 7);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_sbx_children_sum_equals_parents_sum() {
        let a = vec![1.0_f64, 3.0, 5.0];
        let b = vec![2.0_f64, 6.0, 8.0];
        let (c1, c2) = simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        for i in 0..3 {
            assert!((c1[i] + c2[i] - a[i] - b[i]).abs() < 1e-9,
                "SBX conservation violated at gene {}", i);
        }
    }

    #[test]
    fn test_gaussian_mutation_length_preserved() {
        let genes = vec![1.0_f64; 10];
        let result = gaussian_mutation(&genes, 0.1, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_gaussian_mutation_prob_zero_unchanged() {
        let genes = vec![5.0_f64; 5];
        let result = gaussian_mutation(&genes, 1.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_gaussian_mutation_prob_one_changes_genes() {
        let genes = vec![100.0_f64; 20];
        let result = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 0);
        let changed = result.iter().zip(genes.iter())
            .filter(|(r, g)| (r - g).abs() > 1e-12).count();
        assert!(changed > 0);
    }

    #[test]
    fn test_gaussian_mutation_deterministic() {
        let genes = vec![1.0_f64, 2.0, 3.0];
        let r1 = gaussian_mutation(&genes, 0.5, 0.8, 42, 10, 3);
        let r2 = gaussian_mutation(&genes, 0.5, 0.8, 42, 10, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_gaussian_mutation_different_individuals_diverge() {
        let genes = vec![1.0_f64; 5];
        let r1 = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 0);
        let r2 = gaussian_mutation(&genes, 1.0, 1.0, 42, 0, 1);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_uniform_mutation_length_preserved() {
        let genes = vec![0.0_f64; 8];
        let result = uniform_mutation(&genes, -1.0, 1.0, 0.5, 42, 0, 0);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_uniform_mutation_prob_zero_unchanged() {
        let genes = vec![3.0_f64; 5];
        let result = uniform_mutation(&genes, 0.0, 1.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_uniform_mutation_respects_bounds_prob_one() {
        let genes = vec![0.0_f64; 100];
        let result = uniform_mutation(&genes, -1.0, 1.0, 1.0, 42, 0, 0);
        for v in &result {
            assert!(*v >= -1.0 && *v < 1.0, "value {} outside [-1.0, 1.0)", v);
        }
    }

    #[test]
    fn test_uniform_mutation_deterministic() {
        let genes = vec![0.0_f64; 5];
        let r1 = uniform_mutation(&genes, -1.0, 1.0, 0.8, 7, 2, 4);
        let r2 = uniform_mutation(&genes, -1.0, 1.0, 0.8, 7, 2, 4);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test float_ops
```

Expected:
```
test operators::float_ops::tests::test_blend_crossover_alpha_zero_stays_within_parents ... ok
test operators::float_ops::tests::test_blend_crossover_deterministic ... ok
test operators::float_ops::tests::test_blend_crossover_different_generations_diverge ... ok
test operators::float_ops::tests::test_blend_crossover_output_lengths ... ok
test operators::float_ops::tests::test_gaussian_mutation_deterministic ... ok
test operators::float_ops::tests::test_gaussian_mutation_different_individuals_diverge ... ok
test operators::float_ops::tests::test_gaussian_mutation_length_preserved ... ok
test operators::float_ops::tests::test_gaussian_mutation_prob_one_changes_genes ... ok
test operators::float_ops::tests::test_gaussian_mutation_prob_zero_unchanged ... ok
test operators::float_ops::tests::test_sbx_children_sum_equals_parents_sum ... ok
test operators::float_ops::tests::test_sbx_deterministic ... ok
test operators::float_ops::tests::test_sbx_output_lengths ... ok
test operators::float_ops::tests::test_uniform_mutation_deterministic ... ok
test operators::float_ops::tests::test_uniform_mutation_length_preserved ... ok
test operators::float_ops::tests::test_uniform_mutation_prob_zero_unchanged ... ok
test operators::float_ops::tests::test_uniform_mutation_respects_bounds_prob_one ... ok

test result: ok. 16 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/operators/float_ops.rs
git commit -m "feat(rust): float_ops — BLX-α, SBX, gaussian_mutation, uniform_mutation"
```

---

## Task 2: Integer Operators (`src/operators/int_ops.rs`)

Three functions operating on `f64`-encoded integers. Each function rounds its output to the nearest integer **before returning** — the result is still `f64` (matching the PyO3 boundary convention) but will always satisfy `x.fract() == 0.0`.

**Files:**
- Modify: `src/operators/int_ops.rs`

- [ ] **Step 1: Write failing Rust tests**

Replace the stub `src/operators/int_ops.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn is_integer(x: f64) -> bool {
        x.fract() == 0.0
    }

    // ── int_simulated_binary_crossover ───────────────────────────────────────

    #[test]
    fn test_int_sbx_output_lengths() {
        let a = vec![10.0_f64, 50.0, 100.0];
        let b = vec![20.0_f64, 80.0, 200.0];
        let (c1, c2) = int_simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
    }

    #[test]
    fn test_int_sbx_outputs_are_integers() {
        let a = vec![10.0_f64, 50.0, 100.0];
        let b = vec![20.0_f64, 80.0, 200.0];
        let (c1, c2) = int_simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert!(c1.iter().all(|&x| is_integer(x)), "c1 must contain only integers");
        assert!(c2.iter().all(|&x| is_integer(x)), "c2 must contain only integers");
    }

    #[test]
    fn test_int_sbx_deterministic() {
        let a = vec![5.0_f64, 10.0];
        let b = vec![15.0_f64, 20.0];
        let (c1a, c2a) = int_simulated_binary_crossover(&a, &b, 2.0, 7, 3, 1);
        let (c1b, c2b) = int_simulated_binary_crossover(&a, &b, 2.0, 7, 3, 1);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    // ── int_gaussian_mutation ────────────────────────────────────────────────

    #[test]
    fn test_int_gaussian_mutation_length_preserved() {
        let genes = vec![100.0_f64; 8];
        let result = int_gaussian_mutation(&genes, 10.0, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_int_gaussian_mutation_outputs_are_integers() {
        let genes = vec![50.0_f64; 20];
        let result = int_gaussian_mutation(&genes, 5.0, 1.0, 42, 0, 0);
        assert!(result.iter().all(|&x| is_integer(x)),
            "All int_gaussian_mutation outputs must be integers");
    }

    #[test]
    fn test_int_gaussian_mutation_prob_zero_unchanged() {
        let genes = vec![42.0_f64; 5];
        let result = int_gaussian_mutation(&genes, 10.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_int_gaussian_mutation_deterministic() {
        let genes = vec![10.0_f64, 20.0, 30.0];
        let r1 = int_gaussian_mutation(&genes, 3.0, 0.9, 13, 5, 2);
        let r2 = int_gaussian_mutation(&genes, 3.0, 0.9, 13, 5, 2);
        assert_eq!(r1, r2);
    }

    // ── int_uniform_mutation ─────────────────────────────────────────────────

    #[test]
    fn test_int_uniform_mutation_length_preserved() {
        let genes = vec![50.0_f64; 6];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_int_uniform_mutation_outputs_are_integers() {
        let genes = vec![50.0_f64; 50];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        assert!(result.iter().all(|&x| is_integer(x)),
            "All int_uniform_mutation outputs must be integers");
    }

    #[test]
    fn test_int_uniform_mutation_respects_bounds() {
        let genes = vec![50.0_f64; 100];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        for v in &result {
            assert!(*v >= 5.0 && *v <= 200.0,
                "value {} outside integer bounds [5, 200]", v);
        }
    }

    #[test]
    fn test_int_uniform_mutation_prob_zero_unchanged() {
        let genes = vec![77.0_f64; 5];
        let result = int_uniform_mutation(&genes, 1.0, 100.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_int_uniform_mutation_deterministic() {
        let genes = vec![50.0_f64; 4];
        let r1 = int_uniform_mutation(&genes, 10.0, 500.0, 0.8, 9, 1, 2);
        let r2 = int_uniform_mutation(&genes, 10.0, 500.0, 0.8, 9, 1, 2);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cargo test int_ops 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function 'int_simulated_binary_crossover'`

- [ ] **Step 3: Implement src/operators/int_ops.rs**

```rust
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

// ── Integer SBX ───────────────────────────────────────────────────────────────

/// Integer Simulated Binary Crossover.
///
/// Applies SBX to f64-encoded integer genes, then rounds each offspring value
/// to the nearest integer (returned as f64, fract() == 0).
/// Input genes must be encoded as f64 (e.g., 42 → 42.0).
/// Does NOT clamp to gene bounds.
pub fn int_simulated_binary_crossover(
    a: &[f64], b: &[f64], eta: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "int_sbx: parent lengths must match");
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    let mut c1 = Vec::with_capacity(a.len());
    let mut c2 = Vec::with_capacity(a.len());
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let u: f64 = rng.gen();
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };
        c1.push((0.5 * ((1.0 + beta) * ai + (1.0 - beta) * bi)).round());
        c2.push((0.5 * ((1.0 - beta) * ai + (1.0 + beta) * bi)).round());
    }
    (c1, c2)
}

// ── Integer Gaussian Mutation ─────────────────────────────────────────────────

/// Per-gene integer Gaussian mutation.
///
/// Adds N(0, sigma) noise to each gene with probability `prob`, then rounds
/// to the nearest integer (result is still f64 with fract() == 0).
/// sigma is an absolute step size in integer units (not a fraction of range).
/// Does NOT clamp to gene bounds.
pub fn int_gaussian_mutation(
    genes: &[f64], sigma: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    let normal = Normal::new(0.0_f64, sigma).expect("sigma must be > 0");
    genes.iter().map(|&g| {
        if rng.gen::<f64>() < prob {
            (g + normal.sample(&mut rng)).round()
        } else {
            g
        }
    }).collect()
}

// ── Integer Uniform Mutation ─────────────────────────────────────────────────

/// Per-gene integer uniform mutation.
///
/// Resamples each gene uniformly from the integer range [round(low), round(high)]
/// (inclusive on both ends) with probability `prob`.
/// Result values satisfy: fract() == 0 and low <= v <= high.
/// low and high are passed as f64 (the PyO3 boundary representation) but must
/// represent integer bounds (e.g., 5.0 for the integer 5).
/// Does NOT clamp — caller is responsible for passing valid bounds.
pub fn int_uniform_mutation(
    genes: &[f64], low: f64, high: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    let lo = low.round() as i64;
    let hi = high.round() as i64;
    genes.iter().map(|&g| {
        if rng.gen::<f64>() < prob {
            rng.gen_range(lo..=hi) as f64
        } else {
            g
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_integer(x: f64) -> bool {
        x.fract() == 0.0
    }

    #[test]
    fn test_int_sbx_output_lengths() {
        let a = vec![10.0_f64, 50.0, 100.0];
        let b = vec![20.0_f64, 80.0, 200.0];
        let (c1, c2) = int_simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert_eq!(c1.len(), 3);
        assert_eq!(c2.len(), 3);
    }

    #[test]
    fn test_int_sbx_outputs_are_integers() {
        let a = vec![10.0_f64, 50.0, 100.0];
        let b = vec![20.0_f64, 80.0, 200.0];
        let (c1, c2) = int_simulated_binary_crossover(&a, &b, 2.0, 42, 0, 0);
        assert!(c1.iter().all(|&x| is_integer(x)));
        assert!(c2.iter().all(|&x| is_integer(x)));
    }

    #[test]
    fn test_int_sbx_deterministic() {
        let a = vec![5.0_f64, 10.0];
        let b = vec![15.0_f64, 20.0];
        let (c1a, c2a) = int_simulated_binary_crossover(&a, &b, 2.0, 7, 3, 1);
        let (c1b, c2b) = int_simulated_binary_crossover(&a, &b, 2.0, 7, 3, 1);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_int_gaussian_mutation_length_preserved() {
        let genes = vec![100.0_f64; 8];
        let result = int_gaussian_mutation(&genes, 10.0, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_int_gaussian_mutation_outputs_are_integers() {
        let genes = vec![50.0_f64; 20];
        let result = int_gaussian_mutation(&genes, 5.0, 1.0, 42, 0, 0);
        assert!(result.iter().all(|&x| is_integer(x)));
    }

    #[test]
    fn test_int_gaussian_mutation_prob_zero_unchanged() {
        let genes = vec![42.0_f64; 5];
        let result = int_gaussian_mutation(&genes, 10.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_int_gaussian_mutation_deterministic() {
        let genes = vec![10.0_f64, 20.0, 30.0];
        let r1 = int_gaussian_mutation(&genes, 3.0, 0.9, 13, 5, 2);
        let r2 = int_gaussian_mutation(&genes, 3.0, 0.9, 13, 5, 2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_int_uniform_mutation_length_preserved() {
        let genes = vec![50.0_f64; 6];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_int_uniform_mutation_outputs_are_integers() {
        let genes = vec![50.0_f64; 50];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        assert!(result.iter().all(|&x| is_integer(x)));
    }

    #[test]
    fn test_int_uniform_mutation_respects_bounds() {
        let genes = vec![50.0_f64; 100];
        let result = int_uniform_mutation(&genes, 5.0, 200.0, 1.0, 42, 0, 0);
        for v in &result {
            assert!(*v >= 5.0 && *v <= 200.0,
                "value {} outside [5, 200]", v);
        }
    }

    #[test]
    fn test_int_uniform_mutation_prob_zero_unchanged() {
        let genes = vec![77.0_f64; 5];
        let result = int_uniform_mutation(&genes, 1.0, 100.0, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_int_uniform_mutation_deterministic() {
        let genes = vec![50.0_f64; 4];
        let r1 = int_uniform_mutation(&genes, 10.0, 500.0, 0.8, 9, 1, 2);
        let r2 = int_uniform_mutation(&genes, 10.0, 500.0, 0.8, 9, 1, 2);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test int_ops
```

Expected:
```
test operators::int_ops::tests::test_int_gaussian_mutation_deterministic ... ok
test operators::int_ops::tests::test_int_gaussian_mutation_length_preserved ... ok
test operators::int_ops::tests::test_int_gaussian_mutation_outputs_are_integers ... ok
test operators::int_ops::tests::test_int_gaussian_mutation_prob_zero_unchanged ... ok
test operators::int_ops::tests::test_int_sbx_deterministic ... ok
test operators::int_ops::tests::test_int_sbx_output_lengths ... ok
test operators::int_ops::tests::test_int_sbx_outputs_are_integers ... ok
test operators::int_ops::tests::test_int_uniform_mutation_deterministic ... ok
test operators::int_ops::tests::test_int_uniform_mutation_length_preserved ... ok
test operators::int_ops::tests::test_int_uniform_mutation_outputs_are_integers ... ok
test operators::int_ops::tests::test_int_uniform_mutation_prob_zero_unchanged ... ok
test operators::int_ops::tests::test_int_uniform_mutation_respects_bounds ... ok

test result: ok. 12 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/operators/int_ops.rs
git commit -m "feat(rust): int_ops — int_sbx, int_gaussian_mutation, int_uniform_mutation"
```

---

## Task 3: Binary Operators (`src/operators/binary_ops.rs`)

Four functions operating on `f64`-encoded booleans (`0.0` = false, `1.0` = true). All outputs contain only `0.0` or `1.0`. Decoding (`x >= 0.5 → true`) lives in the Python `OperatorSet` (Part 5).

**Files:**
- Modify: `src/operators/binary_ops.rs`

- [ ] **Step 1: Write failing Rust tests**

Replace the stub with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn only_binary(vals: &[f64]) -> bool {
        vals.iter().all(|&x| x == 0.0 || x == 1.0)
    }

    // ── one_point_crossover ──────────────────────────────────────────────────

    #[test]
    fn test_one_point_crossover_lengths() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);
    }

    #[test]
    fn test_one_point_crossover_only_binary_values() {
        let a = vec![1.0_f64; 10];
        let b = vec![0.0_f64; 10];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        assert!(only_binary(&c1), "c1 must contain only 0.0 or 1.0");
        assert!(only_binary(&c2), "c2 must contain only 0.0 or 1.0");
    }

    #[test]
    fn test_one_point_crossover_deterministic() {
        let a = vec![1.0_f64, 1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0_f64, 1.0, 1.0, 0.0, 0.0];
        let (c1a, c2a) = one_point_crossover(&a, &b, 7, 2, 5);
        let (c1b, c2b) = one_point_crossover(&a, &b, 7, 2, 5);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_one_point_crossover_children_partition_parents() {
        // Each position in c1 comes from either a or b.
        // The complement check: c1[i] + c2[i] == a[i] + b[i] for all i.
        let a = vec![1.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let b = vec![0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        for i in 0..8 {
            assert_eq!(c1[i] + c2[i], a[i] + b[i],
                "at position {}: c1+c2 must equal a+b", i);
        }
    }

    // ── two_point_crossover ──────────────────────────────────────────────────

    #[test]
    fn test_two_point_crossover_lengths() {
        let a = vec![1.0_f64; 10];
        let b = vec![0.0_f64; 10];
        let (c1, c2) = two_point_crossover(&a, &b, 42, 0, 0);
        assert_eq!(c1.len(), 10);
        assert_eq!(c2.len(), 10);
    }

    #[test]
    fn test_two_point_crossover_only_binary_values() {
        let a = vec![1.0_f64; 12];
        let b = vec![0.0_f64; 12];
        let (c1, c2) = two_point_crossover(&a, &b, 42, 0, 0);
        assert!(only_binary(&c1));
        assert!(only_binary(&c2));
    }

    #[test]
    fn test_two_point_crossover_deterministic() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1a, c2a) = two_point_crossover(&a, &b, 99, 4, 2);
        let (c1b, c2b) = two_point_crossover(&a, &b, 99, 4, 2);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    // ── uniform_crossover ────────────────────────────────────────────────────

    #[test]
    fn test_uniform_crossover_lengths() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1, c2) = uniform_crossover(&a, &b, 0.5, 42, 0, 0);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);
    }

    #[test]
    fn test_uniform_crossover_only_binary_values() {
        let a = vec![1.0_f64; 20];
        let b = vec![0.0_f64; 20];
        let (c1, c2) = uniform_crossover(&a, &b, 0.5, 42, 0, 0);
        assert!(only_binary(&c1));
        assert!(only_binary(&c2));
    }

    #[test]
    fn test_uniform_crossover_deterministic() {
        let a = vec![1.0_f64; 6];
        let b = vec![0.0_f64; 6];
        let (c1a, c2a) = uniform_crossover(&a, &b, 0.5, 3, 1, 0);
        let (c1b, c2b) = uniform_crossover(&a, &b, 0.5, 3, 1, 0);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    // ── bit_flip_mutation ────────────────────────────────────────────────────

    #[test]
    fn test_bit_flip_mutation_length_preserved() {
        let genes = vec![1.0_f64; 10];
        let result = bit_flip_mutation(&genes, 0.5, 42, 0, 0);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_bit_flip_mutation_only_binary_values() {
        let genes = vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let result = bit_flip_mutation(&genes, 0.5, 42, 0, 0);
        assert!(only_binary(&result), "bit_flip output must contain only 0.0 or 1.0");
    }

    #[test]
    fn test_bit_flip_mutation_prob_zero_unchanged() {
        let genes = vec![1.0_f64, 0.0, 1.0, 0.0];
        let result = bit_flip_mutation(&genes, 0.0, 42, 0, 0);
        assert_eq!(result, genes, "prob=0 must leave genes unchanged");
    }

    #[test]
    fn test_bit_flip_mutation_prob_one_all_flipped() {
        let genes = vec![1.0_f64, 0.0, 1.0, 0.0];
        let result = bit_flip_mutation(&genes, 1.0, 42, 0, 0);
        let expected = vec![0.0_f64, 1.0, 0.0, 1.0];
        assert_eq!(result, expected, "prob=1 must flip every bit");
    }

    #[test]
    fn test_bit_flip_mutation_deterministic() {
        let genes = vec![0.0_f64, 1.0, 0.0, 1.0, 1.0];
        let r1 = bit_flip_mutation(&genes, 0.5, 5, 3, 1);
        let r2 = bit_flip_mutation(&genes, 0.5, 5, 3, 1);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cargo test binary_ops 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function 'one_point_crossover'`

- [ ] **Step 3: Implement src/operators/binary_ops.rs**

```rust
use rand::prelude::*;
use rand::rngs::StdRng;
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

// ── One-point Crossover ───────────────────────────────────────────────────────

/// One-point crossover on f64-encoded boolean genes (0.0 = false, 1.0 = true).
///
/// Selects a random cut point in 1..n, exchanges tails between parents.
/// Outputs contain only 0.0 or 1.0 since inputs are assumed to be valid encodings.
pub fn one_point_crossover(
    a: &[f64], b: &[f64],
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "one_point_crossover: parent lengths must match");
    let n = a.len();
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    // For n <= 1 there is no meaningful cut point — return parents unchanged.
    let point = if n > 1 { rng.gen_range(1..n) } else { 0 };
    let c1: Vec<f64> = a[..point].iter().chain(b[point..].iter()).cloned().collect();
    let c2: Vec<f64> = b[..point].iter().chain(a[point..].iter()).cloned().collect();
    (c1, c2)
}

// ── Two-point Crossover ───────────────────────────────────────────────────────

/// Two-point crossover on f64-encoded boolean genes.
///
/// Selects two random positions p1 <= p2 in 0..n; the segment [p1..p2] is
/// swapped between parents. If p1 == p2, the result is identical to parents.
pub fn two_point_crossover(
    a: &[f64], b: &[f64],
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "two_point_crossover: parent lengths must match");
    let n = a.len();
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    let (p1, p2) = if n > 1 {
        let x = rng.gen_range(0..n);
        let y = rng.gen_range(0..n);
        if x <= y { (x, y) } else { (y, x) }
    } else {
        (0, 0)
    };
    let c1: Vec<f64> = (0..n)
        .map(|i| if i >= p1 && i < p2 { b[i] } else { a[i] })
        .collect();
    let c2: Vec<f64> = (0..n)
        .map(|i| if i >= p1 && i < p2 { a[i] } else { b[i] })
        .collect();
    (c1, c2)
}

// ── Uniform Crossover ─────────────────────────────────────────────────────────

/// Per-gene uniform crossover on f64-encoded boolean genes.
///
/// Each gene is independently swapped between parents with probability `prob`.
/// At prob=0.5 each gene is equally likely to come from either parent.
pub fn uniform_crossover(
    a: &[f64], b: &[f64], prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "uniform_crossover: parent lengths must match");
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER)
    );
    let mut c1 = Vec::with_capacity(a.len());
    let mut c2 = Vec::with_capacity(a.len());
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if rng.gen::<f64>() < prob {
            c1.push(bi);
            c2.push(ai);
        } else {
            c1.push(ai);
            c2.push(bi);
        }
    }
    (c1, c2)
}

// ── Bit-flip Mutation ─────────────────────────────────────────────────────────

/// Per-gene bit-flip mutation on f64-encoded boolean genes.
///
/// Each gene is flipped (0.0 → 1.0, 1.0 → 0.0) with probability `prob`.
/// Inputs are assumed to be valid 0.0/1.0 encodings; outputs are always 0.0/1.0.
pub fn bit_flip_mutation(
    genes: &[f64], prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    genes.iter().map(|&g| {
        if rng.gen::<f64>() < prob {
            if g >= 0.5 { 0.0 } else { 1.0 }
        } else {
            g
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn only_binary(vals: &[f64]) -> bool {
        vals.iter().all(|&x| x == 0.0 || x == 1.0)
    }

    #[test]
    fn test_one_point_crossover_lengths() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);
    }

    #[test]
    fn test_one_point_crossover_only_binary_values() {
        let a = vec![1.0_f64; 10];
        let b = vec![0.0_f64; 10];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        assert!(only_binary(&c1));
        assert!(only_binary(&c2));
    }

    #[test]
    fn test_one_point_crossover_deterministic() {
        let a = vec![1.0_f64, 1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0_f64, 1.0, 1.0, 0.0, 0.0];
        let (c1a, c2a) = one_point_crossover(&a, &b, 7, 2, 5);
        let (c1b, c2b) = one_point_crossover(&a, &b, 7, 2, 5);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_one_point_crossover_children_partition_parents() {
        let a = vec![1.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let b = vec![0.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let (c1, c2) = one_point_crossover(&a, &b, 42, 0, 0);
        for i in 0..8 {
            assert_eq!(c1[i] + c2[i], a[i] + b[i],
                "partition property violated at position {}", i);
        }
    }

    #[test]
    fn test_two_point_crossover_lengths() {
        let a = vec![1.0_f64; 10];
        let b = vec![0.0_f64; 10];
        let (c1, c2) = two_point_crossover(&a, &b, 42, 0, 0);
        assert_eq!(c1.len(), 10);
        assert_eq!(c2.len(), 10);
    }

    #[test]
    fn test_two_point_crossover_only_binary_values() {
        let a = vec![1.0_f64; 12];
        let b = vec![0.0_f64; 12];
        let (c1, c2) = two_point_crossover(&a, &b, 42, 0, 0);
        assert!(only_binary(&c1));
        assert!(only_binary(&c2));
    }

    #[test]
    fn test_two_point_crossover_deterministic() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1a, c2a) = two_point_crossover(&a, &b, 99, 4, 2);
        let (c1b, c2b) = two_point_crossover(&a, &b, 99, 4, 2);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_uniform_crossover_lengths() {
        let a = vec![1.0_f64; 8];
        let b = vec![0.0_f64; 8];
        let (c1, c2) = uniform_crossover(&a, &b, 0.5, 42, 0, 0);
        assert_eq!(c1.len(), 8);
        assert_eq!(c2.len(), 8);
    }

    #[test]
    fn test_uniform_crossover_only_binary_values() {
        let a = vec![1.0_f64; 20];
        let b = vec![0.0_f64; 20];
        let (c1, c2) = uniform_crossover(&a, &b, 0.5, 42, 0, 0);
        assert!(only_binary(&c1));
        assert!(only_binary(&c2));
    }

    #[test]
    fn test_uniform_crossover_deterministic() {
        let a = vec![1.0_f64; 6];
        let b = vec![0.0_f64; 6];
        let (c1a, c2a) = uniform_crossover(&a, &b, 0.5, 3, 1, 0);
        let (c1b, c2b) = uniform_crossover(&a, &b, 0.5, 3, 1, 0);
        assert_eq!(c1a, c1b);
        assert_eq!(c2a, c2b);
    }

    #[test]
    fn test_bit_flip_mutation_length_preserved() {
        let genes = vec![1.0_f64; 10];
        let result = bit_flip_mutation(&genes, 0.5, 42, 0, 0);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_bit_flip_mutation_only_binary_values() {
        let genes = vec![0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let result = bit_flip_mutation(&genes, 0.5, 42, 0, 0);
        assert!(only_binary(&result));
    }

    #[test]
    fn test_bit_flip_mutation_prob_zero_unchanged() {
        let genes = vec![1.0_f64, 0.0, 1.0, 0.0];
        let result = bit_flip_mutation(&genes, 0.0, 42, 0, 0);
        assert_eq!(result, genes);
    }

    #[test]
    fn test_bit_flip_mutation_prob_one_all_flipped() {
        let genes = vec![1.0_f64, 0.0, 1.0, 0.0];
        let result = bit_flip_mutation(&genes, 1.0, 42, 0, 0);
        assert_eq!(result, vec![0.0_f64, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_bit_flip_mutation_deterministic() {
        let genes = vec![0.0_f64, 1.0, 0.0, 1.0, 1.0];
        let r1 = bit_flip_mutation(&genes, 0.5, 5, 3, 1);
        let r2 = bit_flip_mutation(&genes, 0.5, 5, 3, 1);
        assert_eq!(r1, r2);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test binary_ops
```

Expected:
```
test operators::binary_ops::tests::test_bit_flip_mutation_deterministic ... ok
test operators::binary_ops::tests::test_bit_flip_mutation_length_preserved ... ok
test operators::binary_ops::tests::test_bit_flip_mutation_only_binary_values ... ok
test operators::binary_ops::tests::test_bit_flip_mutation_prob_one_all_flipped ... ok
test operators::binary_ops::tests::test_bit_flip_mutation_prob_zero_unchanged ... ok
test operators::binary_ops::tests::test_one_point_crossover_children_partition_parents ... ok
test operators::binary_ops::tests::test_one_point_crossover_deterministic ... ok
test operators::binary_ops::tests::test_one_point_crossover_lengths ... ok
test operators::binary_ops::tests::test_one_point_crossover_only_binary_values ... ok
test operators::binary_ops::tests::test_two_point_crossover_deterministic ... ok
test operators::binary_ops::tests::test_two_point_crossover_lengths ... ok
test operators::binary_ops::tests::test_two_point_crossover_only_binary_values ... ok
test operators::binary_ops::tests::test_uniform_crossover_deterministic ... ok
test operators::binary_ops::tests::test_uniform_crossover_lengths ... ok
test operators::binary_ops::tests::test_uniform_crossover_only_binary_values ... ok

test result: ok. 15 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/operators/binary_ops.rs
git commit -m "feat(rust): binary_ops — one_point, two_point, uniform XO, bit_flip (f64 encoded)"
```

---

## Task 4: PyO3 Exposure — Update `src/lib.rs`

Add 11 wrapper functions and register them in the `_core` module. The wrappers are thin: they receive Python values, call the Rust operator function, and return the result. No logic lives in the wrapper.

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace src/lib.rs with the updated version**

The full `src/lib.rs` after Part 2 (replaces the Part 1 version entirely):

```rust
use pyo3::prelude::*;

mod gene_spec;
mod individual;
pub mod operators;
pub mod utils;
mod selection;
mod reproduce;
mod cmaes;
mod parallel;

use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use utils::{
    py_derive_seed,
    OP_CMAES_ASK, OP_CROSSOVER, OP_CROSSOVER_PROB,
    OP_INIT, OP_MULTI_RUN, OP_MUTATION, OP_SELECTION,
};
use operators::{binary_ops, float_ops, int_ops};

// ── Float operator wrappers ───────────────────────────────────────────────────

#[pyfunction]
fn blend_crossover(
    a: Vec<f64>, b: Vec<f64>, alpha: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    float_ops::blend_crossover(&a, &b, alpha, master_seed, generation, individual_idx)
}

#[pyfunction]
fn simulated_binary_crossover(
    a: Vec<f64>, b: Vec<f64>, eta: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    float_ops::simulated_binary_crossover(&a, &b, eta, master_seed, generation, individual_idx)
}

#[pyfunction]
fn gaussian_mutation(
    genes: Vec<f64>, sigma: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    float_ops::gaussian_mutation(&genes, sigma, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn uniform_mutation(
    genes: Vec<f64>, low: f64, high: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    float_ops::uniform_mutation(&genes, low, high, prob, master_seed, generation, individual_idx)
}

// ── Integer operator wrappers ─────────────────────────────────────────────────

#[pyfunction]
fn int_simulated_binary_crossover(
    a: Vec<f64>, b: Vec<f64>, eta: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    int_ops::int_simulated_binary_crossover(&a, &b, eta, master_seed, generation, individual_idx)
}

#[pyfunction]
fn int_gaussian_mutation(
    genes: Vec<f64>, sigma: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    int_ops::int_gaussian_mutation(&genes, sigma, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn int_uniform_mutation(
    genes: Vec<f64>, low: f64, high: f64, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    int_ops::int_uniform_mutation(&genes, low, high, prob, master_seed, generation, individual_idx)
}

// ── Binary operator wrappers ──────────────────────────────────────────────────

#[pyfunction]
fn one_point_crossover(
    a: Vec<f64>, b: Vec<f64>,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::one_point_crossover(&a, &b, master_seed, generation, individual_idx)
}

#[pyfunction]
fn two_point_crossover(
    a: Vec<f64>, b: Vec<f64>,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::two_point_crossover(&a, &b, master_seed, generation, individual_idx)
}

#[pyfunction]
fn uniform_crossover(
    a: Vec<f64>, b: Vec<f64>, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::uniform_crossover(&a, &b, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn bit_flip_mutation(
    genes: Vec<f64>, prob: f64,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    binary_ops::bit_flip_mutation(&genes, prob, master_seed, generation, individual_idx)
}

// ── Module root ───────────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Rayon thread pool — 8 MB stack on all platforms (prevents Windows nalgebra overflow)
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    // Individual types
    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;

    // Seed architecture
    m.add_function(wrap_pyfunction!(py_derive_seed, m)?)?;
    m.add("OP_INIT",           OP_INIT)?;
    m.add("OP_CROSSOVER",      OP_CROSSOVER)?;
    m.add("OP_MUTATION",       OP_MUTATION)?;
    m.add("OP_SELECTION",      OP_SELECTION)?;
    m.add("OP_CMAES_ASK",      OP_CMAES_ASK)?;
    m.add("OP_MULTI_RUN",      OP_MULTI_RUN)?;
    m.add("OP_CROSSOVER_PROB", OP_CROSSOVER_PROB)?;

    // Float operators
    m.add_function(wrap_pyfunction!(blend_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_mutation, m)?)?;

    // Integer operators
    m.add_function(wrap_pyfunction!(int_simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(int_gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(int_uniform_mutation, m)?)?;

    // Binary operators
    m.add_function(wrap_pyfunction!(one_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(two_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(bit_flip_mutation, m)?)?;

    // Selection, reproduce, cmaes, parallel registered in Parts 3–4

    Ok(())
}
```

- [ ] **Step 2: Compile**

```bash
maturin develop --release
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "feat(rust): expose all 11 operator functions to Python via PyO3"
```

---

## Task 5: Python Smoke Tests + Full Verification

**Files:**
- Create: `tests/unit/test_operators_rust.py`

- [ ] **Step 1: Write the Python operator smoke tests**

```python
"""
Smoke tests for the Rust operator functions exposed via PyO3.
These tests verify the Python-callable API surface — not exhaustive correctness
(that is covered by the Rust unit tests). Focus: correct return types/shapes,
determinism from Python, and the f64 encoding contract.
"""
import pytest
from evocore._core import (
    blend_crossover,
    simulated_binary_crossover,
    gaussian_mutation,
    uniform_mutation,
    int_simulated_binary_crossover,
    int_gaussian_mutation,
    int_uniform_mutation,
    one_point_crossover,
    two_point_crossover,
    uniform_crossover,
    bit_flip_mutation,
    OP_CROSSOVER,
    OP_MUTATION,
    py_derive_seed,
)


# ── Float operators ───────────────────────────────────────────────────────────

class TestFloatOperators:

    def test_blend_crossover_returns_two_lists_of_correct_length(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c1, c2 = blend_crossover(a, b, 0.5, 42, 0, 0)
        assert len(c1) == 3
        assert len(c2) == 3

    def test_blend_crossover_deterministic_from_python(self):
        a = [1.0, 2.0]
        b = [3.0, 4.0]
        c1a, c2a = blend_crossover(a, b, 0.5, 42, 5, 3)
        c1b, c2b = blend_crossover(a, b, 0.5, 42, 5, 3)
        assert c1a == c1b
        assert c2a == c2b

    def test_sbx_returns_two_lists(self):
        a = [0.0, 1.0, 2.0]
        b = [3.0, 4.0, 5.0]
        c1, c2 = simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert len(c1) == 3 and len(c2) == 3

    def test_sbx_conservation(self):
        """c1[i] + c2[i] == a[i] + b[i] for all i."""
        a = [1.0, 3.0, 5.0]
        b = [2.0, 6.0, 8.0]
        c1, c2 = simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        for i in range(3):
            assert abs(c1[i] + c2[i] - a[i] - b[i]) < 1e-9

    def test_gaussian_mutation_prob_zero_unchanged(self):
        genes = [5.0] * 5
        result = gaussian_mutation(genes, 1.0, 0.0, 42, 0, 0)
        assert result == genes

    def test_gaussian_mutation_deterministic(self):
        genes = [1.0, 2.0, 3.0]
        r1 = gaussian_mutation(genes, 0.5, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 0.5, 1.0, 42, 0, 0)
        assert r1 == r2

    def test_uniform_mutation_respects_bounds(self):
        genes = [0.0] * 50
        result = uniform_mutation(genes, -1.0, 1.0, 1.0, 42, 0, 0)
        assert all(-1.0 <= v < 1.0 for v in result)

    def test_uniform_mutation_prob_zero_unchanged(self):
        genes = [3.0] * 5
        result = uniform_mutation(genes, 0.0, 10.0, 0.0, 42, 0, 0)
        assert result == genes


# ── Integer operators ─────────────────────────────────────────────────────────

class TestIntegerOperators:

    def test_int_sbx_returns_correct_length(self):
        a = [10.0, 50.0, 100.0]
        b = [20.0, 80.0, 200.0]
        c1, c2 = int_simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert len(c1) == 3 and len(c2) == 3

    def test_int_sbx_outputs_are_integers(self):
        a = [10.0, 50.0, 100.0]
        b = [20.0, 80.0, 200.0]
        c1, c2 = int_simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
        assert all(v == int(v) for v in c1), "c1 must contain integer-valued floats"
        assert all(v == int(v) for v in c2), "c2 must contain integer-valued floats"

    def test_int_gaussian_mutation_outputs_are_integers(self):
        genes = [50.0] * 20
        result = int_gaussian_mutation(genes, 5.0, 1.0, 42, 0, 0)
        assert all(v == int(v) for v in result)

    def test_int_gaussian_mutation_prob_zero_unchanged(self):
        genes = [42.0] * 5
        result = int_gaussian_mutation(genes, 10.0, 0.0, 42, 0, 0)
        assert result == genes

    def test_int_uniform_mutation_outputs_are_integers(self):
        genes = [50.0] * 30
        result = int_uniform_mutation(genes, 5.0, 200.0, 1.0, 42, 0, 0)
        assert all(v == int(v) for v in result)

    def test_int_uniform_mutation_respects_bounds(self):
        genes = [50.0] * 100
        result = int_uniform_mutation(genes, 5.0, 200.0, 1.0, 42, 0, 0)
        assert all(5.0 <= v <= 200.0 for v in result)

    def test_int_uniform_mutation_prob_zero_unchanged(self):
        genes = [77.0] * 5
        result = int_uniform_mutation(genes, 1.0, 100.0, 0.0, 42, 0, 0)
        assert result == genes


# ── Binary operators ──────────────────────────────────────────────────────────

class TestBinaryOperators:

    def _is_binary(self, vals):
        return all(v == 0.0 or v == 1.0 for v in vals)

    def test_one_point_crossover_lengths(self):
        a = [1.0] * 8
        b = [0.0] * 8
        c1, c2 = one_point_crossover(a, b, 42, 0, 0)
        assert len(c1) == 8 and len(c2) == 8

    def test_one_point_crossover_only_binary(self):
        a = [1.0] * 10
        b = [0.0] * 10
        c1, c2 = one_point_crossover(a, b, 42, 0, 0)
        assert self._is_binary(c1) and self._is_binary(c2)

    def test_two_point_crossover_lengths(self):
        a = [1.0] * 10
        b = [0.0] * 10
        c1, c2 = two_point_crossover(a, b, 42, 0, 0)
        assert len(c1) == 10 and len(c2) == 10

    def test_two_point_crossover_only_binary(self):
        a = [1.0] * 12
        b = [0.0] * 12
        c1, c2 = two_point_crossover(a, b, 42, 0, 0)
        assert self._is_binary(c1) and self._is_binary(c2)

    def test_uniform_crossover_lengths(self):
        a = [1.0] * 8
        b = [0.0] * 8
        c1, c2 = uniform_crossover(a, b, 0.5, 42, 0, 0)
        assert len(c1) == 8 and len(c2) == 8

    def test_uniform_crossover_only_binary(self):
        a = [1.0] * 20
        b = [0.0] * 20
        c1, c2 = uniform_crossover(a, b, 0.5, 42, 0, 0)
        assert self._is_binary(c1) and self._is_binary(c2)

    def test_bit_flip_mutation_prob_zero_unchanged(self):
        genes = [1.0, 0.0, 1.0, 0.0]
        result = bit_flip_mutation(genes, 0.0, 42, 0, 0)
        assert result == genes

    def test_bit_flip_mutation_prob_one_all_flipped(self):
        genes = [1.0, 0.0, 1.0, 0.0]
        result = bit_flip_mutation(genes, 1.0, 42, 0, 0)
        assert result == [0.0, 1.0, 0.0, 1.0]

    def test_bit_flip_mutation_only_binary(self):
        genes = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        result = bit_flip_mutation(genes, 0.5, 42, 0, 0)
        assert self._is_binary(result)


# ── Cross-operator determinism ────────────────────────────────────────────────

class TestOperatorDeterminism:
    """
    Verify that (master_seed, generation, individual_idx) fully determines the
    output of each operator from the Python side — the key v3 invariant.
    """

    def test_different_master_seeds_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 1, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 2, 0, 0)
        assert r1 != r2, "Different master seeds must produce different mutations"

    def test_different_generations_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 42, 1, 0)
        assert r1 != r2

    def test_different_individual_indices_diverge(self):
        genes = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 0)
        r2 = gaussian_mutation(genes, 1.0, 1.0, 42, 0, 1)
        assert r1 != r2

    def test_crossover_and_mutation_use_different_op_constants(self):
        """blend_crossover and gaussian_mutation should use OP_CROSSOVER vs OP_MUTATION.
        Verify they produce different seeds for the same (master, gen, idx)."""
        seed_xo  = py_derive_seed(42, 0, 0, OP_CROSSOVER)
        seed_mut = py_derive_seed(42, 0, 0, OP_MUTATION)
        assert seed_xo != seed_mut, "OP_CROSSOVER and OP_MUTATION must produce different seeds"
```

- [ ] **Step 2: Run the smoke tests — they should already pass (maturin built in Task 4)**

```bash
pytest tests/unit/test_operators_rust.py -v
```

Expected: all tests pass. If any test fails due to compilation, run `maturin develop --release` first.

- [ ] **Step 3: Run all Rust tests — confirm no regressions**

```bash
cargo test
```

Expected:
```
test result: ok. 43 passed; 0 failed
```

(17 from Part 1 + 16 float_ops + 12 int_ops + 15 binary_ops = 60. Wait, let me recount: Part 1 = 7 utils + 2 gene_spec + 8 individual = 17. Part 2 = 16 float + 12 int + 15 binary = 43. Total = 60.)

Expected: `test result: ok. 60 passed; 0 failed`

- [ ] **Step 4: Run all Python tests — confirm no regressions**

```bash
pytest tests/unit/ -v
```

Expected: all 11 prior tests + new operator smoke tests pass.

- [ ] **Step 5: Final smoke test from the REPL**

```bash
python - << 'EOF'
from evocore._core import (
    simulated_binary_crossover,
    gaussian_mutation,
    bit_flip_mutation,
    int_uniform_mutation,
)

# Float SBX
a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
c1, c2 = simulated_binary_crossover(a, b, 2.0, 42, 0, 0)
assert len(c1) == 3
print(f"SBX c1={[round(x,3) for x in c1]}")

# Gaussian mutation — no mu parameter
genes = [0.0] * 5
mutated = gaussian_mutation(genes, 0.5, 1.0, 42, 0, 0)
assert len(mutated) == 5
print(f"Gaussian mutated: {[round(x,3) for x in mutated]}")

# Bit-flip (f64 encoded)
bits = [1.0, 0.0, 1.0, 0.0]
flipped = bit_flip_mutation(bits, 1.0, 42, 0, 0)
assert flipped == [0.0, 1.0, 0.0, 1.0]
print(f"Bit-flip all flipped: {flipped}")

# Integer uniform — result is integer-valued f64
int_genes = [50.0] * 5
int_mutated = int_uniform_mutation(int_genes, 5.0, 200.0, 1.0, 42, 0, 0)
assert all(v == int(v) for v in int_mutated)
print(f"Int uniform mutated: {int_mutated}")

print("\nPart 2 complete — all operator smoke tests passed")
EOF
```

Expected:
```
SBX c1=[...]
Gaussian mutated: [...]
Bit-flip all flipped: [0.0, 1.0, 0.0, 1.0]
Int uniform mutated: [...]

Part 2 complete — all operator smoke tests passed
```

- [ ] **Step 6: Final commit**

```bash
git add tests/unit/test_operators_rust.py
git commit -m "test(python): operator smoke tests — all 11 functions, determinism invariants"
git tag part2-complete
```

---

## Part 2 Exit Criteria Checklist

- [ ] `cargo test` passes **60 Rust tests** (17 from Part 1 + 43 new)
- [ ] `maturin develop --release` succeeds with no errors
- [ ] `pytest tests/unit/` passes all tests including `test_operators_rust.py`
- [ ] All 11 operator functions importable from `evocore._core`
- [ ] Float operators: `gaussian_mutation` has no `mu` parameter
- [ ] Int operators: all outputs satisfy `v == int(v)` (integer-valued f64)
- [ ] Binary operators: all outputs contain only `0.0` or `1.0`
- [ ] All operators accept `(master_seed, generation, individual_idx)` — no raw `seed` parameter
- [ ] Different `(generation, individual_idx)` pairs produce different outputs for the same operator
- [ ] No mutable RNG state escapes any operator function
