# evocore v5 — Part 3: Rust Selection, Reproduce & Parallel Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the remaining hot-path Rust components — NaN-safe selection algorithms, the single-call `reproduce()` function (selection → crossover → mutation → clamp+round), population initialisation, and Rayon/sequential evaluation — then expose all of them to Python via PyO3.

**Architecture:** Three new Rust modules replace their stubs from Part 1. `selection.rs` owns all three selection algorithms with safe NaN/Inf handling. `reproduce.rs` owns population initialisation (`init_population`) and the full generational reproduction loop (`reproduce`) — a single PyO3 call that encapsulates selection, crossover-probability gating, crossover, mutation, and bounds clamping+rounding. `parallel.rs` owns `evaluate_sequential` and `evaluate_parallel_rayon`, both accepting `Vec<Vec<f64>>` (the universal PyO3 boundary encoding). All randomness derives from `derive_seed(master_seed, generation, individual_idx, op)` — no mutable RNG state anywhere. `lib.rs` is updated to register all new exports.

**Tech Stack:** Rust 1.78+, PyO3 0.21, Rayon 1.9, rand 0.8, rand_distr 0.4, maturin 1.5+, Python 3.11+, pytest

**Prerequisite:** Parts 1 and 2 complete — `src/utils.rs`, `src/individual.rs`, `src/gene_spec.rs`, all three operator files, and `src/lib.rs` compile and pass their tests (60 Rust tests, 11+ Python tests).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/selection.rs` | Replace stub | `tournament_selection`, `roulette_selection`, `rank_selection` — NaN-safe, `derive_seed`-based |
| `src/reproduce.rs` | Replace stub | `init_population`, `reproduce` (selection + xo + mutation + clamp+round) |
| `src/parallel.rs` | Replace stub | `evaluate_sequential`, `evaluate_parallel_rayon` |
| `src/lib.rs` | Modify | Register all new PyO3 functions |
| `tests/unit/test_selection_rust.py` | Create | Python smoke tests for selection |
| `tests/unit/test_reproduce_rust.py` | Create | Python smoke tests for reproduce + init_population |
| `tests/unit/test_parallel_rust.py` | Create | Python smoke tests for evaluation functions |

---

## RNG Convention for This Part

Every function follows the same pattern established in Part 2:

```rust
let mut rng = StdRng::seed_from_u64(
    derive_seed(master_seed, generation, individual_idx, OP_*)
);
```

Specific conventions for this part:

| Operation | `individual_idx` | `op` |
|---|---|---|
| Selection | `0` (population-level op) | `OP_SELECTION` |
| Crossover-probability gate for pair `p` | `p as u64` | `OP_CROSSOVER_PROB` |
| Crossover for pair `p` | `p as u64` | `OP_CROSSOVER` |
| Mutation for offspring `i` | `i as u64` | `OP_MUTATION` |
| Population init for individual `i` | `i as u64` | `OP_INIT` |
| Parallel evaluation (Rayon) | n/a — fitness fn is deterministic given genes | n/a |

---

## Task 1: `src/selection.rs` — NaN-Safe Selection Algorithms

Three selection algorithms, all using `derive_seed(master_seed, generation, 0, OP_SELECTION)`.

**NaN/Inf policy** (applied before any comparison):
- `NaN` → replace with `f64::NEG_INFINITY` (treated as worst)
- `f64::NEG_INFINITY` → kept as-is (worst)
- `f64::INFINITY` → replace with `f64::MAX` (treated as finite but very good; no panic)

This normalisation is applied in the private `safe_fitness(f: f64) -> f64` helper, which is called at the start of every selection function.

**Files:**
- Modify: `src/selection.rs`

- [ ] **Step 1: Write failing Rust tests**

Replace the stub `src/selection.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fitnesses() -> Vec<f64> {
        vec![1.0, 5.0, 3.0, 2.0, 4.0]
    }

    // ── safe_fitness ─────────────────────────────────────────────────────────

    #[test]
    fn test_safe_fitness_passes_through_normal_values() {
        assert_eq!(safe_fitness(3.14), 3.14);
        assert_eq!(safe_fitness(-100.0), -100.0);
        assert_eq!(safe_fitness(0.0), 0.0);
    }

    #[test]
    fn test_safe_fitness_replaces_nan_with_neg_infinity() {
        let result = safe_fitness(f64::NAN);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_safe_fitness_replaces_pos_infinity_with_max() {
        assert_eq!(safe_fitness(f64::INFINITY), f64::MAX);
    }

    #[test]
    fn test_safe_fitness_preserves_neg_infinity() {
        assert_eq!(safe_fitness(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    // ── tournament_selection ──────────────────────────────────────────────────

    #[test]
    fn test_tournament_returns_k_indices() {
        let fitnesses = sample_fitnesses();
        let indices = tournament_selection(&fitnesses, 4, 2, 42, 0);
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_tournament_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = tournament_selection(&fitnesses, 6, 3, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len(), "index {} out of range", i);
        }
    }

    #[test]
    fn test_tournament_deterministic() {
        let fitnesses = sample_fitnesses();
        let a = tournament_selection(&fitnesses, 4, 2, 99, 3);
        let b = tournament_selection(&fitnesses, 4, 2, 99, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_tournament_different_generations_diverge() {
        let fitnesses = sample_fitnesses();
        let a = tournament_selection(&fitnesses, 4, 2, 42, 0);
        let b = tournament_selection(&fitnesses, 4, 2, 42, 1);
        assert_ne!(a, b, "different generations must produce different selection");
    }

    #[test]
    fn test_tournament_nan_fitness_never_wins_large_tournament() {
        // With tournament_size = 5 (whole population), the best known-finite
        // individual always wins over NaN individuals.
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 99.0];
        let indices = tournament_selection(&fitnesses, 100, 5, 42, 0);
        // Index 4 (fitness=99.0) must win every tournament
        assert!(indices.iter().all(|&i| i == 4),
            "NaN individuals should never win a full-population tournament");
    }

    // ── roulette_selection ────────────────────────────────────────────────────

    #[test]
    fn test_roulette_returns_k_indices() {
        let fitnesses = vec![1.0, 2.0, 3.0, 4.0];
        let indices = roulette_selection(&fitnesses, 5, 42, 0);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_roulette_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = roulette_selection(&fitnesses, 10, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len());
        }
    }

    #[test]
    fn test_roulette_deterministic() {
        let fitnesses = vec![1.0, 2.0, 3.0];
        let a = roulette_selection(&fitnesses, 5, 7, 2);
        let b = roulette_selection(&fitnesses, 5, 7, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_roulette_nan_fitness_excluded() {
        // NaN individual should be treated as having fitness -inf (nearly 0 weight).
        // With 100 draws and only one valid individual, that individual wins all.
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, 100.0];
        let indices = roulette_selection(&fitnesses, 100, 42, 0);
        assert!(indices.iter().all(|&i| i == 3),
            "NaN individuals should have near-zero selection probability");
    }

    // ── rank_selection ────────────────────────────────────────────────────────

    #[test]
    fn test_rank_returns_k_indices() {
        let fitnesses = sample_fitnesses();
        let indices = rank_selection(&fitnesses, 6, 42, 0);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_rank_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = rank_selection(&fitnesses, 8, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len());
        }
    }

    #[test]
    fn test_rank_deterministic() {
        let fitnesses = vec![3.0, 1.0, 2.0, 5.0, 4.0];
        let a = rank_selection(&fitnesses, 4, 13, 5);
        let b = rank_selection(&fitnesses, 4, 13, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_rank_nan_individual_lowest_rank() {
        // NaN gets rank 1 (lowest). With only one non-NaN individual, it wins all.
        let fitnesses = vec![f64::NAN, f64::NAN, 50.0];
        let indices = rank_selection(&fitnesses, 50, 42, 0);
        assert!(indices.iter().all(|&i| i == 2),
            "NaN individuals must have rank 1 (lowest probability)");
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cargo test selection 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function 'safe_fitness'` or similar compile error.

- [ ] **Step 3: Implement src/selection.rs**

```rust
use rand::prelude::*;
use rand::rngs::StdRng;
use crate::utils::{derive_seed, OP_SELECTION};

// ── NaN/Inf normalisation ─────────────────────────────────────────────────────

/// Normalise a raw fitness value for safe comparison:
/// - NaN         → f64::NEG_INFINITY  (treated as worst)
/// - +∞          → f64::MAX           (treated as very good but finite)
/// - Everything else → unchanged
#[inline]
pub fn safe_fitness(f: f64) -> f64 {
    if f.is_nan() {
        f64::NEG_INFINITY
    } else if f == f64::INFINITY {
        f64::MAX
    } else {
        f
    }
}

// ── Tournament Selection ──────────────────────────────────────────────────────

/// Tournament selection. Returns `k` indices into `fitnesses`.
///
/// For each draw, samples `tournament_size` candidates uniformly at random and
/// returns the index of the candidate with the highest safe fitness.
/// Higher fitness = higher selection pressure.
/// NaN fitness values are treated as −∞ and will never win a tournament.
pub fn tournament_selection(
    fitnesses: &[f64],
    k: usize,
    tournament_size: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "tournament_selection: empty population");
    assert!(tournament_size >= 1, "tournament_selection: tournament_size must be >= 1");
    let safe: Vec<f64> = fitnesses.iter().cloned().map(safe_fitness).collect();
    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));
    (0..k)
        .map(|_| {
            (0..tournament_size)
                .map(|_| rng.gen_range(0..n))
                .max_by(|&a, &b| safe[a].partial_cmp(&safe[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
        })
        .collect()
}

// ── Roulette-wheel (fitness-proportionate) Selection ─────────────────────────

/// Fitness-proportionate (roulette-wheel) selection. Returns `k` indices.
///
/// Shifts fitnesses so the minimum becomes a small positive value before
/// computing proportions, ensuring all individuals have non-negative weight.
/// NaN fitness values → safe_fitness → −∞ → effectively zero selection weight
/// after the shift (since they will be the minimum, making their weight ≈ ε).
pub fn roulette_selection(
    fitnesses: &[f64],
    k: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "roulette_selection: empty population");
    let safe: Vec<f64> = fitnesses.iter().cloned().map(safe_fitness).collect();
    let min_fit = safe.iter().cloned().fold(f64::INFINITY, f64::min);
    // Shift so min → ε (avoids zero-weight individuals entirely, but NaN individuals
    // end up with weight ≈ 1e-12 which is negligible vs real individuals).
    let shifted: Vec<f64> = safe.iter().map(|&f| (f - min_fit) + 1e-12).collect();
    let total: f64 = shifted.iter().sum();
    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));
    (0..k)
        .map(|_| {
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = n - 1; // fallback to last element
            for (i, &w) in shifted.iter().enumerate() {
                r -= w;
                if r <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        })
        .collect()
}

// ── Linear Rank Selection ─────────────────────────────────────────────────────

/// Linear rank selection. Returns `k` indices.
///
/// Assigns each individual a rank from 1 (worst) to n (best) based on sorted
/// fitness order, then selects proportional to rank rather than raw fitness.
/// This reduces the effect of fitness outliers and prevents premature convergence.
/// NaN fitness values receive rank 1 (worst rank, lowest selection probability).
pub fn rank_selection(
    fitnesses: &[f64],
    k: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "rank_selection: empty population");
    let safe: Vec<f64> = fitnesses.iter().cloned().map(safe_fitness).collect();

    // Build (original_index, safe_fitness) pairs, sort ascending by fitness.
    let mut order: Vec<(usize, f64)> = safe.iter().cloned().enumerate().collect();
    order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign rank 1..=n. `ranks[original_idx] = rank`.
    let mut ranks = vec![0usize; n];
    for (rank, (orig_idx, _)) in order.iter().enumerate() {
        ranks[*orig_idx] = rank + 1;
    }

    let total: f64 = (1..=n).map(|r| r as f64).sum::<f64>();
    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));
    (0..k)
        .map(|_| {
            let mut r = rng.gen::<f64>() * total;
            let mut chosen = n - 1;
            // Iterate in rank-ascending order so probability accumulates
            // correctly (low-rank individuals get fewer "chances").
            for (orig_idx, _) in &order {
                r -= ranks[*orig_idx] as f64;
                if r <= 0.0 {
                    chosen = *orig_idx;
                    break;
                }
            }
            chosen
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fitnesses() -> Vec<f64> {
        vec![1.0, 5.0, 3.0, 2.0, 4.0]
    }

    #[test]
    fn test_safe_fitness_passes_through_normal_values() {
        assert_eq!(safe_fitness(3.14), 3.14);
        assert_eq!(safe_fitness(-100.0), -100.0);
        assert_eq!(safe_fitness(0.0), 0.0);
    }

    #[test]
    fn test_safe_fitness_replaces_nan_with_neg_infinity() {
        let result = safe_fitness(f64::NAN);
        assert!(result.is_infinite() && result < 0.0);
    }

    #[test]
    fn test_safe_fitness_replaces_pos_infinity_with_max() {
        assert_eq!(safe_fitness(f64::INFINITY), f64::MAX);
    }

    #[test]
    fn test_safe_fitness_preserves_neg_infinity() {
        assert_eq!(safe_fitness(f64::NEG_INFINITY), f64::NEG_INFINITY);
    }

    #[test]
    fn test_tournament_returns_k_indices() {
        let indices = tournament_selection(&sample_fitnesses(), 4, 2, 42, 0);
        assert_eq!(indices.len(), 4);
    }

    #[test]
    fn test_tournament_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = tournament_selection(&fitnesses, 6, 3, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len());
        }
    }

    #[test]
    fn test_tournament_deterministic() {
        let fitnesses = sample_fitnesses();
        let a = tournament_selection(&fitnesses, 4, 2, 99, 3);
        let b = tournament_selection(&fitnesses, 4, 2, 99, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_tournament_different_generations_diverge() {
        let fitnesses = sample_fitnesses();
        let a = tournament_selection(&fitnesses, 4, 2, 42, 0);
        let b = tournament_selection(&fitnesses, 4, 2, 42, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_tournament_nan_fitness_never_wins_large_tournament() {
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 99.0];
        let indices = tournament_selection(&fitnesses, 100, 5, 42, 0);
        assert!(indices.iter().all(|&i| i == 4));
    }

    #[test]
    fn test_roulette_returns_k_indices() {
        let indices = roulette_selection(&vec![1.0, 2.0, 3.0, 4.0], 5, 42, 0);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_roulette_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = roulette_selection(&fitnesses, 10, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len());
        }
    }

    #[test]
    fn test_roulette_deterministic() {
        let a = roulette_selection(&vec![1.0, 2.0, 3.0], 5, 7, 2);
        let b = roulette_selection(&vec![1.0, 2.0, 3.0], 5, 7, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_roulette_nan_fitness_excluded() {
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, 100.0];
        let indices = roulette_selection(&fitnesses, 100, 42, 0);
        assert!(indices.iter().all(|&i| i == 3));
    }

    #[test]
    fn test_rank_returns_k_indices() {
        let indices = rank_selection(&sample_fitnesses(), 6, 42, 0);
        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_rank_all_indices_in_range() {
        let fitnesses = sample_fitnesses();
        let indices = rank_selection(&fitnesses, 8, 42, 0);
        for &i in &indices {
            assert!(i < fitnesses.len());
        }
    }

    #[test]
    fn test_rank_deterministic() {
        let a = rank_selection(&vec![3.0, 1.0, 2.0, 5.0, 4.0], 4, 13, 5);
        let b = rank_selection(&vec![3.0, 1.0, 2.0, 5.0, 4.0], 4, 13, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_rank_nan_individual_lowest_rank() {
        let fitnesses = vec![f64::NAN, f64::NAN, 50.0];
        let indices = rank_selection(&fitnesses, 50, 42, 0);
        assert!(indices.iter().all(|&i| i == 2));
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test selection
```

Expected:
```
test selection::tests::test_rank_all_indices_in_range ... ok
test selection::tests::test_rank_deterministic ... ok
test selection::tests::test_rank_nan_individual_lowest_rank ... ok
test selection::tests::test_rank_returns_k_indices ... ok
test selection::tests::test_roulette_all_indices_in_range ... ok
test selection::tests::test_roulette_deterministic ... ok
test selection::tests::test_roulette_nan_fitness_excluded ... ok
test selection::tests::test_roulette_returns_k_indices ... ok
test selection::tests::test_safe_fitness_passes_through_normal_values ... ok
test selection::tests::test_safe_fitness_preserves_neg_infinity ... ok
test selection::tests::test_safe_fitness_replaces_nan_with_neg_infinity ... ok
test selection::tests::test_safe_fitness_replaces_pos_infinity_with_max ... ok
test selection::tests::test_tournament_all_indices_in_range ... ok
test selection::tests::test_tournament_deterministic ... ok
test selection::tests::test_tournament_different_generations_diverge ... ok
test selection::tests::test_tournament_nan_fitness_never_wins_large_tournament ... ok
test selection::tests::test_tournament_returns_k_indices ... ok

test result: ok. 17 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/selection.rs
git commit -m "feat(rust): NaN-safe selection algorithms (tournament, roulette, rank)"
```

---

## Task 2: `src/reproduce.rs` — Population Initialisation and Reproduction

Two public functions: `init_population` (deterministically initialises genes for a full population) and `reproduce` (one Rust call per generation: selection → crossover-probability gate → crossover → mutation → clamp+round).

### Design Notes

**`init_population`:** Each individual `i` gets seed `derive_seed(master_seed, 0, i as u64, OP_INIT)`. This means the initialisation is independent of `generation` (always 0) and independent of population size — adding more individuals never changes the genes of existing ones.

**`reproduce` per-gene dispatch:** All genes arrive as `f64` (the universal PyO3 encoding). The `gene_kinds` field tells the function whether to round (int), threshold (bool), or leave as-is (float) after each operator. Clamping to `[low, high]` is applied to all genes after every operator. The crossover operates on the full gene vector regardless of kind — SBX/BLX work fine on encoded integers, and the subsequent `clamp_and_round` enforces correctness.

**Mutation dispatch:** determined by `mutation_type` string:
- `"gaussian"` with `GeneKind::Float`: `g += N(0, sigma_i)`, not rounded
- `"gaussian"` with `GeneKind::Int`: `g += N(0, sigma_i)`, rounded
- `"uniform"` with `GeneKind::Float` or `GeneKind::Int`: resample from `[low_i, high_i)`, rounded if int
- `"bit_flip"` (only valid for `GeneKind::Bool`): flip `0.0 ↔ 1.0`

`mutation_sigmas` is a `Vec<f64>` of per-gene **absolute** sigma values. Uniform mutation ignores sigma and uses `gene_bounds[i]` directly.

**Elitism:** Not handled in `reproduce()`. The Python `GAEngine` calls `reproduce()` and then overwrites the first `elitism` slots with the best individuals from the previous generation. This keeps Rust pure data-transformation.

**Files:**
- Modify: `src/reproduce.rs`

- [ ] **Step 1: Write failing Rust tests**

Replace the stub `src/reproduce.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene_spec::GeneKind;

    fn float_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); n]
    }

    fn float_kinds(n: usize) -> Vec<GeneKind> {
        vec![GeneKind::Float; n]
    }

    fn uniform_sigmas(n: usize, sigma: f64) -> Vec<f64> {
        vec![sigma; n]
    }

    // ── init_population ───────────────────────────────────────────────────────

    #[test]
    fn test_init_population_correct_size() {
        let pop = init_population(&float_bounds(5), &float_kinds(5), 20, 42);
        assert_eq!(pop.len(), 20, "population size must equal requested size");
    }

    #[test]
    fn test_init_population_correct_gene_length() {
        let pop = init_population(&float_bounds(7), &float_kinds(7), 10, 42);
        assert!(pop.iter().all(|ind| ind.len() == 7));
    }

    #[test]
    fn test_init_population_genes_within_bounds() {
        let bounds = vec![(-2.0_f64, 2.0), (0.0, 10.0), (-1.0, 1.0)];
        let kinds = vec![GeneKind::Float; 3];
        let pop = init_population(&bounds, &kinds, 50, 42);
        for ind in &pop {
            for (i, &g) in ind.iter().enumerate() {
                assert!(
                    g >= bounds[i].0 && g < bounds[i].1,
                    "gene[{}]={} outside [{}, {})", i, g, bounds[i].0, bounds[i].1
                );
            }
        }
    }

    #[test]
    fn test_init_population_int_genes_are_integers() {
        let bounds = vec![(5.0_f64, 200.0), (10.0, 500.0)];
        let kinds = vec![GeneKind::Int, GeneKind::Int];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} is not an integer", g);
            }
        }
    }

    #[test]
    fn test_init_population_bool_genes_are_binary() {
        let bounds = vec![(0.0_f64, 1.0); 10];
        let kinds = vec![GeneKind::Bool; 10];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert!(g == 0.0 || g == 1.0, "bool gene {} is neither 0.0 nor 1.0", g);
            }
        }
    }

    #[test]
    fn test_init_population_deterministic() {
        let bounds = float_bounds(4);
        let kinds = float_kinds(4);
        let p1 = init_population(&bounds, &kinds, 10, 7);
        let p2 = init_population(&bounds, &kinds, 10, 7);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_init_population_different_seeds_diverge() {
        let bounds = float_bounds(4);
        let kinds = float_kinds(4);
        let p1 = init_population(&bounds, &kinds, 10, 1);
        let p2 = init_population(&bounds, &kinds, 10, 2);
        assert_ne!(p1[0], p2[0], "different seeds must produce different genes");
    }

    // ── clamp_and_round ───────────────────────────────────────────────────────

    #[test]
    fn test_clamp_and_round_float_clamps_to_bounds() {
        let genes  = vec![-10.0_f64, 10.0];
        let bounds = vec![(-5.0_f64, 5.0), (-5.0, 5.0)];
        let kinds  = vec![GeneKind::Float, GeneKind::Float];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result, vec![-5.0, 5.0]);
    }

    #[test]
    fn test_clamp_and_round_int_rounds_and_clamps() {
        let genes  = vec![3.7_f64, -1.3, 200.9];
        let bounds = vec![(1.0_f64, 100.0), (0.0, 50.0), (0.0, 100.0)];
        let kinds  = vec![GeneKind::Int; 3];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result[0], 4.0);   // 3.7 rounds to 4
        assert_eq!(result[1], 0.0);   // -1.3 rounds to -1 → clamped to 0
        assert_eq!(result[2], 100.0); // 200.9 rounds to 201 → clamped to 100
    }

    #[test]
    fn test_clamp_and_round_bool_thresholds_at_half() {
        let genes  = vec![0.3_f64, 0.7, 0.5, 0.49];
        let bounds = vec![(0.0_f64, 1.0); 4];
        let kinds  = vec![GeneKind::Bool; 4];
        let result = clamp_and_round(&genes, &bounds, &kinds);
        assert_eq!(result, vec![0.0, 1.0, 1.0, 0.0]);
    }

    // ── reproduce ────────────────────────────────────────────────────────────

    fn make_float_config(pop_size: usize, gene_len: usize) -> ReproduceConfig {
        ReproduceConfig {
            crossover_type:  CrossoverType::Sbx,
            crossover_prob:  0.9,
            crossover_eta:   2.0,
            crossover_alpha: 0.5,
            mutation_type:   MutationType::Gaussian,
            mutation_prob:   0.1,
            mutation_sigmas: vec![0.5; gene_len],
            gene_bounds:     vec![(-5.0, 5.0); gene_len],
            gene_kinds:      vec![GeneKind::Float; gene_len],
            selection_type:  SelectionType::Tournament,
            tournament_size: 3,
            population_size: pop_size,
        }
    }

    #[test]
    fn test_reproduce_returns_correct_population_size() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1; 5]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = make_float_config(20, 5);
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        assert_eq!(new_pop.len(), 20);
    }

    #[test]
    fn test_reproduce_returns_correct_gene_length() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 6]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 6);
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        assert!(new_pop.iter().all(|ind| ind.len() == 6));
    }

    #[test]
    fn test_reproduce_float_genes_within_bounds() {
        let pop: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1 - 1.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..30).map(|i| -(i as f64 * 0.1)).collect();
        let config = ReproduceConfig {
            gene_bounds: vec![(-5.0, 5.0); 4],
            ..make_float_config(30, 4)
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(g >= -5.0 && g <= 5.0, "float gene {} outside [-5.0, 5.0]", g);
            }
        }
    }

    #[test]
    fn test_reproduce_int_genes_are_always_integers() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64; 3]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type:  CrossoverType::Sbx,
            crossover_prob:  0.9,
            crossover_eta:   2.0,
            crossover_alpha: 0.5,
            mutation_type:   MutationType::Gaussian,
            mutation_prob:   0.5,
            mutation_sigmas: vec![2.0; 3],
            gene_bounds:     vec![(0.0, 20.0); 3],
            gene_kinds:      vec![GeneKind::Int; 3],
            selection_type:  SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} is not integer-valued", g);
            }
        }
    }

    #[test]
    fn test_reproduce_bool_genes_are_always_binary() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| {
            (0..8).map(|j| if (i + j) % 2 == 0 { 1.0 } else { 0.0 }).collect()
        }).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type:  CrossoverType::OnePoint,
            crossover_prob:  0.8,
            crossover_eta:   2.0,
            crossover_alpha: 0.5,
            mutation_type:   MutationType::BitFlip,
            mutation_prob:   0.1,
            mutation_sigmas: vec![0.0; 8],
            gene_bounds:     vec![(0.0, 1.0); 8],
            gene_kinds:      vec![GeneKind::Bool; 8],
            selection_type:  SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(g == 0.0 || g == 1.0, "bool gene {} is not 0.0 or 1.0", g);
            }
        }
    }

    #[test]
    fn test_reproduce_deterministic() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 77, 5);
        let r2 = reproduce(&pop, &fitnesses, &config, 77, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_reproduce_different_generations_diverge() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 42, 0);
        let r2 = reproduce(&pop, &fitnesses, &config, 42, 1);
        assert_ne!(r1, r2, "different generations must produce different offspring");
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cargo test reproduce 2>&1 | head -10
```

Expected: `error[E0422]: cannot find struct 'ReproduceConfig'` or similar.

- [ ] **Step 3: Implement src/reproduce.rs**

```rust
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use crate::gene_spec::GeneKind;
use crate::utils::{derive_seed, OP_INIT, OP_CROSSOVER, OP_CROSSOVER_PROB, OP_MUTATION, OP_SELECTION};
use crate::selection;
use crate::operators::{float_ops, int_ops, binary_ops};

// ── Configuration Types ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum CrossoverType {
    Sbx,
    Blx,
    OnePoint,
    TwoPoint,
    UniformXO,
}

#[derive(Clone, Debug)]
pub enum MutationType {
    Gaussian,
    Uniform,
    BitFlip,
}

#[derive(Clone, Debug)]
pub enum SelectionType {
    Tournament,
    Roulette,
    Rank,
}

#[derive(Clone, Debug)]
pub struct ReproduceConfig {
    pub crossover_type:  CrossoverType,
    pub crossover_prob:  f64,
    pub crossover_eta:   f64,      // SBX / int_SBX distribution index
    pub crossover_alpha: f64,      // BLX-α alpha parameter
    pub mutation_type:   MutationType,
    pub mutation_prob:   f64,
    pub mutation_sigmas: Vec<f64>, // per-gene absolute sigma (only for Gaussian)
    pub gene_bounds:     Vec<(f64, f64)>,
    pub gene_kinds:      Vec<GeneKind>,
    pub selection_type:  SelectionType,
    pub tournament_size: usize,
    pub population_size: usize,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Clamp each gene to its [low, high] bounds, then round integer genes and
/// threshold boolean genes. Returns a new gene vector.
pub fn clamp_and_round(
    genes: &[f64],
    bounds: &[(f64, f64)],
    kinds: &[GeneKind],
) -> Vec<f64> {
    genes.iter().enumerate().map(|(i, &g)| {
        let (lo, hi) = bounds[i];
        match kinds[i] {
            GeneKind::Float => g.clamp(lo, hi),
            GeneKind::Int   => g.round().clamp(lo, hi),
            GeneKind::Bool  => if g >= 0.5 { 1.0 } else { 0.0 },
        }
    }).collect()
}

/// Apply crossover to a parent pair. Returns two offspring gene vectors
/// before clamping. `pair_idx` is used as `individual_idx` for seed derivation.
fn apply_crossover(
    a: &[f64], b: &[f64],
    config: &ReproduceConfig,
    master_seed: u64, generation: u64, pair_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    match &config.crossover_type {
        CrossoverType::Sbx =>
            float_ops::simulated_binary_crossover(a, b, config.crossover_eta,
                master_seed, generation, pair_idx),
        CrossoverType::Blx =>
            float_ops::blend_crossover(a, b, config.crossover_alpha,
                master_seed, generation, pair_idx),
        CrossoverType::OnePoint =>
            binary_ops::one_point_crossover(a, b, master_seed, generation, pair_idx),
        CrossoverType::TwoPoint =>
            binary_ops::two_point_crossover(a, b, master_seed, generation, pair_idx),
        CrossoverType::UniformXO =>
            binary_ops::uniform_crossover(a, b, 0.5, master_seed, generation, pair_idx),
    }
}

/// Apply mutation to a single individual's genes, returning new gene vector
/// before clamping. Handles per-gene dispatch based on gene_kinds.
/// `individual_idx` is the index of this offspring in the new population.
fn apply_mutation(
    genes: &[f64],
    config: &ReproduceConfig,
    master_seed: u64, generation: u64, individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(
        derive_seed(master_seed, generation, individual_idx, OP_MUTATION)
    );
    genes.iter().enumerate().map(|(i, &g)| {
        match (&config.mutation_type, &config.gene_kinds[i]) {
            // Gaussian mutation for float genes — no rounding
            (MutationType::Gaussian, GeneKind::Float) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let sigma = config.mutation_sigmas[i];
                    g + Normal::new(0.0_f64, sigma.max(1e-20)).unwrap().sample(&mut rng)
                } else { g }
            },
            // Gaussian mutation for int genes — round after adding noise
            (MutationType::Gaussian, GeneKind::Int) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let sigma = config.mutation_sigmas[i];
                    (g + Normal::new(0.0_f64, sigma.max(1e-20)).unwrap().sample(&mut rng)).round()
                } else { g }
            },
            // Uniform mutation for float genes
            (MutationType::Uniform, GeneKind::Float) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let (lo, hi) = config.gene_bounds[i];
                    rng.gen_range(lo..hi)
                } else { g }
            },
            // Uniform mutation for int genes — draw integer uniformly
            (MutationType::Uniform, GeneKind::Int) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let lo = config.gene_bounds[i].0.round() as i64;
                    let hi = config.gene_bounds[i].1.round() as i64;
                    rng.gen_range(lo..=hi) as f64
                } else { g }
            },
            // Bit-flip mutation for bool genes
            (MutationType::BitFlip, GeneKind::Bool) | (_, GeneKind::Bool) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    if g >= 0.5 { 0.0 } else { 1.0 }
                } else { g }
            },
            // Fallback: Gaussian treated as float (should not happen if Python validates config)
            (MutationType::Gaussian, _) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let sigma = config.mutation_sigmas[i];
                    g + Normal::new(0.0_f64, sigma.max(1e-20)).unwrap().sample(&mut rng)
                } else { g }
            },
            (MutationType::Uniform, _) => {
                if rng.gen::<f64>() < config.mutation_prob {
                    let (lo, hi) = config.gene_bounds[i];
                    rng.gen_range(lo..hi)
                } else { g }
            },
        }
    }).collect()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Initialise a full population deterministically.
///
/// Each individual `i` gets seed `derive_seed(master_seed, 0, i, OP_INIT)`.
/// Float genes: sampled from U(lo, hi).
/// Int genes:   sampled from integers in [round(lo), round(hi)] inclusive.
/// Bool genes:  sampled as 0.0 or 1.0 with equal probability.
pub fn init_population(
    gene_bounds: &[(f64, f64)],
    gene_kinds:  &[GeneKind],
    population_size: usize,
    master_seed: u64,
) -> Vec<Vec<f64>> {
    let gene_len = gene_bounds.len();
    assert_eq!(gene_len, gene_kinds.len(), "init_population: bounds and kinds length mismatch");
    (0..population_size).map(|i| {
        let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, 0, i as u64, OP_INIT));
        (0..gene_len).map(|j| {
            let (lo, hi) = gene_bounds[j];
            match gene_kinds[j] {
                GeneKind::Float => rng.gen_range(lo..hi),
                GeneKind::Int   => {
                    let lo_i = lo.round() as i64;
                    let hi_i = hi.round() as i64;
                    rng.gen_range(lo_i..=hi_i) as f64
                },
                GeneKind::Bool  => if rng.gen::<bool>() { 1.0 } else { 0.0 },
            }
        }).collect()
    }).collect()
}

/// Reproduce a full new population from the current one in a single call.
///
/// Steps per pair of offspring:
///   1. Select parent pair via `selection_type`.
///   2. Apply crossover with probability `crossover_prob` using `OP_CROSSOVER_PROB` seed.
///   3. Apply crossover operator (if triggered) or copy parents as-is.
///   4. Apply per-gene mutation to each offspring.
///   5. Clamp all genes to `gene_bounds`, round int genes, threshold bool genes.
///
/// Elitism is NOT handled here — the Python GAEngine overwrites elite slots
/// after this call, keeping this function a pure data transformation.
pub fn reproduce(
    population:  &[Vec<f64>],
    fitnesses:   &[f64],
    config:      &ReproduceConfig,
    master_seed: u64,
    generation:  u64,
) -> Vec<Vec<f64>> {
    let pop_size = config.population_size;

    // Select parent indices (enough for pop_size offspring, possibly +1 for odd sizes)
    let parent_count = pop_size + 1;
    let parent_indices: Vec<usize> = match &config.selection_type {
        SelectionType::Tournament =>
            selection::tournament_selection(fitnesses, parent_count,
                config.tournament_size, master_seed, generation),
        SelectionType::Roulette =>
            selection::roulette_selection(fitnesses, parent_count, master_seed, generation),
        SelectionType::Rank =>
            selection::rank_selection(fitnesses, parent_count, master_seed, generation),
    };

    let mut new_pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    let mut offspring_idx: u64 = 0;

    let pairs_needed = (pop_size + 1) / 2;
    for pair in 0..pairs_needed {
        let a = &population[parent_indices[pair * 2]];
        let b = &population[parent_indices[pair * 2 + 1]];

        // Crossover probability gate — unique seed per pair
        let apply_xo = {
            let mut rng = StdRng::seed_from_u64(
                derive_seed(master_seed, generation, pair as u64, OP_CROSSOVER_PROB)
            );
            rng.gen::<f64>() < config.crossover_prob
        };

        let (c1, c2) = if apply_xo {
            apply_crossover(a, b, config, master_seed, generation, pair as u64)
        } else {
            (a.clone(), b.clone())
        };

        // Mutate each offspring
        let mut m1 = apply_mutation(&c1, config, master_seed, generation, offspring_idx);
        offspring_idx += 1;
        let mut m2 = apply_mutation(&c2, config, master_seed, generation, offspring_idx);
        offspring_idx += 1;

        // Clamp + round
        m1 = clamp_and_round(&m1, &config.gene_bounds, &config.gene_kinds);
        m2 = clamp_and_round(&m2, &config.gene_bounds, &config.gene_kinds);

        new_pop.push(m1);
        if new_pop.len() < pop_size {
            new_pop.push(m2);
        }
    }

    new_pop.truncate(pop_size);
    new_pop
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene_spec::GeneKind;

    fn float_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); n]
    }
    fn float_kinds(n: usize) -> Vec<GeneKind> {
        vec![GeneKind::Float; n]
    }

    fn make_float_config(pop_size: usize, gene_len: usize) -> ReproduceConfig {
        ReproduceConfig {
            crossover_type:  CrossoverType::Sbx,
            crossover_prob:  0.9,
            crossover_eta:   2.0,
            crossover_alpha: 0.5,
            mutation_type:   MutationType::Gaussian,
            mutation_prob:   0.1,
            mutation_sigmas: vec![0.5; gene_len],
            gene_bounds:     vec![(-5.0, 5.0); gene_len],
            gene_kinds:      vec![GeneKind::Float; gene_len],
            selection_type:  SelectionType::Tournament,
            tournament_size: 3,
            population_size: pop_size,
        }
    }

    #[test]
    fn test_init_population_correct_size() {
        let pop = init_population(&float_bounds(5), &float_kinds(5), 20, 42);
        assert_eq!(pop.len(), 20);
    }

    #[test]
    fn test_init_population_correct_gene_length() {
        let pop = init_population(&float_bounds(7), &float_kinds(7), 10, 42);
        assert!(pop.iter().all(|ind| ind.len() == 7));
    }

    #[test]
    fn test_init_population_genes_within_bounds() {
        let bounds = vec![(-2.0_f64, 2.0), (0.0, 10.0), (-1.0, 1.0)];
        let kinds = vec![GeneKind::Float; 3];
        let pop = init_population(&bounds, &kinds, 50, 42);
        for ind in &pop {
            for (i, &g) in ind.iter().enumerate() {
                assert!(g >= bounds[i].0 && g < bounds[i].1,
                    "gene[{}]={} outside [{}, {})", i, g, bounds[i].0, bounds[i].1);
            }
        }
    }

    #[test]
    fn test_init_population_int_genes_are_integers() {
        let bounds = vec![(5.0_f64, 200.0), (10.0, 500.0)];
        let kinds = vec![GeneKind::Int, GeneKind::Int];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} not integer", g);
            }
        }
    }

    #[test]
    fn test_init_population_bool_genes_are_binary() {
        let bounds = vec![(0.0_f64, 1.0); 10];
        let kinds = vec![GeneKind::Bool; 10];
        let pop = init_population(&bounds, &kinds, 20, 42);
        for ind in &pop {
            for &g in ind {
                assert!(g == 0.0 || g == 1.0, "bool gene {} not 0.0 or 1.0", g);
            }
        }
    }

    #[test]
    fn test_init_population_deterministic() {
        let p1 = init_population(&float_bounds(4), &float_kinds(4), 10, 7);
        let p2 = init_population(&float_bounds(4), &float_kinds(4), 10, 7);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_init_population_different_seeds_diverge() {
        let p1 = init_population(&float_bounds(4), &float_kinds(4), 10, 1);
        let p2 = init_population(&float_bounds(4), &float_kinds(4), 10, 2);
        assert_ne!(p1[0], p2[0]);
    }

    #[test]
    fn test_clamp_and_round_float_clamps_to_bounds() {
        let result = clamp_and_round(
            &[-10.0_f64, 10.0],
            &[(-5.0_f64, 5.0), (-5.0, 5.0)],
            &[GeneKind::Float, GeneKind::Float],
        );
        assert_eq!(result, vec![-5.0, 5.0]);
    }

    #[test]
    fn test_clamp_and_round_int_rounds_and_clamps() {
        let result = clamp_and_round(
            &[3.7_f64, -1.3, 200.9],
            &[(1.0_f64, 100.0), (0.0, 50.0), (0.0, 100.0)],
            &[GeneKind::Int; 3],
        );
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 100.0);
    }

    #[test]
    fn test_clamp_and_round_bool_thresholds_at_half() {
        let result = clamp_and_round(
            &[0.3_f64, 0.7, 0.5, 0.49],
            &[(0.0_f64, 1.0); 4],
            &[GeneKind::Bool; 4],
        );
        assert_eq!(result, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_reproduce_returns_correct_population_size() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1; 5]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let new_pop = reproduce(&pop, &fitnesses, &make_float_config(20, 5), 42, 0);
        assert_eq!(new_pop.len(), 20);
    }

    #[test]
    fn test_reproduce_returns_correct_gene_length() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 6]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let new_pop = reproduce(&pop, &fitnesses, &make_float_config(10, 6), 42, 0);
        assert!(new_pop.iter().all(|ind| ind.len() == 6));
    }

    #[test]
    fn test_reproduce_float_genes_within_bounds() {
        let pop: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.1 - 1.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..30).map(|i| -(i as f64 * 0.1)).collect();
        let config = ReproduceConfig {
            gene_bounds: vec![(-5.0, 5.0); 4],
            ..make_float_config(30, 4)
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(g >= -5.0 && g <= 5.0);
            }
        }
    }

    #[test]
    fn test_reproduce_int_genes_are_always_integers() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64; 3]).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type:  CrossoverType::Sbx,
            crossover_prob:  0.9,
            crossover_eta:   2.0,
            crossover_alpha: 0.5,
            mutation_type:   MutationType::Gaussian,
            mutation_prob:   0.5,
            mutation_sigmas: vec![2.0; 3],
            gene_bounds:     vec![(0.0, 20.0); 3],
            gene_kinds:      vec![GeneKind::Int; 3],
            selection_type:  SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert_eq!(g, g.round(), "int gene {} not integer-valued", g);
            }
        }
    }

    #[test]
    fn test_reproduce_bool_genes_are_always_binary() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| {
            (0..8).map(|j| if (i + j) % 2 == 0 { 1.0 } else { 0.0 }).collect()
        }).collect();
        let fitnesses: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let config = ReproduceConfig {
            crossover_type: CrossoverType::OnePoint,
            crossover_prob: 0.8,
            crossover_eta:  2.0, crossover_alpha: 0.5,
            mutation_type:  MutationType::BitFlip,
            mutation_prob:  0.1,
            mutation_sigmas: vec![0.0; 8],
            gene_bounds:    vec![(0.0, 1.0); 8],
            gene_kinds:     vec![GeneKind::Bool; 8],
            selection_type: SelectionType::Tournament,
            tournament_size: 2,
            population_size: 20,
        };
        let new_pop = reproduce(&pop, &fitnesses, &config, 42, 0);
        for ind in &new_pop {
            for &g in ind {
                assert!(g == 0.0 || g == 1.0);
            }
        }
    }

    #[test]
    fn test_reproduce_deterministic() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 77, 5);
        let r2 = reproduce(&pop, &fitnesses, &config, 77, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_reproduce_different_generations_diverge() {
        let pop: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5; 4]).collect();
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let config = make_float_config(10, 4);
        let r1 = reproduce(&pop, &fitnesses, &config, 42, 0);
        let r2 = reproduce(&pop, &fitnesses, &config, 42, 1);
        assert_ne!(r1, r2);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test reproduce
```

Expected:
```
test reproduce::tests::test_clamp_and_round_bool_thresholds_at_half ... ok
test reproduce::tests::test_clamp_and_round_float_clamps_to_bounds ... ok
test reproduce::tests::test_clamp_and_round_int_rounds_and_clamps ... ok
test reproduce::tests::test_init_population_bool_genes_are_binary ... ok
test reproduce::tests::test_init_population_correct_gene_length ... ok
test reproduce::tests::test_init_population_correct_size ... ok
test reproduce::tests::test_init_population_deterministic ... ok
test reproduce::tests::test_init_population_different_seeds_diverge ... ok
test reproduce::tests::test_init_population_genes_within_bounds ... ok
test reproduce::tests::test_init_population_int_genes_are_integers ... ok
test reproduce::tests::test_reproduce_bool_genes_are_always_binary ... ok
test reproduce::tests::test_reproduce_deterministic ... ok
test reproduce::tests::test_reproduce_different_generations_diverge ... ok
test reproduce::tests::test_reproduce_float_genes_within_bounds ... ok
test reproduce::tests::test_reproduce_int_genes_are_always_integers ... ok
test reproduce::tests::test_reproduce_returns_correct_gene_length ... ok
test reproduce::tests::test_reproduce_returns_correct_population_size ... ok

test result: ok. 17 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/reproduce.rs
git commit -m "feat(rust): init_population + reproduce (selection→xo→mutation→clamp+round)"
```

---

## Task 3: `src/parallel.rs` — Sequential and Rayon Parallel Evaluation

Two functions for batch fitness evaluation over `Vec<Vec<f64>>`. Both call a Python `fitness_fn` callable via PyO3. True parallelism in `evaluate_parallel_rayon` requires the fitness function to release the GIL (e.g., NumPy-heavy work); for pure-Python fitness functions, `evaluate_sequential` is recommended and documented.

**Files:**
- Modify: `src/parallel.rs`

- [ ] **Step 1: Write failing Rust tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn double_sum(genes: &[f64]) -> f64 {
        genes.iter().sum::<f64>() * 2.0
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_length() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_values() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert!((results[0] - 6.0).abs() < 1e-10, "expected 6.0, got {}", results[0]);
        assert!((results[1] - 14.0).abs() < 1e-10, "expected 14.0, got {}", results[1]);
    }

    #[test]
    fn test_evaluate_batch_parallel_same_as_sequential() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let seq = evaluate_batch_sequential(&pop, double_sum);
        let par = evaluate_batch_parallel(&pop, double_sum, 4);
        for (i, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
            assert!((s - p).abs() < 1e-10,
                "mismatch at index {}: sequential={}, parallel={}", i, s, p);
        }
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cargo test parallel 2>&1 | head -10
```

Expected: `error[E0425]: cannot find function 'evaluate_batch_sequential'`

- [ ] **Step 3: Implement src/parallel.rs**

```rust
use rayon::prelude::*;
use pyo3::prelude::*;

/// Sequential batch evaluation. Calls `fitness_fn` on each gene vector in order.
///
/// This is an internal Rust-only function used by `evaluate_batch_parallel` tests.
/// The Python-facing functions below use PyO3's `Py<PyAny>` for the callable.
pub fn evaluate_batch_sequential<F>(population: &[Vec<f64>], fitness_fn: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    population.iter().map(|genes| fitness_fn(genes)).collect()
}

/// Rayon parallel batch evaluation (internal, Rust-only).
pub fn evaluate_batch_parallel<F>(population: &[Vec<f64>], fitness_fn: F, _n_threads: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    population.par_iter().map(|genes| fitness_fn(genes)).collect()
}

/// Python-facing sequential evaluation.
///
/// Calls `fitness_fn(genes)` for each gene vector in order. The fitness function
/// receives a Python list of floats. Returns a Vec<f64> of fitness values.
/// Recommended for pure-Python fitness functions (no GIL release needed).
#[pyfunction]
pub fn evaluate_sequential(
    py: Python<'_>,
    genes_list: Vec<Vec<f64>>,
    fitness_fn: Py<PyAny>,
) -> PyResult<Vec<f64>> {
    genes_list.iter().map(|genes| {
        let result = fitness_fn.call1(py, (genes.clone(),))?;
        result.extract::<f64>(py)
    }).collect()
}

/// Python-facing Rayon parallel evaluation.
///
/// Releases the GIL and uses Rayon for parallel dispatch. Each Rayon worker
/// re-acquires the GIL to call the Python fitness function. True parallelism
/// only occurs when `fitness_fn` releases the GIL internally (e.g., NumPy ops).
/// For pure-Python fitness functions, use `evaluate_sequential` instead.
#[pyfunction]
pub fn evaluate_parallel_rayon(
    py: Python<'_>,
    genes_list: Vec<Vec<f64>>,
    fitness_fn: Py<PyAny>,
    n_threads: usize,
) -> PyResult<Vec<f64>> {
    let _ = n_threads; // Rayon pool size is set globally at module init (8MB stack)
    py.allow_threads(|| {
        genes_list
            .par_iter()
            .map(|genes| {
                Python::with_gil(|py| {
                    let result = fitness_fn.call1(py, (genes.clone(),))?;
                    result.extract::<f64>(py)
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_sum(genes: &[f64]) -> f64 {
        genes.iter().sum::<f64>() * 2.0
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_length() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_values() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert!((results[0] - 6.0).abs() < 1e-10);
        assert!((results[1] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_batch_parallel_same_as_sequential() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let seq = evaluate_batch_sequential(&pop, double_sum);
        let par = evaluate_batch_parallel(&pop, double_sum, 4);
        for (s, p) in seq.iter().zip(par.iter()) {
            assert!((s - p).abs() < 1e-10);
        }
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cargo test parallel
```

Expected:
```
test parallel::tests::test_evaluate_batch_parallel_same_as_sequential ... ok
test parallel::tests::test_evaluate_batch_sequential_correct_length ... ok
test parallel::tests::test_evaluate_batch_sequential_correct_values ... ok

test result: ok. 3 passed; 0 failed
```

- [ ] **Step 5: Commit**

```bash
git add src/parallel.rs
git commit -m "feat(rust): evaluate_sequential + evaluate_parallel_rayon with PyO3 callables"
```

---

## Task 4: Update `src/lib.rs` — Register All New Exports

Add PyO3 wrapper functions for `init_population` and `reproduce`, and register the selection, parallel, and new functions in `_core`. The `reproduce` and `init_population` wrappers convert Python-friendly string types (e.g., `"sbx"`, `"int"`) to the Rust enum types used internally.

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Replace src/lib.rs with the full updated version**

```rust
use pyo3::prelude::*;

mod gene_spec;
mod individual;
pub mod operators;
pub mod utils;
pub mod selection;
pub mod reproduce;
pub mod parallel;
mod cmaes;

use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use utils::{
    py_derive_seed,
    OP_CMAES_ASK, OP_CROSSOVER, OP_CROSSOVER_PROB,
    OP_INIT, OP_MULTI_RUN, OP_MUTATION, OP_SELECTION,
};
use operators::{binary_ops, float_ops, int_ops};
use parallel::{evaluate_sequential, evaluate_parallel_rayon};
use gene_spec::GeneKind;
use reproduce::{
    CrossoverType, MutationType, SelectionType, ReproduceConfig,
    init_population as rust_init_population,
    reproduce as rust_reproduce,
};

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

// ── Selection wrappers ────────────────────────────────────────────────────────

#[pyfunction]
fn tournament_selection(
    fitnesses: Vec<f64>, k: usize, tournament_size: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> {
    selection::tournament_selection(&fitnesses, k, tournament_size, master_seed, generation)
}

#[pyfunction]
fn roulette_selection(
    fitnesses: Vec<f64>, k: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> {
    selection::roulette_selection(&fitnesses, k, master_seed, generation)
}

#[pyfunction]
fn rank_selection(
    fitnesses: Vec<f64>, k: usize,
    master_seed: u64, generation: u64,
) -> Vec<usize> {
    selection::rank_selection(&fitnesses, k, master_seed, generation)
}

// ── Population initialisation wrapper ────────────────────────────────────────

/// Initialise a population of `population_size` individuals.
///
/// `gene_bounds` is a list of `(low, high)` tuples, one per gene.
/// `gene_kinds` is a list of strings: `"float"`, `"int"`, or `"bool"`.
/// Returns `Vec<Vec<f64>>` using the universal f64 encoding.
#[pyfunction]
fn init_population(
    gene_bounds:     Vec<(f64, f64)>,
    gene_kinds_str:  Vec<String>,
    population_size: usize,
    master_seed:     u64,
) -> PyResult<Vec<Vec<f64>>> {
    let kinds = parse_gene_kinds(&gene_kinds_str)?;
    Ok(rust_init_population(&gene_bounds, &kinds, population_size, master_seed))
}

// ── Reproduce wrapper ─────────────────────────────────────────────────────────

/// Full reproduction step: selection → crossover → mutation → clamp+round.
///
/// `crossover_type`: `"sbx"` | `"blx"` | `"one_point"` | `"two_point"` | `"uniform_xo"`
/// `mutation_type`:  `"gaussian"` | `"uniform"` | `"bit_flip"`
/// `selection_type`: `"tournament"` | `"roulette"` | `"rank"`
/// `gene_kinds`:     list of `"float"` | `"int"` | `"bool"` per gene
/// `mutation_sigmas`: per-gene absolute sigma values (for Gaussian mutation)
///
/// Returns a new population as `Vec<Vec<f64>>`. Elitism is NOT applied here —
/// the Python GAEngine overwrites elite slots after this call.
#[pyfunction]
#[pyo3(signature = (
    population, fitnesses,
    crossover_type, crossover_prob, crossover_eta, crossover_alpha,
    mutation_type, mutation_prob, mutation_sigmas,
    gene_bounds, gene_kinds,
    selection_type, tournament_size,
    population_size,
    master_seed, generation,
))]
fn reproduce_population(
    population:      Vec<Vec<f64>>,
    fitnesses:       Vec<f64>,
    crossover_type:  String,
    crossover_prob:  f64,
    crossover_eta:   f64,
    crossover_alpha: f64,
    mutation_type:   String,
    mutation_prob:   f64,
    mutation_sigmas: Vec<f64>,
    gene_bounds:     Vec<(f64, f64)>,
    gene_kinds:      Vec<String>,
    selection_type:  String,
    tournament_size: usize,
    population_size: usize,
    master_seed:     u64,
    generation:      u64,
) -> PyResult<Vec<Vec<f64>>> {
    let xo_type = match crossover_type.as_str() {
        "sbx"        => CrossoverType::Sbx,
        "blx"        => CrossoverType::Blx,
        "one_point"  => CrossoverType::OnePoint,
        "two_point"  => CrossoverType::TwoPoint,
        "uniform_xo" => CrossoverType::UniformXO,
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown crossover_type: '{}'. Valid: sbx, blx, one_point, two_point, uniform_xo", other)
        )),
    };
    let mut_type = match mutation_type.as_str() {
        "gaussian" => MutationType::Gaussian,
        "uniform"  => MutationType::Uniform,
        "bit_flip" => MutationType::BitFlip,
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown mutation_type: '{}'. Valid: gaussian, uniform, bit_flip", other)
        )),
    };
    let sel_type = match selection_type.as_str() {
        "tournament" => SelectionType::Tournament,
        "roulette"   => SelectionType::Roulette,
        "rank"       => SelectionType::Rank,
        other => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown selection_type: '{}'. Valid: tournament, roulette, rank", other)
        )),
    };
    let kinds = parse_gene_kinds(&gene_kinds)?;
    let config = ReproduceConfig {
        crossover_type:  xo_type,
        crossover_prob,
        crossover_eta,
        crossover_alpha,
        mutation_type:   mut_type,
        mutation_prob,
        mutation_sigmas,
        gene_bounds,
        gene_kinds:      kinds,
        selection_type:  sel_type,
        tournament_size,
        population_size,
    };
    Ok(rust_reproduce(&population, &fitnesses, &config, master_seed, generation))
}

// ── Shared helper ─────────────────────────────────────────────────────────────

fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>> {
    kinds_str.iter().map(|s| match s.as_str() {
        "float" => Ok(GeneKind::Float),
        "int"   => Ok(GeneKind::Int),
        "bool"  => Ok(GeneKind::Bool),
        other   => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown gene kind: '{}'. Valid: float, int, bool", other)
        )),
    }).collect()
}

// ── Module root ───────────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Rayon thread pool — 8 MB stack on all platforms (prevents Windows nalgebra overflow)
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

    // CMA-ES registered in Part 4

    Ok(())
}
```

- [ ] **Step 2: Compile**

```bash
maturin develop --release
```

Expected: no errors.

- [ ] **Step 3: Smoke test all new exports**

```bash
python - << 'EOF'
from evocore._core import (
    tournament_selection, roulette_selection, rank_selection,
    init_population, reproduce_population,
    evaluate_sequential, evaluate_parallel_rayon,
)

# Selection
idx = tournament_selection([1.0, 5.0, 3.0, 2.0, 4.0], 4, 2, 42, 0)
assert len(idx) == 4
assert all(0 <= i < 5 for i in idx)

# Init population
pop = init_population([(-1.0, 1.0)] * 4, ["float"] * 4, 10, 42)
assert len(pop) == 10
assert all(len(ind) == 4 for ind in pop)

# Reproduce
fitnesses = list(range(10))
new_pop = reproduce_population(
    pop, fitnesses,
    "sbx", 0.9, 2.0, 0.5,
    "gaussian", 0.1, [0.2] * 4,
    [(-1.0, 1.0)] * 4, ["float"] * 4,
    "tournament", 3,
    10,
    42, 0,
)
assert len(new_pop) == 10

# Evaluation
genes = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
seq = evaluate_sequential(genes, sum)
assert seq == [3.0, 7.0, 11.0]
par = evaluate_parallel_rayon(genes, sum, 2)
assert par == [3.0, 7.0, 11.0]

print("Part 3 lib.rs exports ok")
EOF
```

Expected: `Part 3 lib.rs exports ok`

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs
git commit -m "feat(rust): expose selection, init_population, reproduce_population, evaluate_* to Python"
```

---

## Task 5: Python Smoke Tests + Full Part 3 Verification

**Files:**
- Create: `tests/unit/test_selection_rust.py`
- Create: `tests/unit/test_reproduce_rust.py`
- Create: `tests/unit/test_parallel_rust.py`

- [ ] **Step 1: Write tests/unit/test_selection_rust.py**

```python
"""
Smoke tests for the Rust selection functions exposed via PyO3.
Focus: correct return types/shapes, NaN safety, and determinism invariant.
"""
import pytest
from evocore._core import (
    tournament_selection, roulette_selection, rank_selection,
)


class TestTournamentSelection:

    def test_returns_correct_length(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = tournament_selection(fitnesses, 4, 2, 42, 0)
        assert len(idx) == 4

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = tournament_selection(fitnesses, 10, 3, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        fitnesses = [1.0, 5.0, 3.0]
        a = tournament_selection(fitnesses, 5, 2, 42, 0)
        b = tournament_selection(fitnesses, 5, 2, 42, 0)
        assert a == b

    def test_different_generations_diverge(self):
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(fitnesses, 5, 2, 42, 0)
        b = tournament_selection(fitnesses, 5, 2, 42, 1)
        assert a != b

    def test_nan_never_wins_full_tournament(self):
        """NaN fitness individual must never win when tournament_size == population."""
        fitnesses = [float("nan")] * 4 + [99.0]
        idx = tournament_selection(fitnesses, 50, 5, 42, 0)
        assert all(i == 4 for i in idx), "NaN individuals should never win"

    def test_returns_list_of_ints(self):
        idx = tournament_selection([1.0, 2.0, 3.0], 3, 2, 42, 0)
        assert all(isinstance(i, int) for i in idx)


class TestRouletteSelection:

    def test_returns_correct_length(self):
        idx = roulette_selection([1.0, 2.0, 3.0, 4.0], 6, 42, 0)
        assert len(idx) == 6

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = roulette_selection(fitnesses, 20, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        a = roulette_selection([1.0, 2.0, 3.0], 5, 7, 2)
        b = roulette_selection([1.0, 2.0, 3.0], 5, 7, 2)
        assert a == b

    def test_nan_fitness_effectively_excluded(self):
        """NaN fitness should have near-zero weight. With 100 draws and only
        one valid individual, that individual wins nearly all draws."""
        fitnesses = [float("nan"), float("nan"), float("nan"), 100.0]
        idx = roulette_selection(fitnesses, 100, 42, 0)
        assert all(i == 3 for i in idx), "NaN individuals should not be selected"


class TestRankSelection:

    def test_returns_correct_length(self):
        idx = rank_selection([1.0, 5.0, 3.0, 2.0, 4.0], 7, 42, 0)
        assert len(idx) == 7

    def test_all_indices_in_range(self):
        fitnesses = [1.0, 5.0, 3.0, 2.0, 4.0]
        idx = rank_selection(fitnesses, 10, 42, 0)
        assert all(0 <= i < len(fitnesses) for i in idx)

    def test_deterministic(self):
        a = rank_selection([3.0, 1.0, 2.0], 5, 13, 5)
        b = rank_selection([3.0, 1.0, 2.0], 5, 13, 5)
        assert a == b

    def test_nan_individual_lowest_rank(self):
        """NaN individual must have rank 1 and virtually never be selected
        when there are higher-ranked alternatives."""
        fitnesses = [float("nan"), float("nan"), 50.0]
        idx = rank_selection(fitnesses, 50, 42, 0)
        assert all(i == 2 for i in idx), "NaN individual must have lowest rank"


class TestSelectionDeterminismInvariant:
    """The (master_seed, generation) pair must fully determine selection output."""

    def test_different_seeds_diverge(self):
        f = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(f, 5, 2, 1, 0)
        b = tournament_selection(f, 5, 2, 2, 0)
        assert a != b

    def test_different_generations_diverge(self):
        f = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = tournament_selection(f, 5, 2, 42, 0)
        b = tournament_selection(f, 5, 2, 42, 1)
        assert a != b
```

- [ ] **Step 2: Write tests/unit/test_reproduce_rust.py**

```python
"""
Smoke tests for init_population and reproduce_population via PyO3.
Focus: correct sizes, type constraints (int/bool), bounds, and determinism.
"""
import pytest
from evocore._core import init_population, reproduce_population


def make_float_pop(pop_size: int, gene_len: int, seed: int = 42):
    return init_population(
        [(-5.0, 5.0)] * gene_len,
        ["float"] * gene_len,
        pop_size, seed,
    )


def run_reproduce(pop, gene_len: int, pop_size: int, seed: int = 42, gen: int = 0):
    fitnesses = [float(i) for i in range(len(pop))]
    return reproduce_population(
        pop, fitnesses,
        "sbx", 0.9, 2.0, 0.5,
        "gaussian", 0.1, [0.5] * gene_len,
        [(-5.0, 5.0)] * gene_len, ["float"] * gene_len,
        "tournament", 3,
        pop_size,
        seed, gen,
    )


class TestInitPopulation:

    def test_correct_population_size(self):
        pop = make_float_pop(20, 5)
        assert len(pop) == 20

    def test_correct_gene_length(self):
        pop = make_float_pop(10, 7)
        assert all(len(ind) == 7 for ind in pop)

    def test_float_genes_within_bounds(self):
        bounds = [(-2.0, 2.0), (0.0, 10.0), (-1.0, 1.0)]
        pop = init_population(bounds, ["float"] * 3, 50, 42)
        for ind in pop:
            for i, g in enumerate(ind):
                lo, hi = bounds[i]
                assert lo <= g < hi, f"gene[{i}]={g} outside [{lo}, {hi})"

    def test_int_genes_are_integer_valued(self):
        bounds = [(5.0, 200.0), (10.0, 500.0)]
        pop = init_population(bounds, ["int", "int"], 20, 42)
        for ind in pop:
            for g in ind:
                assert g == int(g), f"int gene {g} is not integer-valued"

    def test_bool_genes_are_binary(self):
        bounds = [(0.0, 1.0)] * 8
        pop = init_population(bounds, ["bool"] * 8, 20, 42)
        for ind in pop:
            for g in ind:
                assert g in (0.0, 1.0), f"bool gene {g} is not 0.0 or 1.0"

    def test_deterministic(self):
        p1 = make_float_pop(10, 4, seed=7)
        p2 = make_float_pop(10, 4, seed=7)
        assert p1 == p2

    def test_different_seeds_diverge(self):
        p1 = make_float_pop(10, 4, seed=1)
        p2 = make_float_pop(10, 4, seed=2)
        assert p1[0] != p2[0]

    def test_invalid_gene_kind_raises(self):
        with pytest.raises(Exception, match="gene kind"):
            init_population([(0.0, 1.0)], ["quantum"], 5, 42)


class TestReproducePopulation:

    def test_returns_correct_population_size(self):
        pop = make_float_pop(20, 5)
        new_pop = run_reproduce(pop, 5, 20)
        assert len(new_pop) == 20

    def test_returns_correct_gene_length(self):
        pop = make_float_pop(10, 6)
        new_pop = run_reproduce(pop, 6, 10)
        assert all(len(ind) == 6 for ind in new_pop)

    def test_float_genes_within_bounds(self):
        pop = make_float_pop(30, 4)
        fitnesses = list(range(30))
        new_pop = reproduce_population(
            pop, fitnesses,
            "sbx", 0.9, 2.0, 0.5,
            "gaussian", 0.1, [0.5] * 4,
            [(-5.0, 5.0)] * 4, ["float"] * 4,
            "tournament", 3, 30, 42, 0,
        )
        for ind in new_pop:
            for g in ind:
                assert -5.0 <= g <= 5.0, f"float gene {g} outside [-5, 5]"

    def test_int_genes_always_integer_valued(self):
        bounds = [(0.0, 20.0)] * 3
        pop = init_population(bounds, ["int"] * 3, 20, 42)
        fitnesses = list(range(20))
        new_pop = reproduce_population(
            pop, fitnesses,
            "sbx", 0.9, 2.0, 0.5,
            "gaussian", 0.5, [2.0] * 3,
            bounds, ["int"] * 3,
            "tournament", 2, 20, 42, 0,
        )
        for ind in new_pop:
            for g in ind:
                assert g == int(g), f"int gene {g} not integer-valued"

    def test_bool_genes_always_binary(self):
        bounds = [(0.0, 1.0)] * 8
        pop = init_population(bounds, ["bool"] * 8, 20, 42)
        fitnesses = list(range(20))
        new_pop = reproduce_population(
            pop, fitnesses,
            "one_point", 0.8, 2.0, 0.5,
            "bit_flip", 0.1, [0.0] * 8,
            bounds, ["bool"] * 8,
            "tournament", 2, 20, 42, 0,
        )
        for ind in new_pop:
            for g in ind:
                assert g in (0.0, 1.0), f"bool gene {g} not 0.0 or 1.0"

    def test_deterministic(self):
        pop = make_float_pop(10, 4)
        r1 = run_reproduce(pop, 4, 10, seed=77, gen=5)
        r2 = run_reproduce(pop, 4, 10, seed=77, gen=5)
        assert r1 == r2

    def test_different_generations_diverge(self):
        pop = make_float_pop(10, 4)
        r1 = run_reproduce(pop, 4, 10, seed=42, gen=0)
        r2 = run_reproduce(pop, 4, 10, seed=42, gen=1)
        assert r1 != r2, "different generations must produce different offspring"

    def test_invalid_crossover_type_raises(self):
        pop = make_float_pop(10, 3)
        with pytest.raises(Exception, match="crossover_type"):
            reproduce_population(
                pop, list(range(10)),
                "quadratic_crossover", 0.9, 2.0, 0.5,
                "gaussian", 0.1, [0.5] * 3,
                [(-1.0, 1.0)] * 3, ["float"] * 3,
                "tournament", 3, 10, 42, 0,
            )

    def test_roulette_selection_mode(self):
        pop = make_float_pop(15, 3)
        fitnesses = [float(i) for i in range(15)]
        new_pop = reproduce_population(
            pop, fitnesses,
            "sbx", 0.9, 2.0, 0.5,
            "gaussian", 0.1, [0.3] * 3,
            [(-5.0, 5.0)] * 3, ["float"] * 3,
            "roulette", 3, 15, 42, 0,
        )
        assert len(new_pop) == 15

    def test_rank_selection_mode(self):
        pop = make_float_pop(12, 3)
        fitnesses = [float(i) for i in range(12)]
        new_pop = reproduce_population(
            pop, fitnesses,
            "sbx", 0.9, 2.0, 0.5,
            "gaussian", 0.1, [0.3] * 3,
            [(-5.0, 5.0)] * 3, ["float"] * 3,
            "rank", 3, 12, 42, 0,
        )
        assert len(new_pop) == 12

    def test_blx_crossover_mode(self):
        pop = make_float_pop(10, 4)
        r = run_reproduce(pop, 4, 10)
        # BLX doesn't change the API contract — just verify it runs without error
        pop2 = init_population([(-5.0, 5.0)] * 4, ["float"] * 4, 10, 42)
        new_pop = reproduce_population(
            pop2, list(range(10)),
            "blx", 0.9, 2.0, 0.5,
            "gaussian", 0.1, [0.3] * 4,
            [(-5.0, 5.0)] * 4, ["float"] * 4,
            "tournament", 3, 10, 42, 0,
        )
        assert len(new_pop) == 10
```

- [ ] **Step 3: Write tests/unit/test_parallel_rust.py**

```python
"""
Smoke tests for evaluate_sequential and evaluate_parallel_rayon via PyO3.
"""
import pytest
import numpy as np
from evocore._core import evaluate_sequential, evaluate_parallel_rayon


def neg_sphere(genes):
    return -sum(x**2 for x in genes)


def numpy_neg_sphere(genes):
    """NumPy-based fitness — releases GIL, safe for Rayon parallel mode."""
    return float(-np.sum(np.array(genes) ** 2))


class TestEvaluateSequential:

    def test_correct_length(self):
        pop = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert len(result) == 3

    def test_correct_values(self):
        pop = [[1.0, 2.0], [3.0, 4.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert abs(result[0] - (-5.0)) < 1e-10,  f"expected -5.0, got {result[0]}"
        assert abs(result[1] - (-25.0)) < 1e-10, f"expected -25.0, got {result[1]}"

    def test_returns_list_of_floats(self):
        pop = [[0.0, 1.0], [2.0, 3.0]]
        result = evaluate_sequential(pop, neg_sphere)
        assert all(isinstance(v, float) for v in result)

    def test_empty_population(self):
        result = evaluate_sequential([], neg_sphere)
        assert result == []

    def test_single_individual(self):
        result = evaluate_sequential([[1.0, 1.0, 1.0]], neg_sphere)
        assert abs(result[0] - (-3.0)) < 1e-10


class TestEvaluateParallelRayon:

    def test_correct_length(self):
        pop = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = evaluate_parallel_rayon(pop, numpy_neg_sphere, 2)
        assert len(result) == 3

    def test_matches_sequential_for_numpy_fitness(self):
        """parallel="thread" must produce byte-identical results to sequential
        for a NumPy fitness function (which releases the GIL)."""
        pop = [[float(i), float(i * 2)] for i in range(20)]
        seq = evaluate_sequential(pop, numpy_neg_sphere)
        par = evaluate_parallel_rayon(pop, numpy_neg_sphere, 4)
        for i, (s, p) in enumerate(zip(seq, par)):
            assert abs(s - p) < 1e-10, f"mismatch at {i}: seq={s}, par={p}"

    def test_n_threads_1_matches_sequential(self):
        pop = [[float(i)] for i in range(10)]
        seq = evaluate_sequential(pop, numpy_neg_sphere)
        par = evaluate_parallel_rayon(pop, numpy_neg_sphere, 1)
        for s, p in zip(seq, par):
            assert abs(s - p) < 1e-10

    def test_empty_population(self):
        result = evaluate_parallel_rayon([], numpy_neg_sphere, 2)
        assert result == []


class TestEvaluationDeterminism:
    """Fitness evaluation is deterministic because genes fully determine output."""

    def test_same_genes_same_fitness(self):
        pop = [[1.0, 2.0, 3.0]] * 5
        r1 = evaluate_sequential(pop, neg_sphere)
        r2 = evaluate_sequential(pop, neg_sphere)
        assert r1 == r2

    def test_different_genes_different_fitness(self):
        pop1 = [[1.0, 2.0], [3.0, 4.0]]
        pop2 = [[5.0, 6.0], [7.0, 8.0]]
        r1 = evaluate_sequential(pop1, neg_sphere)
        r2 = evaluate_sequential(pop2, neg_sphere)
        assert r1 != r2
```

- [ ] **Step 4: Run all new Python tests**

```bash
pytest tests/unit/test_selection_rust.py tests/unit/test_reproduce_rust.py tests/unit/test_parallel_rust.py -v
```

Expected: All tests pass. The output should show approximately:
```
tests/unit/test_selection_rust.py::TestTournamentSelection::test_... PASSED
...
tests/unit/test_reproduce_rust.py::TestInitPopulation::test_... PASSED
...
tests/unit/test_parallel_rust.py::TestEvaluateSequential::test_... PASSED
...
```

- [ ] **Step 5: Run the full Rust test suite — confirm no regressions**

```bash
cargo test
```

Expected:
```
test result: ok. 97 passed; 0 failed
```

(60 from Parts 1+2 + 17 selection + 17 reproduce + 3 parallel = 97)

- [ ] **Step 6: Run all Python unit tests — confirm no regressions**

```bash
pytest tests/unit/ -v
```

Expected: All prior tests plus the new 40+ tests pass, 0 failures.

- [ ] **Step 7: Final end-to-end smoke test**

```bash
python - << 'EOF'
from evocore._core import (
    init_population, reproduce_population,
    tournament_selection, roulette_selection, rank_selection,
    evaluate_sequential, evaluate_parallel_rayon,
)
import math

# Full GA mini-loop: init → evaluate → reproduce × 3 generations
bounds  = [(-5.0, 5.0)] * 5
kinds   = ["float"] * 5
sigmas  = [0.5] * 5

pop = init_population(bounds, kinds, 20, seed=42)
assert len(pop) == 20
assert all(all(-5.0 <= g < 5.0 for g in ind) for ind in pop)

def neg_sphere(genes):
    return -sum(x**2 for x in genes)

for gen in range(3):
    fitnesses = evaluate_sequential(pop, neg_sphere)
    assert len(fitnesses) == 20
    # Check NaN safety
    assert all(math.isfinite(f) for f in fitnesses)

    pop = reproduce_population(
        pop, fitnesses,
        "sbx", 0.9, 2.0, 0.5,
        "gaussian", 0.1, sigmas,
        bounds, kinds,
        "tournament", 3, 20,
        42, gen,
    )
    assert len(pop) == 20
    best = max(fitnesses)
    print(f"gen {gen}: best_fitness={best:.4f}")

# Verify fitness improves or at least doesn't crash
print("\nPart 3 complete — mini GA loop ran 3 generations without error")
EOF
```

Expected:
```
gen 0: best_fitness=...
gen 1: best_fitness=...
gen 2: best_fitness=...

Part 3 complete — mini GA loop ran 3 generations without error
```

- [ ] **Step 8: Final commit and tag**

```bash
git add tests/unit/test_selection_rust.py \
        tests/unit/test_reproduce_rust.py \
        tests/unit/test_parallel_rust.py
git commit -m "test(python): selection, reproduce, parallel evaluation smoke tests"
git tag part3-complete
```

---

## Part 3 Exit Criteria Checklist

- [ ] `cargo test` passes **97 Rust tests** (60 from Parts 1+2 + 17 selection + 17 reproduce + 3 parallel)
- [ ] `maturin develop --release` succeeds with no errors
- [ ] `pytest tests/unit/` passes all tests including the three new files
- [ ] `tournament_selection`, `roulette_selection`, `rank_selection` importable from `evocore._core`
- [ ] All three selection functions treat `NaN` fitness as worst (never selected when alternatives exist)
- [ ] `init_population` importable; produces correct sizes, gene types, and bounds compliance
- [ ] `reproduce_population` importable; accepts string enum params; returns correct size
- [ ] Int genes always satisfy `g == int(g)` after both `init_population` and `reproduce_population`
- [ ] Bool genes are always `0.0` or `1.0` after both functions
- [ ] Float genes never exceed their `gene_bounds` after `reproduce_population`
- [ ] `reproduce_population` is deterministic: same `(master_seed, generation)` → same result
- [ ] `reproduce_population` is NOT idempotent across generations: `gen=0` ≠ `gen=1`
- [ ] `evaluate_sequential` and `evaluate_parallel_rayon` produce byte-identical results for the same genes
- [ ] Invalid string enum values (`crossover_type`, `mutation_type`, `selection_type`, `gene_kinds`) raise `ValueError` from Python
- [ ] Elitism is **not** handled in `reproduce_population` — verified by design (Python GA engine will handle it in Part 6)
