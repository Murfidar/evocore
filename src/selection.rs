use rand::prelude::*;
use rand::rngs::StdRng;
use std::cmp::Ordering;

use crate::utils::{derive_seed, OP_SELECTION};

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

fn pick_weighted_index(rng: &mut StdRng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..weights.len());
    }

    let mut r = rng.gen::<f64>() * total;
    for (idx, weight) in weights.iter().enumerate() {
        r -= weight;
        if r <= 0.0 {
            return idx;
        }
    }

    weights.len() - 1
}

pub fn tournament_selection(
    fitnesses: &[f64],
    k: usize,
    tournament_size: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "tournament_selection: empty population");
    assert!(
        tournament_size > 0,
        "tournament_selection: tournament_size must be >= 1"
    );

    let safe: Vec<f64> = fitnesses.iter().copied().map(safe_fitness).collect();
    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));

    (0..k)
        .map(|_| {
            let contestants: Vec<usize> = if tournament_size >= n {
                (0..n).collect()
            } else {
                (0..n).choose_multiple(&mut rng, tournament_size)
            };

            contestants
                .into_iter()
                .max_by(|&left, &right| {
                    safe[left]
                        .partial_cmp(&safe[right])
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap()
        })
        .collect()
}

pub fn roulette_selection(
    fitnesses: &[f64],
    k: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "roulette_selection: empty population");

    let safe: Vec<f64> = fitnesses.iter().copied().map(safe_fitness).collect();
    let finite: Vec<f64> = safe
        .iter()
        .copied()
        .filter(|value| *value != f64::NEG_INFINITY)
        .collect();
    let min_fit = finite.iter().copied().reduce(f64::min);
    let weights: Vec<f64> = match min_fit {
        Some(min_fit) => safe
            .iter()
            .map(|&value| {
                if value == f64::NEG_INFINITY {
                    0.0
                } else {
                    (value - min_fit) + 1.0
                }
            })
            .collect(),
        None => vec![1.0; n],
    };

    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));
    (0..k)
        .map(|_| pick_weighted_index(&mut rng, &weights))
        .collect()
}

pub fn rank_selection(
    fitnesses: &[f64],
    k: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    let n = fitnesses.len();
    assert!(n > 0, "rank_selection: empty population");

    let safe: Vec<f64> = fitnesses.iter().copied().map(safe_fitness).collect();
    let mut order: Vec<(usize, f64)> = safe.iter().copied().enumerate().collect();
    order.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));

    let mut weights = vec![0.0; n];
    let finite_count = safe
        .iter()
        .filter(|&&value| value != f64::NEG_INFINITY)
        .count();
    if finite_count == 0 {
        weights.fill(1.0);
    } else {
        let mut rank = 1.0;
        for (idx, value) in order {
            if value != f64::NEG_INFINITY {
                weights[idx] = rank;
                rank += 1.0;
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(derive_seed(master_seed, generation, 0, OP_SELECTION));
    (0..k)
        .map(|_| pick_weighted_index(&mut rng, &weights))
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
        assert_eq!(safe_fitness(2.5), 2.5);
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
        assert_ne!(
            a, b,
            "different generations must produce different selection"
        );
    }

    #[test]
    fn test_tournament_nan_fitness_never_wins_large_tournament() {
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 99.0];
        let indices = tournament_selection(&fitnesses, 100, 5, 42, 0);
        assert!(
            indices.iter().all(|&i| i == 4),
            "NaN individuals should never win a full-population tournament"
        );
    }

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
        let fitnesses = vec![f64::NAN, f64::NAN, f64::NAN, 100.0];
        let indices = roulette_selection(&fitnesses, 100, 42, 0);
        assert!(
            indices.iter().all(|&i| i == 3),
            "NaN individuals should have near-zero selection probability"
        );
    }

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
        let fitnesses = vec![f64::NAN, f64::NAN, 50.0];
        let indices = rank_selection(&fitnesses, 50, 42, 0);
        assert!(
            indices.iter().all(|&i| i == 2),
            "NaN individuals must have rank 1 (lowest probability)"
        );
    }
}
