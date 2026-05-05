use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

pub fn blend_crossover(
    a: &[f64],
    b: &[f64],
    alpha: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        b.len(),
        "blend_crossover: parent lengths must match"
    );

    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_CROSSOVER,
    ));
    let mut c1 = Vec::with_capacity(a.len());
    let mut c2 = Vec::with_capacity(a.len());

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let diff = (ai - bi).abs();
        let lo = ai.min(bi) - alpha * diff;
        let hi = ai.max(bi) + alpha * diff;

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

pub fn simulated_binary_crossover(
    a: &[f64],
    b: &[f64],
    eta: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "sbx: parent lengths must match");

    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_CROSSOVER,
    ));
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

pub fn gaussian_mutation(
    genes: &[f64],
    sigma: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_MUTATION,
    ));
    let normal = Normal::new(0.0_f64, sigma).expect("sigma must be > 0");

    genes
        .iter()
        .map(|&g| {
            if rng.gen::<f64>() < prob {
                g + normal.sample(&mut rng)
            } else {
                g
            }
        })
        .collect()
}

pub fn uniform_mutation(
    genes: &[f64],
    low: f64,
    high: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_MUTATION,
    ));

    genes
        .iter()
        .map(|&g| {
            if rng.gen::<f64>() < prob {
                rng.gen_range(low..high)
            } else {
                g
            }
        })
        .collect()
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
            assert!(
                c1[i] >= lo - 1e-10 && c1[i] <= hi + 1e-10,
                "c1[{}]={} outside [{}, {}]",
                i,
                c1[i],
                lo,
                hi
            );
            assert!(
                c2[i] >= lo - 1e-10 && c2[i] <= hi + 1e-10,
                "c2[{}]={} outside [{}, {}]",
                i,
                c2[i],
                lo,
                hi
            );
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
            assert!(
                (c1[i] + c2[i] - a[i] - b[i]).abs() < 1e-9,
                "SBX conservation violated at gene {}",
                i
            );
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
        let changed = result
            .iter()
            .zip(genes.iter())
            .filter(|(r, g)| (*r - *g).abs() > 1e-12)
            .count();
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
