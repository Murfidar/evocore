use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

pub fn int_simulated_binary_crossover(
    a: &[f64],
    b: &[f64],
    eta: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), b.len(), "int_sbx: parent lengths must match");

    let mut rng =
        StdRng::seed_from_u64(derive_seed(master_seed, generation, individual_idx, OP_CROSSOVER));
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

pub fn int_gaussian_mutation(
    genes: &[f64],
    sigma: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    let mut rng =
        StdRng::seed_from_u64(derive_seed(master_seed, generation, individual_idx, OP_MUTATION));
    let normal = Normal::new(0.0_f64, sigma).expect("sigma must be > 0");

    genes
        .iter()
        .map(|&g| {
            if rng.gen::<f64>() < prob {
                (g + normal.sample(&mut rng)).round()
            } else {
                g
            }
        })
        .collect()
}

pub fn int_uniform_mutation(
    genes: &[f64],
    low: f64,
    high: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    let mut rng =
        StdRng::seed_from_u64(derive_seed(master_seed, generation, individual_idx, OP_MUTATION));
    let lo = low.round() as i64;
    let hi = high.round() as i64;

    genes
        .iter()
        .map(|&g| {
            if rng.gen::<f64>() < prob {
                rng.gen_range(lo..=hi) as f64
            } else {
                g
            }
        })
        .collect()
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
            assert!(*v >= 5.0 && *v <= 200.0, "value {} outside [5, 200]", v);
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
