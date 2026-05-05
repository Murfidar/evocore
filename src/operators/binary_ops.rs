use rand::prelude::*;
use rand::rngs::StdRng;

use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION};

pub fn one_point_crossover(
    a: &[f64],
    b: &[f64],
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        b.len(),
        "one_point_crossover: parent lengths must match"
    );

    let n = a.len();
    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_CROSSOVER,
    ));
    let point = if n > 1 { rng.gen_range(1..n) } else { 0 };

    let c1 = a[..point]
        .iter()
        .chain(b[point..].iter())
        .copied()
        .collect();
    let c2 = b[..point]
        .iter()
        .chain(a[point..].iter())
        .copied()
        .collect();

    (c1, c2)
}

pub fn two_point_crossover(
    a: &[f64],
    b: &[f64],
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        b.len(),
        "two_point_crossover: parent lengths must match"
    );

    let n = a.len();
    let mut rng = StdRng::seed_from_u64(derive_seed(
        master_seed,
        generation,
        individual_idx,
        OP_CROSSOVER,
    ));
    let (p1, p2) = if n > 1 {
        let x = rng.gen_range(0..n);
        let y = rng.gen_range(0..n);
        if x <= y {
            (x, y)
        } else {
            (y, x)
        }
    } else {
        (0, 0)
    };

    let c1 = (0..n)
        .map(|i| if i >= p1 && i < p2 { b[i] } else { a[i] })
        .collect();
    let c2 = (0..n)
        .map(|i| if i >= p1 && i < p2 { a[i] } else { b[i] })
        .collect();

    (c1, c2)
}

pub fn uniform_crossover(
    a: &[f64],
    b: &[f64],
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        b.len(),
        "uniform_crossover: parent lengths must match"
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

pub fn bit_flip_mutation(
    genes: &[f64],
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
                if g >= 0.5 {
                    0.0
                } else {
                    1.0
                }
            } else {
                g
            }
        })
        .collect()
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
            assert_eq!(
                c1[i] + c2[i],
                a[i] + b[i],
                "partition property violated at position {}",
                i
            );
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
