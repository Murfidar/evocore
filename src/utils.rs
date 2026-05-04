use pyo3::prelude::*;

pub const OP_INIT: u64 = 0;
pub const OP_CROSSOVER: u64 = 1;
pub const OP_MUTATION: u64 = 2;
pub const OP_SELECTION: u64 = 3;
pub const OP_CMAES_ASK: u64 = 4;
pub const OP_MULTI_RUN: u64 = 5;
pub const OP_CROSSOVER_PROB: u64 = 6;

pub fn derive_seed(master: u64, generation: u64, individual_idx: u64, op: u64) -> u64 {
    let mut x = master
        .wrapping_add(generation.wrapping_mul(0x9e3779b97f4a7c15))
        .wrapping_add(individual_idx.wrapping_mul(0x6c62272e07bb0142))
        .wrapping_add(op.wrapping_mul(0xd2b74407b1ce6d93));

    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

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
        assert_ne!(
            a,
            b,
            "derive_seed must not be commutative across generation and individual_idx"
        );
    }

    #[test]
    fn test_derive_seed_avalanche_on_master() {
        let base = derive_seed(0u64, 5, 3, OP_CROSSOVER);
        let flipped = derive_seed(1u64, 5, 3, OP_CROSSOVER);
        let bits_changed = (base ^ flipped).count_ones();
        assert!(
            bits_changed >= 16,
            "Expected >=16 bits to flip (avalanche), got {}",
            bits_changed
        );
    }
}
