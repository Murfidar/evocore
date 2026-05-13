use std::cmp::Ordering;

use crate::selection::safe_fitness;
use crate::utils::{derive_seed, OP_SELECTION};

pub fn candidate_id(master_seed: u64, event_index: u64, candidate_index: u64) -> String {
    let left = derive_seed(master_seed, event_index, candidate_index, OP_SELECTION);
    let right = derive_seed(
        master_seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        candidate_index,
        event_index,
        OP_SELECTION,
    );
    format!("c-{left:016x}{right:016x}")
}

pub fn rank_top_k(scores: &[f64], trusted_mask: &[bool], k: usize) -> Vec<usize> {
    assert_eq!(
        scores.len(),
        trusted_mask.len(),
        "scores and trusted_mask length mismatch"
    );
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(
        |&left, &right| match trusted_mask[right].cmp(&trusted_mask[left]) {
            Ordering::Equal => safe_fitness(scores[right])
                .partial_cmp(&safe_fitness(scores[left]))
                .unwrap_or(Ordering::Equal),
            ordering => ordering,
        },
    );
    indices.truncate(k.min(indices.len()));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_id_deterministic() {
        assert_eq!(candidate_id(42, 1, 2), candidate_id(42, 1, 2));
        assert_ne!(candidate_id(42, 1, 2), candidate_id(42, 1, 3));
    }

    #[test]
    fn test_rank_top_k_prefers_trusted() {
        let ranked = rank_top_k(&[0.9, 10.0, 0.7], &[true, false, true], 2);
        assert_eq!(ranked, vec![0, 2]);
    }
}
