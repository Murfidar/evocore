use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cmp::Ordering;

use crate::gene_codec::{parse_gene_kinds, repair_encoded_value};
use crate::gene_spec::GeneKind;
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION, OP_SELECTION};

const JDE_F_REFRESH_PROBABILITY: f64 = 0.1;
const JDE_CR_REFRESH_PROBABILITY: f64 = 0.1;
const JDE_F_LOW: f64 = 0.1;
const JDE_F_HIGH: f64 = 1.0;

#[derive(Clone, Debug, PartialEq, Eq)]
enum DEStrategy {
    Rand1,
    Best1,
    Rand2,
    CurrentToBest1,
    JdeRand1,
}

type DifferencePairs = Vec<(usize, usize)>;
type RecipeSlots = (usize, DifferencePairs, Option<usize>);

#[derive(Clone, Debug)]
struct JdeCommittedState {
    f_by_slot: Vec<f64>,
    cr_by_slot: Vec<f64>,
}

#[derive(Clone, Debug)]
struct TrialProposal {
    target_slot: usize,
    genes: Vec<f64>,
    strategy: &'static str,
    base_slot: usize,
    best_slot: Option<usize>,
    donor_slots: Vec<usize>,
    difference_pairs: Vec<(usize, usize)>,
    mutation_factor: Option<f64>,
    crossover_rate: Option<f64>,
    adaptive_slot: Option<usize>,
}

fn parse_strategy(strategy: &str) -> PyResult<DEStrategy> {
    match strategy {
        "rand1bin" => Ok(DEStrategy::Rand1),
        "best1bin" => Ok(DEStrategy::Best1),
        "rand2bin" => Ok(DEStrategy::Rand2),
        "current-to-best1bin" => Ok(DEStrategy::CurrentToBest1),
        "jde-rand1bin" => Ok(DEStrategy::JdeRand1),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown DE strategy: {other}"
        ))),
    }
}

fn strategy_name(strategy: &DEStrategy) -> &'static str {
    match strategy {
        DEStrategy::Rand1 => "rand1bin",
        DEStrategy::Best1 => "best1bin",
        DEStrategy::Rand2 => "rand2bin",
        DEStrategy::CurrentToBest1 => "current-to-best1bin",
        DEStrategy::JdeRand1 => "jde-rand1bin",
    }
}

fn min_population(strategy: &DEStrategy) -> usize {
    match strategy {
        DEStrategy::Rand2 => 6,
        _ => 4,
    }
}

fn comparison_score(score: f64, direction: &str) -> PyResult<f64> {
    if !score.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DE scores must be finite for trial generation.",
        ));
    }
    match direction {
        "maximize" => Ok(score),
        "minimize" => Ok(-score),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "direction must be 'maximize' or 'minimize', got {other:?}."
        ))),
    }
}

fn validate_inputs(
    population: &[Vec<f64>],
    scores: &[f64],
    gene_bounds: &[(f64, f64)],
    gene_kinds: &[GeneKind],
    strategy: &DEStrategy,
    target_slots: &[usize],
) -> PyResult<()> {
    if population.len() < min_population(strategy) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "population_size must be at least {} for strategy='{}'.",
            min_population(strategy),
            strategy_name(strategy)
        )));
    }
    if population.len() != scores.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "population and scores must have the same length.",
        ));
    }
    if gene_bounds.len() != gene_kinds.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gene_bounds and gene_kinds must have the same length.",
        ));
    }
    for row in population {
        if row.len() != gene_bounds.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "each population row must match gene count.",
            ));
        }
    }
    for &slot in target_slots {
        if slot >= population.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "target slot {slot} is outside the population."
            )));
        }
    }
    Ok(())
}

fn proposal_to_py(py: Python<'_>, proposal: &TrialProposal) -> PyResult<Py<PyAny>> {
    let item = PyDict::new(py);
    item.set_item("target_slot", proposal.target_slot)?;
    item.set_item("genes", proposal.genes.clone())?;

    let metadata = PyDict::new(py);
    metadata.set_item("strategy", proposal.strategy)?;
    metadata.set_item("target_slot", proposal.target_slot)?;
    metadata.set_item("base_slot", proposal.base_slot)?;
    metadata.set_item("donor_slots", proposal.donor_slots.clone())?;
    let pairs = PyList::empty(py);
    for (left, right) in &proposal.difference_pairs {
        pairs.append(vec![*left, *right])?;
    }
    metadata.set_item("difference_pairs", pairs)?;
    if let Some(best_slot) = proposal.best_slot {
        metadata.set_item("best_slot", best_slot)?;
    }
    if let Some(value) = proposal.mutation_factor {
        metadata.set_item("mutation_factor", value)?;
    }
    if let Some(value) = proposal.crossover_rate {
        metadata.set_item("crossover_rate", value)?;
    }
    if let Some(value) = proposal.adaptive_slot {
        metadata.set_item("adaptive_slot", value)?;
    }
    item.set_item("metadata", metadata)?;
    Ok(item.into_any().unbind())
}

fn extract_jde_state(
    jde_state: Option<Bound<'_, PyAny>>,
    population_size: usize,
    strategy: &DEStrategy,
) -> PyResult<Option<JdeCommittedState>> {
    if !matches!(strategy, DEStrategy::JdeRand1) {
        return Ok(None);
    }
    let Some(payload) = jde_state else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "jde_state is required for strategy='jde-rand1bin'.",
        ));
    };
    let f_by_slot = payload.get_item("f_by_slot")?.extract::<Vec<f64>>()?;
    let cr_by_slot = payload.get_item("cr_by_slot")?.extract::<Vec<f64>>()?;
    if f_by_slot.len() != population_size || cr_by_slot.len() != population_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "jde_state f_by_slot and cr_by_slot must match population size.",
        ));
    }
    Ok(Some(JdeCommittedState {
        f_by_slot,
        cr_by_slot,
    }))
}

fn best_slot(scores: &[f64], direction: &str) -> PyResult<usize> {
    let mut best_idx = 0;
    let mut best_score = comparison_score(scores[0], direction)?;
    for (idx, score) in scores.iter().enumerate().skip(1) {
        let comparison = comparison_score(*score, direction)?;
        if comparison
            .partial_cmp(&best_score)
            .unwrap_or(Ordering::Less)
            == Ordering::Greater
        {
            best_idx = idx;
            best_score = comparison;
        }
    }
    Ok(best_idx)
}

fn rng_for(seed: u64, generation: u64, slot: usize, op: u64) -> StdRng {
    StdRng::seed_from_u64(derive_seed(seed, generation, slot as u64, op))
}

fn jde_rng(seed: u64, generation: u64, target_slot: usize, offset: u64) -> StdRng {
    StdRng::seed_from_u64(derive_seed(
        seed,
        generation,
        target_slot as u64 * 10 + offset,
        OP_MUTATION,
    ))
}

fn propose_jde_params(
    state: &JdeCommittedState,
    seed: u64,
    generation: u64,
    target_slot: usize,
) -> (f64, f64) {
    let mut f_value = state.f_by_slot[target_slot];
    let mut cr_value = state.cr_by_slot[target_slot];
    let mut f_rng = jde_rng(seed, generation, target_slot, 1);
    let mut cr_rng = jde_rng(seed, generation, target_slot, 2);
    if f_rng.gen::<f64>() < JDE_F_REFRESH_PROBABILITY {
        f_value = JDE_F_LOW + f_rng.gen::<f64>() * (JDE_F_HIGH - JDE_F_LOW);
    }
    if cr_rng.gen::<f64>() < JDE_CR_REFRESH_PROBABILITY {
        cr_value = cr_rng.gen::<f64>();
    }
    (f_value, cr_value)
}

fn sample_slots(
    population_size: usize,
    count: usize,
    excluded: &[usize],
    seed: u64,
    generation: u64,
    target_slot: usize,
) -> PyResult<Vec<usize>> {
    let mut choices: Vec<usize> = (0..population_size)
        .filter(|slot| !excluded.contains(slot))
        .collect();
    if choices.len() < count {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "not enough donor slots for DE strategy.",
        ));
    }
    let mut rng = rng_for(seed, generation, target_slot, OP_SELECTION);
    choices.shuffle(&mut rng);
    Ok(choices.into_iter().take(count).collect())
}

fn recipe_slots(
    population_size: usize,
    strategy: &DEStrategy,
    seed: u64,
    generation: u64,
    target_slot: usize,
    best_slot: usize,
) -> PyResult<RecipeSlots> {
    match strategy {
        DEStrategy::Rand1 | DEStrategy::JdeRand1 => {
            let slots = sample_slots(
                population_size,
                3,
                &[target_slot],
                seed,
                generation,
                target_slot,
            )?;
            Ok((slots[0], vec![(slots[1], slots[2])], None))
        }
        DEStrategy::Best1 => {
            let slots = sample_slots(
                population_size,
                2,
                &[target_slot, best_slot],
                seed,
                generation,
                target_slot,
            )?;
            Ok((best_slot, vec![(slots[0], slots[1])], Some(best_slot)))
        }
        DEStrategy::Rand2 => {
            let slots = sample_slots(
                population_size,
                5,
                &[target_slot],
                seed,
                generation,
                target_slot,
            )?;
            Ok((
                slots[0],
                vec![(slots[1], slots[2]), (slots[3], slots[4])],
                None,
            ))
        }
        DEStrategy::CurrentToBest1 => {
            let slots = sample_slots(
                population_size,
                2,
                &[target_slot, best_slot],
                seed,
                generation,
                target_slot,
            )?;
            Ok((target_slot, vec![(slots[0], slots[1])], Some(best_slot)))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mutant_value(
    population: &[Vec<f64>],
    gene_idx: usize,
    base: &[f64],
    target: &[f64],
    difference_pairs: &[(usize, usize)],
    best_slot: Option<usize>,
    mutation_factor: f64,
    current_to_best: bool,
    bool_rng: &mut StdRng,
    kind: &GeneKind,
) -> f64 {
    if matches!(kind, GeneKind::Bool) {
        let mut value = base[gene_idx] >= 0.5;
        if current_to_best {
            if let Some(slot) = best_slot {
                let best = population[slot][gene_idx] >= 0.5;
                let target_value = target[gene_idx] >= 0.5;
                if best != target_value {
                    return if best { 1.0 } else { 0.0 };
                }
            }
        }
        for (left, right) in difference_pairs {
            let left_value = population[*left][gene_idx] >= 0.5;
            let right_value = population[*right][gene_idx] >= 0.5;
            if left_value != right_value && bool_rng.gen::<f64>() < mutation_factor.min(1.0) {
                value = !value;
            }
        }
        return if value { 1.0 } else { 0.0 };
    }

    let mut value = base[gene_idx];
    if current_to_best {
        if let Some(slot) = best_slot {
            value = target[gene_idx]
                + mutation_factor * (population[slot][gene_idx] - target[gene_idx]);
        }
    }
    for (left, right) in difference_pairs {
        value += mutation_factor * (population[*left][gene_idx] - population[*right][gene_idx]);
    }
    value
}

#[allow(clippy::too_many_arguments)]
fn build_trial_genes(
    population: &[Vec<f64>],
    gene_bounds: &[(f64, f64)],
    gene_kinds: &[GeneKind],
    seed: u64,
    generation: u64,
    target_slot: usize,
    base_slot: usize,
    difference_pairs: &[(usize, usize)],
    best_slot: Option<usize>,
    mutation_factor: f64,
    crossover_rate: f64,
    current_to_best: bool,
) -> Vec<f64> {
    let target = &population[target_slot];
    let base = &population[base_slot];
    let mut mask_rng = rng_for(seed, generation, target_slot, OP_CROSSOVER);
    let mut bool_rng = rng_for(seed, generation, target_slot, OP_MUTATION);
    let gene_count = gene_bounds.len();
    let variable_indices: Vec<usize> = gene_bounds
        .iter()
        .enumerate()
        .filter_map(|(idx, (low, high))| (low != high).then_some(idx))
        .collect();
    let forced_index = if variable_indices.is_empty() {
        0
    } else {
        variable_indices[mask_rng.gen_range(0..variable_indices.len())]
    };

    (0..gene_count)
        .map(|gene_idx| {
            let (low, high) = gene_bounds[gene_idx];
            if low == high {
                return repair_encoded_value(low, gene_bounds[gene_idx], &gene_kinds[gene_idx]);
            }
            let selected = gene_idx == forced_index || mask_rng.gen::<f64>() < crossover_rate;
            if !selected {
                return repair_encoded_value(
                    target[gene_idx],
                    gene_bounds[gene_idx],
                    &gene_kinds[gene_idx],
                );
            }
            let value = mutant_value(
                population,
                gene_idx,
                base,
                target,
                difference_pairs,
                best_slot,
                mutation_factor,
                current_to_best,
                &mut bool_rng,
                &gene_kinds[gene_idx],
            );
            repair_encoded_value(value, gene_bounds[gene_idx], &gene_kinds[gene_idx])
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn generate_one_trial(
    population: &[Vec<f64>],
    scores: &[f64],
    gene_bounds: &[(f64, f64)],
    gene_kinds: &[GeneKind],
    strategy: &DEStrategy,
    mutation_factor: f64,
    crossover_rate: f64,
    seed: u64,
    generation: u64,
    target_slot: usize,
    direction: &str,
    jde_state: Option<&JdeCommittedState>,
) -> PyResult<TrialProposal> {
    let best_slot = best_slot(scores, direction)?;
    let (f, cr) = if let (DEStrategy::JdeRand1, Some(state)) = (strategy, jde_state) {
        propose_jde_params(state, seed, generation, target_slot)
    } else {
        (mutation_factor, crossover_rate)
    };
    let (base_slot, pairs, reported_best) = recipe_slots(
        population.len(),
        strategy,
        seed,
        generation,
        target_slot,
        best_slot,
    )?;
    let genes = build_trial_genes(
        population,
        gene_bounds,
        gene_kinds,
        seed,
        generation,
        target_slot,
        base_slot,
        &pairs,
        reported_best,
        f,
        cr,
        matches!(strategy, DEStrategy::CurrentToBest1),
    );
    let donor_slots = std::iter::once(base_slot)
        .chain(pairs.iter().flat_map(|(left, right)| [*left, *right]))
        .collect();
    Ok(TrialProposal {
        target_slot,
        genes,
        strategy: strategy_name(strategy),
        base_slot,
        best_slot: reported_best,
        donor_slots,
        difference_pairs: pairs,
        mutation_factor: matches!(strategy, DEStrategy::JdeRand1).then_some(f),
        crossover_rate: matches!(strategy, DEStrategy::JdeRand1).then_some(cr),
        adaptive_slot: matches!(strategy, DEStrategy::JdeRand1).then_some(target_slot),
    })
}

#[allow(clippy::too_many_arguments)]
fn generate_trials(
    population: &[Vec<f64>],
    scores: &[f64],
    gene_bounds: &[(f64, f64)],
    gene_kinds: &[GeneKind],
    strategy: &DEStrategy,
    mutation_factor: f64,
    crossover_rate: f64,
    seed: u64,
    generation: u64,
    target_slots: &[usize],
    direction: &str,
    jde_state: Option<&JdeCommittedState>,
) -> PyResult<Vec<TrialProposal>> {
    target_slots
        .iter()
        .map(|&target_slot| {
            generate_one_trial(
                population,
                scores,
                gene_bounds,
                gene_kinds,
                strategy,
                mutation_factor,
                crossover_rate,
                seed,
                generation,
                target_slot,
                direction,
                jde_state,
            )
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    population,
    scores,
    gene_bounds,
    gene_kinds,
    strategy,
    mutation_factor,
    crossover_rate,
    seed,
    generation,
    target_slots,
    direction,
    jde_state=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn de_generate_trials(
    py: Python<'_>,
    population: Vec<Vec<f64>>,
    scores: Vec<f64>,
    gene_bounds: Vec<(f64, f64)>,
    gene_kinds: Vec<String>,
    strategy: String,
    mutation_factor: f64,
    crossover_rate: f64,
    seed: u64,
    generation: u64,
    target_slots: Vec<usize>,
    direction: String,
    jde_state: Option<Bound<'_, PyAny>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let parsed_strategy = parse_strategy(&strategy)?;
    let kinds = parse_gene_kinds(&gene_kinds)?;
    validate_inputs(
        &population,
        &scores,
        &gene_bounds,
        &kinds,
        &parsed_strategy,
        &target_slots,
    )?;
    let committed_jde = extract_jde_state(jde_state, population.len(), &parsed_strategy)?;
    let proposals = generate_trials(
        &population,
        &scores,
        &gene_bounds,
        &kinds,
        &parsed_strategy,
        mutation_factor,
        crossover_rate,
        seed,
        generation,
        &target_slots,
        &direction,
        committed_jde.as_ref(),
    )?;
    proposals
        .iter()
        .map(|proposal| proposal_to_py(py, proposal))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0), (0.0, 1.0)]
    }

    fn kinds() -> Vec<GeneKind> {
        vec![
            GeneKind::Float,
            GeneKind::Float,
            GeneKind::Int,
            GeneKind::Bool,
        ]
    }

    fn population() -> Vec<Vec<f64>> {
        vec![
            vec![-4.0, -3.0, 1.0, 0.0],
            vec![-2.0, -1.0, 2.0, 1.0],
            vec![0.0, 1.0, 3.0, 0.0],
            vec![1.5, 2.0, 4.0, 1.0],
            vec![3.0, 4.0, 5.0, 0.0],
            vec![4.0, 5.0, 6.0, 1.0],
        ]
    }

    fn scores() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 5.0]
    }

    fn proposals(strategy: DEStrategy) -> Vec<TrialProposal> {
        generate_trials(
            &population(),
            &scores(),
            &bounds(),
            &kinds(),
            &strategy,
            0.7,
            0.9,
            42,
            3,
            &[0, 1, 2],
            "maximize",
            None,
        )
        .unwrap()
    }

    fn assert_valid_genes(genes: &[f64]) {
        for (idx, value) in genes.iter().enumerate() {
            let (low, high) = bounds()[idx];
            assert!(*value >= low && *value <= high);
        }
        assert_eq!(genes[2], genes[2].round());
        assert!(genes[3] == 0.0 || genes[3] == 1.0);
    }

    #[test]
    fn de_rand1bin_donors_exclude_target() {
        let proposals = proposals(DEStrategy::Rand1);
        for proposal in proposals {
            assert_eq!(proposal.donor_slots.len(), 3);
            assert!(!proposal.difference_pairs.iter().any(|(left, right)| {
                *left == proposal.target_slot || *right == proposal.target_slot
            }));
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_best1bin_uses_best_as_base() {
        let proposals = proposals(DEStrategy::Best1);
        for proposal in proposals {
            assert_eq!(proposal.best_slot, Some(4));
            assert_eq!(proposal.base_slot, 4);
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_rand2bin_uses_five_donor_slots() {
        let proposals = proposals(DEStrategy::Rand2);
        for proposal in proposals {
            assert_eq!(proposal.donor_slots.len(), 5);
            assert_eq!(proposal.difference_pairs.len(), 2);
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_current_to_best_uses_target_as_base() {
        let proposals = proposals(DEStrategy::CurrentToBest1);
        for proposal in proposals {
            assert_eq!(proposal.base_slot, proposal.target_slot);
            assert_eq!(proposal.best_slot, Some(4));
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_best_slot_honors_minimize_direction() {
        assert_eq!(best_slot(&scores(), "minimize").unwrap(), 0);
        assert_eq!(best_slot(&scores(), "maximize").unwrap(), 4);
    }

    #[test]
    fn de_generation_is_deterministic() {
        let first = proposals(DEStrategy::Rand2);
        let second = proposals(DEStrategy::Rand2);
        assert_eq!(first[0].genes, second[0].genes);
        assert_eq!(first[0].donor_slots, second[0].donor_slots);
    }

    #[test]
    fn de_forced_crossover_uses_variable_gene_when_fixed_gene_exists() {
        let population = vec![
            vec![1.5, 0.0],
            vec![1.5, 1.0],
            vec![1.5, 3.0],
            vec![1.5, -2.0],
        ];
        let proposal = generate_one_trial(
            &population,
            &[0.0, 1.0, 2.0, 3.0],
            &[(1.5, 1.5), (-10.0, 10.0)],
            &[GeneKind::Float, GeneKind::Float],
            &DEStrategy::Rand1,
            0.8,
            0.0,
            0,
            0,
            0,
            "maximize",
            None,
        )
        .unwrap();

        assert_eq!(proposal.genes[0], 1.5);
        assert_ne!(proposal.genes[1], population[0][1]);
    }

    #[test]
    fn de_jde_proposal_is_deterministic() {
        let state = JdeCommittedState {
            f_by_slot: vec![0.5; 6],
            cr_by_slot: vec![0.9; 6],
        };
        let first = propose_jde_params(&state, 42, 99, 3);
        let second = propose_jde_params(&state, 42, 99, 3);
        assert_eq!(first, second);
        assert!(first.0 >= JDE_F_LOW && first.0 <= JDE_F_HIGH);
        assert!(first.1 >= 0.0 && first.1 <= 1.0);
    }
}
