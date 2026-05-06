use pyo3::prelude::*;

mod cmaes;
mod gene_spec;
mod individual;
pub mod operators;
pub mod parallel;
pub mod reproduce;
pub mod selection;
pub mod utils;

use cmaes::PyCMAESState;
use gene_spec::GeneKind;
use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use operators::{binary_ops, float_ops, int_ops};
use parallel::{evaluate_parallel_rayon, evaluate_sequential};
use reproduce::{
    init_population as rust_init_population, reproduce as rust_reproduce, CrossoverType,
    MutationType, ReproduceConfig, SelectionType,
};
use utils::{
    py_derive_seed, OP_CMAES_ASK, OP_CROSSOVER, OP_CROSSOVER_PROB, OP_INIT, OP_MULTI_RUN,
    OP_MUTATION, OP_SELECTION,
};

#[pyfunction]
fn blend_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    alpha: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    float_ops::blend_crossover(&a, &b, alpha, master_seed, generation, individual_idx)
}

#[pyfunction]
fn simulated_binary_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    eta: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    float_ops::simulated_binary_crossover(&a, &b, eta, master_seed, generation, individual_idx)
}

#[pyfunction]
fn gaussian_mutation(
    genes: Vec<f64>,
    sigma: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    float_ops::gaussian_mutation(&genes, sigma, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn uniform_mutation(
    genes: Vec<f64>,
    low: f64,
    high: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    float_ops::uniform_mutation(
        &genes,
        low,
        high,
        prob,
        master_seed,
        generation,
        individual_idx,
    )
}

#[pyfunction]
fn int_simulated_binary_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    eta: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    int_ops::int_simulated_binary_crossover(&a, &b, eta, master_seed, generation, individual_idx)
}

#[pyfunction]
fn int_gaussian_mutation(
    genes: Vec<f64>,
    sigma: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    int_ops::int_gaussian_mutation(&genes, sigma, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn int_uniform_mutation(
    genes: Vec<f64>,
    low: f64,
    high: f64,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    int_ops::int_uniform_mutation(
        &genes,
        low,
        high,
        prob,
        master_seed,
        generation,
        individual_idx,
    )
}

#[pyfunction]
fn one_point_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::one_point_crossover(&a, &b, master_seed, generation, individual_idx)
}

#[pyfunction]
fn two_point_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::two_point_crossover(&a, &b, master_seed, generation, individual_idx)
}

#[pyfunction]
fn uniform_crossover(
    a: Vec<f64>,
    b: Vec<f64>,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> (Vec<f64>, Vec<f64>) {
    binary_ops::uniform_crossover(&a, &b, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn bit_flip_mutation(
    genes: Vec<f64>,
    prob: f64,
    master_seed: u64,
    generation: u64,
    individual_idx: u64,
) -> Vec<f64> {
    binary_ops::bit_flip_mutation(&genes, prob, master_seed, generation, individual_idx)
}

#[pyfunction]
fn tournament_selection(
    fitnesses: Vec<f64>,
    k: usize,
    tournament_size: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    selection::tournament_selection(&fitnesses, k, tournament_size, master_seed, generation)
}

#[pyfunction]
fn roulette_selection(
    fitnesses: Vec<f64>,
    k: usize,
    master_seed: u64,
    generation: u64,
) -> Vec<usize> {
    selection::roulette_selection(&fitnesses, k, master_seed, generation)
}

#[pyfunction]
fn rank_selection(fitnesses: Vec<f64>, k: usize, master_seed: u64, generation: u64) -> Vec<usize> {
    selection::rank_selection(&fitnesses, k, master_seed, generation)
}

#[pyfunction]
fn init_population(
    gene_bounds: Vec<(f64, f64)>,
    gene_kinds_str: Vec<String>,
    population_size: usize,
    master_seed: u64,
) -> PyResult<Vec<Vec<f64>>> {
    let kinds = parse_gene_kinds(&gene_kinds_str)?;
    Ok(rust_init_population(
        &gene_bounds,
        &kinds,
        population_size,
        master_seed,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    population, fitnesses,
    crossover_type, crossover_prob, crossover_eta, crossover_alpha,
    mutation_type, mutation_prob, mutation_sigmas,
    gene_bounds, gene_kinds,
    selection_type, tournament_size,
    population_size,
    master_seed, generation,
    mutation_individual_prob=1.0,
))]
#[allow(clippy::too_many_arguments)]
fn reproduce_population(
    population: Vec<Vec<f64>>,
    fitnesses: Vec<f64>,
    crossover_type: String,
    crossover_prob: f64,
    crossover_eta: f64,
    crossover_alpha: f64,
    mutation_type: String,
    mutation_prob: f64,
    mutation_sigmas: Vec<f64>,
    gene_bounds: Vec<(f64, f64)>,
    gene_kinds: Vec<String>,
    selection_type: String,
    tournament_size: usize,
    population_size: usize,
    master_seed: u64,
    generation: u64,
    mutation_individual_prob: f64,
) -> PyResult<Vec<Vec<f64>>> {
    if population.len() != fitnesses.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "population and fitnesses must have the same length",
        ));
    }
    if gene_bounds.len() != gene_kinds.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gene_bounds and gene_kinds must have the same length",
        ));
    }
    if gene_bounds.len() != mutation_sigmas.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mutation_sigmas length must match gene count",
        ));
    }
    if !(0.0..=1.0).contains(&mutation_individual_prob) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mutation_individual_prob must be in [0, 1]",
        ));
    }

    let xo_type = match crossover_type.as_str() {
        "sbx" => CrossoverType::Sbx,
        "blx" => CrossoverType::Blx,
        "one_point" => CrossoverType::OnePoint,
        "two_point" => CrossoverType::TwoPoint,
        "uniform_xo" | "uniform" => CrossoverType::UniformXO,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown crossover_type: '{}'. Valid: sbx, blx, one_point, two_point, uniform_xo",
                other
            )))
        }
    };

    let mut_type = match mutation_type.as_str() {
        "gaussian" => MutationType::Gaussian,
        "uniform" => MutationType::Uniform,
        "bit_flip" => MutationType::BitFlip,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown mutation_type: '{}'. Valid: gaussian, uniform, bit_flip",
                other
            )))
        }
    };

    let sel_type = match selection_type.as_str() {
        "tournament" => SelectionType::Tournament,
        "roulette" => SelectionType::Roulette,
        "rank" => SelectionType::Rank,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown selection_type: '{}'. Valid: tournament, roulette, rank",
                other
            )))
        }
    };

    let kinds = parse_gene_kinds(&gene_kinds)?;
    let config = ReproduceConfig {
        crossover_type: xo_type,
        crossover_prob,
        crossover_eta,
        crossover_alpha,
        mutation_type: mut_type,
        mutation_prob,
        mutation_individual_prob,
        mutation_sigmas,
        gene_bounds,
        gene_kinds: kinds,
        selection_type: sel_type,
        tournament_size,
        population_size,
    };

    Ok(rust_reproduce(
        &population,
        &fitnesses,
        &config,
        master_seed,
        generation,
    ))
}

fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>> {
    kinds_str
        .iter()
        .map(|kind| match kind.as_str() {
            "float" => Ok(GeneKind::Float),
            "int" => Ok(GeneKind::Int),
            "bool" => Ok(GeneKind::Bool),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown gene kind: '{}'. Valid: float, int, bool",
                other
            ))),
        })
        .collect()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;
    m.add_class::<PyCMAESState>()?;

    m.add_function(wrap_pyfunction!(py_derive_seed, m)?)?;
    m.add("OP_INIT", OP_INIT)?;
    m.add("OP_CROSSOVER", OP_CROSSOVER)?;
    m.add("OP_MUTATION", OP_MUTATION)?;
    m.add("OP_SELECTION", OP_SELECTION)?;
    m.add("OP_CMAES_ASK", OP_CMAES_ASK)?;
    m.add("OP_MULTI_RUN", OP_MULTI_RUN)?;
    m.add("OP_CROSSOVER_PROB", OP_CROSSOVER_PROB)?;

    m.add_function(wrap_pyfunction!(blend_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_mutation, m)?)?;

    m.add_function(wrap_pyfunction!(int_simulated_binary_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(int_gaussian_mutation, m)?)?;
    m.add_function(wrap_pyfunction!(int_uniform_mutation, m)?)?;

    m.add_function(wrap_pyfunction!(one_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(two_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(bit_flip_mutation, m)?)?;

    m.add_function(wrap_pyfunction!(tournament_selection, m)?)?;
    m.add_function(wrap_pyfunction!(roulette_selection, m)?)?;
    m.add_function(wrap_pyfunction!(rank_selection, m)?)?;

    m.add_function(wrap_pyfunction!(init_population, m)?)?;
    m.add_function(wrap_pyfunction!(reproduce_population, m)?)?;

    m.add_function(wrap_pyfunction!(evaluate_sequential, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_parallel_rayon, m)?)?;

    Ok(())
}
