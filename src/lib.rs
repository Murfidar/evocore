use pyo3::prelude::*;

mod cmaes;
mod gene_spec;
mod individual;
pub mod operators;
mod parallel;
mod reproduce;
mod selection;
pub mod utils;

use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use operators::{binary_ops, float_ops, int_ops};
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
    float_ops::uniform_mutation(&genes, low, high, prob, master_seed, generation, individual_idx)
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
    int_ops::int_uniform_mutation(&genes, low, high, prob, master_seed, generation, individual_idx)
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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .ok();

    m.add_class::<FloatIndividual>()?;
    m.add_class::<IntegerIndividual>()?;
    m.add_class::<BinaryIndividual>()?;

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

    Ok(())
}
