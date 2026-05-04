use pyo3::prelude::*;

mod cmaes;
pub mod gene_spec;
mod individual;
mod operators;
mod parallel;
mod reproduce;
mod selection;
pub mod utils;

use individual::{BinaryIndividual, FloatIndividual, IntegerIndividual};
use utils::{
    py_derive_seed, OP_CMAES_ASK, OP_CROSSOVER, OP_CROSSOVER_PROB, OP_INIT, OP_MULTI_RUN,
    OP_MUTATION, OP_SELECTION,
};

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

    Ok(())
}
