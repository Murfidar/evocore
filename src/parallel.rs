use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

pub fn evaluate_batch_sequential<F>(population: &[Vec<f64>], fitness_fn: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    population.iter().map(|genes| fitness_fn(genes)).collect()
}

pub fn evaluate_batch_parallel<F>(
    population: &[Vec<f64>],
    fitness_fn: F,
    n_threads: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    assert!(n_threads > 0, "n_threads must be positive");
    if n_threads == 1 {
        return population.iter().map(|genes| fitness_fn(genes)).collect();
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("failed to build rayon thread pool");
    pool.install(|| {
        population
            .par_iter()
            .map(|genes| fitness_fn(genes))
            .collect()
    })
}

#[pyfunction]
pub fn evaluate_sequential(
    py: Python<'_>,
    genes_list: Vec<Vec<f64>>,
    fitness_fn: Py<PyAny>,
) -> PyResult<Vec<f64>> {
    genes_list
        .iter()
        .map(|genes| {
            let result = fitness_fn.call1(py, (genes.clone(),))?;
            result.extract::<f64>(py)
        })
        .collect()
}

#[pyfunction]
pub fn evaluate_parallel_rayon(
    py: Python<'_>,
    genes_list: Vec<Vec<f64>>,
    fitness_fn: Py<PyAny>,
    n_threads: usize,
) -> PyResult<Vec<f64>> {
    if n_threads == 0 {
        return Err(PyValueError::new_err("n_threads must be positive."));
    }
    py.detach(|| {
        if n_threads == 1 {
            return genes_list
                .iter()
                .map(|genes| {
                    Python::attach(|py| {
                        let result = fitness_fn.call1(py, (genes.clone(),))?;
                        result.extract::<f64>(py)
                    })
                })
                .collect();
        }

        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        pool.install(|| {
            genes_list
                .par_iter()
                .map(|genes| {
                    Python::attach(|py| {
                        let result = fitness_fn.call1(py, (genes.clone(),))?;
                        result.extract::<f64>(py)
                    })
                })
                .collect()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    fn double_sum(genes: &[f64]) -> f64 {
        genes.iter().sum::<f64>() * 2.0
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_length() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_evaluate_batch_sequential_correct_values() {
        let pop = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = evaluate_batch_sequential(&pop, double_sum);
        assert!(
            (results[0] - 6.0).abs() < 1e-10,
            "expected 6.0, got {}",
            results[0]
        );
        assert!(
            (results[1] - 14.0).abs() < 1e-10,
            "expected 14.0, got {}",
            results[1]
        );
    }

    #[test]
    fn test_evaluate_batch_parallel_same_as_sequential() {
        let pop: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64, i as f64 * 2.0]).collect();
        let seq = evaluate_batch_sequential(&pop, double_sum);
        let par = evaluate_batch_parallel(&pop, double_sum, 4);
        for (i, (s, p)) in seq.iter().zip(par.iter()).enumerate() {
            assert!(
                (s - p).abs() < 1e-10,
                "mismatch at index {}: sequential={}, parallel={}",
                i,
                s,
                p
            );
        }
    }

    #[test]
    fn test_evaluate_batch_parallel_respects_single_thread_limit() {
        let pop: Vec<Vec<f64>> = (0..64).map(|i| vec![i as f64]).collect();
        let thread_ids = Arc::new(Mutex::new(HashSet::new()));
        let seen = Arc::clone(&thread_ids);

        let results = evaluate_batch_parallel(
            &pop,
            move |genes| {
                seen.lock().unwrap().insert(thread::current().id());
                thread::sleep(Duration::from_millis(1));
                genes[0]
            },
            1,
        );

        assert_eq!(results.len(), pop.len());
        assert_eq!(thread_ids.lock().unwrap().len(), 1);
    }
}
