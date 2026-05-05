use pyo3::prelude::*;
use rayon::prelude::*;

pub fn evaluate_batch_sequential<F>(population: &[Vec<f64>], fitness_fn: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    population.iter().map(|genes| fitness_fn(genes)).collect()
}

pub fn evaluate_batch_parallel<F>(
    population: &[Vec<f64>],
    fitness_fn: F,
    _n_threads: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    population
        .par_iter()
        .map(|genes| fitness_fn(genes))
        .collect()
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
    let _ = n_threads;
    py.detach(|| {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
