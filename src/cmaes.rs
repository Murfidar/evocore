use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::{FromPyObject, IntoPyObject};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;

use crate::selection::safe_fitness;
use crate::utils::{derive_seed, OP_CMAES_ASK};

const MIN_SIGMA: f64 = 1e-20;
const MIN_EIGENVALUE: f64 = 1e-20;
const CMAES_SNAPSHOT_SCHEMA_VERSION: usize = 1;
const CMAES_SNAPSHOT_OPTIMIZER_TYPE: &str = "cmaes";
const COV_SYMMETRY_TOL: f64 = 1e-10;
const COV_MIN_EIGEN_TOL: f64 = -1e-12;

pub fn mirror_fold(x: f64, low: f64, high: f64) -> f64 {
    let range = high - low;
    if range <= 0.0 {
        return low;
    }

    let period = 2.0 * range;
    let mut t = (x - low) % period;
    if t < 0.0 {
        t += period;
    }
    if t > range {
        t = period - t;
    }
    t + low
}

#[derive(Debug)]
struct EigenCache {
    eigenvectors: DMatrix<f64>,
    eigenvalues_sqrt: DVector<f64>,
    valid: bool,
}

impl EigenCache {
    fn invalid(n: usize) -> Self {
        Self {
            eigenvectors: DMatrix::identity(n, n),
            eigenvalues_sqrt: DVector::from_element(n, 1.0),
            valid: false,
        }
    }
}

#[derive(Debug, Clone, IntoPyObject, FromPyObject)]
struct CMAESStateSnapshotEnvelope {
    #[pyo3(item)]
    schema_version: usize,
    #[pyo3(item)]
    optimizer_type: String,
    #[pyo3(item)]
    state: CMAESStateSnapshotPayload,
}

#[derive(Debug, Clone, IntoPyObject, FromPyObject)]
struct CMAESStateSnapshotPayload {
    #[pyo3(item)]
    n: usize,
    #[pyo3(item("lambda"))]
    lambda_: usize,
    #[pyo3(item)]
    generation: usize,
    #[pyo3(item)]
    mean: Vec<f64>,
    #[pyo3(item)]
    sigma: f64,
    #[pyo3(item)]
    cov: Vec<Vec<f64>>,
    #[pyo3(item)]
    pc: Vec<f64>,
    #[pyo3(item)]
    ps: Vec<f64>,
    #[pyo3(item)]
    bounds: Vec<Vec<f64>>,
    #[pyo3(item)]
    eigendecomp_interval: usize,
    #[pyo3(item)]
    pending_eigen_updates: usize,
    #[pyo3(item)]
    eigen_cache: CMAESEigenCacheSnapshot,
}

#[derive(Debug, Clone, IntoPyObject, FromPyObject)]
struct CMAESEigenCacheSnapshot {
    #[pyo3(item)]
    valid: bool,
    #[pyo3(item)]
    eigenvectors: Vec<Vec<f64>>,
    #[pyo3(item)]
    eigenvalues_sqrt: Vec<f64>,
}

fn vector_to_vec(vector: &DVector<f64>) -> Vec<f64> {
    vector.iter().copied().collect()
}

fn matrix_to_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn bounds_to_rows(bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    bounds.iter().map(|(low, high)| vec![*low, *high]).collect()
}

fn ensure_finite_vector(name: &str, values: &[f64], expected_len: usize) -> Result<(), String> {
    if values.len() != expected_len {
        return Err(format!(
            "{name} length must be {expected_len}, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} values must be finite"));
    }
    Ok(())
}

fn matrix_from_rows(name: &str, rows: Vec<Vec<f64>>, n: usize) -> Result<DMatrix<f64>, String> {
    if rows.len() != n {
        return Err(format!("{name} must have {n} rows, got {}", rows.len()));
    }
    let mut flat = Vec::with_capacity(n * n);
    for (row_idx, row) in rows.into_iter().enumerate() {
        if row.len() != n {
            return Err(format!("{name} row {row_idx} must have {n} columns"));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(format!("{name} values must be finite"));
        }
        flat.extend(row);
    }
    Ok(DMatrix::from_row_slice(n, n, &flat))
}

fn bounds_from_rows(rows: Vec<Vec<f64>>, n: usize) -> Result<Vec<(f64, f64)>, String> {
    if rows.len() != n {
        return Err(format!("bounds length must be {n}, got {}", rows.len()));
    }
    rows.into_iter()
        .enumerate()
        .map(|(idx, row)| {
            if row.len() != 2 {
                return Err(format!("bounds row {idx} must contain low and high"));
            }
            let low = row[0];
            let high = row[1];
            if !low.is_finite() || !high.is_finite() || low >= high {
                return Err(format!("bound {idx} must be finite with low < high"));
            }
            Ok((low, high))
        })
        .collect()
}

fn validate_covariance(cov: &DMatrix<f64>) -> Result<(), String> {
    for row in 0..cov.nrows() {
        for col in 0..cov.ncols() {
            if (cov[(row, col)] - cov[(col, row)]).abs() > COV_SYMMETRY_TOL {
                return Err("cov must be symmetric".to_string());
            }
        }
    }
    let eigen = cov.clone().symmetric_eigen();
    if eigen
        .eigenvalues
        .iter()
        .any(|value| *value < COV_MIN_EIGEN_TOL)
    {
        return Err("cov must not have clearly negative eigenvalues".to_string());
    }
    Ok(())
}

#[derive(Debug)]
pub struct CMAESState {
    pub n: usize,
    pub lambda: usize,
    pub mu: usize,
    pub mean: DVector<f64>,
    pub sigma: f64,
    pub cov: DMatrix<f64>,
    pub pc: DVector<f64>,
    pub ps: DVector<f64>,
    pub generation: usize,
    pub eigendecomp_interval: usize,
    bounds: Vec<(f64, f64)>,
    weights: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,
    eigen_cache: RefCell<EigenCache>,
    pending_eigen_updates: Cell<usize>,
}

impl CMAESState {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(mean: Vec<f64>, sigma: f64, lambda: usize, bounds: Vec<(f64, f64)>) -> Self {
        Self::try_new(mean, sigma, lambda, bounds).expect("invalid CMAESState parameters")
    }

    pub fn try_new(
        mean: Vec<f64>,
        sigma: f64,
        lambda: usize,
        bounds: Vec<(f64, f64)>,
    ) -> Result<Self, String> {
        if mean.is_empty() {
            return Err("mean must not be empty".to_string());
        }
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err("sigma must be finite and > 0".to_string());
        }
        if lambda < 2 {
            return Err("lambda must be >= 2".to_string());
        }
        if bounds.len() != mean.len() {
            return Err("bounds length must match mean length".to_string());
        }
        if bounds
            .iter()
            .any(|(low, high)| !low.is_finite() || !high.is_finite() || low >= high)
        {
            return Err("each bound must be finite with low < high".to_string());
        }

        let n = mean.len();
        let mu = lambda / 2;
        let mut raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64) + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let weight_sum: f64 = raw_weights.iter().sum();
        for weight in &mut raw_weights {
            *weight /= weight_sum;
        }
        let mueff = 1.0
            / raw_weights
                .iter()
                .map(|weight| weight * weight)
                .sum::<f64>();
        let n_f = n as f64;
        let cc = (4.0 + (mueff / n_f)) / (n_f + 4.0 + (2.0 * mueff / n_f));
        let cs = (mueff + 2.0) / (n_f + mueff + 5.0);
        let c1 = 2.0 / (((n_f + 1.3).powi(2)) + mueff);
        let cmu_unclamped = (2.0 * (mueff - 2.0 + (1.0 / mueff))) / (((n_f + 2.0).powi(2)) + mueff);
        let cmu = cmu_unclamped.max(0.0).min(1.0 - c1);
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (n_f + 1.0)).sqrt().max(0.0)) + cs;
        let chi_n = n_f.sqrt() * (1.0 - (1.0 / (4.0 * n_f)) + (1.0 / (21.0 * n_f * n_f)));
        let eigendecomp_interval =
            ((1.0 / (10.0 * n_f * (c1 + cmu).max(MIN_EIGENVALUE))).floor() as usize).max(1);

        Ok(Self {
            n,
            lambda,
            mu,
            mean: DVector::from_vec(mean),
            sigma: sigma.max(MIN_SIGMA),
            cov: DMatrix::identity(n, n),
            pc: DVector::zeros(n),
            ps: DVector::zeros(n),
            generation: 0,
            eigendecomp_interval,
            bounds,
            weights: raw_weights,
            mueff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            eigen_cache: RefCell::new(EigenCache::invalid(n)),
            pending_eigen_updates: Cell::new(0),
        })
    }

    fn to_snapshot(&self) -> CMAESStateSnapshotEnvelope {
        let cache = self.eigen_cache.borrow();
        CMAESStateSnapshotEnvelope {
            schema_version: CMAES_SNAPSHOT_SCHEMA_VERSION,
            optimizer_type: CMAES_SNAPSHOT_OPTIMIZER_TYPE.to_string(),
            state: CMAESStateSnapshotPayload {
                n: self.n,
                lambda_: self.lambda,
                generation: self.generation,
                mean: vector_to_vec(&self.mean),
                sigma: self.sigma,
                cov: matrix_to_rows(&self.cov),
                pc: vector_to_vec(&self.pc),
                ps: vector_to_vec(&self.ps),
                bounds: bounds_to_rows(&self.bounds),
                eigendecomp_interval: self.eigendecomp_interval,
                pending_eigen_updates: self.pending_eigen_updates.get(),
                eigen_cache: CMAESEigenCacheSnapshot {
                    valid: cache.valid,
                    eigenvectors: matrix_to_rows(&cache.eigenvectors),
                    eigenvalues_sqrt: vector_to_vec(&cache.eigenvalues_sqrt),
                },
            },
        }
    }

    fn try_from_snapshot(snapshot: CMAESStateSnapshotEnvelope) -> Result<Self, String> {
        if snapshot.schema_version != CMAES_SNAPSHOT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported CMA-ES state snapshot schema_version {}",
                snapshot.schema_version
            ));
        }
        if snapshot.optimizer_type != CMAES_SNAPSHOT_OPTIMIZER_TYPE {
            return Err(format!(
                "expected optimizer_type {CMAES_SNAPSHOT_OPTIMIZER_TYPE}, got {}",
                snapshot.optimizer_type
            ));
        }

        let payload = snapshot.state;
        if payload.n == 0 {
            return Err("n must be positive".to_string());
        }
        if payload.lambda_ < 2 {
            return Err("lambda must be >= 2".to_string());
        }
        if payload.eigendecomp_interval == 0 {
            return Err("eigendecomp_interval must be positive".to_string());
        }
        ensure_finite_vector("mean", &payload.mean, payload.n)?;
        ensure_finite_vector("pc", &payload.pc, payload.n)?;
        ensure_finite_vector("ps", &payload.ps, payload.n)?;
        if payload.sigma <= 0.0 || !payload.sigma.is_finite() {
            return Err("sigma must be finite and > 0".to_string());
        }

        let bounds = bounds_from_rows(payload.bounds, payload.n)?;
        let cov = matrix_from_rows("cov", payload.cov, payload.n)?;
        validate_covariance(&cov)?;

        let eigenvectors = matrix_from_rows(
            "eigen_cache.eigenvectors",
            payload.eigen_cache.eigenvectors,
            payload.n,
        )?;
        ensure_finite_vector(
            "eigen_cache.eigenvalues_sqrt",
            &payload.eigen_cache.eigenvalues_sqrt,
            payload.n,
        )?;
        if payload
            .eigen_cache
            .eigenvalues_sqrt
            .iter()
            .any(|value| *value <= 0.0)
        {
            return Err("eigen_cache.eigenvalues_sqrt values must be > 0".to_string());
        }

        let mut state = Self::try_new(payload.mean, payload.sigma, payload.lambda_, bounds)?;
        state.cov = cov;
        state.pc = DVector::from_vec(payload.pc);
        state.ps = DVector::from_vec(payload.ps);
        state.generation = payload.generation;
        state.eigendecomp_interval = payload.eigendecomp_interval;
        state
            .pending_eigen_updates
            .set(payload.pending_eigen_updates);
        state.eigen_cache.replace(if payload.eigen_cache.valid {
            EigenCache {
                eigenvectors,
                eigenvalues_sqrt: DVector::from_vec(payload.eigen_cache.eigenvalues_sqrt),
                valid: true,
            }
        } else {
            EigenCache::invalid(payload.n)
        });

        Ok(state)
    }

    fn ensure_eigen_cache(&self) {
        let should_refresh = {
            let cache = self.eigen_cache.borrow();
            !cache.valid || self.pending_eigen_updates.get() >= self.eigendecomp_interval
        };
        if !should_refresh {
            return;
        }

        let sym_cov = (&self.cov + self.cov.transpose()) * 0.5;
        let eigen = sym_cov.symmetric_eigen();
        let eigenvalues_sqrt = eigen
            .eigenvalues
            .map(|value| value.max(MIN_EIGENVALUE).sqrt());

        let mut cache = self.eigen_cache.borrow_mut();
        cache.eigenvectors = eigen.eigenvectors;
        cache.eigenvalues_sqrt = eigenvalues_sqrt;
        cache.valid = true;
        self.pending_eigen_updates.set(0);
    }

    fn invsqrt_covariance(&self) -> DMatrix<f64> {
        self.ensure_eigen_cache();
        let cache = self.eigen_cache.borrow();
        let inv_diag =
            DVector::from_iterator(self.n, cache.eigenvalues_sqrt.iter().map(|v| 1.0 / *v));
        let scaled = &cache.eigenvectors * DMatrix::from_diagonal(&inv_diag);
        scaled * cache.eigenvectors.transpose()
    }

    fn validate_samples(&self, samples: &[Vec<f64>], fitnesses: &[f64]) -> Result<(), String> {
        if samples.len() != fitnesses.len() {
            return Err("samples and fitnesses must have the same length".to_string());
        }
        if samples.len() != self.lambda {
            return Err(format!(
                "expected {} samples, got {}",
                self.lambda,
                samples.len()
            ));
        }
        if samples.iter().any(|sample| sample.len() != self.n) {
            return Err(format!("each sample must have {} genes", self.n));
        }
        Ok(())
    }

    pub fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        self.ensure_eigen_cache();
        let cache = self.eigen_cache.borrow();

        (0..self.lambda)
            .map(|sample_idx| {
                let seed = derive_seed(master_seed, generation, sample_idx as u64, OP_CMAES_ASK);
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let z =
                    DVector::from_iterator(self.n, (0..self.n).map(|_| normal.sample(&mut rng)));
                let transformed = &cache.eigenvectors * cache.eigenvalues_sqrt.component_mul(&z);

                (0..self.n)
                    .map(|idx| {
                        let raw = self.mean[idx] + self.sigma * transformed[idx];
                        let (low, high) = self.bounds[idx];
                        mirror_fold(raw, low, high)
                    })
                    .collect()
            })
            .collect()
    }

    pub fn tell(&mut self, samples: &[Vec<f64>], fitnesses: &[f64]) {
        self.validate_samples(samples, fitnesses)
            .expect("invalid samples or fitnesses for tell()");

        let mut ranked: Vec<usize> = (0..samples.len()).collect();
        ranked.sort_by(|&left, &right| {
            safe_fitness(fitnesses[right])
                .partial_cmp(&safe_fitness(fitnesses[left]))
                .unwrap_or(Ordering::Equal)
        });

        let old_mean = self.mean.clone();
        let old_cov = self.cov.clone();

        let mut new_mean = DVector::zeros(self.n);
        for (rank, &sample_idx) in ranked.iter().take(self.mu).enumerate() {
            let sample = DVector::from_column_slice(&samples[sample_idx]);
            new_mean += sample * self.weights[rank];
        }
        let mean_diff = (&new_mean - &old_mean) / self.sigma;

        let invsqrt_c = self.invsqrt_covariance();
        let ps_factor = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        self.ps = (&self.ps * (1.0 - self.cs)) + ((&invsqrt_c * &mean_diff) * ps_factor);

        let n_f = self.n as f64;
        let generation_scale = (1.0 - (1.0 - self.cs).powf(2.0 * (self.generation as f64 + 1.0)))
            .sqrt()
            .max(MIN_EIGENVALUE);
        let hsig_threshold = 1.4 + 2.0 / (n_f + 1.0);
        let hsig = if (self.ps.norm() / (self.chi_n * generation_scale)) < hsig_threshold {
            1.0
        } else {
            0.0
        };

        let pc_factor = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        self.pc = (&self.pc * (1.0 - self.cc)) + (&mean_diff * (hsig * pc_factor));

        let rank_one = (&self.pc * self.pc.transpose())
            + (old_cov.clone() * ((1.0 - hsig) * self.cc * (2.0 - self.cc)));

        let mut rank_mu_acc = DMatrix::zeros(self.n, self.n);
        for (rank, &sample_idx) in ranked.iter().take(self.mu).enumerate() {
            let sample = DVector::from_column_slice(&samples[sample_idx]);
            let artmp = (sample - &old_mean) / self.sigma;
            rank_mu_acc += (&artmp * artmp.transpose()) * self.weights[rank];
        }

        self.cov = (old_cov * (1.0 - self.c1 - self.cmu))
            + (rank_one * self.c1)
            + (rank_mu_acc * self.cmu);
        self.cov = (&self.cov + self.cov.transpose()) * 0.5;

        let sigma_scale = ((self.cs / self.damps) * ((self.ps.norm() / self.chi_n) - 1.0)).exp();
        self.sigma = (self.sigma * sigma_scale).max(MIN_SIGMA);
        self.mean = new_mean;
        self.generation += 1;
        self.pending_eigen_updates
            .set(self.pending_eigen_updates.get().saturating_add(1));
    }
}

#[pyclass(unsendable)]
pub struct PyCMAESState {
    inner: CMAESState,
}

#[pymethods]
impl PyCMAESState {
    #[new]
    #[pyo3(signature = (mean, sigma, lambda_, bounds))]
    fn new(mean: Vec<f64>, sigma: f64, lambda_: usize, bounds: Vec<(f64, f64)>) -> PyResult<Self> {
        let inner = CMAESState::try_new(mean, sigma, lambda_, bounds)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn generation(&self) -> usize {
        self.inner.generation
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean.iter().copied().collect()
    }

    #[getter]
    fn eigendecomp_interval(&self) -> usize {
        self.inner.eigendecomp_interval
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.inner.to_snapshot().into_pyobject(py)?.into_any())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, snapshot: &Bound<'_, PyAny>) -> PyResult<Self> {
        let snapshot = snapshot
            .extract::<CMAESStateSnapshotEnvelope>()
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid CMA-ES state snapshot: {err}"
                ))
            })?;
        let inner = CMAESState::try_from_snapshot(snapshot)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    fn ask(&self, master_seed: u64, generation: u64) -> Vec<Vec<f64>> {
        self.inner.ask(master_seed, generation)
    }

    fn tell(&mut self, samples: Vec<Vec<f64>>, fitnesses: Vec<f64>) -> PyResult<()> {
        self.inner
            .validate_samples(&samples, &fitnesses)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.inner.tell(&samples, &fitnesses);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCMAESState(n={}, lambda={}, sigma={}, generation={})",
            self.inner.n, self.inner.lambda, self.inner.sigma, self.inner.generation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_fold_in_bounds_unchanged() {
        assert!((mirror_fold(0.5, 0.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((mirror_fold(-2.0, -5.0, 5.0) - (-2.0)).abs() < 1e-12);
        assert!((mirror_fold(0.0, -1.0, 1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_at_boundaries_unchanged() {
        assert!((mirror_fold(0.0, 0.0, 1.0) - 0.0).abs() < 1e-12);
        assert!((mirror_fold(1.0, 0.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_fold_slightly_above_high_reflects() {
        let result = mirror_fold(1.1, 0.0, 1.0);
        assert!((result - 0.9).abs() < 1e-12, "expected 0.9, got {}", result);
    }

    #[test]
    fn test_mirror_fold_slightly_below_low_reflects() {
        let result = mirror_fold(-0.1, 0.0, 1.0);
        assert!((result - 0.1).abs() < 1e-12, "expected 0.1, got {}", result);
    }

    #[test]
    fn test_mirror_fold_result_always_in_bounds() {
        let low = -3.0_f64;
        let high = 3.0_f64;
        for i in -50..=50 {
            let x = i as f64 * 0.7;
            let result = mirror_fold(x, low, high);
            assert!(
                result >= low - 1e-10 && result <= high + 1e-10,
                "mirror_fold({}, {}, {}) = {} is outside bounds",
                x,
                low,
                high,
                result
            );
        }
    }

    #[test]
    fn test_mirror_fold_far_outside_still_in_bounds() {
        let result = mirror_fold(25.0, 0.0, 1.0);
        assert!((0.0..=1.0).contains(&result), "got {}", result);
    }

    fn make_state(n: usize, lambda: usize) -> CMAESState {
        let mean = vec![0.0_f64; n];
        let bounds = vec![(-5.0_f64, 5.0); n];
        CMAESState::new(mean, 0.5, lambda, bounds)
    }

    #[test]
    fn test_new_sets_correct_n() {
        let s = make_state(5, 10);
        assert_eq!(s.n, 5);
    }

    #[test]
    fn test_new_sets_correct_lambda() {
        let s = make_state(5, 10);
        assert_eq!(s.lambda, 10);
    }

    #[test]
    fn test_new_mu_is_half_lambda() {
        let s = make_state(5, 10);
        assert_eq!(s.mu, 5);
    }

    #[test]
    fn test_new_sigma_preserved() {
        let mean = vec![0.0_f64; 3];
        let bounds = vec![(-5.0_f64, 5.0); 3];
        let s = CMAESState::new(mean, 0.3, 6, bounds);
        assert!((s.sigma - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_new_covariance_is_identity() {
        let s = make_state(4, 8);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s.cov[(i, j)] - expected).abs() < 1e-12,
                    "cov[{},{}] = {}, expected {}",
                    i,
                    j,
                    s.cov[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_new_evolution_paths_are_zero() {
        let s = make_state(5, 10);
        assert!(s.pc.iter().all(|x| x.abs() < 1e-12), "pc must be zero");
        assert!(s.ps.iter().all(|x| x.abs() < 1e-12), "ps must be zero");
    }

    #[test]
    fn test_new_generation_is_zero() {
        let s = make_state(5, 10);
        assert_eq!(s.generation, 0);
    }

    #[test]
    fn test_new_eigendecomp_interval_at_least_one() {
        for n in [3, 5, 10, 20, 50, 100] {
            let lambda = 4 + (3.0 * (n as f64).ln()) as usize;
            let s = make_state(n, lambda);
            assert!(
                s.eigendecomp_interval >= 1,
                "eigendecomp_interval must be >= 1 for n={}",
                n
            );
        }
    }

    #[test]
    fn test_new_eigendecomp_interval_increases_with_n() {
        let s_small = make_state(5, 4 + (3.0 * 5.0_f64.ln()) as usize);
        let s_large = make_state(200, 4 + (3.0 * 200.0_f64.ln()) as usize);
        assert!(
            s_large.eigendecomp_interval >= s_small.eigendecomp_interval,
            "larger n should have >= eigendecomp_interval; small={}, large={}",
            s_small.eigendecomp_interval,
            s_large.eigendecomp_interval
        );
    }

    #[test]
    fn test_ask_returns_correct_lambda_samples() {
        let s = make_state(5, 12);
        let samples = s.ask(42, 0);
        assert_eq!(samples.len(), 12);
    }

    #[test]
    fn test_ask_returns_correct_n_genes_per_sample() {
        let s = make_state(7, 10);
        let samples = s.ask(42, 0);
        assert!(samples.iter().all(|samp| samp.len() == 7));
    }

    #[test]
    fn test_ask_deterministic_same_inputs() {
        let s = make_state(5, 10);
        let a = s.ask(42, 3);
        let b = s.ask(42, 3);
        assert_eq!(
            a, b,
            "ask() must be deterministic for the same (master_seed, generation)"
        );
    }

    #[test]
    fn test_ask_different_generation_diverges() {
        let s = make_state(5, 10);
        let a = s.ask(42, 0);
        let b = s.ask(42, 1);
        assert_ne!(a, b, "different generation must produce different samples");
    }

    #[test]
    fn test_ask_different_master_seed_diverges() {
        let s = make_state(5, 10);
        let a = s.ask(1, 0);
        let b = s.ask(2, 0);
        assert_ne!(a, b, "different master_seed must produce different samples");
    }

    #[test]
    fn test_ask_samples_within_bounds_after_mirror_folding() {
        let bounds = vec![
            (-2.0_f64, 2.0),
            (-1.0, 1.0),
            (0.0, 5.0),
            (-10.0, 0.0),
            (3.0, 7.0),
        ];
        let mean = bounds.iter().map(|(lo, hi)| (lo + hi) / 2.0).collect();
        let s = CMAESState::new(mean, 5.0, 20, bounds.clone());
        let samples = s.ask(42, 0);
        for (i, sample) in samples.iter().enumerate() {
            for (j, &g) in sample.iter().enumerate() {
                let (lo, hi) = bounds[j];
                assert!(
                    g >= lo - 1e-10 && g <= hi + 1e-10,
                    "sample[{}][{}]={} outside [{}, {}]",
                    i,
                    j,
                    g,
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn test_tell_increments_generation() {
        let mut s = make_state(3, 6);
        let samples = s.ask(42, 0);
        let fitnesses: Vec<f64> = samples
            .iter()
            .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
            .collect();
        s.tell(&samples, &fitnesses);
        assert_eq!(s.generation, 1, "tell() must increment generation");
    }

    #[test]
    fn test_tell_sigma_stays_positive_over_many_generations() {
        let mut s = make_state(4, 8);
        for gen in 0..20 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples
                .iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            assert!(
                s.sigma > 0.0,
                "sigma must stay positive after {} generations",
                gen + 1
            );
        }
    }

    #[test]
    fn test_tell_generation_matches_number_of_tell_calls() {
        let mut s = make_state(3, 6);
        for gen in 0..5 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples
                .iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        assert_eq!(s.generation, 5);
    }

    #[test]
    fn test_tell_mean_moves_toward_optimum() {
        let n = 5;
        let mean_start = vec![3.0_f64; n];
        let bounds = vec![(-10.0_f64, 10.0); n];
        let initial_norm: f64 = mean_start.iter().map(|x| x * x).sum::<f64>().sqrt();

        let mut s = CMAESState::new(mean_start, 1.0, 20, bounds.clone());
        for gen in 0..30 {
            let samples = s.ask(42, gen as u64);
            let fitnesses: Vec<f64> = samples
                .iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        let final_norm: f64 = s.mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            final_norm < initial_norm,
            "mean norm should decrease toward optimum: initial={:.4}, final={:.4}",
            initial_norm,
            final_norm
        );
    }

    #[test]
    fn test_tell_covariance_stays_finite() {
        let mut s = make_state(4, 8);
        for gen in 0..10 {
            let samples = s.ask(99, gen as u64);
            let fitnesses: Vec<f64> = samples
                .iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
            for i in 0..4 {
                for j in 0..4 {
                    assert!(
                        s.cov[(i, j)].is_finite(),
                        "cov[{},{}] is not finite after {} generations",
                        i,
                        j,
                        gen + 1
                    );
                }
            }
        }
    }

    fn evolved_state_with_lazy_eigen_cache() -> CMAESState {
        let mut s = make_state(4, 8);
        for gen in 0..3 {
            let samples = s.ask(99, gen as u64);
            let fitnesses: Vec<f64> = samples
                .iter()
                .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
                .collect();
            s.tell(&samples, &fitnesses);
        }
        s
    }

    #[test]
    fn test_snapshot_round_trip_preserves_next_ask() {
        let s = evolved_state_with_lazy_eigen_cache();
        let restored = CMAESState::try_from_snapshot(s.to_snapshot()).unwrap();

        assert_eq!(restored.generation, s.generation);
        assert_eq!(
            restored.ask(123, restored.generation as u64),
            s.ask(123, s.generation as u64)
        );
    }

    #[test]
    fn test_snapshot_round_trip_after_same_tell_matches_snapshot() {
        let mut s = evolved_state_with_lazy_eigen_cache();
        let mut restored = CMAESState::try_from_snapshot(s.to_snapshot()).unwrap();
        let samples = s.ask(123, s.generation as u64);
        let fitnesses: Vec<f64> = samples
            .iter()
            .map(|samp| -samp.iter().map(|x| x * x).sum::<f64>())
            .collect();

        restored.tell(&samples, &fitnesses);
        s.tell(&samples, &fitnesses);

        assert_eq!(
            restored.ask(456, restored.generation as u64),
            s.ask(456, s.generation as u64)
        );
        assert_eq!(
            restored.to_snapshot().state.pending_eigen_updates,
            s.to_snapshot().state.pending_eigen_updates
        );
    }

    #[test]
    fn test_snapshot_rejects_invalid_schema_version() {
        let mut snapshot = make_state(3, 6).to_snapshot();
        snapshot.schema_version = 2;

        let err = CMAESState::try_from_snapshot(snapshot).unwrap_err();
        assert!(err.contains("schema_version"));
    }

    #[test]
    fn test_snapshot_rejects_wrong_optimizer_type() {
        let mut snapshot = make_state(3, 6).to_snapshot();
        snapshot.optimizer_type = "ga".to_string();

        let err = CMAESState::try_from_snapshot(snapshot).unwrap_err();
        assert!(err.contains("optimizer_type"));
    }

    #[test]
    fn test_snapshot_rejects_nonsymmetric_covariance() {
        let mut snapshot = make_state(3, 6).to_snapshot();
        snapshot.state.cov[0][1] = 0.25;

        let err = CMAESState::try_from_snapshot(snapshot).unwrap_err();
        assert!(err.contains("cov"));
    }

    #[test]
    fn test_snapshot_rejects_negative_covariance_eigenvalue() {
        let mut snapshot = make_state(3, 6).to_snapshot();
        snapshot.state.cov[0][0] = -1.0;

        let err = CMAESState::try_from_snapshot(snapshot).unwrap_err();
        assert!(err.contains("cov"));
    }
}
