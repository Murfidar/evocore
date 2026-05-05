use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;

use crate::selection::safe_fitness;
use crate::utils::{derive_seed, OP_CMAES_ASK};

const MIN_SIGMA: f64 = 1e-20;
const MIN_EIGENVALUE: f64 = 1e-20;

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
}
