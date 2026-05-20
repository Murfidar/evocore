# CMA-ES Rust State Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable `PyCMAESState.to_dict()` and `PyCMAESState.from_dict(...)` so Rust-backed CMA-ES state can be exported, validated, restored, and continued deterministically.

**Architecture:** Keep the snapshot primitive inside `src/cmaes.rs`, because the Rust state owns CMA-ES adaptation, covariance, evolution paths, and lazy eigendecomposition state. Expose only JSON-safe dictionaries through PyO3, update the Python stub, and leave `CMAESOptimizer` checkpoint/resume unsupported until a later optimizer-ledger integration.

**Tech Stack:** Rust, PyO3 0.28, nalgebra, maturin, Python typing stubs, pytest, MkDocs, repo-local `.venv`.

---

## Design Refinement

The approved design said the eigen cache should not be checkpointed because it is derived from covariance. While mapping the implementation, the current Rust code shows a sharper contract:

- `ask(...)` samples from `eigen_cache`.
- `tell(...)` updates `cov` and increments `pending_eigen_updates`.
- `ensure_eigen_cache()` refreshes only when the cache is invalid or `pending_eigen_updates >= eigendecomp_interval`.

That means a state can have a valid, intentionally stale eigendecomposition. If restore discards that cache and rebuilds from the newest covariance, the next `ask(...)` can differ from the uninterrupted state.

This plan therefore treats the lazy eigendecomposition data as checkpointable CMA-ES algorithm state. The public schema should call it `eigen_cache` or `eigen_state`, validate it strictly, and version it under schema V1. This is the smallest way to keep `to_dict()` side-effect-free and make restored continuation exact.

---

## File Structure

- Modify: `src/cmaes.rs`
  - Add schema constants, snapshot payload structs, validation helpers, Rust round-trip helpers, and PyO3 `to_dict/from_dict`.
  - Keep the implementation local to the Rust state; do not add serde or new dependencies.
- Modify: `evocore/_core.pyi`
  - Add the `PyCMAESState.to_dict()` and `PyCMAESState.from_dict(...)` signatures.
- Modify: `tests/unit/test_cmaes_rust.py`
  - Add Python-facing tests for JSON safety, restore determinism, malformed snapshots, and schema shape.
- Modify: `docs/site/cmaes.md`
  - Document Rust state snapshots while keeping full `CMAESOptimizer` checkpoint/resume as future work.
- Modify: `docs/site/callbacks-checkpointing.md`
  - Update the unsupported CMA-ES checkpoint wording to mention the new primitive.
- Modify: `CHANGELOG.md`
  - Add a user-visible `Unreleased` entry for CMA-ES Rust state snapshots.

---

### Task 1: Python API Tests

**Files:**
- Modify: `tests/unit/test_cmaes_rust.py`

- [ ] **Step 1: Add failing tests for JSON-safe snapshot shape**

Append this test class after `TestIntegerGeneWorkflow`:

```python
class TestStateSnapshots:
    def test_to_dict_returns_json_safe_schema_v1_payload(self):
        s = make_state(n=3, lambda_=6, sigma=0.4)
        samples = s.ask(42, 0)
        s.tell(samples, [neg_sphere(sample) for sample in samples])

        payload = s.to_dict()

        assert payload["schema_version"] == 1
        assert payload["optimizer_type"] == "cmaes"
        assert set(payload["state"]) == {
            "n",
            "lambda",
            "generation",
            "mean",
            "sigma",
            "cov",
            "pc",
            "ps",
            "bounds",
            "eigendecomp_interval",
            "pending_eigen_updates",
            "eigen_cache",
        }
        assert payload["state"]["n"] == 3
        assert payload["state"]["lambda"] == 6
        assert payload["state"]["generation"] == 1
        assert len(payload["state"]["mean"]) == 3
        assert len(payload["state"]["cov"]) == 3
        assert len(payload["state"]["cov"][0]) == 3
        assert payload["state"]["eigen_cache"]["valid"] in {True, False}
        assert len(payload["state"]["eigen_cache"]["eigenvectors"]) == 3
        assert len(payload["state"]["eigen_cache"]["eigenvalues_sqrt"]) == 3

        import json

        json.dumps(payload, sort_keys=True)
```

- [ ] **Step 2: Add failing tests for deterministic restore**

Add these methods to `TestStateSnapshots`:

```python
    def test_from_dict_restores_next_ask_after_stale_eigen_cache(self):
        s = make_state(n=4, lambda_=8, sigma=0.5)
        for generation in range(3):
            samples = s.ask(99, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])

        payload = s.to_dict()
        restored = PyCMAESState.from_dict(payload)

        assert restored.generation == s.generation
        assert restored.sigma == pytest.approx(s.sigma)
        assert restored.mean == pytest.approx(s.mean)
        assert restored.ask(123, restored.generation) == s.ask(123, s.generation)

    def test_restored_state_matches_uninterrupted_state_after_same_tell(self):
        s = make_state(n=4, lambda_=8, sigma=0.5)
        for generation in range(3):
            samples = s.ask(99, generation)
            s.tell(samples, [neg_sphere(sample) for sample in samples])

        restored = PyCMAESState.from_dict(s.to_dict())
        samples = s.ask(123, s.generation)
        assert samples == restored.ask(123, restored.generation)
        fitnesses = [neg_sphere(sample) for sample in samples]

        s.tell(samples, fitnesses)
        restored.tell(samples, fitnesses)

        assert restored.to_dict() == s.to_dict()
```

- [ ] **Step 3: Add failing tests for malformed snapshots**

Add these methods to `TestStateSnapshots`:

```python
    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda payload: payload.update({"schema_version": 999}), "schema_version"),
            (lambda payload: payload.update({"optimizer_type": "ga"}), "optimizer_type"),
            (lambda payload: payload["state"].update({"lambda": 1}), "lambda"),
            (lambda payload: payload["state"].update({"sigma": 0.0}), "sigma"),
            (lambda payload: payload["state"].update({"generation": -1}), "invalid|generation"),
            (lambda payload: payload["state"].update({"mean": [0.0]}), "mean"),
            (lambda payload: payload["state"].update({"cov": [[1.0, 0.0]]}), "cov"),
            (lambda payload: payload["state"].update({"bounds": [[1.0, -1.0]] * 3}), "bound"),
            (
                lambda payload: payload["state"]["eigen_cache"].update(
                    {"eigenvalues_sqrt": [1.0]}
                ),
                "eigen_cache",
            ),
        ],
    )
    def test_from_dict_rejects_malformed_snapshots(self, mutate, match):
        payload = make_state(n=3, lambda_=6).to_dict()
        mutate(payload)

        with pytest.raises(ValueError, match=match):
            PyCMAESState.from_dict(payload)

    def test_from_dict_rejects_nonsymmetric_covariance(self):
        payload = make_state(n=3, lambda_=6).to_dict()
        payload["state"]["cov"][0][1] = 0.25

        with pytest.raises(ValueError, match="cov"):
            PyCMAESState.from_dict(payload)

    def test_from_dict_rejects_negative_covariance_eigenvalue(self):
        payload = make_state(n=3, lambda_=6).to_dict()
        payload["state"]["cov"][0][0] = -1.0

        with pytest.raises(ValueError, match="cov"):
            PyCMAESState.from_dict(payload)
```

- [ ] **Step 4: Run the Python tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_rust.py::TestStateSnapshots -v
```

Expected: FAIL because `PyCMAESState` has no `to_dict` or `from_dict` yet.

- [ ] **Step 5: Commit the failing tests**

```powershell
git add tests/unit/test_cmaes_rust.py
git commit -m "test: cover cmaes rust state snapshots"
```

---

### Task 2: Rust Snapshot Payload And Validation

**Files:**
- Modify: `src/cmaes.rs`

- [ ] **Step 1: Add Rust snapshot types and constants**

Near the top of `src/cmaes.rs`, extend the imports and constants:

```rust
use pyo3::types::PyType;
use pyo3::{FromPyObject, IntoPyObject, IntoPyObjectExt};
```

Add these constants after `MIN_EIGENVALUE`:

```rust
const CMAES_SNAPSHOT_SCHEMA_VERSION: usize = 1;
const CMAES_SNAPSHOT_OPTIMIZER_TYPE: &str = "cmaes";
const COV_SYMMETRY_TOL: f64 = 1e-10;
const COV_MIN_EIGEN_TOL: f64 = -1e-12;
```

Add these payload structs before `pub struct CMAESState`:

```rust
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
```

- [ ] **Step 2: Add conversion helpers**

Add these helpers before `impl CMAESState`:

```rust
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
```

- [ ] **Step 3: Add `CMAESState` snapshot methods**

Inside `impl CMAESState`, before `fn ensure_eigen_cache(&self)`, add:

```rust
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
        if payload.generation > (u64::MAX as usize) {
            return Err("generation is too large".to_string());
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

        let mut state = Self::try_new(
            payload.mean,
            payload.sigma,
            payload.lambda_,
            bounds,
        )?;
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
```

- [ ] **Step 4: Add Rust unit tests for snapshot round trip**

Inside `#[cfg(test)] mod tests`, add:

```rust
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
```

- [ ] **Step 5: Add Rust unit tests for validation failures**

Add:

```rust
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
```

- [ ] **Step 6: Run Rust tests and verify the new Rust helpers pass**

Run:

```powershell
cargo test cmaes
```

Expected: PASS for the CMA-ES Rust tests. Python tests still fail because the PyO3 methods and stubs are not exposed yet.

---

### Task 3: PyO3 Methods And Python Stub

**Files:**
- Modify: `src/cmaes.rs`
- Modify: `evocore/_core.pyi`

- [ ] **Step 1: Add `to_dict` and `from_dict` to `PyCMAESState`**

Inside `#[pymethods] impl PyCMAESState`, after `eigendecomp_interval`, add:

```rust
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
```

- [ ] **Step 2: Update the stub**

In `evocore/_core.pyi`, update `PyCMAESState`:

```python
class PyCMAESState:
    generation: int
    sigma: float
    mean: list[float]
    eigendecomp_interval: int

    def __init__(
        self,
        mean: Sequence[float],
        sigma: float,
        lambda_: int,
        bounds: Sequence[tuple[float, float]],
    ) -> None: ...
    def ask(self, master_seed: int, generation: int) -> list[list[float]]: ...
    def tell(self, samples: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None: ...
    def to_dict(self) -> dict[str, object]: ...
    @classmethod
    def from_dict(cls, snapshot: dict[str, object]) -> PyCMAESState: ...
```

- [ ] **Step 3: Format and compile the extension**

Run:

```powershell
cargo fmt --check
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: both commands PASS. If `cargo fmt --check` fails only due formatting, run `cargo fmt`, inspect the diff, then rerun `cargo fmt --check`.

- [ ] **Step 4: Run the targeted Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_rust.py::TestStateSnapshots -v
```

Expected: PASS for all `TestStateSnapshots` tests.

- [ ] **Step 5: Commit implementation and stub**

```powershell
git add src/cmaes.rs evocore/_core.pyi
git commit -m "feat: add cmaes rust state snapshots"
```

---

### Task 4: Documentation And Changelog

**Files:**
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update the CMA-ES docs**

In `docs/site/cmaes.md`, replace the current `## Checkpoint Resume` section with:

````markdown
## Rust State Snapshots

`PyCMAESState` exposes a Rust-backed state snapshot primitive:

```python
from evocore._core import PyCMAESState

state = PyCMAESState([0.0, 0.0], 0.5, 6, [(-5.0, 5.0), (-5.0, 5.0)])
samples = state.ask(42, state.generation)
state.tell(samples, [-sum(value * value for value in sample) for sample in samples])

snapshot = state.to_dict()
restored = PyCMAESState.from_dict(snapshot)

assert restored.ask(42, restored.generation) == state.ask(42, state.generation)
```

The snapshot is a schema-versioned optimizer-state payload. It preserves the
CMA-ES adaptation state needed for deterministic continuation, including mean,
sigma, covariance, evolution paths, generation, bounds, and lazy
eigendecomposition state.

`CMAESOptimizer` checkpoint/resume is still unsupported in checkpoint v1. The
Rust state primitive is the foundation for that later optimizer-level work,
which also needs Python candidate ledgers, pending batches, telemetry, and event
indexes.

Use `OptimizationResult.to_dict()` for completed-run export and `engine.events`
for ask/tell audit rows. Those exports are not checkpoint files and are not
replayed to rebuild CMA-ES state.
````

- [ ] **Step 2: Update callbacks/checkpointing docs**

In `docs/site/callbacks-checkpointing.md`, replace the CMA-ES paragraph under `Unsupported Checkpoint Surfaces` with:

```markdown
`CMAESOptimizer` checkpoint/resume is unsupported in checkpoint v1. The
Rust-backed `PyCMAESState` now has a stable state snapshot primitive, but full
optimizer resume still needs the Python optimizer ledger and pending-batch state
to be wired into the checkpoint envelope. CMA-ES result export and event audit
history remain available.
```

- [ ] **Step 3: Update the changelog**

In `CHANGELOG.md`, under `[Unreleased]` / `### Added`, add:

```markdown
- Rust-backed `PyCMAESState.to_dict()` and `PyCMAESState.from_dict(...)`
  snapshots for deterministic CMA-ES state continuation primitives.
```

- [ ] **Step 4: Build the docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS. Existing known MkDocs warnings about Material/blog or an unlisted generated parity doc are acceptable only if they already existed before this change.

- [ ] **Step 5: Remove generated site output if present**

Run:

```powershell
if (Test-Path site) { Remove-Item -Recurse -Force -LiteralPath site }
git status --short
```

Expected: `site/` is not present in the working tree and only intended source files remain modified.

- [ ] **Step 6: Commit docs**

```powershell
git add docs/site/cmaes.md docs/site/callbacks-checkpointing.md CHANGELOG.md
git commit -m "docs: document cmaes rust snapshots"
```

---

### Task 5: Full Verification

**Files:**
- Verify: Rust extension, Python tests, docs, formatting, linting

- [ ] **Step 1: Run Rust formatting and linting**

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 2: Run Rust tests**

```powershell
cargo test
```

Expected: PASS.

- [ ] **Step 3: Rebuild the Python extension**

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: PASS.

- [ ] **Step 4: Run targeted and broad Python tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_rust.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 5: Run Python formatting and linting**

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS.

- [ ] **Step 6: Build docs**

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS with no new warnings introduced by this change.

- [ ] **Step 7: Clean generated docs output and inspect status**

```powershell
if (Test-Path site) { Remove-Item -Recurse -Force -LiteralPath site }
git status --short --branch
```

Expected: clean branch after committed tasks, or only intentional uncommitted changes if verification forced a small fix.

- [ ] **Step 8: Push the branch**

```powershell
git push
```

Expected: branch pushes to the existing PR branch.

---

## Self-Review Checklist

- Spec coverage:
  - Stable Rust-backed state snapshots: Tasks 2 and 3.
  - JSON-safe schema V1: Tasks 1, 2, and 3.
  - Strict validation: Tasks 1 and 2.
  - Deterministic continuation: Tasks 1 and 2.
  - PyO3 API and stubs: Task 3.
  - Docs and changelog: Task 4.
  - No `CMAESOptimizer` checkpoint/resume: File structure, docs, and Task 4 keep this explicit.
- Design correction:
  - Lazy eigendecomposition state is included because current CMA-ES sampling uses the cached basis between refresh intervals. This keeps snapshots deterministic without mutating state during export.
- Placeholder scan:
  - No placeholder steps remain.
  - All implementation steps include exact file paths, code snippets, commands, and expected results.
- Verification:
  - The plan uses `.venv\Scripts\python.exe` for Python commands and includes Rust, Python, docs, and cleanup verification.
