# Rust Gene Codec Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate Rust-internal gene kind parsing and encoded repair so DE and reproduction kernels share one implementation while preserving current PyO3 signatures.

**Architecture:** Add `src/gene_codec.rs` as an internal module that reuses `crate::gene_spec::GeneKind`. `src/de.rs`, `src/reproduce.rs`, and `src/lib.rs` call this module instead of maintaining local parsing or repair copies. Python-facing behavior remains enforced through existing Rust kernel tests.

**Tech Stack:** Rust 2021, PyO3, cargo test, cargo fmt, cargo clippy, maturin, pytest.

---

## File Structure

- Create: `src/gene_codec.rs`
  - Owns `parse_gene_kind`, `parse_gene_kinds`, `repair_encoded_value`, and `repair_encoded_values`.
- Modify: `src/lib.rs`
  - Adds `mod gene_codec;`, imports shared parser, and removes local `parse_gene_kinds`.
- Modify: `src/de.rs`
  - Uses shared parser and repair helper.
- Modify: `src/reproduce.rs`
  - Makes `clamp_and_round(...)` delegate to shared vector repair.
- Modify: `tests/unit/test_de_rust_kernel.py`
  - Adds one cross-boundary assertion that Rust DE repair stays encoded-valid for edge values.
- Modify: `tests/unit/test_operators_rust.py`
  - Keeps Rust reproduction repair behavior covered from Python.

---

### Task 1: Add Failing Rust Module Tests

**Files:**
- Create: `src/gene_codec.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Register the new internal module**

In `src/lib.rs`, add the module near the other module declarations:

```rust
mod gene_codec;
```

- [ ] **Step 2: Create module tests before implementing helpers**

Create `src/gene_codec.rs` with:

```rust
use pyo3::prelude::*;

use crate::gene_spec::GeneKind;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repair_encoded_value_matches_python_contract() {
        assert_eq!(
            repair_encoded_value(99.0, (-1.0, 1.0), &GeneKind::Float),
            1.0
        );
        assert_eq!(
            repair_encoded_value(20.8, (2.0, 20.0), &GeneKind::Int),
            20.0
        );
        assert_eq!(
            repair_encoded_value(0.49, (0.0, 1.0), &GeneKind::Bool),
            0.0
        );
        assert_eq!(
            repair_encoded_value(0.5, (0.0, 1.0), &GeneKind::Bool),
            1.0
        );
    }

    #[test]
    fn test_repair_encoded_values_repairs_full_vector() {
        let genes = vec![99.0, 1.2, 0.8];
        let bounds = vec![(-1.0, 1.0), (2.0, 20.0), (0.0, 1.0)];
        let kinds = vec![GeneKind::Float, GeneKind::Int, GeneKind::Bool];

        assert_eq!(repair_encoded_values(&genes, &bounds, &kinds), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_parse_gene_kinds_reports_unknown_kind() {
        let error = parse_gene_kinds(&["float".to_string(), "bad".to_string()])
            .expect_err("unknown kind should fail");

        assert!(error.to_string().contains("Unknown gene kind"));
    }
}
```

The file intentionally references helpers that do not exist yet.

- [ ] **Step 3: Run the Rust module tests and verify they fail**

Run:

```powershell
cargo test gene_codec
```

Expected: FAIL with unresolved function errors for `repair_encoded_value`, `repair_encoded_values`, and `parse_gene_kinds`.

---

### Task 2: Implement Shared Rust Gene Codec Helpers

**Files:**
- Modify: `src/gene_codec.rs`

- [ ] **Step 1: Add parser and repair implementations**

In `src/gene_codec.rs`, add the helper functions above the test module:

```rust
pub(crate) fn parse_gene_kind(kind: &str) -> PyResult<GeneKind> {
    match kind {
        "float" => Ok(GeneKind::Float),
        "int" => Ok(GeneKind::Int),
        "bool" => Ok(GeneKind::Bool),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown gene kind: '{other}'. Valid: float, int, bool"
        ))),
    }
}

pub(crate) fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>> {
    kinds_str
        .iter()
        .map(|kind| parse_gene_kind(kind.as_str()))
        .collect()
}

pub(crate) fn repair_encoded_value(value: f64, bounds: (f64, f64), kind: &GeneKind) -> f64 {
    let (low, high) = bounds;
    match kind {
        GeneKind::Float => value.clamp(low, high),
        GeneKind::Int => value.round().clamp(low, high),
        GeneKind::Bool => {
            if value >= 0.5 {
                1.0
            } else {
                0.0
            }
        }
    }
}

pub(crate) fn repair_encoded_values(
    genes: &[f64],
    bounds: &[(f64, f64)],
    kinds: &[GeneKind],
) -> Vec<f64> {
    assert_eq!(
        genes.len(),
        bounds.len(),
        "repair_encoded_values: genes/bounds mismatch"
    );
    assert_eq!(
        genes.len(),
        kinds.len(),
        "repair_encoded_values: genes/kinds mismatch"
    );

    genes
        .iter()
        .enumerate()
        .map(|(idx, &value)| repair_encoded_value(value, bounds[idx], &kinds[idx]))
        .collect()
}
```

- [ ] **Step 2: Run the Rust module tests**

Run:

```powershell
cargo test gene_codec
```

Expected: PASS.

---

### Task 3: Make Reproduction Delegate To Shared Repair

**Files:**
- Modify: `src/reproduce.rs`

- [ ] **Step 1: Import shared vector repair**

In `src/reproduce.rs`, add:

```rust
use crate::gene_codec::repair_encoded_values;
```

- [ ] **Step 2: Replace `clamp_and_round(...)` internals**

Replace the body of `clamp_and_round(...)` with:

```rust
pub fn clamp_and_round(genes: &[f64], bounds: &[(f64, f64)], kinds: &[GeneKind]) -> Vec<f64> {
    repair_encoded_values(genes, bounds, kinds)
}
```

Keep the function name because existing reproduction code and Rust tests already use it.

- [ ] **Step 3: Run reproduction tests**

Run:

```powershell
cargo test reproduce
```

Expected: PASS, including the existing `test_clamp_and_round_*` tests.

---

### Task 4: Make DE And PyO3 Entry Points Use Shared Parsing And Repair

**Files:**
- Modify: `src/de.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Update `src/de.rs` imports**

In `src/de.rs`, add:

```rust
use crate::gene_codec::{parse_gene_kinds, repair_encoded_value};
```

Delete the local `parse_gene_kinds(...)` and `repair_value(...)` functions.

- [ ] **Step 2: Replace DE repair calls**

Replace each `repair_value(...)` call with `repair_encoded_value(...)`. For example:

```rust
return repair_encoded_value(low, gene_bounds[gene_idx], &gene_kinds[gene_idx]);
```

and:

```rust
repair_encoded_value(value, gene_bounds[gene_idx], &gene_kinds[gene_idx])
```

- [ ] **Step 3: Update `src/lib.rs` parser import and remove local parser**

In `src/lib.rs`, add:

```rust
use gene_codec::parse_gene_kinds;
```

Remove the local `fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>>` block.

If `use gene_spec::GeneKind;` becomes unused in `src/lib.rs`, remove it.

- [ ] **Step 4: Run DE and lib Rust tests**

Run:

```powershell
cargo test de
cargo test
```

Expected: both PASS.

---

### Task 5: Add Python Parity Assertions For Rust Outputs

**Files:**
- Modify: `tests/unit/test_de_rust_kernel.py`
- Modify: `tests/unit/test_operators_rust.py`

- [ ] **Step 1: Add a DE encoded-repair edge test**

In `tests/unit/test_de_rust_kernel.py`, add:

```python
def test_de_generate_trials_repairs_encoded_outputs_for_each_kind() -> None:
    population = [
        [-100.0, 1.0, 0.0],
        [100.0, 20.0, 1.0],
        [-50.0, 2.0, 0.0],
        [50.0, 19.0, 1.0],
    ]

    proposals = _core.de_generate_trials(
        population,
        [0.0, 1.0, 2.0, 3.0],
        [(-1.0, 1.0), (2.0, 20.0), (0.0, 1.0)],
        ["float", "int", "bool"],
        "rand1bin",
        2.0,
        1.0,
        123,
        0,
        [0, 1, 2, 3],
        "maximize",
    )

    for proposal in proposals:
        x, period, enabled = proposal["genes"]
        assert -1.0 <= x <= 1.0
        assert 2.0 <= period <= 20.0
        assert period == round(period)
        assert enabled in (0.0, 1.0)
```

- [ ] **Step 2: Add or keep Rust operator parity coverage**

If `tests/unit/test_operators_rust.py` does not already check mixed int/bool repair through `reproduce_population`, add:

```python
def test_reproduce_population_repairs_int_and_bool_outputs() -> None:
    result = _core.reproduce_population(
        [[0.0, 2.0, 0.0], [1.0, 20.0, 1.0], [-1.0, 10.0, 0.0], [0.5, 5.0, 1.0]],
        [0.0, 1.0, 2.0, 3.0],
        "uniform",
        1.0,
        2.0,
        0.5,
        "gaussian",
        1.0,
        [0.5, 5.0, 0.0],
        [(-1.0, 1.0), (2.0, 20.0), (0.0, 1.0)],
        ["float", "int", "bool"],
        "tournament",
        2,
        4,
        123,
        0,
    )

    for row in result:
        assert -1.0 <= row[0] <= 1.0
        assert 2.0 <= row[1] <= 20.0
        assert row[1] == round(row[1])
        assert row[2] in (0.0, 1.0)
```

If an equivalent test already exists, do not duplicate it; update its assertion message to mention shared Rust repair.

- [ ] **Step 3: Rebuild extension and run Python parity tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py tests/unit/test_operators_rust.py -v
```

Expected: PASS.

---

### Task 6: Final Verification And Commit

**Files:**
- All files touched in Tasks 1-5.

- [ ] **Step 1: Run Rust verification**

Run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all PASS.

- [ ] **Step 2: Run Python extension and focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py tests/unit/test_operators_rust.py -v
```

Expected: both PASS.

- [ ] **Step 3: Commit task-related files only**

Run:

```powershell
git status --short
git add src/gene_codec.rs src/lib.rs src/de.rs src/reproduce.rs tests/unit/test_de_rust_kernel.py tests/unit/test_operators_rust.py
git commit -m "refactor(rust): share gene codec repair"
```

Expected: commit succeeds with only Rust gene codec parity files staged.
