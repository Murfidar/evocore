# DE Rust Kernel Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Differential Evolution trial proposal generation for all current built-in strategies into one Rust/PyO3 kernel while preserving Python-owned optimizer lifecycle semantics.

**Architecture:** Add `src/de.rs` with a single `_core.de_generate_trials(...)` batch proposal function. Rust owns deterministic proposal math and returns encoded trial genes plus metadata; Python owns `Candidate` creation, target replacement, jDE commit/discard, checkpoints, events, telemetry, policies, callbacks, and evaluator integration.

**Tech Stack:** Rust 2021, PyO3, rand, Python 3.11+, maturin, pytest, ruff.

---

## File Structure

- Create: `src/de.rs`
  - Owns DE strategy dispatch, donor sampling, crossover masks, mixed repair, jDE F/CR proposal, input validation, and proposal metadata.
- Modify: `src/lib.rs`
  - Adds `mod de;` and exposes `_core.de_generate_trials(...)`.
- Modify: `evocore/_core.pyi`
  - Adds the Python stub for `de_generate_trials(...)`.
- Modify: `evocore/optimizers/de/ask_tell.py`
  - Encodes target population, calls `_core.de_generate_trials(...)`, decodes returned genes, creates candidates, and registers pending jDE params.
- Modify: `evocore/optimizers/de/strategies.py`
  - Keeps strategy specs and validation; removes Rust-migrated math from the production path.
- Modify: `evocore/optimizers/de/adaptive.py`
  - Adds a small committed-state export helper for Rust input while keeping commit/discard in Python.
- Create: `tests/unit/test_de_rust_kernel.py`
  - Tests the direct Rust/PyO3 proposal kernel contract.
- Modify: `tests/unit/test_de_strategies.py`
  - Narrows strategy tests to registry validation and production wrapper behavior.
- Modify: `tests/unit/test_de_ask_tell.py`
  - Covers Rust-backed trial proposals through `ask(...)` and `tell(...)`.
- Modify: `tests/unit/test_de_jde.py`
  - Covers Rust-proposed jDE params and Python commit/discard semantics.
- Modify: `tests/unit/test_de_checkpointing.py`
  - Covers checkpoint restore after Rust-backed trial ask, including pending jDE params.
- Modify: `tests/integration/test_de_mixed_gene_space.py`
  - Confirms mixed search-space runs remain valid with Rust-generated trials.
- Modify: `docs/site/de.md`
  - Documents Rust-backed DE trial proposals and deterministic sequence note.
- Modify: `CHANGELOG.md`
  - Records the user-visible kernel migration and seeded sequence impact.

---

### Task 1: Add Direct PyO3 Kernel Contract Tests

**Files:**
- Create: `tests/unit/test_de_rust_kernel.py`

- [ ] **Step 1: Write failing direct-kernel tests**

Create `tests/unit/test_de_rust_kernel.py` with:

```python
from __future__ import annotations

import math

import pytest

from evocore import _core


BOUNDS = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0), (0.0, 1.0)]
KINDS = ["float", "float", "int", "bool"]
POPULATION = [
    [-4.0, -3.0, 1.0, 0.0],
    [-2.0, -1.0, 2.0, 1.0],
    [0.0, 1.0, 3.0, 0.0],
    [1.5, 2.0, 4.0, 1.0],
    [3.0, 4.0, 5.0, 0.0],
    [4.0, 5.0, 6.0, 1.0],
]
SCORES = [1.0, 2.0, 3.0, 4.0, 9.0, 5.0]


def _generate(strategy: str, *, direction: str = "maximize", target_slots=(0, 1, 2)):
    return _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        strategy,
        0.7,
        0.9,
        42,
        3,
        list(target_slots),
        direction,
    )


def _assert_valid_gene_vector(genes: list[float]) -> None:
    assert len(genes) == len(BOUNDS)
    for value, (low, high), kind in zip(genes, BOUNDS, KINDS, strict=True):
        assert low <= value <= high
        if kind == "int":
            assert value == round(value)
        if kind == "bool":
            assert value in (0.0, 1.0)


@pytest.mark.parametrize(
    ("strategy", "min_population", "donor_count"),
    [
        ("rand1bin", 4, 3),
        ("best1bin", 4, 3),
        ("rand2bin", 6, 5),
        ("current-to-best1bin", 4, 3),
    ],
)
def test_de_generate_trials_stateless_strategies_are_deterministic(
    strategy: str,
    min_population: int,
    donor_count: int,
) -> None:
    first = _generate(strategy)
    second = _generate(strategy)

    assert first == second
    assert len(first) == 3
    assert min_population <= len(POPULATION)

    for expected_slot, proposal in zip([0, 1, 2], first, strict=True):
        assert proposal["target_slot"] == expected_slot
        _assert_valid_gene_vector(proposal["genes"])
        metadata = proposal["metadata"]
        assert metadata["strategy"] == strategy
        assert metadata["target_slot"] == expected_slot
        assert len(metadata["donor_slots"]) == donor_count
        assert len(set(metadata["donor_slots"])) == donor_count
        assert expected_slot not in metadata["donor_slots"]


def test_de_generate_trials_best_strategy_reports_best_slot() -> None:
    proposals = _generate("best1bin")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 4
        assert metadata["base_slot"] == 4


def test_de_generate_trials_current_to_best_reports_target_base() -> None:
    proposals = _generate("current-to-best1bin")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 4
        assert metadata["base_slot"] == metadata["target_slot"]


def test_de_generate_trials_minimize_uses_lowest_score_as_best_slot() -> None:
    proposals = _generate("best1bin", direction="minimize")

    for proposal in proposals:
        metadata = proposal["metadata"]
        assert metadata["best_slot"] == 0
        assert metadata["base_slot"] == 0


def test_de_generate_trials_jde_returns_trial_parameters() -> None:
    first = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        3,
        [0, 1, 2],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )
    second = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        3,
        [0, 1, 2],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )

    assert first == second
    for proposal in first:
        _assert_valid_gene_vector(proposal["genes"])
        metadata = proposal["metadata"]
        assert metadata["strategy"] == "jde-rand1bin"
        assert metadata["adaptive_slot"] == metadata["target_slot"]
        assert 0.0 <= metadata["crossover_rate"] <= 1.0
        assert math.isfinite(metadata["mutation_factor"])
        assert metadata["mutation_factor"] >= 0.0


def test_de_generate_trials_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown DE strategy"):
        _generate("unknown")
```

- [ ] **Step 2: Run the new tests and verify they fail because the Rust export is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py -v
```

Expected: FAIL with an error mentioning that `_core` has no attribute `de_generate_trials`.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/unit/test_de_rust_kernel.py
git commit -m "test(de): cover rust trial kernel contract"
```

---

### Task 2: Add Rust Module, PyO3 Export, And Type Stub

**Files:**
- Create: `src/de.rs`
- Modify: `src/lib.rs`
- Modify: `evocore/_core.pyi`
- Test: `tests/unit/test_de_rust_kernel.py`

- [ ] **Step 1: Add `src/de.rs` with public entry point and validation types**

Create `src/de.rs` with this structure:

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cmp::Ordering;

use crate::gene_spec::GeneKind;
use crate::utils::{derive_seed, OP_CROSSOVER, OP_MUTATION, OP_SELECTION};

const JDE_F_REFRESH_PROBABILITY: f64 = 0.1;
const JDE_CR_REFRESH_PROBABILITY: f64 = 0.1;
const JDE_F_LOW: f64 = 0.1;
const JDE_F_HIGH: f64 = 1.0;

#[derive(Clone, Debug, PartialEq, Eq)]
enum DEStrategy {
    Rand1Bin,
    Best1Bin,
    Rand2Bin,
    CurrentToBest1Bin,
    JdeRand1Bin,
}

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
        "rand1bin" => Ok(DEStrategy::Rand1Bin),
        "best1bin" => Ok(DEStrategy::Best1Bin),
        "rand2bin" => Ok(DEStrategy::Rand2Bin),
        "current-to-best1bin" => Ok(DEStrategy::CurrentToBest1Bin),
        "jde-rand1bin" => Ok(DEStrategy::JdeRand1Bin),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown DE strategy: {other}"
        ))),
    }
}

fn strategy_name(strategy: &DEStrategy) -> &'static str {
    match strategy {
        DEStrategy::Rand1Bin => "rand1bin",
        DEStrategy::Best1Bin => "best1bin",
        DEStrategy::Rand2Bin => "rand2bin",
        DEStrategy::CurrentToBest1Bin => "current-to-best1bin",
        DEStrategy::JdeRand1Bin => "jde-rand1bin",
    }
}

fn min_population(strategy: &DEStrategy) -> usize {
    match strategy {
        DEStrategy::Rand2Bin => 6,
        _ => 4,
    }
}

fn parse_gene_kinds(kinds_str: &[String]) -> PyResult<Vec<GeneKind>> {
    kinds_str
        .iter()
        .map(|kind| match kind.as_str() {
            "float" => Ok(GeneKind::Float),
            "int" => Ok(GeneKind::Int),
            "bool" => Ok(GeneKind::Bool),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown gene kind: '{other}'. Valid: float, int, bool"
            ))),
        })
        .collect()
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

fn repair_value(value: f64, bounds: (f64, f64), kind: &GeneKind) -> f64 {
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
```

Also add the helper signatures below the entry point. They may return simple deterministic proposals in this task and will be completed in later tasks:

```rust
fn extract_jde_state(
    jde_state: Option<Bound<'_, PyAny>>,
    population_size: usize,
    strategy: &DEStrategy,
) -> PyResult<Option<JdeCommittedState>> {
    if !matches!(strategy, DEStrategy::JdeRand1Bin) {
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
```

- [ ] **Step 2: Add a minimal `generate_one_trial(...)` body that compiles**

Append:

```rust
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
    let f = if let (DEStrategy::JdeRand1Bin, Some(state)) = (strategy, jde_state) {
        state.f_by_slot[target_slot]
    } else {
        mutation_factor
    };
    let cr = if let (DEStrategy::JdeRand1Bin, Some(state)) = (strategy, jde_state) {
        state.cr_by_slot[target_slot]
    } else {
        crossover_rate
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
        matches!(strategy, DEStrategy::CurrentToBest1Bin),
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
        mutation_factor: matches!(strategy, DEStrategy::JdeRand1Bin).then_some(f),
        crossover_rate: matches!(strategy, DEStrategy::JdeRand1Bin).then_some(cr),
        adaptive_slot: matches!(strategy, DEStrategy::JdeRand1Bin).then_some(target_slot),
    })
}
```

- [ ] **Step 3: Add the remaining helper skeletons used by `generate_one_trial(...)`**

Append:

```rust
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
) -> PyResult<(usize, Vec<(usize, usize)>, Option<usize>)> {
    match strategy {
        DEStrategy::Rand1Bin | DEStrategy::JdeRand1Bin => {
            let slots = sample_slots(population_size, 3, &[target_slot], seed, generation, target_slot)?;
            Ok((slots[0], vec![(slots[1], slots[2])], None))
        }
        DEStrategy::Best1Bin => {
            let slots = sample_slots(population_size, 2, &[target_slot, best_slot], seed, generation, target_slot)?;
            Ok((best_slot, vec![(slots[0], slots[1])], Some(best_slot)))
        }
        DEStrategy::Rand2Bin => {
            let slots = sample_slots(population_size, 5, &[target_slot], seed, generation, target_slot)?;
            Ok((slots[0], vec![(slots[1], slots[2]), (slots[3], slots[4])], None))
        }
        DEStrategy::CurrentToBest1Bin => {
            let slots = sample_slots(population_size, 2, &[target_slot, best_slot], seed, generation, target_slot)?;
            Ok((target_slot, vec![(slots[0], slots[1])], Some(best_slot)))
        }
    }
}
```

- [ ] **Step 4: Add the actual trial gene builder**

Append:

```rust
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
    let forced_index = if gene_count == 0 {
        0
    } else {
        mask_rng.gen_range(0..gene_count)
    };

    (0..gene_count)
        .map(|gene_idx| {
            let (low, high) = gene_bounds[gene_idx];
            if low == high {
                return repair_value(low, gene_bounds[gene_idx], &gene_kinds[gene_idx]);
            }
            let selected = gene_idx == forced_index || mask_rng.gen::<f64>() < crossover_rate;
            if !selected {
                return repair_value(target[gene_idx], gene_bounds[gene_idx], &gene_kinds[gene_idx]);
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
            repair_value(value, gene_bounds[gene_idx], &gene_kinds[gene_idx])
        })
        .collect()
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
            value = target[gene_idx] + mutation_factor * (population[slot][gene_idx] - target[gene_idx]);
        }
    }
    for (left, right) in difference_pairs {
        value += mutation_factor * (population[*left][gene_idx] - population[*right][gene_idx]);
    }
    value
}
```

- [ ] **Step 5: Wire the Rust module into `src/lib.rs`**

Modify `src/lib.rs`:

```rust
mod de;
```

Add the import near other `use` lines:

```rust
use de::de_generate_trials;
```

Add the PyO3 function registration in `_core(...)` near other functions:

```rust
m.add_function(wrap_pyfunction!(de_generate_trials, m)?)?;
```

- [ ] **Step 6: Add `_core.pyi` stub**

Append to `evocore/_core.pyi`:

```python
def de_generate_trials(
    population: Sequence[Sequence[float]],
    scores: Sequence[float],
    gene_bounds: Sequence[tuple[float, float]],
    gene_kinds: Sequence[str],
    strategy: str,
    mutation_factor: float,
    crossover_rate: float,
    seed: int,
    generation: int,
    target_slots: Sequence[int],
    direction: str,
    jde_state: dict[str, object] | None = None,
) -> list[dict[str, object]]: ...
```

- [ ] **Step 7: Build the extension and run direct kernel tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py -v
```

Expected: direct kernel tests pass.

- [ ] **Step 8: Run Rust format/test checks for touched Rust surface**

Run:

```powershell
cargo fmt --check
cargo test de
```

Expected: both pass. If `cargo test de` reports zero filtered tests, keep going and add Rust unit tests in Task 3.

- [ ] **Step 9: Commit the Rust export**

```powershell
git add src/de.rs src/lib.rs evocore/_core.pyi
git commit -m "feat(de): add rust trial proposal kernel"
```

---

### Task 3: Add Rust Unit Tests For Strategy Invariants

**Files:**
- Modify: `src/de.rs`
- Test: `cargo test de`

- [ ] **Step 1: Add Rust test module**

Append this test module to `src/de.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0), (-5.0, 5.0), (0.0, 10.0), (0.0, 1.0)]
    }

    fn kinds() -> Vec<GeneKind> {
        vec![GeneKind::Float, GeneKind::Float, GeneKind::Int, GeneKind::Bool]
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
        let proposals = proposals(DEStrategy::Rand1Bin);
        for proposal in proposals {
            assert_eq!(proposal.donor_slots.len(), 3);
            assert!(!proposal.donor_slots.contains(&proposal.target_slot));
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_best1bin_uses_best_as_base() {
        let proposals = proposals(DEStrategy::Best1Bin);
        for proposal in proposals {
            assert_eq!(proposal.best_slot, Some(4));
            assert_eq!(proposal.base_slot, 4);
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_rand2bin_uses_five_donor_slots() {
        let proposals = proposals(DEStrategy::Rand2Bin);
        for proposal in proposals {
            assert_eq!(proposal.donor_slots.len(), 5);
            assert_eq!(proposal.difference_pairs.len(), 2);
            assert_valid_genes(&proposal.genes);
        }
    }

    #[test]
    fn de_current_to_best_uses_target_as_base() {
        let proposals = proposals(DEStrategy::CurrentToBest1Bin);
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
        let first = proposals(DEStrategy::Rand2Bin);
        let second = proposals(DEStrategy::Rand2Bin);
        assert_eq!(first[0].genes, second[0].genes);
        assert_eq!(first[0].donor_slots, second[0].donor_slots);
    }
}
```

- [ ] **Step 2: Run Rust tests**

Run:

```powershell
cargo test de
```

Expected: PASS.

- [ ] **Step 3: Run direct Python kernel tests again**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit Rust invariant tests**

```powershell
git add src/de.rs tests/unit/test_de_rust_kernel.py
git commit -m "test(de): cover rust kernel invariants"
```

---

### Task 4: Implement Rust jDE Parameter Proposal

**Files:**
- Modify: `src/de.rs`
- Test: `tests/unit/test_de_rust_kernel.py`
- Test: `cargo test de`

- [ ] **Step 1: Write a focused jDE refresh test in Python**

Add this test to `tests/unit/test_de_rust_kernel.py`:

```python
def test_de_generate_trials_jde_can_refresh_parameters_from_seed() -> None:
    proposals = _core.de_generate_trials(
        POPULATION,
        SCORES,
        BOUNDS,
        KINDS,
        "jde-rand1bin",
        0.5,
        0.9,
        42,
        99,
        [0, 1, 2, 3, 4, 5],
        "maximize",
        {"f_by_slot": [0.5] * 6, "cr_by_slot": [0.9] * 6},
    )

    params = [
        (proposal["metadata"]["mutation_factor"], proposal["metadata"]["crossover_rate"])
        for proposal in proposals
    ]
    assert any(f != 0.5 or cr != 0.9 for f, cr in params)
    assert all(0.1 <= f <= 1.0 for f, _ in params)
    assert all(0.0 <= cr <= 1.0 for _, cr in params)
```

- [ ] **Step 2: Run the jDE test and verify it fails before refresh implementation**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py::test_de_generate_trials_jde_can_refresh_parameters_from_seed -v
```

Expected: FAIL because all returned jDE params still equal committed params.

- [ ] **Step 3: Implement jDE parameter refresh in `src/de.rs`**

Add:

```rust
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
```

Change the `generate_one_trial(...)` F/CR selection to:

```rust
let (f, cr) = if let (DEStrategy::JdeRand1Bin, Some(state)) = (strategy, jde_state) {
    propose_jde_params(state, seed, generation, target_slot)
} else {
    (mutation_factor, crossover_rate)
};
```

- [ ] **Step 4: Add Rust unit test for jDE proposal determinism**

Append inside the `#[cfg(test)]` module:

```rust
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
```

- [ ] **Step 5: Run jDE checks**

Run:

```powershell
cargo test de
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit jDE kernel support**

```powershell
git add src/de.rs tests/unit/test_de_rust_kernel.py
git commit -m "feat(de): generate jde trial parameters in rust"
```

---

### Task 5: Integrate Rust Kernel Into DE `ask(...)`

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `evocore/optimizers/de/adaptive.py`
- Test: `tests/unit/test_de_ask_tell.py`
- Test: `tests/unit/test_de_jde.py`

- [ ] **Step 1: Add a Python test proving non-default strategies use Rust-backed `ask(...)`**

Add to `tests/unit/test_de_ask_tell.py`:

```python
def test_de_trial_ask_uses_rust_kernel_metadata_for_each_strategy():
    strategies = ["rand1bin", "best1bin", "rand2bin", "current-to-best1bin"]
    for strategy in strategies:
        optimizer = DifferentialEvolutionOptimizer(
            _mixed_space(),
            population_size=6,
            strategy=strategy,
            seed=42,
        )
        initial = optimizer.ask()
        optimizer.tell(
            [
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(index),
                    confidence="trusted_full",
                    stage="full",
                )
                for index, candidate in enumerate(initial)
            ]
        )

        trials = optimizer.ask(3)

        assert len(trials) == 3
        for trial in trials:
            assert trial.metadata["strategy"] == strategy
            assert "donor_slots" in trial.metadata
            assert "difference_pairs" in trial.metadata
            assert "target_slot" in trial.metadata
```

This test uses the existing `_mixed_space()` and `_records(...)` helpers already defined in `tests/unit/test_de_ask_tell.py`.

- [ ] **Step 2: Run the focused test and verify it fails before integration**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_trial_ask_uses_rust_kernel_metadata_for_each_strategy -v
```

Expected: FAIL because the current Python strategy path may produce tuple metadata types or does not call the Rust kernel.

- [ ] **Step 3: Add committed-state export to `JDEAdaptiveState`**

In `evocore/optimizers/de/adaptive.py`, add:

```python
    def to_rust_committed_state(self) -> dict[str, list[float]]:
        """Return committed per-slot values for Rust jDE proposal generation."""
        return {
            "f_by_slot": list(self.f_by_slot),
            "cr_by_slot": list(self.cr_by_slot),
        }
```

- [ ] **Step 4: Add Rust proposal wrapper to `DifferentialEvolutionAskTellMixin`**

In `evocore/optimizers/de/ask_tell.py`, add this method inside `DifferentialEvolutionAskTellMixin`:

```python
    def _rust_trial_proposals(self, count: int) -> list[TrialProposal]:
        target_population = self._target_population()
        trial_count = min(int(count), len(target_population))
        target_slots = list(range(trial_count))
        population_encoded = [
            [
                1.0 if value is True else 0.0 if value is False else float(value)
                for value in candidate.genes
            ]
            for candidate in target_population
        ]
        scores = [
            candidate.state_comparison_score(self.direction)
            for candidate in target_population
        ]
        jde_state = None
        to_rust_committed_state = getattr(self._de_strategy_state, "to_rust_committed_state", None)
        if callable(to_rust_committed_state):
            jde_state = to_rust_committed_state()

        raw_proposals = _core.de_generate_trials(
            population_encoded,
            scores,
            self.gene_space.rust_bounds,
            self.gene_space.kinds,
            self.strategy,
            self.mutation_factor,
            self.crossover_rate,
            self.seed,
            self.generation,
            target_slots,
            self.direction,
            jde_state,
        )
        proposals: list[TrialProposal] = []
        for raw in raw_proposals:
            genes = _decode_de_values(self.gene_space, raw["genes"])
            metadata = dict(raw["metadata"])
            self.gene_space.validate_genes(genes)
            proposals.append(TrialProposal(genes=genes, metadata=metadata))
        return proposals
```

- [ ] **Step 5: Change `_trial_candidates(...)` to call Rust proposals**

Replace the `proposal = self._trial_proposal_for_slot(target_slot)` line and surrounding target-slot loop setup with:

```python
        proposals = self._rust_trial_proposals(trial_count)
        for target_slot, proposal in enumerate(proposals):
            target = self._target_candidate(target_slot)
            genes = proposal.genes
            metadata = dict(proposal.metadata)
            metadata["target_candidate_id"] = target.candidate_id
```

Keep the existing candidate creation, pending target maps, `_record_pending_strategy_trial(candidate)`, and append behavior unchanged.

- [ ] **Step 6: Run focused ask/tell integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py::test_de_trial_ask_uses_rust_kernel_metadata_for_each_strategy -v
```

Expected: PASS.

- [ ] **Step 7: Add jDE pending-param integration test**

Add to `tests/unit/test_de_jde.py`:

```python
def test_jde_trial_ask_registers_rust_proposed_pending_params():
    optimizer = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        seed=42,
    )
    initial = optimizer.ask()
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(initial)
        ]
    )

    trials = optimizer.ask(3)

    assert len(trials) == 3
    for trial in trials:
        assert trial.candidate_id in optimizer._de_strategy_state.pending_trial_params
        pending = optimizer._de_strategy_state.pending_trial_params[trial.candidate_id]
        assert pending.target_slot == trial.metadata["adaptive_slot"]
        assert pending.mutation_factor == trial.metadata["mutation_factor"]
        assert pending.crossover_rate == trial.metadata["crossover_rate"]
```

- [ ] **Step 8: Run jDE integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_jde.py::test_jde_trial_ask_registers_rust_proposed_pending_params -v
```

Expected: PASS.

- [ ] **Step 9: Run related DE ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_jde.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit Python integration**

```powershell
git add evocore/optimizers/de/ask_tell.py evocore/optimizers/de/adaptive.py tests/unit/test_de_ask_tell.py tests/unit/test_de_jde.py
git commit -m "feat(de): use rust kernel for trial ask"
```

---

### Task 6: Retire Python Strategy Math From Production Path

**Files:**
- Modify: `evocore/optimizers/de/strategies.py`
- Modify: `tests/unit/test_de_strategies.py`
- Test: `tests/unit/test_de_strategies.py`

- [ ] **Step 1: Update strategy tests to focus on registry and validation**

In `tests/unit/test_de_strategies.py`, keep tests that validate:

```python
def test_supported_strategy_names_include_all_builtins() -> None:
    assert supported_strategy_names() == (
        "rand1bin",
        "best1bin",
        "rand2bin",
        "current-to-best1bin",
        "jde-rand1bin",
    )


@pytest.mark.parametrize(
    ("strategy", "minimum"),
    [
        ("rand1bin", 4),
        ("best1bin", 4),
        ("rand2bin", 6),
        ("current-to-best1bin", 4),
        ("jde-rand1bin", 4),
    ],
)
def test_validate_strategy_population_size(strategy: str, minimum: int) -> None:
    validate_strategy_population_size(strategy, minimum)
    with pytest.raises(ConfigurationError, match="population_size must be at least"):
        validate_strategy_population_size(strategy, minimum - 1)
```

Remove direct tests for `_rand1bin_trial`, `_best1bin_trial`, `_rand2bin_trial`, `_current_to_best1bin_trial`, and `_jde_rand1bin_trial` after the Rust-backed ask path is covered.

- [ ] **Step 2: Trim production strategy math exports**

In `evocore/optimizers/de/strategies.py`, keep:

```python
__all__ = [
    "SUPPORTED_DE_STRATEGIES",
    "DEStrategySpec",
    "TrialContext",
    "TrialProposal",
    "strategy_spec_for",
    "supported_strategy_names",
    "validate_strategy_population_size",
]
```

If `TrialContext` becomes unused after integration, remove it from runtime code and tests in this same task.

- [ ] **Step 3: Remove obsolete imports in `ask_tell.py`**

In `evocore/optimizers/de/ask_tell.py`, remove imports that are no longer used:

```python
from evocore.optimizers.de.strategies import (
    TrialProposal,
)
```

Keep only `TrialProposal` if `_rust_trial_proposals(...)` still constructs it.

- [ ] **Step 4: Run strategy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategies.py -v
```

Expected: PASS.

- [ ] **Step 5: Run all DE unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_strategies.py tests/unit/test_de_jde.py tests/unit/test_de_rust_kernel.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit strategy cleanup**

```powershell
git add evocore/optimizers/de/strategies.py evocore/optimizers/de/ask_tell.py tests/unit/test_de_strategies.py
git commit -m "refactor(de): keep strategy registry in python"
```

---

### Task 7: Checkpoint, Run, And Multi-Run Coverage

**Files:**
- Modify: `tests/unit/test_de_checkpointing.py`
- Modify: `tests/unit/test_de_multi_run.py`
- Modify: `tests/integration/test_de_mixed_gene_space.py`

- [ ] **Step 1: Add checkpoint restore test after Rust-backed trial ask**

Add to `tests/unit/test_de_checkpointing.py`:

```python
def test_de_checkpoint_restores_after_rust_backed_trial_ask(tmp_path):
    optimizer = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="rand2bin",
        seed=42,
    )
    initial = optimizer.ask()
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(initial)
        ]
    )
    trials = optimizer.ask(3)
    checkpoint = optimizer.ask_tell_checkpoint(metadata={"phase": "trial_ask"})
    path = tmp_path / "de-rust-trial.evocore-checkpoint.json"
    optimizer.save_checkpoint(path, checkpoint)

    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="rand2bin",
        seed=42,
    )
    summary = restored.resume_ask_tell_checkpoint(path)

    assert summary.pending_batch_ids
    assert [candidate.candidate_id for candidate in trials] == [
        candidate_id
        for batch_id in summary.pending_batch_ids
        for candidate_id in restored._batches_by_id[batch_id].candidate_ids
    ]
    assert restored._trial_target_slots == optimizer._trial_target_slots
    assert restored._trial_target_candidate_ids == optimizer._trial_target_candidate_ids
```

- [ ] **Step 2: Add jDE checkpoint restore test with pending params**

Add to `tests/unit/test_de_checkpointing.py`:

```python
def test_de_checkpoint_restores_jde_pending_params_after_rust_trial_ask(tmp_path):
    optimizer = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        seed=42,
    )
    initial = optimizer.ask()
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(initial)
        ]
    )
    optimizer.ask(3)
    path = tmp_path / "de-jde-rust-trial.evocore-checkpoint.json"
    optimizer.save_checkpoint(path, optimizer.ask_tell_checkpoint())

    restored = DifferentialEvolutionOptimizer(
        _space(),
        population_size=6,
        strategy="jde-rand1bin",
        seed=42,
    )
    restored.resume_ask_tell_checkpoint(path)

    assert restored._de_strategy_state.pending_trial_params
    assert restored._de_strategy_state.to_checkpoint() == optimizer._de_strategy_state.to_checkpoint()
```

- [ ] **Step 3: Run checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_checkpointing.py -v
```

Expected: PASS. If golden fixtures assert exact old Python-generated trial values, update fixtures only after confirming the payload shape remains compatible.

- [ ] **Step 4: Add non-default strategy multi-run smoke test**

Add to `tests/unit/test_de_multi_run.py`:

```python
def test_de_run_multiple_supports_rust_backed_strategy():
    optimizer = DifferentialEvolutionOptimizer(
        GeneSpace.uniform(-2.0, 2.0, 3),
        population_size=6,
        max_generations=2,
        strategy="current-to-best1bin",
        seed=42,
    )

    batch = optimizer.run_multiple(SphereEvaluator(), n_runs=2)

    assert batch.n_runs == 2
    assert len(batch.all_runs) == 2
    assert batch.best in batch.all_runs
    assert all(run.optimizer_type == "DifferentialEvolutionOptimizer" for run in batch.all_runs)
```

- [ ] **Step 5: Run multi-run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_multi_run.py -v
```

Expected: PASS.

- [ ] **Step 6: Add mixed integration assertion for jDE**

Add to `tests/integration/test_de_mixed_gene_space.py`:

```python
def test_de_jde_mixed_space_run_uses_valid_gene_types():
    optimizer = DifferentialEvolutionOptimizer(
        _mixed_space(),
        population_size=6,
        max_generations=2,
        strategy="jde-rand1bin",
        seed=42,
    )

    result = optimizer.run(MixedSwitchEvaluator())

    for solution in result.final_solutions:
        assert isinstance(solution.values[0], float)
        assert isinstance(solution.values[1], int)
        assert isinstance(solution.values[2], bool)
```

This test uses the existing `_mixed_space()` helper and `MixedSwitchEvaluator` class already defined in `tests/integration/test_de_mixed_gene_space.py`.

- [ ] **Step 7: Run DE integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit checkpoint and run coverage**

```powershell
git add tests/unit/test_de_checkpointing.py tests/unit/test_de_multi_run.py tests/integration/test_de_mixed_gene_space.py
git commit -m "test(de): cover rust kernel lifecycle integration"
```

---

### Task 8: Documentation, Changelog, And Final Verification

**Files:**
- Modify: `docs/site/de.md`
- Modify: `CHANGELOG.md`
- Test: repository verification commands

- [ ] **Step 1: Update DE docs**

In `docs/site/de.md`, add this paragraph under the strategy section:

```markdown
DE trial proposals are generated by EvoCore's Rust extension. The Python
optimizer still owns ask/tell state, target replacement, checkpoint envelopes,
events, telemetry, callbacks, and evaluator integration. The Rust kernel owns
deterministic proposal math for the built-in strategies.
```

Add this paragraph under reproducibility:

```markdown
DE remains deterministic for the same EvoCore version, `GeneSpace`, optimizer
configuration, direction, and seed. The Rust proposal kernel may produce a
different exact trial sequence than older Python-only DE releases, so compare
seeded replay within the same EvoCore version when exact candidate sequences
matter.
```

- [ ] **Step 2: Update changelog**

In `CHANGELOG.md`, add an entry in the unreleased/current section:

```markdown
- Migrated Differential Evolution trial proposal generation for built-in
  strategies to the Rust extension while keeping Python ask/tell, replacement,
  checkpoint, event, telemetry, and policy semantics unchanged. Seeded DE runs
  remain deterministic within the same EvoCore version, but exact trial
  sequences may differ from the previous Python-generated strategy path.
```

- [ ] **Step 3: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 4: Rebuild extension and run relevant tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_rust_kernel.py tests/unit/test_de_ask_tell.py tests/unit/test_de_strategies.py tests/unit/test_de_jde.py tests/unit/test_de_checkpointing.py tests/unit/test_de_multi_run.py tests/integration/test_de_mixed_gene_space.py -v
```

Expected: PASS.

- [ ] **Step 5: Run broader unit/integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
cargo test
```

Expected: PASS.

- [ ] **Step 6: Commit docs and verification-ready state**

```powershell
git add docs/site/de.md CHANGELOG.md
git commit -m "docs(de): document rust proposal kernel"
```

- [ ] **Step 7: Inspect final status**

Run:

```powershell
git status --short --branch
git log --oneline -8
```

Expected: branch contains the task commits and no unstaged task changes.

---

## Self-Review Checklist

- Spec coverage:
  - Rust kernel for all current strategies is covered by Tasks 1-4.
  - Python lifecycle ownership is preserved by Task 5.
  - Runtime strategy math cleanup is covered by Task 6.
  - Checkpoint, jDE pending params, run, and multi-run coverage are covered by Task 7.
  - Docs, changelog, stubs, Rust checks, Python checks, and maturin rebuild are covered by Tasks 2 and 8.
- Placeholder scan:
  - The plan avoids placeholder-only steps and red-flag filler phrases.
  - Every code-changing task includes concrete files, code snippets, commands, and expected results.
- Type consistency:
  - Rust export name is `_core.de_generate_trials(...)` throughout.
  - Python jDE export helper is `JDEAdaptiveState.to_rust_committed_state()`.
  - Returned proposal dictionaries use `target_slot`, `genes`, and `metadata`.
  - Metadata names match the design: `strategy`, `base_slot`, `best_slot`, `donor_slots`, `difference_pairs`, `mutation_factor`, `crossover_rate`, and `adaptive_slot`.
