# DE Strategy Metadata Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tests that prevent Python and Rust Differential Evolution strategy metadata from drifting after the Rust proposal-kernel migration.

**Architecture:** Keep Python as the public strategy registry and Rust as the built-in strategy math owner. Add a focused contract test module that compares Python `DEStrategySpec` values with Rust kernel acceptance, rejection, and proposal metadata behavior. Do not expose a new PyO3 metadata endpoint in this slice.

**Tech Stack:** Python 3.11+, pytest, maturin-built PyO3 extension.

---

## File Structure

- Create: `tests/unit/test_de_strategy_metadata_contract.py`
  - Contract tests comparing Python strategy specs against Rust kernel behavior.
- Modify: `evocore/optimizers/de/strategies.py`
  - Only if tests reveal a Python metadata mismatch.
- Modify: `src/de.rs`
  - Only if tests reveal a Rust metadata mismatch.

---

### Task 1: Add Python/Rust Strategy Contract Tests

**Files:**
- Create: `tests/unit/test_de_strategy_metadata_contract.py`

- [ ] **Step 1: Write direct contract tests**

Create `tests/unit/test_de_strategy_metadata_contract.py` with:

```python
from __future__ import annotations

import pytest

from evocore import _core
from evocore.optimizers.de.strategies import (
    SUPPORTED_DE_STRATEGIES,
    supported_strategy_names,
)


STRATEGY_CONTRACT = {
    "rand1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": False,
    },
    "best1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": True,
        "base_is_target": False,
        "adaptive": False,
    },
    "rand2bin": {
        "donor_count": 5,
        "difference_pair_count": 2,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": False,
    },
    "current-to-best1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": True,
        "base_is_target": True,
        "adaptive": False,
    },
    "jde-rand1bin": {
        "donor_count": 3,
        "difference_pair_count": 1,
        "uses_best_slot": False,
        "base_is_target": False,
        "adaptive": True,
    },
}


def _population(size: int) -> list[list[float]]:
    return [[float(index), float(index + 1)] for index in range(size)]


def _scores(size: int) -> list[float]:
    return [float(index) for index in range(size)]


def _jde_state(size: int) -> dict[str, list[float]]:
    return {"f_by_slot": [0.5] * size, "cr_by_slot": [0.9] * size}


def _generate(strategy: str, population_size: int):
    jde_state = _jde_state(population_size) if strategy == "jde-rand1bin" else None
    return _core.de_generate_trials(
        _population(population_size),
        _scores(population_size),
        [(-10.0, 10.0), (-10.0, 10.0)],
        ["float", "float"],
        strategy,
        0.7,
        0.9,
        42,
        0,
        [0],
        "maximize",
        jde_state,
    )


def test_python_strategy_names_have_contract_entries() -> None:
    assert set(supported_strategy_names()) == set(STRATEGY_CONTRACT)


@pytest.mark.parametrize("strategy", supported_strategy_names())
def test_rust_accepts_every_python_strategy_at_min_population(strategy: str) -> None:
    spec = SUPPORTED_DE_STRATEGIES[strategy]

    proposals = _generate(strategy, spec.min_population_size)

    assert len(proposals) == 1
    metadata = proposals[0]["metadata"]
    expected = STRATEGY_CONTRACT[strategy]
    assert metadata["strategy"] == strategy
    assert len(metadata["donor_slots"]) == expected["donor_count"]
    assert len(metadata["difference_pairs"]) == expected["difference_pair_count"]
    assert ("best_slot" in metadata) is expected["uses_best_slot"]
    if expected["base_is_target"]:
        assert metadata["base_slot"] == metadata["target_slot"]
    assert ("adaptive_slot" in metadata) is expected["adaptive"]
    assert ("mutation_factor" in metadata) is expected["adaptive"]
    assert ("crossover_rate" in metadata) is expected["adaptive"]


@pytest.mark.parametrize("strategy", supported_strategy_names())
def test_rust_min_population_matches_python_spec(strategy: str) -> None:
    spec = SUPPORTED_DE_STRATEGIES[strategy]

    with pytest.raises(ValueError, match=f"at least {spec.min_population_size}"):
        _generate(strategy, spec.min_population_size - 1)


def test_rust_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown DE strategy"):
        _generate("not-a-strategy", 6)


def test_rust_requires_jde_state_for_adaptive_strategy() -> None:
    with pytest.raises(ValueError, match="jde_state is required"):
        _core.de_generate_trials(
            _population(4),
            _scores(4),
            [(-10.0, 10.0), (-10.0, 10.0)],
            ["float", "float"],
            "jde-rand1bin",
            0.7,
            0.9,
            42,
            0,
            [0],
            "maximize",
        )
```

- [ ] **Step 2: Run the new contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategy_metadata_contract.py -v
```

Expected: PASS if Python and Rust are already aligned. FAIL identifies the specific drift to correct.

---

### Task 2: Correct Any Metadata Drift Without Adding A New API

**Files:**
- Modify: `evocore/optimizers/de/strategies.py` only if Python metadata is wrong.
- Modify: `src/de.rs` only if Rust metadata is wrong.

- [ ] **Step 1: If Python minimum population is wrong, update `SUPPORTED_DE_STRATEGIES`**

Use this shape in `evocore/optimizers/de/strategies.py`:

```python
SUPPORTED_DE_STRATEGIES: dict[str, DEStrategySpec] = {
    "rand1bin": DEStrategySpec(name="rand1bin", min_population_size=4),
    "best1bin": DEStrategySpec(name="best1bin", min_population_size=4),
    "rand2bin": DEStrategySpec(name="rand2bin", min_population_size=6),
    "current-to-best1bin": DEStrategySpec(
        name="current-to-best1bin",
        min_population_size=4,
    ),
    "jde-rand1bin": DEStrategySpec(
        name="jde-rand1bin",
        min_population_size=4,
        is_adaptive=True,
        checkpoint_state_schema=1,
    ),
}
```

- [ ] **Step 2: If Rust minimum population is wrong, update `min_population(...)`**

Use this shape in `src/de.rs`:

```rust
fn min_population(strategy: &DEStrategy) -> usize {
    match strategy {
        DEStrategy::Rand2 => 6,
        _ => 4,
    }
}
```

- [ ] **Step 3: If Rust strategy parsing is wrong, update `parse_strategy(...)` and `strategy_name(...)`**

Use these canonical names:

```rust
fn parse_strategy(strategy: &str) -> PyResult<DEStrategy> {
    match strategy {
        "rand1bin" => Ok(DEStrategy::Rand1),
        "best1bin" => Ok(DEStrategy::Best1),
        "rand2bin" => Ok(DEStrategy::Rand2),
        "current-to-best1bin" => Ok(DEStrategy::CurrentToBest1),
        "jde-rand1bin" => Ok(DEStrategy::JdeRand1),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown DE strategy: {other}"
        ))),
    }
}

fn strategy_name(strategy: &DEStrategy) -> &'static str {
    match strategy {
        DEStrategy::Rand1 => "rand1bin",
        DEStrategy::Best1 => "best1bin",
        DEStrategy::Rand2 => "rand2bin",
        DEStrategy::CurrentToBest1 => "current-to-best1bin",
        DEStrategy::JdeRand1 => "jde-rand1bin",
    }
}
```

- [ ] **Step 4: Rebuild extension and rerun contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategy_metadata_contract.py tests/unit/test_de_rust_kernel.py tests/unit/test_de_strategies.py -v
```

Expected: PASS.

---

### Task 3: Final Verification And Commit

**Files:**
- All files touched in Tasks 1-2.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_strategy_metadata_contract.py tests/unit/test_de_rust_kernel.py tests/unit/test_de_strategies.py -v
```

Expected: PASS.

- [ ] **Step 2: Run Rust verification only if `src/de.rs` changed**

Run when Rust changed:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: PASS.

- [ ] **Step 3: Run Python formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both PASS.

- [ ] **Step 4: Commit task-related files only**

Run:

```powershell
git status --short
git add tests/unit/test_de_strategy_metadata_contract.py evocore/optimizers/de/strategies.py src/de.rs
git commit -m "test(de): lock python rust strategy metadata"
```

Expected: commit succeeds. If `evocore/optimizers/de/strategies.py` or `src/de.rs` did not change, leave those paths out of `git add`.
