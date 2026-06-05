# Search-Space Codec Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add neutral Python search-space encode, decode, and repair helpers, then make operator, DE, and CMA-ES call sites delegate to them without changing public optimizer behavior.

**Architecture:** `evocore/search_space/codec.py` becomes the canonical Python owner of decoded-value repair and Rust-boundary encoding/decoding. `OperatorCodec` remains the compatibility facade, `apply_bounds_policy` delegates through a local import to avoid an operator/codec import cycle, and CMA-ES continues storing continuous samples separately for Rust `tell(...)`.

**Tech Stack:** Python 3.11+, pytest, ruff, maturin-built PyO3 extension already present in `.venv`.

---

## File Structure

- Modify: `evocore/search_space/codec.py`
  - Add module-level helper functions and make `OperatorCodec` delegate to them.
- Modify: `evocore/search_space/__init__.py`
  - Re-export the new helpers from the search-space package surface.
- Modify: `evocore/optimizers/operators.py`
  - Make clamp bounds policy delegate to `repair_gene_values(...)`.
- Modify: `evocore/optimizers/de/ask_tell.py`
  - Remove `_decode_de_values(...)`; use shared encode/decode helpers.
- Modify: `evocore/optimizers/cmaes/engine.py`
  - Use shared repair semantics for user-facing encoded samples while preserving continuous samples.
- Create: `tests/unit/test_search_space_codec.py`
  - Covers the new helper contract directly.
- Modify: `tests/unit/test_operators.py`
  - Verifies `OperatorCodec` delegates to repaired decode semantics.
- Modify: `tests/unit/test_operator_contract.py`
  - Keeps bounds-policy clamp behavior covered after delegation.
- Modify: `tests/unit/test_de_ask_tell.py`
  - Verifies DE decodes Rust proposal outputs through the shared codec.
- Modify: `tests/unit/test_cmaes_engine.py`
  - Verifies CMA-ES user-facing repair still returns encoded repaired samples.

---

### Task 1: Add Direct Shared Codec Tests

**Files:**
- Create: `tests/unit/test_search_space_codec.py`

- [ ] **Step 1: Write failing tests for the helper contract**

Create `tests/unit/test_search_space_codec.py` with:

```python
from __future__ import annotations

import pytest

from evocore import ConfigurationError, Gene, GeneSpace
from evocore.search_space import (
    decode_gene_values,
    encode_gene_values,
    repair_gene_value,
    repair_gene_values,
)


def _mixed_space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
            Gene("fixed", "float", 0.5, 0.5),
        ]
    )


def test_repair_gene_value_clamps_rounds_thresholds_and_preserves_types() -> None:
    space = _mixed_space()

    assert repair_gene_value(99.0, space.genes[0]) == pytest.approx(1.0)
    assert repair_gene_value(20.8, space.genes[1]) == 20
    assert repair_gene_value(0.49, space.genes[2]) is False
    assert repair_gene_value(0.5, space.genes[2]) is True
    assert repair_gene_value(False, space.genes[2]) is False
    assert repair_gene_value(99.0, space.genes[3]) == pytest.approx(0.5)


def test_repair_gene_values_validates_length_and_repaired_values() -> None:
    space = _mixed_space()

    assert repair_gene_values(space, [99.0, 20.8, 0.2, -9.0]) == [
        1.0,
        20,
        False,
        0.5,
    ]

    with pytest.raises(ConfigurationError, match="expected 4 genes, got 3"):
        repair_gene_values(space, [0.0, 3, True])


def test_decode_gene_values_repairs_encoded_numeric_vectors() -> None:
    space = _mixed_space()

    assert decode_gene_values(space, [-9.0, 1.2, 0.8, 99.0]) == [
        -1.0,
        2,
        True,
        0.5,
    ]


def test_encode_gene_values_validates_decoded_values_before_encoding() -> None:
    space = _mixed_space()

    assert encode_gene_values(space, [0.25, 7, True, 0.5]) == [0.25, 7.0, 1.0, 0.5]

    with pytest.raises(ConfigurationError, match="Gene 'period' at index 1 expects int"):
        encode_gene_values(space, [0.25, 7.0, True, 0.5])


def test_repair_gene_value_rejects_incompatible_inputs() -> None:
    space = _mixed_space()

    with pytest.raises(ConfigurationError, match="expects numeric-compatible value"):
        repair_gene_value("bad", space.genes[0])

    with pytest.raises(ConfigurationError, match="expects bool-compatible value"):
        repair_gene_value("bad", space.genes[2])
```

- [ ] **Step 2: Run the new tests and verify they fail because helpers are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_search_space_codec.py -v
```

Expected: FAIL during import with missing `decode_gene_values`, `encode_gene_values`, `repair_gene_value`, or `repair_gene_values`.

---

### Task 2: Implement Shared Python Codec Helpers

**Files:**
- Modify: `evocore/search_space/codec.py`
- Modify: `evocore/search_space/__init__.py`

- [ ] **Step 1: Add imports and helper functions**

In `evocore/search_space/codec.py`, update the search-space imports and add helpers above `class OperatorCodec`:

```python
from evocore.search_space.genes import Gene, GeneSpace
from evocore.search_space.solutions import GeneValue, Solution


def _validate_gene_count(gene_space: GeneSpace, values: Sequence[object], *, label: str) -> None:
    if len(values) != gene_space.length:
        raise ConfigurationError(f"{label} expected {gene_space.length} genes, got {len(values)}.")


def repair_gene_value(value: object, gene: Gene) -> GeneValue:
    """Repair one decoded or encoded value according to a gene definition."""
    if gene.kind == "bool":
        if type(value) is bool:
            return value
        if isinstance(value, int | float) and type(value) is not bool:
            return float(value) >= 0.5
        raise ConfigurationError(
            f"Gene {gene.name!r} expects bool-compatible value, got {type(value).__name__}."
        )

    if not isinstance(value, int | float) or type(value) is bool:
        raise ConfigurationError(
            f"Gene {gene.name!r} expects numeric-compatible value, got {type(value).__name__}."
        )

    low = float(gene.low)
    high = float(gene.high)
    if gene.kind == "int":
        rounded = float(round(float(value)))
        return int(min(max(rounded, low), high))
    return float(min(max(float(value), low), high))


def repair_gene_values(gene_space: GeneSpace, values: Sequence[object]) -> list[GeneValue]:
    """Repair a full gene vector and validate it against the gene space."""
    _validate_gene_count(gene_space, values, label="Gene repair")
    repaired = [
        repair_gene_value(value, gene)
        for value, gene in zip(values, gene_space.genes, strict=False)
    ]
    gene_space.validate_genes(repaired)
    return repaired


def encode_gene_values(gene_space: GeneSpace, values: Sequence[GeneValue]) -> list[float]:
    """Encode validated Python gene values into Rust/operator floats."""
    gene_space.validate_genes(values)
    encoded: list[float] = []
    for value, gene in zip(values, gene_space.genes, strict=False):
        if gene.kind == "bool":
            encoded.append(1.0 if bool(value) else 0.0)
        elif gene.kind == "int":
            encoded.append(float(int(value)))
        else:
            encoded.append(float(value))
    return encoded


def decode_gene_values(gene_space: GeneSpace, encoded: Sequence[float]) -> list[GeneValue]:
    """Decode and repair Rust/operator floats into Python gene values."""
    return repair_gene_values(gene_space, encoded)
```

- [ ] **Step 2: Re-export helpers from the package entrance**

In `evocore/search_space/__init__.py`, update imports and `__all__`:

```python
from evocore.search_space.codec import (
    OperatorCodec,
    decode_gene_values,
    encode_gene_values,
    repair_gene_value,
    repair_gene_values,
)
```

Add these names to `__all__`:

```python
    "decode_gene_values",
    "encode_gene_values",
    "repair_gene_value",
    "repair_gene_values",
```

- [ ] **Step 3: Run direct helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_search_space_codec.py -v
```

Expected: PASS.

---

### Task 3: Delegate Existing Python Codec And Bounds Call Sites

**Files:**
- Modify: `evocore/search_space/codec.py`
- Modify: `evocore/optimizers/operators.py`
- Modify: `tests/unit/test_operators.py`
- Modify: `tests/unit/test_operator_contract.py`

- [ ] **Step 1: Make `OperatorCodec` delegate encode/decode**

Replace `OperatorCodec.encode_values(...)` with:

```python
    def encode_values(self, values: Sequence[GeneValue]) -> list[float]:
        """Encode Python values into the float vector used by Rust."""
        return encode_gene_values(self.gene_space, values)
```

Replace `OperatorCodec.decode_values(...)` with:

```python
    def decode_values(self, genes_f64: Sequence[float]) -> list[GeneValue]:
        """Decode Rust float vectors back into Python gene values."""
        return decode_gene_values(self.gene_space, genes_f64)
```

- [ ] **Step 2: Make clamp bounds policy delegate with a local import**

In `evocore/optimizers/operators.py`, replace the body after the unsupported-policy check with:

```python
    if bounds_policy.name != "clamp":
        raise ConfigurationError(f"Unsupported bounds policy: {bounds_policy.name!r}.")

    from evocore.search_space.codec import repair_gene_values

    return repair_gene_values(gene_space, values)
```

Use a local import so `operators.py` does not create a top-level cycle with `codec.py`.

- [ ] **Step 3: Add an `OperatorCodec` repaired-decode assertion**

In `tests/unit/test_operators.py`, add:

```python
def test_operator_codec_decode_values_repairs_encoded_values():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("flag", "bool"),
            Gene("fixed", "float", 0.5, 0.5),
        ]
    )
    ops = OperatorCodec(space, "uniform", "gaussian")

    assert ops.decode_values([99.0, 20.8, 0.2, -9.0]) == [1.0, 20, False, 0.5]
```

- [ ] **Step 4: Run operator-focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_search_space_codec.py tests/unit/test_operators.py tests/unit/test_operator_contract.py -v
```

Expected: PASS.

---

### Task 4: Move DE To Shared Encode/Decode Helpers

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `tests/unit/test_de_ask_tell.py`

- [ ] **Step 1: Update DE imports**

In `evocore/optimizers/de/ask_tell.py`, remove the now-unused `Sequence` import only if it becomes unused after edits. Add shared helper imports:

```python
from evocore.search_space import Solution, decode_gene_values, encode_gene_values
```

- [ ] **Step 2: Remove `_decode_de_values(...)` and use shared helpers**

Delete the private `_decode_de_values(...)` function.

In `_initial_candidates(...)`, replace:

```python
_decode_de_values(self.gene_space, encoded),
```

with:

```python
decode_gene_values(self.gene_space, encoded),
```

In `_rust_trial_proposals(...)`, replace the manual `population_encoded` list comprehension with:

```python
        population_encoded = [
            encode_gene_values(self.gene_space, candidate.genes)
            for candidate in target_population
        ]
```

In the Rust proposal loop, replace:

```python
genes = _decode_de_values(self.gene_space, raw["genes"])
```

with:

```python
genes = decode_gene_values(self.gene_space, raw["genes"])
```

- [ ] **Step 3: Strengthen the DE Rust-output decode test**

In `tests/unit/test_de_ask_tell.py`, in `test_de_trial_ask_uses_rust_kernel_output`, change the fake Rust genes to repaired edge values:

```python
"genes": [99.0, 20.8, 0.2, -9.0],
```

Update the expected trial genes to:

```python
    assert [trial.genes for trial in trials] == [
        [5.0, 20, False, pytest.approx(1.5)],
        [5.0, 20, False, pytest.approx(1.5)],
        [5.0, 20, False, pytest.approx(1.5)],
    ]
```

- [ ] **Step 4: Run DE ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py -v
```

Expected: PASS.

---

### Task 5: Move CMA-ES User-Facing Repair To Shared Helpers

**Files:**
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`

- [ ] **Step 1: Import shared helpers in CMA-ES engine**

In `evocore/optimizers/cmaes/engine.py`, add:

```python
from evocore.search_space import decode_gene_values, encode_gene_values
```

If `decode_gene_values` is unused, import only `encode_gene_values` and `repair_gene_values`.

- [ ] **Step 2: Replace `_apply_bounds_and_round(...)` body**

Replace `_apply_bounds_and_round(...)` with:

```python
    def _apply_bounds_and_round(self, genes_f64: Sequence[float]) -> list[float]:
        repaired = decode_gene_values(self.gene_space, genes_f64)
        return encode_gene_values(self.gene_space, repaired)
```

This keeps the method returning encoded floats for `_decode_solution(...)`, while the original continuous samples remain stored in `continuous_samples_by_id` for Rust CMA-ES `tell(...)`.

- [ ] **Step 3: Add a CMA-ES continuous-sample preservation assertion**

In `tests/unit/test_cmaes_ask_tell_vnext.py`, add:

```python
def test_cma_ask_keeps_continuous_samples_separate_from_repaired_candidate_genes() -> None:
    space = GeneSpace([Gene("period", "int", 5, 20), Gene("x", "float", -1.0, 1.0)])
    engine = CMAESOptimizer(space, population_size=6, seed=42)

    candidates = engine.ask()
    batch = engine._batches_by_id[candidates[0].batch_id]

    assert set(batch.continuous_samples_by_id) == {
        candidate.candidate_id for candidate in candidates
    }
    for candidate in candidates:
        continuous = batch.continuous_samples_by_id[candidate.candidate_id]
        assert isinstance(candidate.genes[0], int)
        assert isinstance(continuous[0], float)
```

- [ ] **Step 4: Run CMA-ES tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: PASS.

---

### Task 6: Final Verification And Commit

**Files:**
- All files touched in Tasks 1-5.

- [ ] **Step 1: Run focused Python tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_search_space_codec.py tests/unit/test_operators.py tests/unit/test_operator_contract.py tests/unit/test_de_ask_tell.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: PASS.

- [ ] **Step 2: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both PASS.

- [ ] **Step 3: Commit task-related files only**

Run:

```powershell
git status --short
git add evocore/search_space/codec.py evocore/search_space/__init__.py evocore/optimizers/operators.py evocore/optimizers/de/ask_tell.py evocore/optimizers/cmaes/engine.py tests/unit/test_search_space_codec.py tests/unit/test_operators.py tests/unit/test_operator_contract.py tests/unit/test_de_ask_tell.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py
git commit -m "refactor(search-space): centralize gene codec repair"
```

Expected: commit succeeds with only task-related files staged.
