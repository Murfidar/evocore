# GeneSpace Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GeneSpace` the canonical owner of flat search-space signature, hash, JSON export, and decoded-gene validation while keeping result reproducibility metadata aligned.

**Architecture:** Add the canonical schema and validation methods to `evocore.gene_space.GeneSpace`. Keep `evocore.stats.gene_space_signature()` and `gene_space_hash()` as thin compatibility helpers, and update GA/CMA reproducibility metadata to consume `GeneSpace` methods. `OperatorSet` keeps Rust-boundary encoding/decoding, but calls the shared `GeneSpace` validator before encoding.

**Tech Stack:** Python 3.11+, dataclasses, PyO3 extension boundary through `OperatorSet`, pytest, Hypothesis, Ruff, MkDocs.

---

## File Structure

- Modify `evocore/gene_space.py`: add canonical `signature()`, `hash()`, `to_dict()`, `to_json()`, and `validate_genes()` methods.
- Modify `evocore/stats.py`: make `gene_space_signature()` delegate to `GeneSpace.signature()` and keep `gene_space_hash()` as a hash-only helper.
- Modify `evocore/operators.py`: call `GeneSpace.validate_genes(...)` before Rust-boundary encoding.
- Modify `evocore/ga.py`: use `self.gene_space.signature()` and `self.gene_space.hash()` in reproducibility metadata.
- Modify `evocore/cmaes.py`: use `self.gene_space.signature()` and `self.gene_space.hash()` in reproducibility metadata.
- Modify `tests/unit/test_gene_space.py`: add canonical export and validation tests.
- Modify `tests/unit/test_stats.py`: update helper compatibility tests for the new signature shape.
- Modify `tests/unit/test_operators.py`: prove encoding uses the shared validator.
- Modify `tests/unit/test_ga_engine.py`: assert GA result metadata equals the canonical `GeneSpace` signature/hash.
- Modify `tests/unit/test_cmaes_engine.py`: assert CMA result metadata equals the canonical `GeneSpace` signature/hash.
- Modify `tests/property/test_gene_space_properties.py`: add JSON-safety and hash-stability properties.
- Create `docs/site/gene-space.md`: document the stabilized flat contract.
- Modify `mkdocs.yml`: add the Gene Spaces page to navigation.
- Modify `CHANGELOG.md`: note new `GeneSpace` export/validation API and updated reproducibility signature shape.

---

### Task 0: Confirm Branch And Worktree

**Files:**
- Read-only: git worktree metadata

- [ ] **Step 1: Check branch and uncommitted files**

Run:

```powershell
git status --short --branch
```

Expected: on `feature/general-optimizer-framework` or another task branch, not `main`. If unrelated uncommitted files are present, leave them untouched and only stage files listed in this plan.

---

### Task 1: Add GeneSpace Canonical Export Tests

**Files:**
- Modify: `tests/unit/test_gene_space.py`

- [ ] **Step 1: Add failing export tests**

Append this test code to `tests/unit/test_gene_space.py`:

```python
import json

from evocore.stats import gene_space_hash, gene_space_signature


def test_gene_space_signature_to_dict_hash_and_json_are_canonical():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0, sigma=0.2),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    expected = {
        "schema_version": 1,
        "genes": [
            {
                "name": "x",
                "kind": "float",
                "low": -1.0,
                "high": 1.0,
                "sigma": 0.2,
                "is_fixed": False,
            },
            {
                "name": "period",
                "kind": "int",
                "low": 2,
                "high": 20,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "enabled",
                "kind": "bool",
                "low": None,
                "high": None,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "fixed_threshold",
                "kind": "float",
                "low": 0.5,
                "high": 0.5,
                "sigma": None,
                "is_fixed": True,
            },
        ],
        "has_names": True,
        "length": 4,
    }

    assert space.signature() == expected
    assert space.to_dict() == expected
    assert gene_space_signature(space) == expected
    assert space.hash() == gene_space_hash(expected)
    assert json.loads(space.to_json()) == expected
    assert space.to_json() == space.to_json()


def test_uniform_gene_space_signature_preserves_unnamed_contract():
    space = GeneSpace.uniform(-5.0, 5.0, 2)

    assert space.signature() == {
        "schema_version": 1,
        "genes": [
            {
                "name": "gene_0",
                "kind": "float",
                "low": -5.0,
                "high": 5.0,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "gene_1",
                "kind": "float",
                "low": -5.0,
                "high": 5.0,
                "sigma": None,
                "is_fixed": False,
            },
        ],
        "has_names": False,
        "length": 2,
    }
```

- [ ] **Step 2: Run export tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_gene_space.py::test_gene_space_signature_to_dict_hash_and_json_are_canonical tests/unit/test_gene_space.py::test_uniform_gene_space_signature_preserves_unnamed_contract -v
```

Expected: FAIL because `GeneSpace.signature`, `GeneSpace.to_dict`, `GeneSpace.hash`, and `GeneSpace.to_json` do not exist yet.

---

### Task 2: Implement GeneSpace Canonical Export Methods

**Files:**
- Modify: `evocore/gene_space.py`

- [ ] **Step 1: Add export imports**

In `evocore/gene_space.py`, replace:

```python
from typing import Literal
```

with:

```python
from typing import Any, Literal
```

Then add this import after `from evocore.exceptions import ConfigurationError`:

```python
from evocore.exporting import canonical_json_hash, stable_json_dumps
```

- [ ] **Step 2: Add canonical export methods**

In `class GeneSpace`, insert this block after the `has_names` property and before `params_for(...)`:

```python
    def signature(self) -> dict[str, Any]:
        """Return the stable canonical signature for this gene space."""
        return {
            "schema_version": 1,
            "genes": [
                {
                    "name": gene.name,
                    "kind": gene.kind,
                    "low": gene.low,
                    "high": gene.high,
                    "sigma": gene.sigma,
                    "is_fixed": gene.is_fixed,
                }
                for gene in self._genes
            ],
            "has_names": self._has_names,
            "length": self.length,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export this gene space as its stable canonical signature."""
        return self.signature()

    def hash(self) -> str:
        """Return a stable SHA-256 hash for this gene-space signature."""
        return canonical_json_hash(self.signature())

    def to_json(self, *, indent: int | None = None) -> str:
        """Export this gene space as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)
```

- [ ] **Step 3: Run export tests and observe helper mismatch**

Run:

```powershell
python -m pytest tests/unit/test_gene_space.py::test_gene_space_signature_to_dict_hash_and_json_are_canonical tests/unit/test_gene_space.py::test_uniform_gene_space_signature_preserves_unnamed_contract -v
```

Expected: the direct `GeneSpace` assertions pass, and the helper assertion still fails because `gene_space_signature(space)` still builds the old payload in `evocore/stats.py`.

- [ ] **Step 4: Commit direct GeneSpace export API**

Run:

```powershell
git add evocore/gene_space.py tests/unit/test_gene_space.py
git commit -m "feat: add genespace export methods"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 3: Add And Implement GeneSpace Decoded-Gene Validation

**Files:**
- Modify: `tests/unit/test_gene_space.py`
- Modify: `evocore/gene_space.py`

- [ ] **Step 1: Add failing validation tests**

Append this test code to `tests/unit/test_gene_space.py`:

```python
def test_validate_genes_accepts_valid_decoded_values():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    assert space.validate_genes([0.25, 10, True, 0.5]) is None
    assert space.validate_genes([0, 2, False, 0.5]) is None


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.25], "GeneSpace expected 4 genes, got 1."),
        ([True, 10, True, 0.5], "Gene 'x' at index 0 expects float"),
        ([float("nan"), 10, True, 0.5], "Gene 'x' at index 0 must be finite"),
        ([2.0, 10, True, 0.5], "Gene 'x' at index 0 must be within"),
        ([0.25, True, True, 0.5], "Gene 'period' at index 1 expects int"),
        ([0.25, 10.0, True, 0.5], "Gene 'period' at index 1 expects int"),
        ([0.25, 21, True, 0.5], "Gene 'period' at index 1 must be within"),
        ([0.25, 10, 1, 0.5], "Gene 'enabled' at index 2 expects bool"),
        ([0.25, 10, True, 0.6], "Gene 'fixed_threshold' at index 3 must be within"),
    ],
)
def test_validate_genes_rejects_invalid_decoded_values(values, message):
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
            GeneDef("fixed_threshold", "float", 0.5, 0.5),
        ]
    )

    with pytest.raises(ConfigurationError, match=message):
        space.validate_genes(values)
```

- [ ] **Step 2: Run validation tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_gene_space.py::test_validate_genes_accepts_valid_decoded_values tests/unit/test_gene_space.py::test_validate_genes_rejects_invalid_decoded_values -v
```

Expected: FAIL because `GeneSpace.validate_genes` does not exist.

- [ ] **Step 3: Implement validation method**

In `class GeneSpace`, insert this block after `to_json(...)` and before `params_for(...)`:

```python
    def validate_genes(self, values: Sequence[float | int | bool]) -> None:
        """Validate decoded Python gene values against this gene space."""
        if len(values) != self.length:
            raise ConfigurationError(f"GeneSpace expected {self.length} genes, got {len(values)}.")

        for index, (value, gene) in enumerate(zip(values, self._genes, strict=False)):
            label = f"Gene {gene.name!r} at index {index}"

            if gene.kind == "bool":
                if type(value) is not bool:
                    raise ConfigurationError(
                        f"{label} expects bool, got {type(value).__name__}."
                    )
                continue

            if gene.kind == "int":
                if type(value) is not int:
                    raise ConfigurationError(
                        f"{label} expects int, got {type(value).__name__}."
                    )
                if value < gene.low or value > gene.high:
                    raise ConfigurationError(
                        f"{label} must be within [{gene.low}, {gene.high}], got {value}."
                    )
                continue

            if type(value) not in (int, float):
                raise ConfigurationError(f"{label} expects float, got {type(value).__name__}.")

            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ConfigurationError(f"{label} must be finite, got {value}.")
            if numeric_value < float(gene.low) or numeric_value > float(gene.high):
                raise ConfigurationError(
                    f"{label} must be within [{gene.low}, {gene.high}], got {value}."
                )
```

- [ ] **Step 4: Run gene-space unit tests**

Run:

```powershell
python -m pytest tests/unit/test_gene_space.py -v
```

Expected: all tests in `tests/unit/test_gene_space.py` pass except the helper assertion from Task 1 may still fail until Task 5 updates `evocore/stats.py`. If it fails only on `gene_space_signature(space)`, continue to Task 5.

- [ ] **Step 5: Commit validation method and tests**

Run:

```powershell
git add evocore/gene_space.py tests/unit/test_gene_space.py
git commit -m "feat: validate decoded genespaces"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 4: Wire OperatorSet Encoding To Shared Validation

**Files:**
- Modify: `tests/unit/test_operators.py`
- Modify: `evocore/operators.py`

- [ ] **Step 1: Add failing OperatorSet validation test**

Append this test to `tests/unit/test_operators.py`:

```python
def test_encode_genes_uses_gene_space_validator_for_invalid_decoded_values():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0),
            GeneDef("period", "int", 2, 20),
        ]
    )
    ops = OperatorSet(space, "sbx", "gaussian")

    with pytest.raises(ConfigurationError, match="Gene 'x' at index 0 expects float"):
        ops.encode_genes([True, 10])

    with pytest.raises(ConfigurationError, match="Gene 'period' at index 1 expects int"):
        ops.encode_genes([0.5, 10.0])
```

- [ ] **Step 2: Run the new OperatorSet test and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_operators.py::test_encode_genes_uses_gene_space_validator_for_invalid_decoded_values -v
```

Expected: FAIL because `OperatorSet.encode_genes(...)` currently coerces invalid values instead of using `GeneSpace.validate_genes(...)`.

- [ ] **Step 3: Update `OperatorSet.encode_genes(...)`**

In `evocore/operators.py`, replace the first length check in `encode_genes(...)`:

```python
        if len(genes) != self.gene_space.length:
            raise ConfigurationError(f"Expected {self.gene_space.length} genes, got {len(genes)}.")
```

with:

```python
        self.gene_space.validate_genes(genes)
```

Keep the existing encoding loop unchanged.

- [ ] **Step 4: Run operator tests**

Run:

```powershell
python -m pytest tests/unit/test_operators.py -v
```

Expected: all operator tests pass.

- [ ] **Step 5: Commit operator validation wiring**

Run:

```powershell
git add evocore/operators.py tests/unit/test_operators.py
git commit -m "fix: validate genes before encoding"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 5: Update Compatibility Helpers And Result Metadata Consumers

**Files:**
- Modify: `tests/unit/test_stats.py`
- Modify: `tests/unit/test_ga_engine.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `evocore/stats.py`
- Modify: `evocore/ga.py`
- Modify: `evocore/cmaes.py`

- [ ] **Step 1: Update stats helper test for canonical signature**

In `tests/unit/test_stats.py`, replace `test_gene_space_signature_preserves_gene_order_and_fields()` with:

```python
def test_gene_space_signature_preserves_gene_order_and_fields():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0, sigma=0.2),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
        ]
    )

    expected = {
        "schema_version": 1,
        "genes": [
            {
                "name": "x",
                "kind": "float",
                "low": -1.0,
                "high": 1.0,
                "sigma": 0.2,
                "is_fixed": False,
            },
            {
                "name": "period",
                "kind": "int",
                "low": 2,
                "high": 20,
                "sigma": None,
                "is_fixed": False,
            },
            {
                "name": "enabled",
                "kind": "bool",
                "low": None,
                "high": None,
                "sigma": None,
                "is_fixed": False,
            },
        ],
        "has_names": True,
        "length": 3,
    }

    assert gene_space_signature(space) == expected
    assert gene_space_signature(space) == space.signature()
```

- [ ] **Step 2: Update GA reproducibility test**

In `tests/unit/test_ga_engine.py`, inside `test_ga_vnext_run_attaches_history_and_reproducibility_metadata()`, replace the engine construction with:

```python
    space = GeneSpace.uniform(-1.0, 1.0, 2)
    engine = GAEngine(space, population_size=4, generations=2, seed=42)
```

Then replace:

```python
    assert result.reproducibility.gene_space_signature["length"] == 2
    assert result.reproducibility.gene_space_hash
```

with:

```python
    assert result.reproducibility.gene_space_signature == space.signature()
    assert result.reproducibility.gene_space_hash == space.hash()
```

- [ ] **Step 3: Update CMA reproducibility test**

In `tests/unit/test_cmaes_engine.py`, inside `test_cma_generation_loop_result_attaches_history_and_reproducibility()`, replace the engine construction with:

```python
    space = GeneSpace.uniform(-2.0, 2.0, 3)
    engine = CMAESEngine(space, population_size=6, generations=2, seed=42)
```

Then add these assertions after `assert result.reproducibility.engine_type == "CMAESEngine"`:

```python
    assert result.reproducibility.gene_space_signature == space.signature()
    assert result.reproducibility.gene_space_hash == space.hash()
```

- [ ] **Step 4: Run compatibility and result tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_gene_space_signature_preserves_gene_order_and_fields tests/unit/test_ga_engine.py::test_ga_vnext_run_attaches_history_and_reproducibility_metadata tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: FAIL because `gene_space_signature(...)`, `GAEngine`, and `CMAESEngine` still use the old helper-owned signature path.

- [ ] **Step 5: Make stats helper delegate**

In `evocore/stats.py`, replace `gene_space_signature(...)` with:

```python
def gene_space_signature(gene_space: GeneSpace) -> dict[str, Any]:
    """Return the canonical signature for a gene space."""
    return gene_space.signature()
```

Keep `gene_space_hash(...)` unchanged:

```python
def gene_space_hash(signature: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a gene-space signature."""
    return canonical_json_hash(signature)
```

- [ ] **Step 6: Update GA reproducibility metadata**

In `evocore/ga.py`, remove `gene_space_hash` and `gene_space_signature` from the `evocore.stats` import list. Keep `ReproducibilityMetadata`.

Replace `_reproducibility_metadata(...)` with:

```python
    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            engine_type="GAEngine",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
        )
```

- [ ] **Step 7: Update CMA reproducibility metadata**

In `evocore/cmaes.py`, remove `gene_space_hash` and `gene_space_signature` from the `evocore.stats` import list. Keep `ReproducibilityMetadata`.

Replace `_reproducibility_metadata(...)` with:

```python
    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = self.gene_space.signature()
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            engine_type="CMAESEngine",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=self.gene_space.hash(),
            optimizer_config=self._optimizer_config(),
        )
```

- [ ] **Step 8: Run compatibility and result tests**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_gene_space_signature_preserves_gene_order_and_fields tests/unit/test_stats.py::test_gene_space_hash_is_stable_for_equivalent_spaces tests/unit/test_ga_engine.py::test_ga_vnext_run_attaches_history_and_reproducibility_metadata tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: PASS.

- [ ] **Step 9: Commit helper and result metadata updates**

Run:

```powershell
git add evocore/stats.py evocore/ga.py evocore/cmaes.py tests/unit/test_stats.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py
git commit -m "refactor: centralize genespace signatures"
```

Expected: commit succeeds with only the six listed files staged.

---

### Task 6: Add GeneSpace Property Tests

**Files:**
- Modify: `tests/property/test_gene_space_properties.py`

- [ ] **Step 1: Add JSON and hash property tests**

At the top of `tests/property/test_gene_space_properties.py`, add:

```python
import json
```

Then append this strategy and tests:

```python
@st.composite
def valid_flat_gene_spaces(draw):
    kinds = draw(st.lists(st.sampled_from(["float", "int", "bool"]), min_size=1, max_size=8))
    genes = []
    for index, kind in enumerate(kinds):
        name = f"gene_{index}"
        if kind == "float":
            low = draw(
                st.floats(
                    min_value=-1000.0,
                    max_value=999.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            fixed = draw(st.booleans())
            if fixed:
                high = low
            else:
                span = draw(
                    st.floats(
                        min_value=1e-6,
                        max_value=1000.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                )
                high = low + span
            genes.append(GeneDef(name, "float", low, high))
        elif kind == "int":
            low = draw(st.integers(min_value=-1000, max_value=999))
            fixed = draw(st.booleans())
            high = low if fixed else draw(st.integers(min_value=low + 1, max_value=low + 1000))
            genes.append(GeneDef(name, "int", low, high))
        else:
            genes.append(GeneDef(name, "bool"))
    return GeneSpace(genes)


@given(valid_flat_gene_spaces())
def test_gene_space_signature_json_round_trips(space):
    signature = space.signature()

    assert json.loads(space.to_json()) == signature
    assert space.to_dict() == signature


@given(valid_flat_gene_spaces())
def test_gene_space_hash_is_stable_for_equivalent_flat_spaces(space):
    equivalent = GeneSpace(list(space.genes), has_names=space.has_names)

    assert equivalent.signature() == space.signature()
    assert equivalent.hash() == space.hash()
```

- [ ] **Step 2: Run property tests**

Run:

```powershell
python -m pytest tests/property/test_gene_space_properties.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit property tests**

Run:

```powershell
git add tests/property/test_gene_space_properties.py
git commit -m "test: cover genespace export properties"
```

Expected: commit succeeds with only the property test file staged.

---

### Task 7: Document GeneSpace Contract

**Files:**
- Create: `docs/site/gene-space.md`
- Modify: `mkdocs.yml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Create the Gene Spaces docs page**

Create `docs/site/gene-space.md` with this content:

````markdown
# Gene Spaces

`GeneSpace` defines the flat search-space schema shared by EvoCore optimizers.

Supported gene kinds are:

- `float`: bounded continuous values.
- `int`: bounded integer values.
- `bool`: unbounded binary values represented as Python `bool`.

```python
from evocore import GeneDef, GeneSpace

space = GeneSpace(
    [
        GeneDef("period", "int", 2, 50, sigma=0.05),
        GeneDef("threshold", "float", 0.0, 1.0),
        GeneDef("enabled", "bool"),
        GeneDef("fixed_mode", "int", 2, 2),
    ]
)
```

Numeric bounds are inclusive. Equal numeric bounds define a fixed gene that remains part
of the full genome and named parameter mapping.

`GeneSpace.uniform(low, high, length)` creates an unnamed float space:

```python
space = GeneSpace.uniform(-5.0, 5.0, 10)
assert space.has_names is False
```

Named spaces expose `params` on decoded candidates and individuals:

```python
params = space.params_for([10, 0.25, True, 2])
assert params == {
    "period": 10,
    "threshold": 0.25,
    "enabled": True,
    "fixed_mode": 2,
}
```

## Validation

`validate_genes(...)` checks decoded Python values without coercing, clamping, or mutating
them:

```python
space.validate_genes([10, 0.25, True, 2])
```

Invalid values raise `ConfigurationError`. Float genes reject booleans, non-finite values,
and out-of-bounds values. Int genes reject booleans, floats, and out-of-bounds values.
Bool genes accept only Python `bool`.

## Stable Signature

`GeneSpace` owns its reproducibility signature:

```python
signature = space.signature()
stable_hash = space.hash()
payload = space.to_dict()
json_text = space.to_json(indent=2)
```

`signature()` and `to_dict()` return the same payload. The signature includes
`schema_version`, ordered gene definitions, `has_names`, `length`, and per-gene
`is_fixed` metadata. `RunResult.reproducibility.gene_space_signature` and
`RunResult.reproducibility.gene_space_hash` use the same canonical values.

This contract is intentionally flat. Categorical, permutation, conditional, and
multi-variable spaces are not part of this slice.
````

- [ ] **Step 2: Add docs page to MkDocs navigation**

In `mkdocs.yml`, add the Gene Spaces page after Quickstart:

```yaml
  - Quickstart: quickstart.md
  - Gene Spaces: gene-space.md
  - Genetic Algorithms: ga.md
```

- [ ] **Step 3: Update changelog**

In `CHANGELOG.md`, under `[Unreleased]` `### Added`, add:

```markdown
- `GeneSpace` now owns stable `signature()`, `hash()`, `to_dict()`, `to_json()`,
  and `validate_genes(...)` helpers for the flat search-space contract.
```

Under `[Unreleased]` `### Changed`, add:

```markdown
- Run reproducibility metadata now uses the canonical `GeneSpace` signature and hash,
  including `schema_version` and per-gene `is_fixed` metadata.
```

- [ ] **Step 4: Verify docs references**

Run:

```powershell
python -m ruff format --check
python -m ruff check
```

Expected: PASS. These commands do not build MkDocs, but they confirm the Python changes remain formatted and lint-clean before final verification.

- [ ] **Step 5: Commit docs and changelog**

Run:

```powershell
git add docs/site/gene-space.md mkdocs.yml CHANGELOG.md
git commit -m "docs: document genespace contract"
```

Expected: commit succeeds with only docs/changelog files staged.

---

### Task 8: Final Verification

**Files:**
- Read-only verification over changed surfaces

- [ ] **Step 1: Run targeted unit tests**

Run:

```powershell
python -m pytest tests/unit/test_gene_space.py tests/unit/test_stats.py tests/unit/test_operators.py tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py -v
```

Expected: PASS.

- [ ] **Step 2: Run property tests**

Run:

```powershell
python -m pytest tests/property/test_gene_space_properties.py -v
```

Expected: PASS.

- [ ] **Step 3: Run Python formatting and lint checks**

Run:

```powershell
python -m ruff format --check
python -m ruff check
```

Expected: PASS.

- [ ] **Step 4: Run Rust/PyO3 verification only if Rust or stub files changed**

If implementation changed files under `src/` or `evocore/_core.pyi`, run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
python -m maturin develop --release
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS. If no Rust or PyO3 stub files changed, skip this step and record that it was skipped because the slice stayed Python/docs-only.

- [ ] **Step 5: Inspect final status**

Run:

```powershell
git status --short --branch
```

Expected: clean worktree on the task branch after all task commits.

---

## Self-Review

Spec coverage:

- `GeneSpace` owns signature/hash/dict/JSON export in Tasks 1 and 2.
- `GeneSpace` owns decoded-gene validation in Task 3.
- `OperatorSet` uses the shared validator in Task 4.
- Compatibility helpers delegate through Task 5.
- GA and CMA result metadata consume canonical `GeneSpace` methods in Task 5.
- Property coverage for JSON safety and stable hashes is in Task 6.
- Docs and changelog updates are in Task 7.
- Final verification commands are in Task 8.

Placeholder scan:

- The plan contains concrete file paths, code snippets, commands, and expected outcomes.
- No incomplete sections remain.

Type consistency:

- Method names match the approved spec: `signature`, `hash`, `to_dict`, `to_json`, and `validate_genes`.
- Compatibility helper names remain `gene_space_signature` and `gene_space_hash`.
- Reproducibility fields remain `gene_space_signature` and `gene_space_hash`.
