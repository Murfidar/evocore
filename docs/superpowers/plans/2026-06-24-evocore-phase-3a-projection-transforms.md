# EvoCore Phase 3A Projection Transforms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the named projection and transform foundation that compiles domain parameters into optimizer-native flat `GeneSpace` coordinates.

**Architecture:** Keep `GeneSpace` flat and schema version 1. Add sibling modules under `evocore/search_space/` for transforms, projection snapshots, active subspace compilation, canonical projected hashes, and portable violation/repair records used by later phases.

**Tech Stack:** Python dataclasses, protocols, EvoCore `GeneSpace`, stable JSON hashing, pytest, hypothesis, ruff, MkDocs.

---

## Dependency

- Source design: `docs/superpowers/specs/2026-06-22-evocore-phase-3-projection-cma-design.md`
- This plan must run before Phase 3B, 3C, and 3D.

## File Structure

- Create: `evocore/search_space/transforms.py`
  - `ParameterTransform`, `IdentityTransform`, `BinaryThresholdTransform`, `ExponentialIntegerTransform`, `OutputNameTransform`.
- Create: `evocore/search_space/constraints.py`
  - Minimal durable `ConstraintViolation` and `RepairRecord` records exposed by `ProjectionResult`; Phase 3B extends this file.
- Create: `evocore/search_space/projection.py`
  - `ParameterProjection`, `ProjectionSnapshot`, `ProjectionResult`, `ActiveGeneProjection`.
- Modify: `evocore/search_space/__init__.py`
  - Re-export public projection, transform, and constraint names.
- Modify: `evocore/__init__.py`
  - Re-export top-level convenience names.
- Create: `tests/unit/test_projection.py`
  - Named-space, canonical active order, transform, and hash tests.
- Create: `tests/property/test_projection_properties.py`
  - Round-trip and inactive-hash invariance properties.
- Modify: `tests/unit/test_package_init.py`
  - Public import smoke tests.

## Public API

Export these names from `evocore.search_space` and top-level `evocore`:

```python
ActiveGeneProjection
BinaryThresholdTransform
ConstraintViolation
ExponentialIntegerTransform
IdentityTransform
OutputNameTransform
ParameterProjection
ParameterTransform
ProjectionResult
ProjectionSnapshot
RepairRecord
```

## Task 1: Projection Tests

**Files:**
- Create: `tests/unit/test_projection.py`
- Create: `tests/property/test_projection_properties.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/unit/test_projection.py`:

```python
import pytest

from evocore import Gene, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ExponentialIntegerTransform,
    IdentityTransform,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("family", "int", 0, 2),
            Gene("fast", "float", 2.0, 20.0),
            Gene("slow", "float", 10.0, 80.0),
            Gene("use_filter", "float", 0.0, 1.0),
            Gene("inactive", "float", -1.0, 1.0),
        ]
    )


def test_projection_requires_named_source_space() -> None:
    with pytest.raises(ConfigurationError, match="named GeneSpace"):
        ActiveGeneProjection(
            source_space=GeneSpace.uniform(-1.0, 1.0, 2),
            active_names=["gene_0"],
        )


def test_projection_canonicalizes_active_order_and_reconstructs() -> None:
    projection = ActiveGeneProjection(
        source_space=_space(),
        active_names=["slow", "fast", "use_filter"],
        structural_bindings={"family": 1, "inactive": 0.25},
        transforms={"use_filter": BinaryThresholdTransform(threshold=0.5)},
        schema_id="template-a",
        schema_version="1",
    )

    result = projection.reconstruct([5.5, 34.0, 0.75])

    assert projection.optimizer_space.names == ["fast", "slow", "use_filter"]
    assert result.parameters == {
        "family": 1,
        "fast": 5.5,
        "slow": 34.0,
        "use_filter": True,
        "inactive": 0.25,
    }
    assert result.optimizer_values == (5.5, 34.0, 0.75)
    assert result.active_names == ("fast", "slow", "use_filter")
    assert result.repairs == ()
    assert result.violations == ()
    assert result.valid is True


def test_inactive_changes_do_not_change_projected_hash() -> None:
    left = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1, "inactive": -1.0},
        schema_id="template-a",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1, "inactive": 1.0},
        schema_id="template-a",
        schema_version="1",
    )

    values = {"family": 1, "fast": 4.0, "slow": 40.0, "inactive": 0.0}

    assert left.value_hash(values) == right.value_hash(values)


def test_structural_changes_do_change_projected_hash() -> None:
    left = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 1},
        identity_keys=("family",),
        schema_id="template-a",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast", "slow"],
        structural_bindings={"family": 2},
        identity_keys=("family",),
        schema_id="template-a",
        schema_version="1",
    )

    assert left.value_hash({"fast": 4.0, "slow": 40.0}) != right.value_hash(
        {"fast": 4.0, "slow": 40.0}
    )


def test_transform_versions_participate_in_snapshot_hash() -> None:
    first = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast"],
        transforms={"fast": IdentityTransform()},
        schema_id="template-a",
        schema_version="1",
    )
    second = ActiveGeneProjection(
        source_space=_space(),
        active_names=["fast"],
        transforms={"fast": ExponentialIntegerTransform(base=2.0)},
        schema_id="template-a",
        schema_version="1",
    )

    assert first.snapshot().signature_hash != second.snapshot().signature_hash
```

- [ ] **Step 2: Write failing property tests**

Create `tests/property/test_projection_properties.py`:

```python
from hypothesis import given, strategies as st

from evocore import Gene, GeneSpace
from evocore.search_space import ActiveGeneProjection


@given(
    fast=st.floats(min_value=2.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    slow=st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False),
)
def test_projection_round_trip_active_values(fast: float, slow: float) -> None:
    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
            ]
        ),
        active_names=["fast", "slow"],
        schema_id="round-trip",
        schema_version="1",
    )

    projected = projection.project({"fast": fast, "slow": slow})
    reconstructed = projection.reconstruct(projected.optimizer_values)

    assert reconstructed.parameters == projected.parameters
    assert reconstructed.projection_hash == projected.projection_hash


@given(
    inactive_left=st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    inactive_right=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_inactive_hash_invariance(inactive_left: float, inactive_right: float) -> None:
    space = GeneSpace(
        [
            Gene("active", "float", 0.0, 1.0),
            Gene("inactive", "float", -1.0, 1.0),
        ]
    )
    left = ActiveGeneProjection(
        source_space=space,
        active_names=["active"],
        structural_bindings={"inactive": inactive_left},
        schema_id="hash",
        schema_version="1",
    )
    right = ActiveGeneProjection(
        source_space=space,
        active_names=["active"],
        structural_bindings={"inactive": inactive_right},
        schema_id="hash",
        schema_version="1",
    )

    assert left.value_hash({"active": 0.5}) == right.value_hash({"active": 0.5})
```

- [ ] **Step 3: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_projection.py tests/property/test_projection_properties.py -v
```

Expected: fails with import errors for `ActiveGeneProjection` and transform names.

## Task 2: Transform and Constraint Records

**Files:**
- Create: `evocore/search_space/transforms.py`
- Create: `evocore/search_space/constraints.py`

- [ ] **Step 1: Implement transform module**

Create `evocore/search_space/transforms.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.search_space.genes import GeneValue


@runtime_checkable
class ParameterTransform(Protocol):
    checkpointable: bool

    def decode(self, value: GeneValue) -> object:
        raise NotImplementedError

    def encode(self, value: object) -> GeneValue:
        raise NotImplementedError

    def signature(self) -> dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityTransform:
    checkpointable: bool = True

    def decode(self, value: GeneValue) -> object:
        return value

    def encode(self, value: object) -> GeneValue:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return float(value)
        raise ConfigurationError("IdentityTransform requires a finite bool, int, or float.")

    def signature(self) -> dict[str, object]:
        return {"type": "identity", "version": 1}


@dataclass(frozen=True)
class BinaryThresholdTransform:
    threshold: float = 0.5
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.threshold)):
            raise ConfigurationError("BinaryThresholdTransform threshold must be finite.")

    def decode(self, value: GeneValue) -> bool:
        return float(value) >= float(self.threshold)

    def encode(self, value: object) -> float:
        return 1.0 if bool(value) else 0.0

    def signature(self) -> dict[str, object]:
        return {"type": "binary_threshold", "version": 1, "threshold": float(self.threshold)}


@dataclass(frozen=True)
class ExponentialIntegerTransform:
    base: float
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.base)) or float(self.base) <= 1.0:
            raise ConfigurationError("ExponentialIntegerTransform base must be finite and > 1.")

    def decode(self, value: GeneValue) -> int:
        return max(0, int(round(float(self.base) ** float(value))))

    def encode(self, value: object) -> float:
        numeric = int(value)
        if numeric <= 0:
            raise ConfigurationError("ExponentialIntegerTransform encode requires value > 0.")
        return math.log(float(numeric), float(self.base))

    def signature(self) -> dict[str, object]:
        return {"type": "exponential_integer", "version": 1, "base": float(self.base)}


@dataclass(frozen=True)
class OutputNameTransform:
    output_name: str
    checkpointable: bool = True

    def __post_init__(self) -> None:
        if not self.output_name:
            raise ConfigurationError("OutputNameTransform output_name must be non-empty.")

    def decode(self, value: GeneValue) -> object:
        return value

    def encode(self, value: object) -> GeneValue:
        return IdentityTransform().encode(value)

    def signature(self) -> dict[str, object]:
        return {"type": "output_name", "version": 1, "output_name": self.output_name}
```

- [ ] **Step 2: Implement durable violation and repair records**

Create `evocore/search_space/constraints.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConstraintViolation:
    code: str
    message: str
    names: tuple[str, ...] = ()
    hook_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairRecord:
    name: str
    previous: object
    repaired: object
    reason: str
    hook_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
```

## Task 3: Active Projection Implementation

**Files:**
- Create: `evocore/search_space/projection.py`
- Modify: `evocore/search_space/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Implement projection dataclasses and protocol**

Create `evocore/search_space/projection.py` with these public records and a `ParameterProjection` protocol:

```python
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import canonical_json_hash, json_safe
from evocore.search_space.constraints import ConstraintViolation, RepairRecord
from evocore.search_space.genes import Gene, GeneSpace, GeneValue
from evocore.search_space.transforms import IdentityTransform, ParameterTransform


@dataclass(frozen=True)
class ProjectionSnapshot:
    schema_version: int
    schema_id: str
    user_schema_version: str
    source_space_signature: Mapping[str, object]
    source_space_hash: str
    optimizer_space_signature: Mapping[str, object]
    optimizer_space_hash: str
    active_names: tuple[str, ...]
    structural_bindings: Mapping[str, object]
    transform_signatures: Mapping[str, Mapping[str, object]]
    identity_keys: tuple[str, ...]
    checkpointable: bool
    signature_hash: str


@dataclass(frozen=True)
class ProjectionResult:
    parameters: Mapping[str, object]
    optimizer_values: tuple[GeneValue, ...]
    active_names: tuple[str, ...]
    structural_bindings: Mapping[str, object]
    repairs: tuple[RepairRecord, ...] = ()
    violations: tuple[ConstraintViolation, ...] = ()
    projection_hash: str = ""
    parameter_hash: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    valid: bool = True
    checkpointable: bool = True


@runtime_checkable
class ParameterProjection(Protocol):
    optimizer_space: GeneSpace
    checkpointable: bool

    def project(self, parameters: Mapping[str, object]) -> ProjectionResult:
        raise NotImplementedError

    def reconstruct(self, values: Sequence[GeneValue]) -> ProjectionResult:
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        raise NotImplementedError

    def snapshot(self) -> ProjectionSnapshot:
        raise NotImplementedError

    def value_hash(self, parameters: Mapping[str, object]) -> str:
        raise NotImplementedError
```

- [ ] **Step 2: Implement `ActiveGeneProjection`**

In `evocore/search_space/projection.py`, add `ActiveGeneProjection` with these exact behaviors:

- `source_space.has_names` must be true, otherwise raise `ConfigurationError("ActiveGeneProjection requires a named GeneSpace.")`.
- Active names must be non-empty, unique, known source names.
- Canonical active order must follow `source_space.names`, not caller order.
- `optimizer_space` must be a new `GeneSpace` built from active source `Gene` objects.
- `project(parameters)` must encode active domain values through transforms, validate via `optimizer_space.validate_genes`, then call `reconstruct()` and return the reconstructed result.
- `reconstruct(values)` must validate vector length and `optimizer_space.validate_genes`, decode active values, merge JSON-safe structural bindings, compute `parameter_hash` from complete parameters, and compute `projection_hash` from projection signature plus active values and configured identity keys.
- `snapshot()` must return detached JSON-safe data with a stable `signature_hash`.
- Runtime-only transforms are allowed only when `checkpointable=False`; `snapshot()` must raise `ConfigurationError` if any transform lacks stable checkpoint identity.

- [ ] **Step 3: Export public names**

Update `evocore/search_space/__init__.py` and `evocore/__init__.py` to export the Public API names listed above.

Append this test to `tests/unit/test_package_init.py`:

```python
def test_phase3a_projection_public_exports():
    from evocore import ActiveGeneProjection, BinaryThresholdTransform, ConstraintViolation

    assert ActiveGeneProjection is not None
    assert BinaryThresholdTransform is not None
    assert ConstraintViolation is not None
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_projection.py tests/property/test_projection_properties.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Phase 3A**

Run:

```powershell
git add evocore/search_space/transforms.py evocore/search_space/constraints.py evocore/search_space/projection.py evocore/search_space/__init__.py evocore/__init__.py tests/unit/test_projection.py tests/property/test_projection_properties.py tests/unit/test_package_init.py
git commit -m "feat(search-space): add active parameter projection"
```

## Verification

- [ ] **Step 1: Run format and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both commands pass.

- [ ] **Step 2: Run docs build**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: strict docs build passes. Existing repo warnings unrelated to this phase may still appear.

## Self-Review Notes

- Spec coverage: projection protocol, active subspace, transform signatures, detached snapshots, stable projected identity, and named parameter identity are covered.
- Compatibility: flat `GeneSpace` behavior and schema version 1 remain unchanged.
- Downstream dependency: Phase 3B extends `constraints.py`; Phase 3C consumes `ParameterProjection` and `optimizer_space`.
