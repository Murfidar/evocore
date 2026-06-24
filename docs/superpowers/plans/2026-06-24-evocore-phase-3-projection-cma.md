# EvoCore Phase 3 Projection CMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 3's projection, constraint-penalty, mixed-numeric CMA, restart, and external expensive-optimization recipe APIs while preserving existing flat optimizer behavior.

**Architecture:** Add projection and constraint modules beside `GeneSpace`, keep `GeneSpace` schema version 1, and make optimizers consume shared confidence and penalty semantics rather than local callbacks. CMA remains a flat numeric optimizer; projection owns active subspaces and reconstruction, while integer-margin and restart helpers extend CMA through explicit opt-in configuration.

**Tech Stack:** Python dataclasses/protocols, Rust/PyO3 CMA state where needed, EvoCore lifecycle records, stable JSON hashing, pytest, hypothesis property tests, ruff, maturin, MkDocs.

---

## Source Design

- `docs/superpowers/specs/2026-06-22-evocore-phase-3-projection-cma-design.md`

## Scope Boundary

Implement in four dependent slices:

- **3A Projection and transforms:** public projection contracts, active subspace compilation, hashes, snapshots, and portable transforms.
- **3B Constraints and penalty records:** repair/validation hooks, `constraint_penalty`, trusted/state confidence helpers, telemetry, snapshots, archives, and batch completion.
- **3C Active and mixed-numeric CMA:** projected warm starts, integer strategy config, real margin ask/tell/checkpoint behavior, and lifecycle-managed restart helpers.
- **3D Recipe and compatibility:** docs, synthetic Trading-Algo-style integration, changelog, golden/checkpoint/property regressions.

Do not add a hierarchical `GeneSpace` DSL, native categorical CMA, `OptimizationSession`, async worker runtime, or trading-specific API names.

## File Structure

- Create: `evocore/search_space/transforms.py`
  - `ParameterTransform` protocol and portable transforms.
- Create: `evocore/search_space/constraints.py`
  - repair/validation protocols, violation and repair records, penalty helper.
- Create: `evocore/search_space/projection.py`
  - `ParameterProjection`, `ActiveGeneProjection`, `ProjectionResult`, `ProjectionSnapshot`.
- Modify: `evocore/search_space/__init__.py`
  - Re-export projection, transform, and constraint names.
- Modify: `evocore/lifecycle/records.py`
  - Add `TRUSTED_CONFIDENCES`, `constraint_penalty`, and helper predicates.
- Modify: `evocore/lifecycle/ask_tell_helpers.py`, `telemetry.py`, `checkpointing.py`, `external.py`, `archives.py`, `selection.py`, `conversion.py`
  - Add penalty accounting and keep trusted APIs excluding penalties by default.
- Modify: `evocore/optimizers/ga/ask_tell.py`, `evocore/optimizers/de/ask_tell.py`, `evocore/optimizers/de/engine.py`, `evocore/optimizers/cmaes/ask_tell.py`
  - Let penalties complete batches and update optimizer state, while marking candidates eliminated.
- Create: `evocore/optimizers/cmaes/projection.py`
  - Projected warm-start helpers and mean construction.
- Modify: `evocore/optimizers/cmaes/mixed.py`
  - Integrate integer strategy helpers with deterministic margin sampling.
- Modify: `evocore/optimizers/cmaes/engine.py`, `config.py`, `ask_tell.py`, `checkpointing.py`
  - Add `integer_strategy`, `integer_min_probability`, margin sample bookkeeping, config/checkpoint identity.
- Create: `evocore/optimizers/cmaes/restarts.py`
  - Restart policy, decision, and fresh optimizer factory.
- Modify: `evocore/optimizers/cmaes/__init__.py`, `evocore/lifecycle/__init__.py`, `evocore/__init__.py`
  - Re-export public names.
- Modify if Rust state access is required: `src/`, `evocore/_core.pyi`
  - Expose only minimal CMA coordinate statistics required by integer-margin sampling.
- Tests:
  - Create `tests/unit/test_projection.py`
  - Create `tests/property/test_projection_properties.py`
  - Create `tests/unit/test_constraints_penalty.py`
  - Modify `tests/unit/test_vnext_evaluation.py`
  - Modify `tests/unit/test_ask_tell_checkpointing.py`
  - Create `tests/unit/test_cmaes_projection.py`
  - Modify `tests/unit/test_cmaes_engine.py`, `tests/unit/test_cmaes_ask_tell_vnext.py`, `tests/unit/test_cmaes_ask_tell_checkpointing.py`, `tests/unit/test_mixed_cma_vnext.py`, `tests/unit/test_optimizer_config.py`
  - Create `tests/property/test_integer_margin_properties.py`
  - Create `tests/unit/test_cmaes_restarts.py`
  - Create `tests/integration/test_phase3_expensive_projection_recipe.py`
  - Modify `tests/unit/test_package_init.py`
- Docs:
  - Modify `docs/site/api.md`
  - Modify `docs/site/mixed-variable-search.md`
  - Modify `docs/site/expensive-external-evaluations.md`
  - Modify `CHANGELOG.md`

## Public API Shape

Use these names unless implementation discovers a concrete conflict:

```python
from evocore.search_space import (
    ActiveGeneProjection,
    BinaryThresholdTransform,
    ConstraintViolation,
    ExponentialIntegerTransform,
    IdentityTransform,
    OutputNameTransform,
    ParameterProjection,
    ParameterRepair,
    ParameterTransform,
    ParameterValidator,
    ProjectionResult,
    ProjectionSnapshot,
    RepairRecord,
)
from evocore.lifecycle import (
    STATE_UPDATE_CONFIDENCES,
    TRUSTED_CONFIDENCES,
    constraint_penalty_record,
    is_state_update_confidence,
    is_trusted_confidence,
)
from evocore.optimizers.cmaes import (
    CMAESRestartDecision,
    CMAESRestartPolicy,
    FixedCMAESRestartPolicy,
    IPOPCMAESRestartPolicy,
    ProjectedWarmStartResult,
    build_projected_cma_mean,
    create_cmaes_restart,
)
```

## Task 1: Phase 3A Projection Tests

**Files:**
- Create: `tests/unit/test_projection.py`
- Create: `tests/property/test_projection_properties.py`

- [ ] **Step 1: Write failing projection unit tests**

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
        ActiveGeneProjection(source_space=GeneSpace.uniform(-1.0, 1.0, 2), active_names=["gene_0"])


def test_projection_canonicalizes_active_order_and_reconstructs() -> None:
    projection = ActiveGeneProjection(
        source_space=_space(),
        active_names=["slow", "fast", "use_filter"],
        structural_bindings={"family": 1, "inactive": 0.25},
        transforms={"use_filter": BinaryThresholdTransform(threshold=0.5)},
        schema_id="template-a",
        schema_version="1",
    )

    assert projection.optimizer_space.names == ["fast", "slow", "use_filter"]
    result = projection.reconstruct([5.5, 34.0, 0.75])

    assert result.parameters == {
        "family": 1,
        "fast": 5.5,
        "slow": 34.0,
        "use_filter": True,
        "inactive": 0.25,
    }
    assert result.optimizer_values == (5.5, 34.0, 0.75)
    assert result.active_names == ("fast", "slow", "use_filter")
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

- [ ] **Step 2: Write failing projection property tests**

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
        source_space=GeneSpace([Gene("fast", "float", 2.0, 20.0), Gene("slow", "float", 10.0, 80.0)]),
        active_names=["fast", "slow"],
        schema_id="round-trip",
        schema_version="1",
    )

    projected = projection.project({"fast": fast, "slow": slow})
    reconstructed = projection.reconstruct(projected.optimizer_values)

    assert reconstructed.parameters == projected.parameters
    assert reconstructed.projection_hash == projected.projection_hash
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_projection.py tests/property/test_projection_properties.py -v
```

Expected: import failures for projection and transform names.

## Task 2: Phase 3A Projection Implementation

**Files:**
- Create: `evocore/search_space/transforms.py`
- Create: `evocore/search_space/constraints.py`
- Create: `evocore/search_space/projection.py`
- Modify: `evocore/search_space/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Implement portable transforms**

Create `evocore/search_space/transforms.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from evocore.core.errors import ConfigurationError
from evocore.search_space.genes import GeneValue


@runtime_checkable
class ParameterTransform(Protocol):
    checkpointable: bool

    def decode(self, value: GeneValue) -> object: ...
    def encode(self, value: object) -> GeneValue: ...
    def signature(self) -> dict[str, object]: ...


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

- [ ] **Step 2: Add minimal projection result support records**

Create `evocore/search_space/constraints.py` with the durable data records that `ProjectionResult` exposes. Task 4 extends this module with hook protocols and penalty helpers.

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

- [ ] **Step 3: Implement projection dataclasses and active projection**

Create `evocore/search_space/projection.py` with frozen `ProjectionSnapshot` and `ProjectionResult`, then `ActiveGeneProjection`. Required behavior:

```python
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
    repairs: tuple[RepairRecord, ...]
    violations: tuple[ConstraintViolation, ...]
    projection_hash: str
    parameter_hash: str
    metadata: Mapping[str, object]
    valid: bool
    checkpointable: bool
```

`ActiveGeneProjection.__init__` must:

- require `source_space.has_names`;
- reject duplicate, unknown, or empty active names;
- sort active names by `source_space.names`;
- build `optimizer_space` from the active source genes;
- JSON-sanitize structural bindings;
- default transforms to `IdentityTransform`;
- include `schema_id`, `schema_version`, active names, structural bindings for `identity_keys`, transform signatures, source hash, and optimizer hash in `snapshot().signature_hash`.

`project(parameters)` must encode active values through transforms, validate optimizer values with `optimizer_space.validate_genes`, reconstruct canonical parameters, and return detached mappings.

`reconstruct(values)` must validate the optimizer vector, decode transforms, merge structural bindings and inactive base values, compute hashes, and return `ProjectionResult`.

- [ ] **Step 4: Export public names**

Add projection and transform names to:

```text
evocore/search_space/__init__.py
evocore/__init__.py
tests/unit/test_package_init.py
```

Use this export smoke test:

```python
def test_phase3_projection_exports():
    from evocore import ActiveGeneProjection, BinaryThresholdTransform, ConstraintViolation, ProjectionResult

    assert ActiveGeneProjection is not None
    assert BinaryThresholdTransform is not None
    assert ConstraintViolation is not None
    assert ProjectionResult is not None
```

- [ ] **Step 5: Run Phase 3A tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_projection.py tests/property/test_projection_properties.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Phase 3A**

Run:

```powershell
git add evocore/search_space/transforms.py evocore/search_space/constraints.py evocore/search_space/projection.py evocore/search_space/__init__.py evocore/__init__.py tests/unit/test_projection.py tests/property/test_projection_properties.py tests/unit/test_package_init.py
git commit -m "feat(search-space): add active parameter projection"
```

## Task 3: Phase 3B Constraint and Penalty Tests

**Files:**
- Create: `tests/unit/test_constraints_penalty.py`
- Modify: `tests/unit/test_vnext_evaluation.py`
- Modify: `tests/unit/test_ask_tell_checkpointing.py`

- [ ] **Step 1: Write failing constraint helper tests**

Create `tests/unit/test_constraints_penalty.py`:

```python
from evocore import Candidate, EvaluationRecord, Gene, GeneSpace
from evocore.lifecycle import (
    STATE_UPDATE_CONFIDENCES,
    TRUSTED_CONFIDENCES,
    constraint_penalty_record,
    is_state_update_confidence,
    is_trusted_confidence,
)
from evocore.search_space import ActiveGeneProjection, ConstraintViolation


def test_constraint_penalty_is_state_update_but_not_trusted() -> None:
    assert "constraint_penalty" in STATE_UPDATE_CONFIDENCES
    assert "constraint_penalty" not in TRUSTED_CONFIDENCES
    assert is_state_update_confidence("constraint_penalty")
    assert not is_trusted_confidence("constraint_penalty")


def test_constraint_penalty_record_has_finite_score_zero_cost_and_metadata() -> None:
    candidate = Candidate("c-1", [2.0], batch_id="b-1")
    record = constraint_penalty_record(
        candidate=candidate,
        stage="projection",
        direction="maximize",
        violations=[ConstraintViolation(code="bad", message="invalid", names=("x",))],
    )

    assert isinstance(record, EvaluationRecord)
    assert record.confidence == "constraint_penalty"
    assert record.score is not None and record.score < 0.0
    assert record.cost == 0.0
    assert record.metadata["constraint_violations"][0]["code"] == "bad"


def test_penalty_candidate_status_is_eliminated() -> None:
    candidate = Candidate("c-1", [2.0], batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=-1.0e300,
            confidence="constraint_penalty",
            stage="projection",
        )
    )

    assert candidate.status == "eliminated"
    assert candidate.best_state_score("maximize") == -1.0e300


def test_constraint_validator_metadata_round_trips_through_projection() -> None:
    def validate(params):
        if params["fast"] >= params["slow"]:
            return [ConstraintViolation(code="ordering", message="fast must be below slow", names=("fast", "slow"))]
        return []

    projection = ActiveGeneProjection(
        source_space=GeneSpace([Gene("fast", "float", 1.0, 20.0), Gene("slow", "float", 1.0, 40.0)]),
        active_names=["fast", "slow"],
        validators=[validate],
        schema_id="constraints",
        schema_version="1",
    )

    result = projection.reconstruct([10.0, 5.0])

    assert result.valid is False
    assert result.violations[0].code == "ordering"
```

- [ ] **Step 2: Add ask/tell penalty completion tests for GA, DE, and CMA**

Add one test per optimizer that asks a full batch, returns `constraint_penalty` for every candidate, and asserts:

```python
assert update.state_accepted_count == len(candidates)
assert update.trusted_count == 0
assert update.consumed_batch_ids == (candidates[0].batch_id,)
assert optimizer.candidate_snapshot(scope="trusted").candidates == ()
```

Use existing optimizer fixtures in `tests/unit/test_ask_tell_checkpointing.py`, `tests/unit/test_de_ask_tell_vnext.py`, and `tests/unit/test_cmaes_ask_tell_vnext.py`.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_constraints_penalty.py tests/unit/test_vnext_evaluation.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: failures because `constraint_penalty` and constraint types are missing.

## Task 4: Phase 3B Constraint and Penalty Implementation

**Files:**
- Modify: `evocore/search_space/constraints.py`
- Modify: `evocore/search_space/projection.py`
- Modify: `evocore/search_space/__init__.py`
- Modify: `evocore/lifecycle/records.py`
- Modify: `evocore/lifecycle/ask_tell_helpers.py`
- Modify: `evocore/lifecycle/telemetry.py`
- Modify: `evocore/lifecycle/checkpointing.py`
- Modify: `evocore/lifecycle/external.py`
- Modify: `evocore/lifecycle/archives.py`
- Modify: `evocore/lifecycle/selection.py`
- Modify: `evocore/lifecycle/conversion.py`
- Modify: GA/DE/CMA ask/tell files listed in File Structure.

- [ ] **Step 1: Extend constraint dataclasses with repair and validation protocols**

Update `evocore/search_space/constraints.py` so it keeps the existing `ConstraintViolation` and `RepairRecord` records from Task 2 and adds hook protocols only. Keep lifecycle record construction out of `search_space` to avoid a package import cycle.

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class ParameterRepair(Protocol):
    checkpointable: bool
    def repair(self, parameters: Mapping[str, object]) -> tuple[Mapping[str, object], Sequence[RepairRecord]]: ...
    def signature(self) -> Mapping[str, object]: ...


@runtime_checkable
class ParameterValidator(Protocol):
    checkpointable: bool
    def validate(self, parameters: Mapping[str, object]) -> Sequence[ConstraintViolation]: ...
    def signature(self) -> Mapping[str, object]: ...
```

- [ ] **Step 2: Update confidence semantics**

In `evocore/lifecycle/records.py`:

```python
from collections.abc import Mapping, Sequence

from evocore.core.serialization import json_safe
from evocore.search_space.constraints import ConstraintViolation

EvaluationConfidence = Literal[
    "surrogate",
    "partial",
    "cached",
    "trusted_full",
    "constraint_penalty",
    "rejected",
]
TRUSTED_CONFIDENCES: tuple[EvaluationConfidence, ...] = ("trusted_full", "cached")
STATE_UPDATE_CONFIDENCES: tuple[EvaluationConfidence, ...] = (
    "trusted_full",
    "cached",
    "constraint_penalty",
)


def is_trusted_confidence(confidence: EvaluationConfidence) -> bool:
    return confidence in TRUSTED_CONFIDENCES


def constraint_penalty_record(
    *,
    candidate: Candidate,
    stage: str,
    direction: Direction,
    violations: Sequence[ConstraintViolation],
    penalty_score: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> EvaluationRecord:
    score = float(penalty_score) if penalty_score is not None else (
        -1.0e300 if direction == "maximize" else 1.0e300
    )
    payload = dict(metadata or {})
    payload["constraint_violations"] = [
        {
            "code": violation.code,
            "message": violation.message,
            "names": list(violation.names),
            "hook_id": violation.hook_id,
            "metadata": json_safe(dict(violation.metadata)),
        }
        for violation in violations
    ]
    safe_payload = json_safe(payload)
    if not isinstance(safe_payload, dict):
        raise FitnessError("constraint penalty metadata must be JSON-safe.")
    return EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="constraint_penalty",
        stage=stage,
        cost=0.0,
        metadata=safe_payload,
    )
```

Update `EvaluationStage`, `EvaluationRecord`, `ScoreObservation`, checkpoint literal validators, and docs/examples to accept `constraint_penalty`. In `Candidate.apply_record`, keep penalty state-eligible but set `status="eliminated"`.

- [ ] **Step 3: Update telemetry and UpdateResult**

Add `candidates_constraint_penalized` and `penalty_count` to telemetry/update summaries. `record_evaluation_telemetry()` should return `"constraint_penalty"` and call `telemetry.record_constraint_penalty(1, stage=record.stage)` for penalty records.

- [ ] **Step 4: Keep trusted APIs excluding penalties**

Replace raw `("trusted_full", "cached")` checks with `TRUSTED_CONFIDENCES` in external-state snapshots, top-k defaults, archives, warm starts, and selection promotion helpers. Keep batch completion and optimizer state update checks using `is_state_update_confidence`.

- [ ] **Step 5: Run Phase 3B focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_constraints_penalty.py tests/unit/test_vnext_evaluation.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: selected tests pass.

- [ ] **Step 6: Commit Phase 3B**

Run:

```powershell
git add evocore tests
git commit -m "feat(lifecycle): add constraint penalty semantics"
```

## Task 5: Phase 3C Projected CMA Warm Starts

**Files:**
- Create: `evocore/optimizers/cmaes/projection.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`
- Modify: `evocore/__init__.py`
- Create: `tests/unit/test_cmaes_projection.py`

- [ ] **Step 1: Write failing projected warm-start tests**

Create `tests/unit/test_cmaes_projection.py`:

```python
import pytest

from evocore import Gene, GeneSpace, WarmStartRecord
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import build_projected_cma_mean
from evocore.search_space import ActiveGeneProjection, BinaryThresholdTransform


def _projection() -> ActiveGeneProjection:
    return ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": BinaryThresholdTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )


def test_build_projected_cma_mean_from_best_matching_record() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[
            WarmStartRecord(params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True}, score=12.0),
            WarmStartRecord(params={"template": 1, "fast": 8.0, "slow": 50.0, "flag": False}, score=10.0),
        ],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean == [5.0, 40.0, 1.0]
    assert result.accepted_count == 2


def test_projected_mean_rejects_template_mismatch() -> None:
    result = build_projected_cma_mean(
        projection=_projection(),
        records=[WarmStartRecord(params={"template": 2, "fast": 5.0, "slow": 40.0, "flag": True}, score=12.0)],
        direction="maximize",
        strategy="best",
    )

    assert result.initial_mean is None
    assert result.rejected[0]["reason"] == "projection_mismatch"


def test_projected_mean_reports_non_invertible_transform() -> None:
    class DecodeOnlyTransform:
        checkpointable = True

        def decode(self, value):
            return bool(value)

        def encode(self, value):
            raise ConfigurationError("DecodeOnlyTransform cannot encode historical values.")

        def signature(self):
            return {"type": "decode_only", "version": 1}

    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("template", "int", 0, 2),
                Gene("fast", "float", 2.0, 20.0),
                Gene("slow", "float", 10.0, 80.0),
                Gene("flag", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast", "slow", "flag"],
        structural_bindings={"template": 1},
        transforms={"flag": DecodeOnlyTransform()},
        identity_keys=("template",),
        schema_id="template-1",
        schema_version="1",
    )

    with pytest.raises(ConfigurationError, match="encode"):
        build_projected_cma_mean(
            projection=projection,
            records=[WarmStartRecord(params={"template": 1, "fast": 5.0, "slow": 40.0, "flag": True}, score=12.0)],
            direction="maximize",
            strategy="best",
        )
```

If frozen transforms prevent monkeypatching, define a local transform class with `decode()` only and assert the same error.

- [ ] **Step 2: Implement projected mean helper**

Create `evocore/optimizers/cmaes/projection.py` with:

```python
@dataclass(frozen=True)
class ProjectedWarmStartResult:
    initial_mean: list[float] | None
    accepted_count: int
    rejected: tuple[Mapping[str, object], ...]
    source_candidate_hashes: tuple[str, ...]
    metadata: Mapping[str, object]


def build_projected_cma_mean(
    *,
    projection: ParameterProjection,
    records: Sequence[WarmStartRecord],
    direction: Direction,
    strategy: Literal["best", "top_k_centroid"] = "best",
    top_k: int | None = None,
) -> ProjectedWarmStartResult:
    ranked = []
    rejected = []
    for index, record in enumerate(records):
        try:
            projected = projection.project(dict(record.params or {}))
        except ConfigurationError as exc:
            rejected.append({"record_index": index, "reason": "projection_mismatch", "message": str(exc)})
            continue
        ranked.append((score_for_direction(record.score, direction), record, projected))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ProjectedWarmStartResult(None, 0, tuple(rejected), (), {"strategy": strategy})
    selected = ranked[: (len(ranked) if top_k is None else int(top_k))]
    if strategy == "best":
        mean = [float(value) for value in selected[0][2].optimizer_values]
    else:
        mean = [
            sum(float(item[2].optimizer_values[index]) for item in selected) / len(selected)
            for index in range(projection.optimizer_space.length)
        ]
    projection.optimizer_space.validate_genes(mean)
    return ProjectedWarmStartResult(
        mean,
        len(ranked),
        tuple(rejected),
        tuple(projected.projection_hash for _, _, projected in selected),
        {"strategy": strategy, "top_k": top_k},
    )
```

Rank records with `score_for_direction`, project `record.params`, reject structural mismatches as metadata, compute either best vector or centroid, validate against `projection.optimizer_space`, and return a detached list usable as `CMAESOptimizer(initial_mean=...)`.

- [ ] **Step 3: Run and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_projection.py tests/unit/test_cmaes_external_state.py -v
```

Expected: selected tests pass.

Commit:

```powershell
git add evocore/optimizers/cmaes/projection.py evocore/optimizers/cmaes/__init__.py evocore/__init__.py tests/unit/test_cmaes_projection.py
git commit -m "feat(cmaes): add projected warm-start helpers"
```

## Task 6: Phase 3C Integer Strategy and Margin CMA

**Files:**
- Modify: `evocore/optimizers/cmaes/mixed.py`
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `evocore/optimizers/cmaes/config.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
- Modify: `evocore/optimizers/cmaes/checkpointing.py`
- Modify if required: `src/`, `evocore/_core.pyi`
- Modify: `tests/unit/test_mixed_cma_vnext.py`
- Modify: `tests/unit/test_cmaes_engine.py`
- Modify: `tests/unit/test_cmaes_ask_tell_checkpointing.py`
- Modify: `tests/unit/test_optimizer_config.py`
- Create: `tests/property/test_integer_margin_properties.py`

- [ ] **Step 1: Write failing integer-strategy tests**

Add tests asserting:

```python
def test_cmaes_default_integer_strategy_is_round() -> None:
    optimizer = CMAESOptimizer(GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)]), population_size=4, seed=1)
    assert optimizer.integer_strategy == "round"


def test_cmaes_rejects_invalid_integer_strategy() -> None:
    with pytest.raises(ConfigurationError, match="integer_strategy"):
        CMAESOptimizer(GeneSpace([Gene("x", "int", 0, 3)]), integer_strategy="bad")


def test_margin_strategy_changes_config_hash() -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])
    assert CMAESOptimizer(space, integer_strategy="round").config_hash() != CMAESOptimizer(space, integer_strategy="margin").config_hash()
```

Add checkpoint resume test:

```python
def test_margin_cma_resume_next_ask_matches_uninterrupted(tmp_path) -> None:
    space = GeneSpace([Gene("x", "int", 0, 3), Gene("y", "float", -1.0, 1.0)])
    uninterrupted = CMAESOptimizer(space, population_size=4, seed=12, integer_strategy="margin")
    restored = CMAESOptimizer(space, population_size=4, seed=12, integer_strategy="margin")

    batch = uninterrupted.ask()
    uninterrupted.tell([
        EvaluationRecord(candidate_id=c.candidate_id, batch_id=c.batch_id, score=float(i), confidence="trusted_full", stage="full")
        for i, c in enumerate(batch)
    ])
    checkpoint = uninterrupted.ask_tell_checkpoint()
    restored.resume_ask_tell_checkpoint(checkpoint.to_dict())

    assert [c.genes for c in restored.ask()] == [c.genes for c in uninterrupted.ask()]
```

- [ ] **Step 2: Implement config and validation**

Add constructor args:

```python
integer_strategy: Literal["round", "margin"] = "round"
integer_min_probability: float = 0.02
```

Validate strategy, probability, and integer ranges. Include both values in `build_cmaes_config()` parameters. Keep default `round` payload and behavior compatible.

- [ ] **Step 3: Implement deterministic margin sampling**

For `integer_strategy="margin"`:

- keep Rust CMA continuous sample unchanged in `continuous_samples_by_id`;
- for each integer gene, build `IntegerMarginDistribution(low, high, integer_min_probability)`;
- derive a deterministic coordinate RNG from optimizer seed, event index, candidate index, gene index, and continuous latent sample hash;
- sample an integer from margin probabilities using mean equal to the continuous coordinate and sigma from `_sigma_abs()` unless Rust exposes a better per-coordinate sigma;
- clamp only as a final safety check;
- return user-facing integer genes while `tell()` still uses original continuous samples.

If correct per-coordinate statistics require Rust access, add minimal PyO3 methods and update `evocore/_core.pyi`; do not move reconstruction into Rust.

- [ ] **Step 4: Add property tests**

`tests/property/test_integer_margin_properties.py` should assert probability sums, bounds, and floor:

```python
@given(low=st.integers(-5, 0), high=st.integers(1, 8), mean=st.floats(-10, 10, allow_nan=False, allow_infinity=False), sigma=st.floats(0.05, 10, allow_nan=False, allow_infinity=False))
def test_integer_margin_probabilities_are_bounded_and_normalized(low, high, mean, sigma):
    if low > high:
        low, high = high, low
    margin = IntegerMarginDistribution(low=low, high=high, min_probability=0.01)
    probabilities = margin.probabilities(mean=mean, sigma=sigma)
    assert set(probabilities) == set(range(low, high + 1))
    assert abs(sum(probabilities.values()) - 1.0) < 1.0e-12
    assert all(value >= 0.01 for value in probabilities.values())
```

- [ ] **Step 5: Run and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_mixed_cma_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_optimizer_config.py tests/property/test_integer_margin_properties.py -v
```

Expected: selected tests pass.

Commit:

```powershell
git add evocore/optimizers/cmaes evocore/_core.pyi src tests/unit/test_mixed_cma_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_optimizer_config.py tests/property/test_integer_margin_properties.py
git commit -m "feat(cmaes): integrate integer margin strategy"
```

## Task 7: Phase 3C CMA Restart Helpers

**Files:**
- Create: `evocore/optimizers/cmaes/restarts.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`
- Modify: `evocore/__init__.py`
- Create: `tests/unit/test_cmaes_restarts.py`

- [ ] **Step 1: Write failing restart tests**

Create `tests/unit/test_cmaes_restarts.py`:

```python
import pytest

from evocore import CMAESOptimizer, GeneSpace
from evocore.core.errors import ConfigurationError
from evocore.optimizers.cmaes import FixedCMAESRestartPolicy, IPOPCMAESRestartPolicy, create_cmaes_restart


def test_fixed_restart_derives_fresh_child_seed() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=4).decide(parent=parent, restart_index=1, reason="stall")

    assert decision.restart_index == 1
    assert decision.reason == "stall"
    assert decision.population_size == 4
    assert decision.seed != parent.seed


def test_ipop_restart_grows_population() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = IPOPCMAESRestartPolicy(base_population_size=4, growth_factor=2).decide(parent=parent, restart_index=2, reason="stall")

    assert decision.population_size == 16


def test_restart_rejects_pending_batch() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    parent.ask()

    with pytest.raises(ConfigurationError, match="pending"):
        FixedCMAESRestartPolicy(population_size=4).decide(parent=parent, restart_index=1, reason="stall")


def test_create_restart_returns_fresh_optimizer() -> None:
    parent = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=10)
    decision = FixedCMAESRestartPolicy(population_size=6).decide(parent=parent, restart_index=1, reason="stall")
    child = create_cmaes_restart(parent=parent, decision=decision)

    assert child.population_size == 6
    assert child.seed == decision.seed
    assert child.generation == 0
```

- [ ] **Step 2: Implement restart policies**

Create `restarts.py` with frozen `CMAESRestartDecision`, protocol `CMAESRestartPolicy`, `FixedCMAESRestartPolicy`, `IPOPCMAESRestartPolicy`, and `create_cmaes_restart(parent, decision)`. Use `derive_child_seed` with parent seed, restart index, reason, and gene-space hash. Reject pending batches by checking `parent.state_summary().pending_batch_ids`.

- [ ] **Step 3: Run and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_restarts.py tests/unit/test_package_init.py -v
```

Expected: selected tests pass.

Commit:

```powershell
git add evocore/optimizers/cmaes/restarts.py evocore/optimizers/cmaes/__init__.py evocore/__init__.py tests/unit/test_cmaes_restarts.py tests/unit/test_package_init.py
git commit -m "feat(cmaes): add restart planning helpers"
```

## Task 8: Phase 3D External Recipe, Docs, and Compatibility

**Files:**
- Create: `tests/integration/test_phase3_expensive_projection_recipe.py`
- Modify: `docs/site/api.md`
- Modify: `docs/site/mixed-variable-search.md`
- Modify: `docs/site/expensive-external-evaluations.md`
- Modify: `CHANGELOG.md`
- Modify: `tests/unit/test_checkpoint_golden_fixtures.py` only if new fixtures are required.

- [ ] **Step 1: Add synthetic Trading-Algo-style integration test**

Create `tests/integration/test_phase3_expensive_projection_recipe.py`:

```python
from evocore import CMAESOptimizer, EvaluationRecord, Gene, GeneSpace, GeneticAlgorithmOptimizer
from evocore.lifecycle import constraint_penalty_record
from evocore.search_space import ActiveGeneProjection, BinaryThresholdTransform, ConstraintViolation, ExponentialIntegerTransform


def test_template_outer_ga_inner_cma_projection_recipe_is_deterministic() -> None:
    outer_space = GeneSpace([Gene("family", "int", 0, 2), Gene("mode", "int", 0, 1)])
    outer = GeneticAlgorithmOptimizer(outer_space, population_size=6, seed=44)
    outer_candidate = outer.ask(1)[0]
    family = int(outer_candidate.genes[0])
    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("family", "int", 0, 2),
                Gene("fast_log", "float", 1.0, 4.0),
                Gene("use_filter", "float", 0.0, 1.0),
            ]
        ),
        active_names=["fast_log", "use_filter"],
        structural_bindings={"family": family},
        transforms={
            "fast_log": ExponentialIntegerTransform(base=2.0),
            "use_filter": BinaryThresholdTransform(),
        },
        identity_keys=("family",),
        schema_id="synthetic-template",
        schema_version="1",
    )
    inner = CMAESOptimizer(projection.optimizer_space, population_size=4, seed=101, integer_strategy="margin")
    inner_batch = inner.ask()
    records = []
    for candidate in inner_batch:
        decoded = projection.reconstruct(candidate.genes)
        if decoded.parameters["fast_log"] < 3:
            records.append(
                constraint_penalty_record(
                    candidate=candidate,
                    stage="projection",
                    direction="maximize",
                    violations=[
                        ConstraintViolation(
                            code="min_fast_period",
                            message="fast period must be at least 3",
                            names=("fast_log",),
                        )
                    ],
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )
        else:
            records.append(
                EvaluationRecord(
                    candidate_id=candidate.candidate_id,
                    batch_id=candidate.batch_id,
                    score=float(decoded.parameters["fast_log"]),
                    confidence="trusted_full",
                    stage="full",
                    metadata={"projection_hash": decoded.projection_hash},
                )
            )
    update = inner.tell(records)

    assert update.state_accepted_count == 4
    assert inner.state_summary().pending_batch_ids == ()
```

- [ ] **Step 2: Update docs and changelog**

Update:

- `docs/site/api.md` with projection, constraints, projected CMA, and restart entries.
- `docs/site/mixed-variable-search.md` with `integer_strategy="round"` default and `integer_strategy="margin"` opt-in.
- `docs/site/expensive-external-evaluations.md` with a template-controlled outer GA / projected inner CMA recipe, cached records, warm starts, family/specialist policies, penalties, and restart lineage.
- `CHANGELOG.md` with public API, checkpoint/config identity, and compatibility notes.

- [ ] **Step 3: Run full verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

If Rust/PyO3 changed, also run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: all commands pass.

- [ ] **Step 4: Commit Phase 3D**

Run:

```powershell
git add docs CHANGELOG.md tests evocore src
git commit -m "docs: add phase 3 expensive optimization recipe"
```

## Compatibility Checklist

- Existing flat GA, DE, and CMA tests pass unchanged when projection APIs are unused.
- Existing CMA `integer_strategy` default is `round`.
- Existing round-only checkpoints still load.
- Margin checkpoints include strategy/config identity and reproduce uninterrupted next asks.
- Penalties are state-update eligible but excluded from trusted snapshots, warm starts, archives, promotion decisions, and top-k defaults.
- Runtime-only hooks cannot produce durable projection snapshots or checkpointable workflows.
- Projection snapshot mismatch errors name the differing source: active names, structural binding, transform signature, hook signature, or identity key.

## Final Verification and PR Update

- [ ] **Step 1: Check working tree**

Run:

```powershell
git status --short --branch
```

Expected: on the Phase 3 implementation branch with only intended changes before each commit.

- [ ] **Step 2: Run complete verification**

Run all commands from Task 8 Step 3.

- [ ] **Step 3: Push and update draft PR**

Run:

```powershell
git push
```

Update the existing draft PR description with:

- all new public API names;
- compatibility notes;
- checkpoint/config identity notes;
- verification commands and results.

## Self-Review Notes

- Spec coverage: 3A covers projection/hash/snapshot/transforms; 3B covers repair/validation, penalties, trusted semantics, telemetry, and batch completion; 3C covers projected CMA warm starts, native integer margin, config/checkpoint identity, and restarts; 3D covers integration recipe, docs, changelog, and regression verification.
- Placeholder scan: the plan intentionally leaves no deferred Phase 3 requirement; unsupported future features are listed only in Scope Boundary.
- Type consistency: `constraint_penalty` is an `EvaluationConfidence`, trusted checks use `TRUSTED_CONFIDENCES`, optimizer-state checks use `STATE_UPDATE_CONFIDENCES`, and CMA margin remains opt-in through `integer_strategy="margin"`.
- Risk note: the integration test snippet should use direct `EvaluationRecord` construction unless a helper for converting warm-start records to evaluation records is added in the same task.
