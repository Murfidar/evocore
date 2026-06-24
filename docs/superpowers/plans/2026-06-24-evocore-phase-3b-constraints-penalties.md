# EvoCore Phase 3B Constraints Penalties Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic repair/validation hooks and `constraint_penalty` records that complete optimizer batches without becoming trusted evidence.

**Architecture:** Keep violation and repair records in `evocore/search_space/constraints.py`; keep evaluation-record construction and confidence semantics in `evocore/lifecycle/records.py` to avoid import cycles. Optimizers use `STATE_UPDATE_CONFIDENCES` for state updates and `TRUSTED_CONFIDENCES` for archives, warm starts, promotions, and top-k defaults.

**Tech Stack:** Python dataclasses/protocols, EvoCore lifecycle records, GA/DE/CMA ask-tell paths, checkpoint serializers, pytest, ruff, MkDocs.

---

## Dependency

- Complete Phase 3A first.
- Source design: `docs/superpowers/specs/2026-06-22-evocore-phase-3-projection-cma-design.md`

## File Structure

- Modify: `evocore/search_space/constraints.py`
  - Add `ParameterRepair` and `ParameterValidator` protocols.
- Modify: `evocore/search_space/projection.py`
  - Run repair and validation hooks; expose ordered repairs and violations.
- Modify: `evocore/lifecycle/records.py`
  - Add `TRUSTED_CONFIDENCES`, `constraint_penalty`, `is_trusted_confidence`, `constraint_penalty_record`.
- Modify: `evocore/lifecycle/ask_tell_helpers.py`
  - Add penalty telemetry label.
- Modify: `evocore/lifecycle/telemetry.py`
  - Track penalty counts.
- Modify: `evocore/lifecycle/checkpointing.py`
  - Accept the new confidence literal.
- Modify: `evocore/lifecycle/external.py`, `archives.py`, `selection.py`, `conversion.py`
  - Keep trusted workflows excluding penalties by default.
- Modify: `evocore/optimizers/ga/ask_tell.py`
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `evocore/optimizers/de/engine.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
  - Let penalties complete state-update batches.
- Modify: `evocore/lifecycle/__init__.py`, `evocore/__init__.py`
  - Export confidence helpers and `constraint_penalty_record`.
- Create: `tests/unit/test_constraints_penalty.py`
- Modify: `tests/unit/test_vnext_evaluation.py`
- Modify: `tests/unit/test_ask_tell_checkpointing.py`
- Modify: `tests/unit/test_de_ask_tell_vnext.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `tests/unit/test_package_init.py`

## Public API

Export these names from `evocore.lifecycle` and top-level `evocore`:

```python
TRUSTED_CONFIDENCES
STATE_UPDATE_CONFIDENCES
constraint_penalty_record
is_state_update_confidence
is_trusted_confidence
```

Export these names from `evocore.search_space` and top-level `evocore`:

```python
ParameterRepair
ParameterValidator
```

## Task 1: Constraint and Penalty Tests

**Files:**
- Create: `tests/unit/test_constraints_penalty.py`
- Modify: `tests/unit/test_vnext_evaluation.py`

- [ ] **Step 1: Write confidence and record tests**

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
```

- [ ] **Step 2: Write projection validation tests**

Append to `tests/unit/test_constraints_penalty.py`:

```python
def test_constraint_validator_metadata_round_trips_through_projection() -> None:
    def validate(params):
        if params["fast"] >= params["slow"]:
            return [
                ConstraintViolation(
                    code="ordering",
                    message="fast must be below slow",
                    names=("fast", "slow"),
                )
            ]
        return []

    projection = ActiveGeneProjection(
        source_space=GeneSpace(
            [
                Gene("fast", "float", 1.0, 20.0),
                Gene("slow", "float", 1.0, 40.0),
            ]
        ),
        active_names=["fast", "slow"],
        validators=[validate],
        schema_id="constraints",
        schema_version="1",
    )

    result = projection.reconstruct([10.0, 5.0])

    assert result.valid is False
    assert result.violations[0].code == "ordering"
```

- [ ] **Step 3: Add lifecycle tests**

In `tests/unit/test_vnext_evaluation.py`, add:

```python
def test_evaluation_record_accepts_constraint_penalty_score() -> None:
    record = EvaluationRecord(
        candidate_id="c-1",
        score=-1.0e300,
        confidence="constraint_penalty",
        stage="projection",
    )

    assert record.confidence == "constraint_penalty"
```

- [ ] **Step 4: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_constraints_penalty.py tests/unit/test_vnext_evaluation.py -v
```

Expected: fails because confidence helpers, penalty helper, and validator support are missing.

## Task 2: Confidence and Penalty Implementation

**Files:**
- Modify: `evocore/search_space/constraints.py`
- Modify: `evocore/search_space/projection.py`
- Modify: `evocore/lifecycle/records.py`
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`

- [ ] **Step 1: Add repair and validation protocols**

Extend `evocore/search_space/constraints.py`:

```python
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class ParameterRepair(Protocol):
    checkpointable: bool

    def repair(
        self,
        parameters: Mapping[str, object],
    ) -> tuple[Mapping[str, object], Sequence[RepairRecord]]:
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        raise NotImplementedError


@runtime_checkable
class ParameterValidator(Protocol):
    checkpointable: bool

    def validate(self, parameters: Mapping[str, object]) -> Sequence[ConstraintViolation]:
        raise NotImplementedError

    def signature(self) -> Mapping[str, object]:
        raise NotImplementedError
```

- [ ] **Step 2: Add confidence helpers and penalty record**

Update `evocore/lifecycle/records.py`:

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
```

Add this helper below `score_for_direction`:

```python
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

Update `EvaluationStage.__post_init__`, `EvaluationRecord.__post_init__`, `ScoreObservation`, and `Candidate.apply_record()` so penalties are valid, state-update eligible, and set candidate status to `"eliminated"`.

- [ ] **Step 3: Add projection repair and validation execution**

Update `ActiveGeneProjection` to accept `repairs=()` and `validators=()`. In `reconstruct()`:

1. Decode active values.
2. Merge structural bindings.
3. Run repair hooks in order and collect `RepairRecord`s.
4. Run validators in order and collect `ConstraintViolation`s.
5. Return `valid=False` when any violation exists.

Do not turn invalid projection results into evaluation records inside search-space code.

- [ ] **Step 4: Export public names**

Update `evocore/search_space/__init__.py`, `evocore/lifecycle/__init__.py`, and `evocore/__init__.py`. Add package smoke tests:

```python
def test_phase3b_penalty_public_exports():
    from evocore import TRUSTED_CONFIDENCES, constraint_penalty_record, is_trusted_confidence

    assert TRUSTED_CONFIDENCES == ("trusted_full", "cached")
    assert constraint_penalty_record is not None
    assert is_trusted_confidence("cached")
```

## Task 3: Optimizer Batch Semantics

**Files:**
- Modify: `evocore/lifecycle/ask_tell_helpers.py`
- Modify: `evocore/lifecycle/telemetry.py`
- Modify: `evocore/lifecycle/checkpointing.py`
- Modify: `evocore/lifecycle/external.py`
- Modify: `evocore/lifecycle/archives.py`
- Modify: `evocore/lifecycle/selection.py`
- Modify: `evocore/lifecycle/conversion.py`
- Modify: `evocore/optimizers/ga/ask_tell.py`
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `evocore/optimizers/de/engine.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`

- [ ] **Step 1: Write optimizer batch completion tests**

Add one test each to `tests/unit/test_ask_tell_checkpointing.py`, `tests/unit/test_de_ask_tell_vnext.py`, and `tests/unit/test_cmaes_ask_tell_vnext.py`:

```python
def test_constraint_penalties_complete_batch_without_trusted_candidates() -> None:
    optimizer = _optimizer()
    candidates = optimizer.ask()
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=-1.0e300,
            confidence="constraint_penalty",
            stage="projection",
        )
        for candidate in candidates
    ]

    update = optimizer.tell(records)

    assert update.state_accepted_count == len(candidates)
    assert update.trusted_count == 0
    assert update.consumed_batch_ids == (candidates[0].batch_id,)
    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()
```

Use each file's existing `_optimizer()` or local optimizer fixture.

- [ ] **Step 2: Update telemetry**

Add `candidates_constraint_penalized: int = 0` to `OptimizationTelemetry`, include it in `to_dict()`, and add:

```python
def record_constraint_penalty(self, count: int, *, stage: str) -> None:
    self.candidates_constraint_penalized += int(count)
    self.eliminated_by_stage[stage] = self.eliminated_by_stage.get(stage, 0) + int(count)
```

Update `UpdateResult` with `penalty_count: int = 0`.

- [ ] **Step 3: Update ask/tell helper labels**

In `evocore/lifecycle/ask_tell_helpers.py`, extend `TelemetryLabel` to include `"constraint_penalty"` and update `record_evaluation_telemetry()`:

```python
if record.confidence == "constraint_penalty":
    telemetry.record_constraint_penalty(1, stage=record.stage)
    return "constraint_penalty"
```

- [ ] **Step 4: Keep trusted APIs trusted-only**

Replace raw `("trusted_full", "cached")` checks with `TRUSTED_CONFIDENCES` in:

```text
evocore/lifecycle/external.py
evocore/lifecycle/archives.py
evocore/lifecycle/selection.py
evocore/lifecycle/conversion.py
```

Keep batch completion and optimizer state updates using `is_state_update_confidence()`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_constraints_penalty.py tests/unit/test_vnext_evaluation.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_de_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Phase 3B**

Run:

```powershell
git add evocore/search_space/constraints.py evocore/search_space/projection.py evocore/search_space/__init__.py evocore/lifecycle evocore/optimizers/ga/ask_tell.py evocore/optimizers/de evocore/optimizers/cmaes/ask_tell.py evocore/__init__.py tests/unit/test_constraints_penalty.py tests/unit/test_vnext_evaluation.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_de_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_package_init.py
git commit -m "feat(lifecycle): add constraint penalty semantics"
```

## Verification

- [ ] **Step 1: Run lifecycle regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_external_state_optimizer_contract.py tests/unit/test_phase2a_external_optimizer_integration.py tests/unit/test_phase2b_stop_policies.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run format, lint, and docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: all commands pass.

## Self-Review Notes

- Spec coverage: repair/validation hooks, `constraint_penalty`, trusted/state confidence separation, batch completion, telemetry, and default trusted exclusions are covered.
- Compatibility: existing trusted and cached workflows remain trusted; penalties are state-update eligible but excluded from promotions and external warm starts by default.
- Downstream dependency: Phase 3C can use penalty records to complete CMA batches for invalid projected samples.
