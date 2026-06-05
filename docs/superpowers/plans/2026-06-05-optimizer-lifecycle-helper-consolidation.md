# Optimizer Lifecycle Helper Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate mechanical GA, DE, and CMA-ES ask/tell lifecycle helper logic while preserving optimizer-specific state transitions.

**Architecture:** Add `evocore/lifecycle/ask_tell_helpers.py` with small pure or narrowly mutating helpers for event append, record lookup, evaluator-record validation, evaluation context creation, and telemetry confidence recording. Optimizer modules keep their own replacement, reproduction, jDE, and CMA-ES state-update behavior.

**Tech Stack:** Python 3.11+, pytest, ruff.

---

## Prerequisite

Implement this after the codec and DE adapter plans. Lifecycle consolidation has the widest behavioral surface and should happen after boundary semantics are already stable.

---

## File Structure

- Create: `evocore/lifecycle/ask_tell_helpers.py`
  - Shared helper functions for ask/tell lifecycle mechanics.
- Create: `tests/unit/test_lifecycle_ask_tell_helpers.py`
  - Direct tests for helper behavior.
- Modify: `evocore/optimizers/ga/ask_tell.py`
  - Uses shared helper functions while preserving GA state population logic.
- Modify: `evocore/optimizers/cmaes/ask_tell.py`
  - Uses shared helper functions while preserving continuous sample `tell(...)`.
- Modify: `evocore/optimizers/de/ask_tell.py`
  - Uses shared helper functions while preserving target replacement and jDE state.
- Modify: `evocore/optimizers/de/engine.py`
  - Uses shared evaluator context and record validation helpers.

---

### Task 1: Add Direct Helper Tests

**Files:**
- Create: `tests/unit/test_lifecycle_ask_tell_helpers.py`

- [ ] **Step 1: Write failing tests for helper behavior**

Create `tests/unit/test_lifecycle_ask_tell_helpers.py` with:

```python
from __future__ import annotations

import pytest

from evocore import FitnessError, Gene, GeneSpace
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    EvaluationStage,
    EventHistory,
    OptimizationTelemetry,
)
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    evaluation_context_for_candidates,
    record_evaluation_telemetry,
    validate_evaluator_records,
)


def _space() -> GeneSpace:
    return GeneSpace([Gene("x", "float", -1.0, 1.0)])


def _candidate(candidate_id: str = "c-1", batch_id: str = "b-1") -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        genes=[0.25],
        batch_id=batch_id,
        origin="random",
        event_index=7,
        generation=2,
    )


def _record(candidate: Candidate, *, score: float | None = 1.0) -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="trusted_full" if score is not None else "rejected",
        stage="full",
        cost=1.5,
        metrics={"loss": 0.1} if score is not None else {},
        metadata={"source": "unit"},
    )


def test_append_candidate_ask_events_matches_optimizer_event_shape() -> None:
    events = EventHistory()
    candidate = _candidate()

    append_candidate_ask_events(events, [candidate], _space())

    rows = events.to_rows()
    assert len(rows) == 1
    assert rows[0]["event_index"] == 0
    assert rows[0]["event_type"] == "ask"
    assert rows[0]["batch_id"] == "b-1"
    assert rows[0]["candidate_id"] == "c-1"
    assert rows[0]["candidate_hash"] == _space().value_hash(candidate.genes)
    assert rows[0]["genes"] == [0.25]


def test_append_candidate_tell_event_merges_extra_metadata() -> None:
    events = EventHistory()
    candidate = _candidate()
    record = _record(candidate)
    candidate.apply_record(record)

    append_candidate_tell_event(
        events,
        candidate,
        record,
        _space(),
        "minimize",
        metadata={"accepted_for_state": True},
    )

    row = events.to_rows()[0]
    assert row["event_type"] == "tell"
    assert row["raw_score"] == pytest.approx(1.0)
    assert row["comparison_score"] == pytest.approx(-1.0)
    assert row["status"] == "trusted"
    assert row["metrics"] == {"loss": 0.1}
    assert row["metadata"] == {"source": "unit", "accepted_for_state": True}


def test_candidate_and_batch_for_record_rejects_unknown_ids() -> None:
    candidate = _candidate()
    batch = CandidateBatch(batch_id="b-1", candidate_ids=("c-1",))

    found_candidate, found_batch = candidate_and_batch_for_record(
        _record(candidate),
        {"c-1": candidate},
        {"b-1": batch},
    )

    assert found_candidate is candidate
    assert found_batch is batch

    with pytest.raises(FitnessError, match="unknown candidate_id"):
        candidate_and_batch_for_record(
            EvaluationRecord(
                candidate_id="missing",
                batch_id="b-1",
                score=1.0,
                confidence="trusted_full",
                stage="full",
            ),
            {"c-1": candidate},
            {"b-1": batch},
        )


def test_record_evaluation_telemetry_updates_counts_and_returns_label() -> None:
    telemetry = OptimizationTelemetry()
    candidate = _candidate()

    label = record_evaluation_telemetry(telemetry, _record(candidate))

    assert label == "trusted"
    assert telemetry.candidates_full_evaluated == 1
    assert telemetry.cost_by_stage["full"] == pytest.approx(1.5)


def test_evaluation_context_for_candidates_requires_one_batch() -> None:
    stage = EvaluationStage("full", budget=1.0, promote_fraction=1.0, confidence="trusted_full")

    context = evaluation_context_for_candidates(
        [_candidate()],
        stage,
        direction="maximize",
        fallback_event_index=99,
        batch_error_message="Assigned candidates must belong to exactly one batch.",
    )

    assert context.batch_id == "b-1"
    assert context.event_index == 7
    assert context.budget == pytest.approx(1.0)

    with pytest.raises(FitnessError, match="exactly one batch"):
        evaluation_context_for_candidates(
            [_candidate("c-1", "b-1"), _candidate("c-2", "b-2")],
            stage,
            direction="maximize",
            fallback_event_index=99,
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )


def test_validate_evaluator_records_rejects_missing_unexpected_duplicate_and_batch_mismatch() -> None:
    assigned = [_candidate("c-1", "b-1"), _candidate("c-2", "b-1")]

    validate_evaluator_records(
        assigned,
        [_record(assigned[0]), _record(assigned[1])],
        batch_error_message="Assigned candidates must belong to exactly one batch.",
    )

    with pytest.raises(FitnessError, match="missing evaluation records"):
        validate_evaluator_records(
            assigned,
            [_record(assigned[0])],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="unknown evaluation records"):
        validate_evaluator_records(
            assigned,
            [
                _record(assigned[0]),
                EvaluationRecord(
                    candidate_id="c-3",
                    batch_id="b-1",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                ),
            ],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="duplicate evaluation records"):
        validate_evaluator_records(
            assigned,
            [_record(assigned[0]), _record(assigned[0]), _record(assigned[1])],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )

    with pytest.raises(FitnessError, match="record batch_id"):
        validate_evaluator_records(
            assigned,
            [
                _record(assigned[0]),
                EvaluationRecord(
                    candidate_id="c-2",
                    batch_id="wrong",
                    score=1.0,
                    confidence="trusted_full",
                    stage="full",
                ),
            ],
            batch_error_message="Assigned candidates must belong to exactly one batch.",
        )
```

- [ ] **Step 2: Run helper tests and verify they fail because the module is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_ask_tell_helpers.py -v
```

Expected: FAIL during import with missing `evocore.lifecycle.ask_tell_helpers`.

---

### Task 2: Implement Shared Ask/Tell Helper Module

**Files:**
- Create: `evocore/lifecycle/ask_tell_helpers.py`

- [ ] **Step 1: Create helper module**

Create `evocore/lifecycle/ask_tell_helpers.py` with:

```python
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from evocore.core.errors import FitnessError
from evocore.lifecycle.batches import CandidateBatch
from evocore.lifecycle.events import EventHistory, EventRecord
from evocore.lifecycle.records import (
    Candidate,
    Direction,
    EvaluationContext,
    EvaluationRecord,
    EvaluationStage,
    score_for_direction,
)
from evocore.lifecycle.telemetry import OptimizationTelemetry
from evocore.search_space import GeneSpace

TelemetryLabel = Literal["trusted", "partial", "surrogate", "cached", "rejected"]


def candidate_and_batch_for_record(
    record: EvaluationRecord,
    candidates_by_id: Mapping[str, Candidate],
    batches_by_id: Mapping[str, CandidateBatch],
) -> tuple[Candidate, CandidateBatch]:
    candidate = candidates_by_id.get(record.candidate_id)
    if candidate is None:
        raise FitnessError(f"tell() received unknown candidate_id: {record.candidate_id!r}")
    if record.batch_id is not None and record.batch_id not in batches_by_id:
        raise FitnessError(f"tell() received unknown batch_id: {record.batch_id!r}")
    batch = batches_by_id.get(candidate.batch_id)
    if batch is None:
        raise FitnessError(f"tell() received unknown batch_id: {candidate.batch_id!r}")
    return candidate, batch


def append_candidate_ask_events(
    events: EventHistory,
    candidates: Sequence[Candidate],
    gene_space: GeneSpace,
) -> None:
    for candidate in candidates:
        events.append(
            EventRecord(
                event_index=len(events),
                event_type="ask",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(gene_space),
                generation=candidate.generation,
                origin=candidate.origin,
                parents=tuple(candidate.parents),
                genes=tuple(candidate.genes),
                params=dict(candidate.params) if candidate.params is not None else None,
                metadata=dict(candidate.metadata),
            )
        )


def append_candidate_tell_event(
    events: EventHistory,
    candidate: Candidate,
    record: EvaluationRecord,
    gene_space: GeneSpace,
    direction: Direction,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    raw_score = float(record.score) if record.score is not None else None
    comparison_score = (
        score_for_direction(raw_score, direction)
        if raw_score is not None and math.isfinite(raw_score)
        else None
    )
    event_metadata = dict(record.metadata)
    event_metadata.update(dict(metadata or {}))
    events.append(
        EventRecord(
            event_index=len(events),
            event_type="tell",
            batch_id=candidate.batch_id,
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.candidate_hash(gene_space),
            generation=candidate.generation,
            stage=record.stage,
            confidence=record.confidence,
            raw_score=raw_score,
            comparison_score=comparison_score,
            cost=record.cost,
            status=candidate.status,
            origin=candidate.origin,
            parents=tuple(candidate.parents),
            genes=tuple(candidate.genes),
            params=dict(candidate.params) if candidate.params is not None else None,
            metrics=dict(record.metrics),
            metadata=event_metadata,
        )
    )


def record_evaluation_telemetry(
    telemetry: OptimizationTelemetry,
    record: EvaluationRecord,
) -> TelemetryLabel:
    if record.confidence == "trusted_full":
        telemetry.record_full(1, stage=record.stage, cost=record.cost)
        return "trusted"
    if record.confidence == "cached":
        telemetry.record_cached(1, stage=record.stage, cost=record.cost)
        return "cached"
    if record.confidence == "partial":
        telemetry.record_partial(1, stage=record.stage, cost=record.cost)
        return "partial"
    if record.confidence == "surrogate":
        telemetry.record_screened(1)
        return "surrogate"
    telemetry.record_eliminated(1, stage=record.stage)
    return "rejected"


def evaluation_context_for_candidates(
    assigned: Sequence[Candidate],
    stage: EvaluationStage,
    *,
    direction: Direction,
    fallback_event_index: int,
    batch_error_message: str,
) -> EvaluationContext:
    batch_ids = {candidate.batch_id for candidate in assigned}
    if len(batch_ids) != 1:
        raise FitnessError(batch_error_message)
    return EvaluationContext(
        stage=stage,
        batch_id=next(iter(batch_ids)),
        event_index=assigned[0].event_index if assigned else fallback_event_index,
        direction=direction,
        budget=stage.budget,
    )


def validate_evaluator_records(
    assigned: Sequence[Candidate],
    records: Sequence[EvaluationRecord],
    *,
    batch_error_message: str,
) -> None:
    expected_ids = [candidate.candidate_id for candidate in assigned]
    returned_ids = [record.candidate_id for record in records]
    expected_counts = Counter(expected_ids)
    returned_counts = Counter(returned_ids)

    missing_ids = [
        candidate_id for candidate_id in expected_ids if returned_counts[candidate_id] == 0
    ]
    unexpected_ids = [
        candidate_id for candidate_id in returned_counts if candidate_id not in expected_counts
    ]
    duplicate_ids = [
        candidate_id
        for candidate_id, count in returned_counts.items()
        if count > expected_counts[candidate_id]
    ]

    if missing_ids:
        raise FitnessError(
            "Evaluator returned missing evaluation records for candidate_ids: "
            f"{sorted(set(missing_ids))!r}."
        )
    if unexpected_ids:
        raise FitnessError(
            "Evaluator returned unknown evaluation records for candidate_ids: "
            f"{sorted(unexpected_ids)!r}."
        )
    if duplicate_ids:
        raise FitnessError(
            "Evaluator returned duplicate evaluation records for candidate_ids: "
            f"{sorted(duplicate_ids)!r}."
        )

    batch_ids = {candidate.batch_id for candidate in assigned}
    if len(batch_ids) != 1:
        raise FitnessError(batch_error_message)
    expected_batch_id = next(iter(batch_ids))
    for record in records:
        if record.batch_id is not None and record.batch_id != expected_batch_id:
            raise FitnessError(
                f"Evaluator returned record batch_id {record.batch_id!r} for batch "
                f"{expected_batch_id!r}."
            )


__all__ = [
    "TelemetryLabel",
    "append_candidate_ask_events",
    "append_candidate_tell_event",
    "candidate_and_batch_for_record",
    "evaluation_context_for_candidates",
    "record_evaluation_telemetry",
    "validate_evaluator_records",
]
```

- [ ] **Step 2: Run direct helper tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_ask_tell_helpers.py -v
```

Expected: PASS.

---

### Task 3: Adopt Helpers In GA Ask/Tell

**Files:**
- Modify: `evocore/optimizers/ga/ask_tell.py`

- [ ] **Step 1: Add helper imports**

Add:

```python
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    evaluation_context_for_candidates,
    record_evaluation_telemetry,
    validate_evaluator_records,
)
```

Remove now-unused imports such as `Counter`, `EventRecord`, `score_for_direction`, or `math` only after ruff confirms they are unused.

- [ ] **Step 2: Delegate event append methods**

Replace `_append_ask_events(...)` with:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed candidates."""
        append_candidate_ask_events(self.events, candidates, self.gene_space)
```

Replace `_append_tell_event(...)` with:

```python
    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        append_candidate_tell_event(
            self.events,
            candidate,
            record,
            self.gene_space,
            self.direction,
        )
```

- [ ] **Step 3: Delegate evaluator context and validation**

Replace `_evaluation_context(...)` with:

```python
    def _evaluation_context(
        self,
        assigned: Sequence[Candidate],
        stage,
    ) -> EvaluationContext:
        return evaluation_context_for_candidates(
            assigned,
            stage,
            direction=self.direction,
            fallback_event_index=self._event_index,
            batch_error_message=(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            ),
        )
```

Replace `_validate_evaluator_records(...)` with:

```python
    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        """Reject incomplete or mismatched synchronous evaluator results."""
        validate_evaluator_records(
            assigned,
            records,
            batch_error_message=(
                "Assigned candidates must belong to exactly one batch for synchronous evaluation."
            ),
        )
```

- [ ] **Step 4: Use shared candidate lookup and telemetry in `tell(...)`**

Inside `tell(...)`, replace the inline candidate/batch lookup block with:

```python
            candidate, batch = candidate_and_batch_for_record(
                record,
                self._candidates_by_id,
                self._batches_by_id,
            )
```

Replace the confidence telemetry branch with:

```python
            confidence = record_evaluation_telemetry(self.vnext_telemetry, record)
            if confidence == "trusted":
                trusted += 1
            elif confidence == "cached":
                cached += 1
            elif confidence == "partial":
                partial += 1
            elif confidence == "surrogate":
                surrogate += 1
            else:
                rejected += 1
```

- [ ] **Step 5: Run GA lifecycle tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_ask_tell_helpers.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py -v
```

Expected: PASS.

---

### Task 4: Adopt Helpers In CMA-ES Ask/Tell

**Files:**
- Modify: `evocore/optimizers/cmaes/ask_tell.py`

- [ ] **Step 1: Add helper imports**

Add:

```python
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    record_evaluation_telemetry,
)
```

Remove now-unused imports such as `math`, `FitnessError`, `EventRecord`, or `score_for_direction` only when ruff confirms they are unused. Keep `FitnessError` if `_consume_complete_batch(...)` still raises it.

- [ ] **Step 2: Delegate candidate lookup and event append methods**

Replace `_candidate_and_batch_for_record(...)` with:

```python
    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        return candidate_and_batch_for_record(
            record,
            self._candidates_by_id,
            self._batches_by_id,
        )
```

Replace `_append_ask_events(...)` with:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed CMA candidates."""
        append_candidate_ask_events(self.events, candidates, self.gene_space)
```

Replace `_append_tell_event(...)` with:

```python
    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        append_candidate_tell_event(
            self.events,
            candidate,
            record,
            self.gene_space,
            self.direction,
        )
```

- [ ] **Step 3: Delegate confidence telemetry while preserving CMA-ES state behavior**

In `_apply_record_confidence(...)`, keep the best-candidate and trusted-record logic, then delegate telemetry:

```python
        if is_state_update_confidence(record.confidence) and (
            self.best_candidate is None
            or candidate.state_comparison_score(self.direction)
            > self.best_candidate.state_comparison_score(self.direction)
        ):
            self.best_candidate = candidate
        if record.confidence == "trusted_full":
            trusted_records.append(record)
        return record_evaluation_telemetry(self.vnext_telemetry, record)
```

- [ ] **Step 4: Run CMA-ES lifecycle tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_cmaes_engine.py -v
```

Expected: PASS.

---

### Task 5: Adopt Helpers In DE Ask/Tell And Run Helpers

**Files:**
- Modify: `evocore/optimizers/de/ask_tell.py`
- Modify: `evocore/optimizers/de/engine.py`

- [ ] **Step 1: Add helper imports to DE ask/tell**

In `evocore/optimizers/de/ask_tell.py`, add:

```python
from evocore.lifecycle.ask_tell_helpers import (
    append_candidate_ask_events,
    append_candidate_tell_event,
    candidate_and_batch_for_record,
    record_evaluation_telemetry,
)
```

- [ ] **Step 2: Delegate DE candidate lookup, ask events, tell events, and telemetry**

Replace `_candidate_and_batch_for_record(...)` with:

```python
    def _candidate_and_batch_for_record(
        self, record: EvaluationRecord
    ) -> tuple[Candidate, CandidateBatch]:
        return candidate_and_batch_for_record(
            record,
            self._candidates_by_id,
            self._batches_by_id,
        )
```

Replace `_append_ask_events(...)` with:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        append_candidate_ask_events(self.events, candidates, self.gene_space)
```

Replace `_apply_telemetry_for_record(...)` with:

```python
    def _apply_telemetry_for_record(self, record: EvaluationRecord) -> str:
        return record_evaluation_telemetry(self.vnext_telemetry, record)
```

Replace `_append_tell_event(...)` body with:

```python
        append_candidate_tell_event(
            self.events,
            candidate,
            record,
            self.gene_space,
            self.direction,
            metadata=metadata,
        )
```

- [ ] **Step 3: Add helper imports to DE engine**

In `evocore/optimizers/de/engine.py`, add:

```python
from evocore.lifecycle.ask_tell_helpers import (
    evaluation_context_for_candidates,
    validate_evaluator_records,
)
```

- [ ] **Step 4: Delegate DE run context and evaluator-record validation**

Replace `_evaluation_context(...)` with:

```python
    def _evaluation_context(self, candidates, stage: EvaluationStage) -> EvaluationContext:
        return evaluation_context_for_candidates(
            candidates,
            stage,
            direction=self.direction,
            fallback_event_index=self._event_index,
            batch_error_message="DE run candidates must belong to exactly one batch.",
        )
```

Replace `_validate_evaluator_records(...)` with:

```python
    def _validate_evaluator_records(
        self,
        assigned: Sequence[Candidate],
        records: Sequence[EvaluationRecord],
    ) -> None:
        """Reject incomplete or mismatched synchronous evaluator results."""
        validate_evaluator_records(
            assigned,
            records,
            batch_error_message="DE run candidates must belong to exactly one batch.",
        )
```

- [ ] **Step 5: Run DE lifecycle tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/unit/test_de_jde.py tests/unit/test_de_checkpointing.py -v
```

Expected: PASS.

---

### Task 6: Full Focused Verification And Commit

**Files:**
- All files touched in Tasks 1-5.

- [ ] **Step 1: Run focused optimizer lifecycle tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_ask_tell_helpers.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_de_ask_tell.py tests/unit/test_de_engine.py tests/unit/test_cmaes_ask_tell_checkpointing.py tests/unit/test_de_checkpointing.py tests/unit/test_ask_tell_checkpointing.py -v
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
git add evocore/lifecycle/ask_tell_helpers.py evocore/optimizers/ga/ask_tell.py evocore/optimizers/cmaes/ask_tell.py evocore/optimizers/de/ask_tell.py evocore/optimizers/de/engine.py tests/unit/test_lifecycle_ask_tell_helpers.py
git commit -m "refactor(lifecycle): share ask tell helpers"
```

Expected: commit succeeds with only lifecycle helper consolidation files staged.
