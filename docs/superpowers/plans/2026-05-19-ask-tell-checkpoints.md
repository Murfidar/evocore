# Ask/Tell Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable GA ask/tell checkpoint snapshots and resume so external-evaluation workflows can recover with pending batches and partial tells intact.

**Architecture:** Reuse the existing `CheckpointSnapshot` envelope and add ask/tell-specific runtime payloads under `state.payload.state_kind = "ga_ask_tell"`. Put generic lifecycle serialization in a focused `evocore.lifecycle.checkpointing` module, then keep GA snapshot/resume orchestration in `evocore.optimizers.ga.checkpointing`.

**Tech Stack:** Python dataclasses, EvoCore lifecycle records, stable JSON checkpoint helpers, pytest, ruff, repo-local `.venv`.

---

## File Structure

- Create: `evocore/lifecycle/checkpointing.py`
  - Owns JSON-safe round-trip helpers for `Candidate`, `ScoreObservation`, `EvaluationRecord`, `CandidateBatch`, `OptimizationTelemetry`, `EventRecord`, and `EventHistory`.
  - Raises `CheckpointError` for malformed checkpoint lifecycle payloads.
- Modify: `evocore/lifecycle/__init__.py`
  - Re-export lifecycle checkpoint helpers only if tests or optimizer modules benefit from the package entrance.
- Modify: `evocore/optimizers/ga/checkpointing.py`
  - Add GA ask/tell constants, snapshot creation, payload validation, state restore, and `resume_ask_tell_checkpoint(...)`.
  - Preserve generation-loop checkpoint and legacy pickle behavior.
- Create: `tests/unit/test_ask_tell_checkpointing.py`
  - Covers lifecycle serialization and GA ask/tell checkpoint/resume behavior.
- Modify: `tests/unit/test_checkpointing.py`
  - Keep shared-envelope and generation-loop coverage where it is; only add cross-state-kind rejection tests if they fit better there.
- Modify: `docs/site/callbacks-checkpointing.md`
  - Mention stable checkpoints now cover generation-loop checkpoints and GA ask/tell snapshots.
- Modify: `docs/site/ga.md`
  - Add a short GA ask/tell checkpoint example.
- Modify: `docs/site/optimizer-telemetry.md`
  - Document that telemetry can be restored from stable ask/tell checkpoints.
- Modify: `CHANGELOG.md`
  - Add a user-visible entry for GA ask/tell checkpoint/resume.

---

### Task 1: Lifecycle Checkpoint Serialization

**Files:**
- Create: `evocore/lifecycle/checkpointing.py`
- Test: `tests/unit/test_ask_tell_checkpointing.py`

- [ ] **Step 1: Write failing lifecycle round-trip tests**

Add this new test file:

```python
import pytest

from evocore import CheckpointError, FitnessError, GeneSpace, GeneticAlgorithmOptimizer
from evocore.lifecycle import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    EventHistory,
    EventRecord,
    OptimizationTelemetry,
)
from evocore.lifecycle.checkpointing import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)


def _record(candidate_id: str, *, batch_id: str, stage: str = "full") -> EvaluationRecord:
    return EvaluationRecord(
        candidate_id=candidate_id,
        batch_id=batch_id,
        score=1.5,
        confidence="trusted_full",
        stage=stage,
        cost=2.0,
        metrics={"loss": 0.25},
        metadata={"worker": "a"},
    )


def test_candidate_checkpoint_round_trip_preserves_scores_and_metadata() -> None:
    candidate = Candidate(
        candidate_id="c-1",
        genes=[0.25, 2, True],
        batch_id="b-1",
        params={"x": 0.25, "n": 2, "enabled": True},
        origin="random",
        parents=("p-1",),
        event_index=3,
        generation=7,
        metadata={"source": "unit"},
    )
    candidate.apply_record(_record("c-1", batch_id="b-1"))

    payload = candidate_to_checkpoint(candidate)
    restored = candidate_from_checkpoint(payload)

    assert restored.candidate_id == candidate.candidate_id
    assert restored.genes == candidate.genes
    assert restored.params == candidate.params
    assert restored.batch_id == candidate.batch_id
    assert restored.parents == candidate.parents
    assert restored.event_index == candidate.event_index
    assert restored.generation == candidate.generation
    assert restored.status == "trusted"
    assert restored.confidence == "trusted_full"
    assert restored.cost == pytest.approx(2.0)
    assert restored.scores["full"].score == pytest.approx(1.5)
    assert restored.scores["full"].metrics == {"loss": 0.25}
    assert restored.metadata["source"] == "unit"


def test_batch_checkpoint_round_trip_preserves_candidate_order_records_and_consumed() -> None:
    batch = CandidateBatch(
        batch_id="b-1",
        candidate_ids=("c-1", "c-2"),
        continuous_samples_by_id={"c-1": [0.1], "c-2": [0.2]},
    )
    batch.accept_record(_record("c-1", batch_id="b-1"))
    batch.consumed = True

    payload = batch_to_checkpoint(batch)
    restored = batch_from_checkpoint(payload)

    assert restored.batch_id == "b-1"
    assert restored.candidate_ids == ("c-1", "c-2")
    assert restored.consumed is True
    assert restored.continuous_samples_by_id == {"c-1": [0.1], "c-2": [0.2]}
    assert list(restored.records_by_key) == [("c-1", "full")]
    assert restored.records_by_key[("c-1", "full")].score == pytest.approx(1.5)


def test_batch_checkpoint_rejects_record_for_candidate_outside_batch() -> None:
    payload = {
        "batch_id": "b-1",
        "candidate_ids": ["c-1"],
        "records": [
            {
                "candidate_id": "c-2",
                "batch_id": "b-1",
                "score": 1.0,
                "confidence": "trusted_full",
                "stage": "full",
                "cost": 0.0,
                "metrics": {},
                "metadata": {},
            }
        ],
        "consumed": False,
        "continuous_samples_by_id": {},
    }

    with pytest.raises(CheckpointError, match="does not belong to batch"):
        batch_from_checkpoint(payload)


def test_telemetry_checkpoint_round_trip_restores_unique_hash_set() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_proposed(3)
    telemetry.unique_candidate_hashes.update({"h2", "h1"})
    telemetry.record_full(2, stage="full", cost=5.0)
    telemetry.record_cached(1, stage="cache", cost=0.5)

    restored = telemetry_from_checkpoint(telemetry_to_checkpoint(telemetry))

    assert restored.total_candidates_proposed == 3
    assert restored.unique_candidate_hashes == {"h1", "h2"}
    assert restored.candidates_full_evaluated == 2
    assert restored.candidates_cached == 1
    assert restored.cost_by_stage == {"cache": 0.5, "full": 5.0}


def test_event_history_checkpoint_round_trip_preserves_append_order() -> None:
    history = EventHistory()
    history.append(EventRecord(event_index=0, event_type="ask", batch_id="b-1"))
    history.append(
        EventRecord(
            event_index=1,
            event_type="tell",
            batch_id="b-1",
            candidate_id="c-1",
            raw_score=1.5,
            comparison_score=1.5,
            status="trusted",
        )
    )

    restored = event_history_from_checkpoint(event_history_to_checkpoint(history))

    assert restored.to_rows() == history.to_rows()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: fail during collection with `ModuleNotFoundError: No module named 'evocore.lifecycle.checkpointing'`.

- [ ] **Step 3: Implement lifecycle serialization helpers**

Create `evocore/lifecycle/checkpointing.py`:

```python
"""Stable checkpoint helpers for lifecycle runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.core.serialization import json_safe
from evocore.lifecycle.batches import CandidateBatch
from evocore.lifecycle.events import EventHistory, EventRecord
from evocore.lifecycle.records import Candidate, EvaluationRecord, ScoreObservation
from evocore.lifecycle.telemetry import OptimizationTelemetry


def _require_mapping(payload: object, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise CheckpointError(f"checkpoint {label} must be an object.")
    return payload


def _require_list(payload: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise CheckpointError(f"checkpoint {label}.{key} must be a list.")
    return value


def score_observation_to_checkpoint(observation: ScoreObservation) -> dict[str, Any]:
    return json_safe(
        {
            "score": observation.score,
            "confidence": observation.confidence,
            "stage": observation.stage,
            "cost": observation.cost,
            "metrics": dict(observation.metrics),
            "metadata": dict(observation.metadata),
        }
    )


def score_observation_from_checkpoint(payload: object) -> ScoreObservation:
    data = _require_mapping(payload, "score observation")
    return ScoreObservation(
        score=data.get("score"),
        confidence=data.get("confidence"),
        stage=str(data.get("stage") or ""),
        cost=float(data.get("cost", 0.0)),
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def candidate_to_checkpoint(candidate: Candidate) -> dict[str, Any]:
    return json_safe(
        {
            "candidate_id": candidate.candidate_id,
            "genes": list(candidate.genes),
            "batch_id": candidate.batch_id,
            "params": dict(candidate.params) if candidate.params is not None else None,
            "origin": candidate.origin,
            "parents": list(candidate.parents),
            "event_index": candidate.event_index,
            "generation": candidate.generation,
            "stage": candidate.stage,
            "status": candidate.status,
            "confidence": candidate.confidence,
            "cost": candidate.cost,
            "scores": {
                stage: score_observation_to_checkpoint(observation)
                for stage, observation in sorted(candidate.scores.items())
            },
            "metadata": dict(candidate.metadata),
        }
    )


def candidate_from_checkpoint(payload: object) -> Candidate:
    data = _require_mapping(payload, "candidate")
    genes = _require_list(data, "genes", "candidate")
    scores_payload = data.get("scores") or {}
    if not isinstance(scores_payload, Mapping):
        raise CheckpointError("checkpoint candidate.scores must be an object.")
    candidate = Candidate(
        candidate_id=str(data.get("candidate_id") or ""),
        genes=list(genes),
        batch_id=str(data.get("batch_id") or ""),
        params=dict(data["params"]) if isinstance(data.get("params"), Mapping) else None,
        origin=data.get("origin", "random"),
        parents=tuple(data.get("parents") or ()),
        event_index=int(data.get("event_index", 0)),
        generation=data.get("generation"),
        stage=data.get("stage"),
        status=data.get("status", "proposed"),
        confidence=data.get("confidence"),
        cost=float(data.get("cost", 0.0)),
        metadata=dict(data.get("metadata") or {}),
    )
    candidate.scores = {
        str(stage): score_observation_from_checkpoint(observation)
        for stage, observation in scores_payload.items()
    }
    return candidate


def evaluation_record_to_checkpoint(record: EvaluationRecord) -> dict[str, Any]:
    return json_safe(
        {
            "candidate_id": record.candidate_id,
            "batch_id": record.batch_id,
            "score": record.score,
            "confidence": record.confidence,
            "stage": record.stage,
            "cost": record.cost,
            "metrics": dict(record.metrics),
            "metadata": dict(record.metadata),
        }
    )


def evaluation_record_from_checkpoint(payload: object) -> EvaluationRecord:
    data = _require_mapping(payload, "evaluation record")
    return EvaluationRecord(
        candidate_id=str(data.get("candidate_id") or ""),
        batch_id=data.get("batch_id"),
        score=data.get("score"),
        confidence=data.get("confidence"),
        stage=str(data.get("stage") or ""),
        cost=float(data.get("cost", 0.0)),
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def batch_to_checkpoint(batch: CandidateBatch) -> dict[str, Any]:
    records = [
        evaluation_record_to_checkpoint(record)
        for _, record in sorted(batch.records_by_key.items())
    ]
    return json_safe(
        {
            "batch_id": batch.batch_id,
            "candidate_ids": list(batch.candidate_ids),
            "records": records,
            "consumed": bool(batch.consumed),
            "continuous_samples_by_id": {
                candidate_id: list(sample)
                for candidate_id, sample in sorted(batch.continuous_samples_by_id.items())
            },
        }
    )


def batch_from_checkpoint(payload: object) -> CandidateBatch:
    data = _require_mapping(payload, "batch")
    candidate_ids = tuple(str(candidate_id) for candidate_id in _require_list(data, "candidate_ids", "batch"))
    samples_payload = data.get("continuous_samples_by_id") or {}
    if not isinstance(samples_payload, Mapping):
        raise CheckpointError("checkpoint batch.continuous_samples_by_id must be an object.")
    batch = CandidateBatch(
        batch_id=str(data.get("batch_id") or ""),
        candidate_ids=candidate_ids,
        continuous_samples_by_id={
            str(candidate_id): [float(value) for value in values]
            for candidate_id, values in samples_payload.items()
        },
    )
    for record_payload in _require_list(data, "records", "batch"):
        batch.accept_record(evaluation_record_from_checkpoint(record_payload))
    batch.consumed = bool(data.get("consumed", False))
    return batch


def telemetry_to_checkpoint(telemetry: OptimizationTelemetry) -> dict[str, Any]:
    return telemetry.to_dict()


def telemetry_from_checkpoint(payload: object) -> OptimizationTelemetry:
    data = _require_mapping(payload, "telemetry")
    telemetry = OptimizationTelemetry(
        total_candidates_proposed=int(data.get("total_candidates_proposed", 0)),
        unique_candidate_hashes=set(data.get("unique_candidate_hashes") or ()),
        candidates_screened=int(data.get("candidates_screened", 0)),
        candidates_partial_evaluated=int(data.get("candidates_partial_evaluated", 0)),
        candidates_full_evaluated=int(data.get("candidates_full_evaluated", 0)),
        candidates_cached=int(data.get("candidates_cached", 0)),
        promoted_by_stage={
            str(stage): int(count)
            for stage, count in dict(data.get("promoted_by_stage") or {}).items()
        },
        eliminated_by_stage={
            str(stage): int(count)
            for stage, count in dict(data.get("eliminated_by_stage") or {}).items()
        },
        cost_by_stage={
            str(stage): float(cost)
            for stage, cost in dict(data.get("cost_by_stage") or {}).items()
        },
    )
    return telemetry


def event_record_to_checkpoint(event: EventRecord) -> dict[str, Any]:
    return event.to_dict()


def event_record_from_checkpoint(payload: object) -> EventRecord:
    data = _require_mapping(payload, "event")
    return EventRecord(
        event_index=int(data.get("event_index", 0)),
        event_type=data.get("event_type"),
        batch_id=data.get("batch_id"),
        candidate_id=data.get("candidate_id"),
        candidate_hash=data.get("candidate_hash"),
        generation=data.get("generation"),
        stage=data.get("stage"),
        confidence=data.get("confidence"),
        raw_score=data.get("raw_score"),
        comparison_score=data.get("comparison_score"),
        cost=float(data.get("cost", 0.0)),
        status=data.get("status"),
        origin=data.get("origin"),
        parents=tuple(data.get("parents") or ()),
        genes=tuple(data.get("genes") or ()),
        params=dict(data["params"]) if isinstance(data.get("params"), Mapping) else None,
        metrics=dict(data.get("metrics") or {}),
        metadata=dict(data.get("metadata") or {}),
    )


def event_history_to_checkpoint(history: EventHistory) -> list[dict[str, Any]]:
    return [event_record_to_checkpoint(event) for event in history]


def event_history_from_checkpoint(payload: object) -> EventHistory:
    if not isinstance(payload, list):
        raise CheckpointError("checkpoint events must be a list.")
    history = EventHistory()
    for row in payload:
        history.append(event_record_from_checkpoint(row))
    return history
```

- [ ] **Step 4: Export helpers from the lifecycle package**

Modify `evocore/lifecycle/__init__.py` to import and include these helper names in `__all__`:

```python
from evocore.lifecycle.checkpointing import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
```

Add the same eight names to `__all__`, sorted in the file's existing style.

- [ ] **Step 5: Run lifecycle tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: the lifecycle serialization tests pass. Later GA tests in the same file do not exist yet.

- [ ] **Step 6: Commit lifecycle serialization**

Run:

```powershell
git add evocore/lifecycle/checkpointing.py evocore/lifecycle/__init__.py tests/unit/test_ask_tell_checkpointing.py
git commit -m "feat(lifecycle): serialize checkpoint runtime state"
```

---

### Task 2: GA Ask/Tell Checkpoint Snapshot

**Files:**
- Modify: `evocore/optimizers/ga/checkpointing.py`
- Test: `tests/unit/test_ask_tell_checkpointing.py`

- [ ] **Step 1: Add failing tests for GA ask/tell snapshot shape**

Append these tests to `tests/unit/test_ask_tell_checkpointing.py`. The import block already
contains `GeneSpace` and `GeneticAlgorithmOptimizer` from Task 1.

```python
def _ga() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=5,
        seed=123,
    )


def test_ga_ask_tell_checkpoint_after_ask_contains_pending_batch_state() -> None:
    optimizer = _ga()
    candidates = optimizer.ask(4)

    snapshot = optimizer.ask_tell_checkpoint(metadata={"reason": "unit"})
    payload = snapshot.to_dict()
    state_payload = payload["state"]["payload"]

    assert state_payload["state_kind"] == "ga_ask_tell"
    assert state_payload["event_index"] == 1
    assert set(state_payload["candidates_by_id"]) == {
        candidate.candidate_id for candidate in candidates
    }
    assert list(state_payload["batches_by_id"]) == [candidates[0].batch_id]
    assert state_payload["trusted_candidate_ids"] == []
    assert state_payload["best_candidate_id"] is None
    assert payload["position"]["mode"] == "ask_tell"
    assert payload["position"]["event_index"] == 1
    assert payload["position"]["pending_batch_ids"] == [candidates[0].batch_id]
    assert payload["metadata"] == {"reason": "unit"}


def test_ga_ask_tell_checkpoint_after_partial_tell_contains_accepted_record() -> None:
    optimizer = _ga()
    candidates = optimizer.ask(4)
    optimizer.tell([_record(candidates[0].candidate_id, batch_id=candidates[0].batch_id)])

    payload = optimizer.ask_tell_checkpoint().to_dict()
    batch_payload = payload["state"]["payload"]["batches_by_id"][candidates[0].batch_id]

    assert len(batch_payload["records"]) == 1
    assert batch_payload["records"][0]["candidate_id"] == candidates[0].candidate_id
    assert payload["position"]["pending_batch_ids"] == [candidates[0].batch_id]
    assert payload["state"]["payload"]["best_candidate_id"] == candidates[0].candidate_id
    assert payload["state"]["payload"]["trusted_candidate_ids"] == [candidates[0].candidate_id]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py::test_ga_ask_tell_checkpoint_after_ask_contains_pending_batch_state tests/unit/test_ask_tell_checkpointing.py::test_ga_ask_tell_checkpoint_after_partial_tell_contains_accepted_record -v
```

Expected: fail with `AttributeError: 'GeneticAlgorithmOptimizer' object has no attribute 'ask_tell_checkpoint'`.

- [ ] **Step 3: Implement GA ask/tell snapshot creation**

Modify imports in `evocore/optimizers/ga/checkpointing.py`:

```python
from evocore.lifecycle.checkpointing import (
    batch_to_checkpoint,
    candidate_to_checkpoint,
    event_history_to_checkpoint,
    telemetry_to_checkpoint,
)
```

Add a module constant near the helper functions:

```python
GA_GENERATION_LOOP_STATE_KIND = "ga_generation_loop"
GA_ASK_TELL_STATE_KIND = "ga_ask_tell"
GA_CHECKPOINT_STATE_SCHEMA_VERSION = 1
```

Change the existing generation-loop payload to use `GA_GENERATION_LOOP_STATE_KIND`.

Add this method to `GeneticAlgorithmCheckpointingMixin`:

```python
    def ask_tell_checkpoint(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        """Return a stable GA ask/tell runtime checkpoint snapshot."""
        trusted_ids = [candidate.candidate_id for candidate in self._trusted_population_vnext]
        best_candidate_id = (
            None if self.best_candidate is None else self.best_candidate.candidate_id
        )
        state_payload = {
            "state_kind": GA_ASK_TELL_STATE_KIND,
            "event_index": self._event_index,
            "candidates_by_id": {
                candidate_id: candidate_to_checkpoint(candidate)
                for candidate_id, candidate in sorted(self._candidates_by_id.items())
            },
            "batches_by_id": {
                batch_id: batch_to_checkpoint(batch)
                for batch_id, batch in sorted(self._batches_by_id.items())
            },
            "trusted_candidate_ids": trusted_ids,
            "best_candidate_id": best_candidate_id,
            "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            "events": event_history_to_checkpoint(self.events),
        }
        return CheckpointSnapshot(
            optimizer_type="GeneticAlgorithmOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "mode": "ask_tell",
                "event_index": self._event_index,
                "pending_batch_ids": list(self._pending_batch_ids()),
                "best_candidate_id": best_candidate_id,
            },
            state={
                "optimizer_type": "GeneticAlgorithmOptimizer",
                "schema_version": GA_CHECKPOINT_STATE_SCHEMA_VERSION,
                "payload": state_payload,
            },
            audit={
                "events": event_history_to_checkpoint(self.events),
                "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            },
            metadata=dict(metadata or {}),
        )
```

- [ ] **Step 4: Run GA snapshot tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: all current ask/tell checkpoint tests pass.

- [ ] **Step 5: Run existing generation-loop checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py -v
```

Expected: all existing checkpoint tests pass, proving the new state-kind constant did not break generation-loop checkpointing.

- [ ] **Step 6: Commit GA snapshot support**

Run:

```powershell
git add evocore/optimizers/ga/checkpointing.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py
git commit -m "feat(ga): snapshot ask tell checkpoints"
```

---

### Task 3: GA Ask/Tell Resume

**Files:**
- Modify: `evocore/optimizers/ga/checkpointing.py`
- Test: `tests/unit/test_ask_tell_checkpointing.py`

- [ ] **Step 1: Add failing tests for resume from pending and partial batches**

Append to `tests/unit/test_ask_tell_checkpointing.py`:

```python
def _records_for(candidates):
    return [
        _record(candidate.candidate_id, batch_id=candidate.batch_id, stage="full")
        for candidate in candidates
    ]


def test_ga_resume_ask_tell_checkpoint_after_ask_accepts_pending_records(tmp_path) -> None:
    source = _ga()
    candidates = source.ask(4)
    checkpoint_path = tmp_path / "ga-ask-tell.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _ga()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_records_for(candidates))

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.trusted_count == 4
    assert result.pending_batch_ids == ()
    assert restored.state_summary().trusted_count == 4
    assert restored.best_candidate is not None


def test_ga_resume_ask_tell_checkpoint_after_partial_tell_accepts_missing_records(tmp_path) -> None:
    source = _ga()
    candidates = source.ask(4)
    source.tell([_record(candidates[0].candidate_id, batch_id=candidates[0].batch_id)])
    checkpoint_path = tmp_path / "ga-partial.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _ga()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_records_for(candidates[1:]))

    assert summary.best_candidate_id == candidates[0].candidate_id
    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.accepted_count == 3
    assert result.pending_batch_ids == ()
    assert restored.state_summary().trusted_count == 4


def test_ga_resume_ask_tell_checkpoint_rejects_duplicate_tell(tmp_path) -> None:
    source = _ga()
    candidates = source.ask(4)
    source.tell([_record(candidates[0].candidate_id, batch_id=candidates[0].batch_id)])
    checkpoint_path = tmp_path / "ga-duplicate.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _ga()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    with pytest.raises(FitnessError, match="already has a state update record"):
        restored.tell([_record(candidates[0].candidate_id, batch_id=candidates[0].batch_id)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_after_ask_accepts_pending_records tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_after_partial_tell_accepts_missing_records tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_rejects_duplicate_tell -v
```

Expected: fail with `AttributeError: 'GeneticAlgorithmOptimizer' object has no attribute 'resume_ask_tell_checkpoint'`.

- [ ] **Step 3: Implement resume helpers**

Modify imports in `evocore/optimizers/ga/checkpointing.py`:

```python
from evocore.lifecycle.checkpointing import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
```

Add these methods to `GeneticAlgorithmCheckpointingMixin`:

```python
    def _ask_tell_payload_from_checkpoint(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state = payload["state"]
        if state.get("schema_version") != GA_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise CheckpointError("checkpoint state.schema_version must be 1.")
        state_payload = state["payload"]
        if state_payload.get("state_kind") != GA_ASK_TELL_STATE_KIND:
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by "
                "GA ask/tell resume."
            )
        return state_payload

    def _restore_ask_tell_state(self, state_payload: Mapping[str, Any]) -> None:
        raw_candidates = state_payload.get("candidates_by_id")
        if not isinstance(raw_candidates, Mapping):
            raise CheckpointError("checkpoint state.payload.candidates_by_id must be an object.")
        candidates = {
            str(candidate_id): candidate_from_checkpoint(candidate_payload)
            for candidate_id, candidate_payload in raw_candidates.items()
        }
        for candidate_id, candidate in candidates.items():
            if candidate.candidate_id != candidate_id:
                raise CheckpointError(
                    f"checkpoint candidate key {candidate_id!r} does not match "
                    f"candidate_id {candidate.candidate_id!r}."
                )

        raw_batches = state_payload.get("batches_by_id")
        if not isinstance(raw_batches, Mapping):
            raise CheckpointError("checkpoint state.payload.batches_by_id must be an object.")
        batches = {
            str(batch_id): batch_from_checkpoint(batch_payload)
            for batch_id, batch_payload in raw_batches.items()
        }
        for batch_id, batch in batches.items():
            if batch.batch_id != batch_id:
                raise CheckpointError(
                    f"checkpoint batch key {batch_id!r} does not match "
                    f"batch_id {batch.batch_id!r}."
                )
            for candidate_id in batch.candidate_ids:
                if candidate_id not in candidates:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} references unknown "
                        f"candidate_id {candidate_id!r}."
                    )

        trusted_ids = list(state_payload.get("trusted_candidate_ids") or ())
        missing_trusted = [candidate_id for candidate_id in trusted_ids if candidate_id not in candidates]
        if missing_trusted:
            raise CheckpointError(
                "checkpoint trusted_candidate_ids reference unknown candidate_ids: "
                f"{missing_trusted!r}."
            )

        best_candidate_id = state_payload.get("best_candidate_id")
        if best_candidate_id is not None and best_candidate_id not in candidates:
            raise CheckpointError(
                f"checkpoint best_candidate_id {best_candidate_id!r} is unknown."
            )

        self._candidates_by_id = candidates
        self._batches_by_id = batches
        self._trusted_population_vnext = [candidates[candidate_id] for candidate_id in trusted_ids]
        self.best_candidate = None if best_candidate_id is None else candidates[best_candidate_id]
        self.vnext_telemetry = telemetry_from_checkpoint(state_payload.get("telemetry") or {})
        self.events = event_history_from_checkpoint(state_payload.get("events") or [])
        self._event_index = int(state_payload.get("event_index", 0))

    def resume_ask_tell_checkpoint(
        self,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    ):
        """Resume GA ask/tell runtime state from a stable checkpoint."""
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        state_payload = self._ask_tell_payload_from_checkpoint(payload)
        self._restore_ask_tell_state(state_payload)
        return self.state_summary()
```

- [ ] **Step 4: Run resume tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: all ask/tell checkpoint tests pass.

- [ ] **Step 5: Commit GA resume support**

Run:

```powershell
git add evocore/optimizers/ga/checkpointing.py tests/unit/test_ask_tell_checkpointing.py
git commit -m "feat(ga): resume ask tell checkpoints"
```

---

### Task 4: Determinism and Mismatch Rejection

**Files:**
- Modify: `tests/unit/test_ask_tell_checkpointing.py`
- Modify: `evocore/optimizers/ga/checkpointing.py`

- [ ] **Step 1: Add failing tests for next ask determinism and identity mismatches**

Append to `tests/unit/test_ask_tell_checkpointing.py`:

```python
def test_ga_resume_ask_tell_checkpoint_next_ask_matches_uninterrupted(tmp_path) -> None:
    source = _ga()
    first_batch = source.ask(4)
    source.tell(_records_for(first_batch))
    checkpoint_path = tmp_path / "ga-complete.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _ga()
    restored.resume_ask_tell_checkpoint(checkpoint_path)
    restored_next = restored.ask(4)
    source_next = source.ask(4)

    assert [candidate.candidate_id for candidate in restored_next] == [
        candidate.candidate_id for candidate in source_next
    ]
    assert [candidate.batch_id for candidate in restored_next] == [
        candidate.batch_id for candidate in source_next
    ]
    assert [candidate.genes for candidate in restored_next] == [
        candidate.genes for candidate in source_next
    ]


def test_ga_resume_ask_tell_checkpoint_rejects_config_mismatch(tmp_path) -> None:
    source = _ga()
    source.ask(4)
    checkpoint_path = tmp_path / "ga-config.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    mismatched = GeneticAlgorithmOptimizer(
        source.gene_space,
        population_size=6,
        max_generations=5,
        seed=123,
    )

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        mismatched.resume_ask_tell_checkpoint(checkpoint_path)


def test_ga_resume_ask_tell_checkpoint_rejects_seed_and_direction_mismatch(tmp_path) -> None:
    source = _ga()
    source.ask(4)
    checkpoint_path = tmp_path / "ga-identity.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    with pytest.raises(CheckpointError, match="seed"):
        GeneticAlgorithmOptimizer(
            source.gene_space,
            population_size=4,
            max_generations=5,
            seed=999,
        ).resume_ask_tell_checkpoint(checkpoint_path)

    with pytest.raises(CheckpointError, match="direction"):
        GeneticAlgorithmOptimizer(
            source.gene_space,
            population_size=4,
            max_generations=5,
            seed=123,
            direction="minimize",
        ).resume_ask_tell_checkpoint(checkpoint_path)


def test_ga_resume_ask_tell_checkpoint_rejects_wrong_state_kind() -> None:
    source = _ga()
    checkpoint = source.checkpoint(generation=0, population=source._initial_population())

    with pytest.raises(CheckpointError, match="ask/tell resume"):
        source.resume_ask_tell_checkpoint(checkpoint.to_dict())
```

- [ ] **Step 2: Run mismatch tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_next_ask_matches_uninterrupted tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_rejects_config_mismatch tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_rejects_seed_and_direction_mismatch tests/unit/test_ask_tell_checkpointing.py::test_ga_resume_ask_tell_checkpoint_rejects_wrong_state_kind -v
```

Expected: mismatch tests pass if Task 3 identity validation is correct. If the wrong-state-kind message differs, update `_ask_tell_payload_from_checkpoint(...)` to include the phrase `ask/tell resume`.

- [ ] **Step 3: Add malformed-reference tests**

Append to `tests/unit/test_ask_tell_checkpointing.py`:

```python
def test_ga_resume_ask_tell_checkpoint_rejects_unknown_best_candidate() -> None:
    source = _ga()
    source.ask(4)
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["best_candidate_id"] = "c-missing"

    with pytest.raises(CheckpointError, match="best_candidate_id"):
        source.resume_ask_tell_checkpoint(payload)


def test_ga_resume_ask_tell_checkpoint_rejects_batch_unknown_candidate_reference() -> None:
    source = _ga()
    candidates = source.ask(4)
    payload = source.ask_tell_checkpoint().to_dict()
    batch_payload = payload["state"]["payload"]["batches_by_id"][candidates[0].batch_id]
    batch_payload["candidate_ids"].append("c-missing")

    with pytest.raises(CheckpointError, match="references unknown candidate_id"):
        source.resume_ask_tell_checkpoint(payload)
```

- [ ] **Step 4: Run all ask/tell checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: all ask/tell checkpoint tests pass.

- [ ] **Step 5: Commit determinism and rejection coverage**

Run:

```powershell
git add evocore/optimizers/ga/checkpointing.py tests/unit/test_ask_tell_checkpointing.py
git commit -m "test(ga): cover ask tell checkpoint identity"
```

---

### Task 5: Public Docs and Changelog

**Files:**
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update checkpointing docs**

In `docs/site/callbacks-checkpointing.md`, add a section titled `GA Ask/Tell Checkpoints`:

````markdown
## GA Ask/Tell Checkpoints

Stable checkpoints also cover manual GA ask/tell workflows. This is the
recommended checkpoint boundary when evaluation work happens outside EvoCore,
for example in a job queue or remote worker pool.

```python
from evocore import EvaluationRecord, GeneticAlgorithmOptimizer

optimizer = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "ga-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = GeneticAlgorithmOptimizer(gene_space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("ga-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score,
        confidence="trusted_full",
        stage="full",
    )
    for candidate, score in zip(candidates, scores, strict=False)
]
restored.tell(records)
```

Pending batches and partial tells are valid checkpoint state. Resume restores
candidate and batch ledgers directly; event history is audit data and is not
replayed to rebuild optimizer state.
````

- [ ] **Step 2: Update GA docs**

In `docs/site/ga.md`, add a short note near the ask/tell section:

```markdown
GA ask/tell checkpoints are stable JSON files. Use
`optimizer.ask_tell_checkpoint()` after `ask()` or after partial `tell()` calls
when external evaluation work may outlive the Python process. Restore with
`resume_ask_tell_checkpoint(...)`, inspect `state_summary().pending_batch_ids`,
and continue with normal `tell()` or `ask()` calls.
```

- [ ] **Step 3: Update telemetry docs**

In `docs/site/optimizer-telemetry.md`, add:

```markdown
Stable GA ask/tell checkpoints include telemetry counters and unique candidate
hashes. Restored telemetry continues counting from the checkpoint rather than
being rebuilt from event rows.
```

- [ ] **Step 4: Update changelog**

Add an entry under the current unreleased section in `CHANGELOG.md`:

```markdown
- Added stable GA ask/tell checkpoints with pending-batch and partial-tell
  resume support.
```

- [ ] **Step 5: Run docs diff check**

Run:

```powershell
git diff --check
```

Expected: exits 0. CRLF warnings are acceptable if the exit code is 0.

- [ ] **Step 6: Commit docs**

Run:

```powershell
git add docs/site/callbacks-checkpointing.md docs/site/ga.md docs/site/optimizer-telemetry.md CHANGELOG.md
git commit -m "docs: document ask tell checkpoints"
```

---

### Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py tests/unit/test_ga_ask_tell_vnext.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run formatting check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: all files already formatted.

- [ ] **Step 3: Run lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: all checks pass.

- [ ] **Step 4: Run unit and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: all unit and integration tests pass.

- [ ] **Step 5: Confirm git status**

Run:

```powershell
git status --short --branch
```

Expected: branch is clean and ahead of origin by the new implementation commits.

- [ ] **Step 6: Push branch and update draft PR**

Run:

```powershell
git push origin feature/general-optimizer-framework
gh pr view 13 --json url,isDraft,state,headRefName,baseRefName
```

Expected: push succeeds; PR #13 remains open and draft against `main`.

---

## Self-Review Notes

- Spec coverage: lifecycle serialization, GA snapshot, GA resume, mismatch rejection, pending batches, partial tells, event audit semantics, docs, changelog, and verification are each covered by a task.
- Scope boundary: policy-driven `run(...)` resume and CMA-ES state restore stay outside this plan.
- Type consistency: public methods are `ask_tell_checkpoint(...)` and `resume_ask_tell_checkpoint(...)`; checkpoint state kind is `ga_ask_tell`; schema version is `1`.
