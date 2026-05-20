# CMA-ES Ask/Tell Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable `CMAESOptimizer` ask/tell checkpoint snapshots and resume so external-evaluation CMA-ES workflows can recover pending batches, partial tells, telemetry, events, best state, and Rust-backed CMA-ES adaptation state.

**Architecture:** Mirror GA ask/tell checkpointing in a focused `evocore.optimizers.cmaes.checkpointing` module. Reuse the shared checkpoint envelope and lifecycle serializers, store Rust state through `PyCMAESState.to_dict()`, restore it through `PyCMAESState.from_dict(...)`, and leave `CMAESOptimizer.run()` unchanged.

**Tech Stack:** Python mixins, EvoCore lifecycle checkpoint serializers, `CheckpointSnapshot`, PyO3-backed `PyCMAESState` snapshots, pytest, ruff, MkDocs, repo-local `.venv`.

---

## File Structure

- Create: `tests/unit/test_cmaes_ask_tell_checkpointing.py`
  - Focused CMA-ES ask/tell checkpoint contract tests.
  - Covers pending batches, partial tells, consumed batches, identity mismatch, malformed ledgers, malformed Rust state, and audit event continuity.
- Create: `evocore/optimizers/cmaes/checkpointing.py`
  - Owns CMA-ES ask/tell checkpoint snapshot creation and restore.
  - Defines `CMAESCheckpointingMixin`, `CMAES_ASK_TELL_STATE_KIND`, and `CMAES_CHECKPOINT_STATE_SCHEMA_VERSION`.
  - Delegates file I/O to shared result checkpoint helpers.
- Modify: `evocore/optimizers/cmaes/engine.py`
  - Import and inherit `CMAESCheckpointingMixin`.
  - Align `_pending_batch_ids()` with state-update completeness so partial-confidence-only batches remain pending.
  - Keep `run()` unchanged.
- Modify: `evocore/optimizers/cmaes/__init__.py`
  - Export `CMAESCheckpointingMixin`, matching the package style for `CMAESAskTellMixin`.
- Modify: `docs/site/cmaes.md`
  - Document CMA-ES ask/tell checkpoints and keep generation-loop resume unsupported.
- Modify: `docs/site/callbacks-checkpointing.md`
  - Move CMA-ES ask/tell out of unsupported surfaces.
  - Keep policy-run and CMA-ES generation-loop resume unsupported.
- Modify: `CHANGELOG.md`
  - Add a user-visible Unreleased entry for CMA-ES ask/tell checkpoints.

---

### Task 1: CMA-ES Ask/Tell Checkpoint Tests

**Files:**
- Create: `tests/unit/test_cmaes_ask_tell_checkpointing.py`

- [ ] **Step 1: Write the failing checkpoint contract tests**

Create `tests/unit/test_cmaes_ask_tell_checkpointing.py` with this content:

```python
import pytest

from evocore import (
    CheckpointError,
    CMAESOptimizer,
    EvaluationRecord,
    FitnessError,
    Gene,
    GeneSpace,
)


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
        ]
    )


def _optimizer(**overrides) -> CMAESOptimizer:
    params = {
        "population_size": 4,
        "max_generations": 5,
        "seed": 7,
    }
    params.update(overrides)
    return CMAESOptimizer(_space(), **params)


def _score(candidate) -> float:
    return -sum(float(value) ** 2 for value in candidate.genes)


def _trusted_records(candidates) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=_score(candidate),
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
        for candidate in candidates
    ]


def _partial_records(candidates) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=_score(candidate),
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
        for candidate in candidates
    ]


def test_cma_ask_tell_checkpoint_after_ask_contains_pending_batch_and_rust_state() -> None:
    optimizer = _optimizer()
    candidates = optimizer.ask()

    snapshot = optimizer.ask_tell_checkpoint(metadata={"reason": "unit"})
    payload = snapshot.to_dict()
    state_payload = payload["state"]["payload"]

    assert payload["optimizer"]["optimizer_type"] == "CMAESOptimizer"
    assert payload["position"]["mode"] == "ask_tell"
    assert payload["position"]["event_index"] == 1
    assert payload["position"]["pending_batch_ids"] == [candidates[0].batch_id]
    assert payload["metadata"] == {"reason": "unit"}
    assert state_payload["state_kind"] == "cmaes_ask_tell"
    assert state_payload["event_index"] == 1
    assert state_payload["best_candidate_id"] is None
    assert state_payload["cmaes_state"]["schema_version"] == 1
    assert state_payload["cmaes_state"]["optimizer_type"] == "cmaes"
    assert state_payload["cmaes_state"]["state"]["generation"] == 0
    assert set(state_payload["candidates_by_id"]) == {
        candidate.candidate_id for candidate in candidates
    }
    assert list(state_payload["batches_by_id"]) == [candidates[0].batch_id]


def test_cma_resume_ask_tell_checkpoint_after_ask_accepts_pending_records(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    checkpoint_path = tmp_path / "cmaes-ask-tell.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_trusted_records(candidates))

    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.trusted_count == 4
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()
    assert restored.generation == 1
    assert restored.state_summary().trusted_count == 4
    assert restored.best_candidate is not None


def test_cma_resume_ask_tell_checkpoint_after_partial_tell_accepts_missing_records(
    tmp_path,
) -> None:
    source = _optimizer()
    candidates = source.ask()
    records = _trusted_records(candidates)
    source.tell(records[:2])
    checkpoint_path = tmp_path / "cmaes-partial.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(records[2:])

    assert summary.best_candidate_id == candidates[0].candidate_id
    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.accepted_count == 2
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()
    assert restored.generation == 1
    assert restored.state_summary().trusted_count == 4


def test_cma_resume_ask_tell_checkpoint_next_ask_matches_uninterrupted(tmp_path) -> None:
    source = _optimizer()
    first_batch = source.ask()
    source.tell(_trusted_records(first_batch))
    checkpoint_path = tmp_path / "cmaes-complete.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)
    restored_next = restored.ask()
    source_next = source.ask()

    assert [candidate.candidate_id for candidate in restored_next] == [
        candidate.candidate_id for candidate in source_next
    ]
    assert [candidate.batch_id for candidate in restored_next] == [
        candidate.batch_id for candidate in source_next
    ]
    assert [candidate.genes for candidate in restored_next] == [
        candidate.genes for candidate in source_next
    ]

    with pytest.raises(FitnessError, match="consumed"):
        restored.tell([_trusted_records(first_batch)[0]])


def test_cma_resume_partial_confidence_records_keeps_batch_pending(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    partial_result = source.tell(_partial_records(candidates))
    checkpoint_path = tmp_path / "cmaes-partial-confidence.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(_trusted_records(candidates))

    assert partial_result.consumed_batch_ids == ()
    assert partial_result.pending_batch_ids == (candidates[0].batch_id,)
    assert summary.pending_batch_ids == (candidates[0].batch_id,)
    assert result.consumed_batch_ids == (candidates[0].batch_id,)
    assert result.pending_batch_ids == ()


def test_cma_resume_ask_tell_checkpoint_rejects_config_mismatch(tmp_path) -> None:
    source = _optimizer()
    source.ask()
    checkpoint_path = tmp_path / "cmaes-config.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    mismatched = _optimizer(population_size=6)

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        mismatched.resume_ask_tell_checkpoint(checkpoint_path)


def test_cma_resume_ask_tell_checkpoint_rejects_seed_and_direction_mismatch(tmp_path) -> None:
    source = _optimizer()
    source.ask()
    checkpoint_path = tmp_path / "cmaes-identity.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    with pytest.raises(CheckpointError, match="seed"):
        _optimizer(seed=999).resume_ask_tell_checkpoint(checkpoint_path)

    with pytest.raises(CheckpointError, match="direction"):
        _optimizer(direction="minimize").resume_ask_tell_checkpoint(checkpoint_path)


def test_cma_resume_ask_tell_checkpoint_rejects_wrong_state_kind() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["state_kind"] = "ga_ask_tell"

    with pytest.raises(CheckpointError, match="CMA-ES ask/tell resume"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_unknown_best_candidate() -> None:
    source = _optimizer()
    candidates = source.ask()
    source.tell([_trusted_records(candidates)[0]])
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["best_candidate_id"] = "c-missing"

    with pytest.raises(CheckpointError, match="best_candidate_id"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_batch_unknown_candidate_reference() -> None:
    source = _optimizer()
    candidates = source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    batch_payload = payload["state"]["payload"]["batches_by_id"][candidates[0].batch_id]
    batch_payload["candidate_ids"].append("c-missing")

    with pytest.raises(CheckpointError, match="references unknown candidate_id"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_missing_cmaes_state() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    del payload["state"]["payload"]["cmaes_state"]

    with pytest.raises(CheckpointError, match="cmaes_state"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_ask_tell_checkpoint_rejects_malformed_cmaes_state() -> None:
    source = _optimizer()
    source.ask()
    payload = source.ask_tell_checkpoint().to_dict()
    payload["state"]["payload"]["cmaes_state"]["schema_version"] = 999

    with pytest.raises(CheckpointError, match="cmaes_state"):
        source.resume_ask_tell_checkpoint(payload)


def test_cma_resume_restores_events_as_audit_data_without_replay(tmp_path) -> None:
    source = _optimizer()
    candidates = source.ask()
    records = _trusted_records(candidates)
    source.tell(records[:1])
    source_event_rows = source.events.to_rows()
    checkpoint_path = tmp_path / "cmaes-events.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint())

    restored = _optimizer()
    summary = restored.resume_ask_tell_checkpoint(checkpoint_path)
    result = restored.tell(records[1:])

    assert restored.events.to_rows()[: len(source_event_rows)] == source_event_rows
    assert summary.trusted_count == 1
    assert result.trusted_count == 3
    assert restored.state_summary().trusted_count == 4
```

- [ ] **Step 2: Run the new tests and verify the expected red state**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: FAIL because `CMAESOptimizer` does not yet expose `ask_tell_checkpoint`, `save_checkpoint`, or `resume_ask_tell_checkpoint`.

- [ ] **Step 3: Commit the failing tests**

```powershell
git add tests/unit/test_cmaes_ask_tell_checkpointing.py
git commit -m "test: cover cmaes ask tell checkpoints"
```

---

### Task 2: CMA-ES Checkpointing Implementation

**Files:**
- Create: `evocore/optimizers/cmaes/checkpointing.py`
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`

- [ ] **Step 1: Create the CMA-ES checkpointing mixin**

Create `evocore/optimizers/cmaes/checkpointing.py` with this content:

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from evocore import _core
from evocore.core.errors import CheckpointError
from evocore.lifecycle import (
    batch_from_checkpoint,
    batch_to_checkpoint,
    candidate_from_checkpoint,
    candidate_to_checkpoint,
    event_history_from_checkpoint,
    event_history_to_checkpoint,
    telemetry_from_checkpoint,
    telemetry_to_checkpoint,
)
from evocore.results import CheckpointSnapshot, validate_checkpoint_identity
from evocore.results import load_checkpoint as load_checkpoint_payload
from evocore.results import save_checkpoint as save_checkpoint_payload

CMAES_ASK_TELL_STATE_KIND = "cmaes_ask_tell"
CMAES_CHECKPOINT_STATE_SCHEMA_VERSION = 1


class CMAESCheckpointingMixin:
    """Stable checkpoint helpers for CMA-ES ask/tell workflows."""

    @staticmethod
    def load_checkpoint(checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
        """Load a stable checkpoint file."""
        return load_checkpoint_payload(checkpoint)

    @staticmethod
    def save_checkpoint(
        checkpoint: str | os.PathLike[str],
        snapshot: CheckpointSnapshot | Mapping[str, Any],
    ) -> None:
        """Save a stable checkpoint file."""
        save_checkpoint_payload(checkpoint, snapshot)

    def _validate_stable_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        validate_checkpoint_identity(
            payload,
            optimizer_type="CMAESOptimizer",
            gene_space_hash=self.gene_space.hash(),
            optimizer_config_hash=self.config_hash(),
            seed=self.seed,
            direction=self.direction,
        )

    def ask_tell_checkpoint(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        """Return a stable CMA-ES ask/tell runtime checkpoint snapshot."""
        best_candidate_id = (
            None if self.best_candidate is None else self.best_candidate.candidate_id
        )
        state_payload = {
            "state_kind": CMAES_ASK_TELL_STATE_KIND,
            "event_index": self._event_index,
            "cmaes_state": self._ensure_state().to_dict(),
            "candidates_by_id": {
                candidate_id: candidate_to_checkpoint(candidate)
                for candidate_id, candidate in sorted(self._candidates_by_id.items())
            },
            "batches_by_id": {
                batch_id: batch_to_checkpoint(batch)
                for batch_id, batch in sorted(self._batches_by_id.items())
            },
            "best_candidate_id": best_candidate_id,
            "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            "events": event_history_to_checkpoint(self.events),
        }
        return CheckpointSnapshot(
            optimizer_type="CMAESOptimizer",
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
                "optimizer_type": "CMAESOptimizer",
                "schema_version": CMAES_CHECKPOINT_STATE_SCHEMA_VERSION,
                "payload": state_payload,
            },
            audit={
                "events": event_history_to_checkpoint(self.events),
                "telemetry": telemetry_to_checkpoint(self.vnext_telemetry),
            },
            metadata=dict(metadata or {}),
        )

    def _ask_tell_payload_from_checkpoint(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state = payload["state"]
        if state.get("schema_version") != CMAES_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise CheckpointError("checkpoint state.schema_version must be 1.")
        state_payload = state["payload"]
        if state_payload.get("state_kind") != CMAES_ASK_TELL_STATE_KIND:
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by "
                "CMA-ES ask/tell resume."
            )
        return state_payload

    def _cmaes_state_from_checkpoint(
        self,
        state_payload: Mapping[str, Any],
    ):
        cmaes_state_payload = state_payload.get("cmaes_state")
        if not isinstance(cmaes_state_payload, Mapping):
            raise CheckpointError("checkpoint state.payload.cmaes_state must be an object.")
        try:
            return _core.PyCMAESState.from_dict(cmaes_state_payload)
        except ValueError as exc:
            raise CheckpointError(
                f"checkpoint state.payload.cmaes_state is invalid: {exc}"
            ) from exc

    def _restore_ask_tell_state(self, state_payload: Mapping[str, Any]) -> None:
        cmaes_state = self._cmaes_state_from_checkpoint(state_payload)

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
            if not batch.consumed:
                missing_samples = [
                    candidate_id
                    for candidate_id in batch.candidate_ids
                    if candidate_id not in batch.continuous_samples_by_id
                ]
                if missing_samples:
                    raise CheckpointError(
                        f"checkpoint batch {batch_id!r} is missing continuous samples "
                        f"for candidate_ids: {missing_samples!r}."
                    )

        best_candidate_id = state_payload.get("best_candidate_id")
        if best_candidate_id is not None and best_candidate_id not in candidates:
            raise CheckpointError(
                f"checkpoint best_candidate_id {best_candidate_id!r} is unknown."
            )

        try:
            event_index = int(state_payload.get("event_index", 0))
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                "checkpoint state.payload.event_index must be a non-negative integer."
            ) from exc
        if event_index < 0:
            raise CheckpointError(
                "checkpoint state.payload.event_index must be a non-negative integer."
            )

        self._state = cmaes_state
        self._candidates_by_id = candidates
        self._batches_by_id = batches
        self.best_candidate = None if best_candidate_id is None else candidates[best_candidate_id]
        self.vnext_telemetry = telemetry_from_checkpoint(state_payload.get("telemetry") or {})
        self.events = event_history_from_checkpoint(state_payload.get("events") or [])
        self._event_index = event_index

    def resume_ask_tell_checkpoint(
        self,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    ):
        """Resume CMA-ES ask/tell runtime state from a stable checkpoint."""
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        state_payload = self._ask_tell_payload_from_checkpoint(payload)
        self._restore_ask_tell_state(state_payload)
        return self.state_summary()


__all__ = [
    "CMAES_ASK_TELL_STATE_KIND",
    "CMAES_CHECKPOINT_STATE_SCHEMA_VERSION",
    "CMAESCheckpointingMixin",
]
```

- [ ] **Step 2: Wire the mixin into `CMAESOptimizer`**

In `evocore/optimizers/cmaes/engine.py`, add this import below the existing ask/tell import:

```python
from evocore.optimizers.cmaes.checkpointing import CMAESCheckpointingMixin
```

Change the class declaration from:

```python
class CMAESOptimizer(CMAESAskTellMixin):
```

to:

```python
class CMAESOptimizer(CMAESCheckpointingMixin, CMAESAskTellMixin):
```

- [ ] **Step 3: Keep partial-confidence-only batches pending**

In `evocore/optimizers/cmaes/engine.py`, replace `_pending_batch_ids()` with:

```python
    def _pending_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            batch_id
            for batch_id, batch in self._batches_by_id.items()
            if not batch.consumed and batch.ordered_state_update_records() is None
        )
```

This matches the CMA-ES state-update contract: a batch is pending until every candidate has a state-update confidence record, even if all candidates already have partial records.

- [ ] **Step 4: Export the mixin from the CMA-ES package**

In `evocore/optimizers/cmaes/__init__.py`, add:

```python
from evocore.optimizers.cmaes.checkpointing import CMAESCheckpointingMixin
```

Update `__all__` to include:

```python
    "CMAESCheckpointingMixin",
```

- [ ] **Step 5: Run the focused CMA-ES checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 6: Run adjacent ask/tell and checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 7: Run ruff checks for touched Python files**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check evocore/optimizers/cmaes/checkpointing.py evocore/optimizers/cmaes/engine.py evocore/optimizers/cmaes/__init__.py tests/unit/test_cmaes_ask_tell_checkpointing.py
.\.venv\Scripts\python.exe -m ruff check evocore/optimizers/cmaes/checkpointing.py evocore/optimizers/cmaes/engine.py evocore/optimizers/cmaes/__init__.py tests/unit/test_cmaes_ask_tell_checkpointing.py
```

Expected: PASS.

- [ ] **Step 8: Commit implementation**

```powershell
git add evocore/optimizers/cmaes/checkpointing.py evocore/optimizers/cmaes/engine.py evocore/optimizers/cmaes/__init__.py tests/unit/test_cmaes_ask_tell_checkpointing.py
git commit -m "feat: add cmaes ask tell checkpoints"
```

---

### Task 3: Documentation And Changelog

**Files:**
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update CMA-ES user docs**

In `docs/site/cmaes.md`, replace the paragraph that starts with `` `CMAESOptimizer` checkpoint/resume is still unsupported`` with:

````markdown
`CMAESOptimizer` supports stable ask/tell checkpoints for manual external-evaluation
workflows:

```python
from evocore import CMAESOptimizer, EvaluationRecord, GeneSpace

space = GeneSpace.uniform(-2.0, 2.0, 3)
optimizer = CMAESOptimizer(space, population_size=8, seed=42)
candidates = optimizer.ask()

optimizer.save_checkpoint(
    "cmaes-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = CMAESOptimizer(space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("cmaes-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

The checkpoint combines the Rust CMA-ES state snapshot with Python candidate
ledgers, pending batches, telemetry, and audit events. Generation-loop and
policy-driven CMA-ES resume remain unsupported.
````

Leave the existing `OptimizationResult.to_dict()` and `engine.events` paragraph after this new text.

- [ ] **Step 2: Update checkpoint docs**

In `docs/site/callbacks-checkpointing.md`, replace the CMA-ES paragraph under `Unsupported Checkpoint Surfaces` with:

```markdown
CMA-ES generation-loop resume and policy-driven `run(evaluator, policy=...)`
resume remain unsupported in checkpoint v1. Manual CMA-ES ask/tell checkpoints
are supported through `CMAESOptimizer.ask_tell_checkpoint()` and
`resume_ask_tell_checkpoint(...)`.
```

After the GA ask/tell checkpoint section, add:

````markdown
## CMA-ES Ask/Tell Checkpoints

CMA-ES ask/tell checkpoints use the same stable envelope as GA and additionally
store the Rust-backed CMA-ES state snapshot. This preserves covariance
adaptation, pending batches, partial records, telemetry, and audit events for
manual external-evaluation workflows.

```python
from evocore import CMAESOptimizer, EvaluationRecord, GeneSpace

gene_space = GeneSpace.uniform(-1.0, 1.0, 3)
optimizer = CMAESOptimizer(gene_space, population_size=8, seed=42)
candidates = optimizer.ask()
optimizer.save_checkpoint(
    "cmaes-ask-tell.evocore-checkpoint.json",
    optimizer.ask_tell_checkpoint(metadata={"phase": "submitted"}),
)

restored = CMAESOptimizer(gene_space, population_size=8, seed=42)
summary = restored.resume_ask_tell_checkpoint("cmaes-ask-tell.evocore-checkpoint.json")

records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=-sum(float(value) ** 2 for value in candidate.genes),
        confidence="trusted_full",
        stage="full",
    )
    for candidate in candidates
]
restored.tell(records)
```

Events are restored as audit history. Resume restores structured optimizer
state directly and does not replay events to rebuild CMA-ES state.
````

- [ ] **Step 3: Update the changelog**

In `CHANGELOG.md`, under `[Unreleased]` / `### Added`, add:

```markdown
- Added stable CMA-ES ask/tell checkpoints with Rust state snapshot resume,
  pending-batch, and partial-tell support.
```

- [ ] **Step 4: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS. Existing Material for MkDocs warnings and the existing `ga-benchmark-parity.md` nav notice are acceptable if unchanged.

- [ ] **Step 5: Remove generated docs output**

Run:

```powershell
if (Test-Path site) { Remove-Item -Recurse -Force -LiteralPath site }
git status --short
```

Expected: `site/` is not present in `git status`.

- [ ] **Step 6: Commit docs**

```powershell
git add docs/site/cmaes.md docs/site/callbacks-checkpointing.md CHANGELOG.md
git commit -m "docs: document cmaes ask tell checkpoints"
```

---

### Task 4: Full Verification And Push

**Files:**
- Verify: CMA-ES checkpoint implementation, adjacent checkpoint surfaces, docs, formatting, package install

- [ ] **Step 1: Reinstall the extension into the repo virtualenv**

Run:

```powershell
.\.venv\Scripts\python.exe -m maturin develop --release
```

Expected: PASS. This keeps the Python test environment aligned with the current mixed Python/Rust package.

- [ ] **Step 2: Run focused and adjacent checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_checkpointing.py -v
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py tests/unit/test_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 3: Run the broad Python test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 4: Run formatting and linting**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS.

- [ ] **Step 5: Build documentation**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS with no new warnings introduced by this change.

- [ ] **Step 6: Clean generated docs output**

Run:

```powershell
if (Test-Path site) { Remove-Item -Recurse -Force -LiteralPath site }
git status --short --branch
```

Expected: branch is clean and ahead of origin by the new task commits.

- [ ] **Step 7: Push the branch**

Run:

```powershell
git push
```

Expected: branch pushes to the existing PR branch.

---

## Self-Review Checklist

- Spec coverage:
  - Stable CMA-ES ask/tell checkpoint API: Task 2.
  - Rust CMA-ES state snapshot in payload: Task 2.
  - Candidate, batch, telemetry, events, and best candidate restore: Task 2.
  - Continuous samples preserved and validated for pending batches: Task 2.
  - Identity validation before mutation: Task 2.
  - Deterministic continuation tests: Task 1.
  - Docs and changelog: Task 3.
  - `run()` resume remains untouched: File Structure and Task 2 only wire ask/tell checkpointing.
- Completeness scan:
  - No incomplete steps remain.
  - Every code-writing step includes concrete code or exact replacement text.
- Type consistency:
  - Public methods are `ask_tell_checkpoint(...)`, `resume_ask_tell_checkpoint(...)`, `save_checkpoint(...)`, and `load_checkpoint(...)`.
  - State kind is consistently `cmaes_ask_tell`.
  - Schema constant is consistently `CMAES_CHECKPOINT_STATE_SCHEMA_VERSION = 1`.
