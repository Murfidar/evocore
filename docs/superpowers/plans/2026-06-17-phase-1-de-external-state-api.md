# Phase 1 DE External State API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the shared external-state API on `DifferentialEvolutionOptimizer`.

**Architecture:** Add a DE-specific external-state mixin that fills target slots before normal DE initialization and exposes honest DE capabilities after the target population is initialized. Keep replacement semantics inside the existing DE `tell(...)` flow; Phase 1 does not add arbitrary post-initialization target replacement.

**Tech Stack:** Python mixins, EvoCore DE ask/tell state, lifecycle snapshot helpers, pytest, ruff.

---

## Dependency

Complete `docs/superpowers/plans/2026-06-17-phase-1-shared-external-state-api.md` first.

## File Structure

- Create: `evocore/optimizers/de/external.py`
  - `DifferentialEvolutionExternalStateMixin`.
  - DE target-slot warm-start logic.
  - DE proposed/tracked injection rules.
  - DE snapshot scopes.
- Modify: `evocore/optimizers/de/engine.py`
  - Import and add the mixin to `DifferentialEvolutionOptimizer`.
- Modify: `evocore/optimizers/de/__init__.py`
  - Re-export `DifferentialEvolutionExternalStateMixin`.
- Create: `tests/unit/test_de_external_state.py`
  - DE warm-start, injection, top-k, duplicate, and checkpoint tests.

## DE Semantics

- `warm_start(mode="state")` is allowed only before a full target population exists and when no pending batch exists.
- State warm start appends accepted candidates to `_target_candidate_ids`, ranks by state score, and trims to `population_size`.
- `warm_start(mode="tracked")` is always allowed when it does not collide with existing candidate IDs or value hashes.
- `inject_candidates(mode="proposed")` is allowed only while initial target slots remain open. Later `tell(...)` accepts them through the existing initial target path.
- `inject_candidates(mode="tracked")` is allowed at any point for reporting and archives.
- Mid-run external target replacement remains a Phase 2 archive/diversity policy concern.

## Task 1: Write DE External State Tests

**Files:**
- Create: `tests/unit/test_de_external_state.py`

- [ ] **Step 1: Add fixtures**

```python
import pytest

from evocore import (
    DifferentialEvolutionOptimizer,
    EvaluationRecord,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    WarmStartRecord,
)
from evocore.core import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )


def _optimizer() -> DifferentialEvolutionOptimizer:
    return DifferentialEvolutionOptimizer(_space(), population_size=4, max_generations=3, seed=456)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "y": -1.0}, score=5.0),
        WarmStartRecord(params={"x": 3.0, "y": 3.0}, score=15.0),
        WarmStartRecord(params={"x": 4.0, "y": 4.0}, score=30.0),
    ]
```

- [ ] **Step 2: Test runtime protocol and capabilities**

```python
def test_de_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is False
    assert capabilities.proposed_candidate_injection is True
    assert capabilities.population_snapshots is True
```

- [ ] **Step 3: Test state warm start fills and trims target population**

```python
def test_de_warm_start_state_fills_target_slots_by_rank() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records())

    assert result.accepted_count == 4
    assert result.cached_count == 4
    assert result.state_accepted_count == 4
    assert optimizer.state_summary().trusted_count == 4
    assert [item.score for item in optimizer.top_candidates(4)] == [30.0, 20.0, 15.0, 10.0]

    batch = optimizer.ask(4)
    assert {candidate.origin for candidate in batch} == {"mutation"}
```

- [ ] **Step 4: Test state warm start rejects after initialization starts**

```python
def test_de_state_warm_start_rejects_after_ask() -> None:
    optimizer = _optimizer()
    optimizer.ask(2)

    with pytest.raises(ConfigurationError, match="before DE target initialization"):
        optimizer.warm_start(_records())
```

- [ ] **Step 5: Test tracked warm start after initialization**

```python
def test_de_tracked_warm_start_after_initialization_does_not_change_targets() -> None:
    optimizer = _optimizer()
    first_batch = optimizer.ask(4)
    optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="full",
            )
            for index, candidate in enumerate(first_batch)
        ]
    )

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.state_accepted_count == 0
    assert optimizer.state_summary().trusted_count == 4
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 9
```

- [ ] **Step 6: Test proposed injection before target population is full**

```python
def test_de_inject_candidates_before_initialization_creates_initial_pending_batch() -> None:
    optimizer = _optimizer()

    injected = optimizer.inject_candidates(
        [WarmStartRecord(params={"x": 0.25, "y": 0.5}, score=0.0)],
        metadata={"candidate_source": "domain_seed"},
    )

    assert len(injected.accepted) == 1
    accepted = injected.accepted[0]
    assert optimizer.state_summary().pending_batch_ids == (accepted.batch_id,)

    result = optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=accepted.candidate_id,
                batch_id=accepted.batch_id,
                score=8.0,
                confidence="cached",
                stage="search_memory",
            )
        ]
    )

    assert result.state_accepted_count == 1
    assert optimizer.state_summary().trusted_count == 1
```

- [ ] **Step 7: Test proposed injection rejects after target population is full**

```python
def test_de_proposed_injection_rejects_after_target_population_is_full() -> None:
    optimizer = _optimizer()
    optimizer.warm_start(_records())

    with pytest.raises(ConfigurationError, match="target population is full"):
        optimizer.inject_candidates(
            [WarmStartRecord(params={"x": 0.0, "y": 0.0}, score=0.0)],
            mode="proposed",
        )
```

- [ ] **Step 8: Test checkpoint round trip after DE warm start**

```python
def test_de_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records())
    checkpoint_path = tmp_path / "de-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.score for item in restored.top_candidates(2)] == [30.0, 20.0]
    assert restored.state_summary().trusted_count == 4
```

- [ ] **Step 9: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_external_state.py -v
```

Expected: fails because DE does not yet implement the external-state methods.

## Task 2: Implement `DifferentialEvolutionExternalStateMixin`

**Files:**
- Create: `evocore/optimizers/de/external.py`

- [ ] **Step 1: Create mixin with imports**

Use imports matching the GA plan, replacing the mixin and optimizer-specific helpers:

```python
from collections.abc import Mapping, Sequence

from evocore import _core
from evocore.core.errors import ConfigurationError
from evocore.lifecycle import (
    AcceptanceDecision,
    Candidate,
    CandidateBatch,
    CandidateOrigin,
    EvaluationRecord,
    UpdateResult,
    batch_id_from_seed,
)
from evocore.lifecycle.ask_tell_helpers import record_evaluation_telemetry
from evocore.lifecycle.external import (
    CmaMeanStrategy,
    ExternalStateCapabilities,
    InjectionMode,
    InjectionResult,
    PopulationSnapshot,
    SnapshotScope,
    WarmStartMode,
    WarmStartRecord,
    build_population_snapshot,
    resolve_warm_start_values,
    top_candidate_snapshots,
)
from evocore.search_space import Solution
```

- [ ] **Step 2: Add capability method**

```python
class DifferentialEvolutionExternalStateMixin:
    """External-state integration API for DifferentialEvolutionOptimizer."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        return ExternalStateCapabilities(
            warm_start_before_ask=True,
            warm_start_after_ask=False,
            proposed_candidate_injection=True,
            state_candidate_injection=False,
            tracked_only_injection=True,
            population_snapshots=True,
            top_candidate_snapshots=True,
            cached_record_helpers=True,
        )
```

- [ ] **Step 3: Add DE scope helpers**

Implement:

- `_external_target_candidates()` returns candidates for `_target_candidate_ids`.
- `_external_known_candidates()` returns all candidates.
- `_external_pending_candidates()` returns candidates from `_pending_batch_ids()`.
- `_external_scored_candidates()` returns candidates with scores.
- `_external_existing_hashes()` hashes all known candidates through `self.gene_space.value_hash`.

- [ ] **Step 4: Implement snapshots and top-k**

Rules:

- `scope="trusted"` maps to current target candidates.
- `scope="known"`, `"pending"`, and `"scored"` match the shared scope names.
- The snapshot `trusted_count` must equal `len(self._target_candidate_ids)`.
- `top_candidates(scope="trusted")` ranks target candidates, not every scored candidate.

- [ ] **Step 5: Implement state warm start**

Rules:

- Reject `mode="state"` if `self._pending_batch_ids()` is non-empty.
- Reject `mode="state"` if `len(self._target_candidate_ids) >= self.population_size`.
- Reject `mode="state"` if `self.generation != 0`.
- Create accepted candidates in one deterministic batch.
- Apply `EvaluationRecord` objects from each accepted `WarmStartRecord`.
- Record telemetry with `record_evaluation_telemetry`.
- Add accepted candidates to `_target_candidate_ids`.
- Sort target candidates by `state_comparison_score(self.direction)` descending and trim to `population_size`.
- Recompute `self.best_candidate` from target candidates after trimming.
- Return `UpdateResult` with `reason="warm_start_target_accepted"` acceptance decisions for candidates kept in target slots.

- [ ] **Step 6: Implement tracked warm start**

Rules:

- Create scored current-run candidates and consumed batches.
- Apply records, append ask/tell events, and record telemetry.
- Do not append to `_target_candidate_ids`.
- Return `UpdateResult(state_accepted_count=0)`.

- [ ] **Step 7: Implement injection**

Rules:

- `mode="proposed"` is allowed only when `len(self._target_candidate_ids) < self.population_size` and `self._pending_batch_ids()` is empty.
- `mode="proposed"` creates a normal pending batch that DE `tell(...)` can consume as initial targets.
- `mode="tracked"` creates a consumed batch and does not affect DE state.
- Reject unsupported modes with `ConfigurationError`.

- [ ] **Step 8: Run focused DE tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_external_state.py -v
```

Expected: all DE external-state tests pass.

## Task 3: Wire the Mixin Into DE

**Files:**
- Modify: `evocore/optimizers/de/engine.py`
- Modify: `evocore/optimizers/de/__init__.py`

- [ ] **Step 1: Import the mixin in `engine.py`**

Add:

```python
from evocore.optimizers.de.external import DifferentialEvolutionExternalStateMixin
```

- [ ] **Step 2: Add the mixin to class inheritance**

Change the class header:

```python
class DifferentialEvolutionOptimizer(
    DifferentialEvolutionExternalStateMixin,
    DifferentialEvolutionCheckpointingMixin,
    DifferentialEvolutionAskTellMixin,
    DifferentialEvolutionMultiRunMixin,
):
```

- [ ] **Step 3: Export the mixin**

Modify `evocore/optimizers/de/__init__.py`:

```python
from evocore.optimizers.de.engine import DifferentialEvolutionOptimizer
from evocore.optimizers.de.external import DifferentialEvolutionExternalStateMixin

__all__ = ["DifferentialEvolutionExternalStateMixin", "DifferentialEvolutionOptimizer"]
```

- [ ] **Step 4: Run DE regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_external_state.py tests/unit/test_de_ask_tell.py tests/unit/test_de_checkpointing.py -v
```

Expected: all selected tests pass.

## Task 4: Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run lint and focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m pytest tests/unit/test_de_external_state.py tests/unit/test_de_ask_tell.py tests/unit/test_de_checkpointing.py -v
```

Expected: all commands pass.

- [ ] **Step 2: Commit DE implementation**

Run:

```powershell
git status --short
git add evocore/optimizers/de/external.py evocore/optimizers/de/engine.py evocore/optimizers/de/__init__.py tests/unit/test_de_external_state.py
git commit -m "feat(de): support external state integration"
```

Expected: commit succeeds and contains only DE external-state changes.

## Compatibility Notes

- Existing DE behavior is unchanged when the new methods are not called.
- Phase 1 intentionally avoids mid-run target replacement by external injection.
- Checkpoint schema does not change because target IDs, candidates, batches, telemetry, and events already round-trip.
- The public API is shared, but DE capability flags document the stricter DE state rules.

## Self-Review Notes

- Spec coverage: DE warm start, pre-initialization seed pools, cached telemetry, top-k snapshots, checkpointing, and conservative post-init behavior are covered.
- Scope boundary: specialist caps, family quotas, novelty pressure, and target replacement policies are deferred.
