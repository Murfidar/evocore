# Phase 1 GA External State API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the shared external-state API on `GeneticAlgorithmOptimizer`.

**Architecture:** Add a focused GA mixin in `evocore.optimizers.ga.external` and place it before ask/tell mixins in the engine inheritance order. The mixin creates current-run candidates and batches, applies warm-start records through existing candidate lifecycle logic, and uses the shared snapshot helpers from `evocore.lifecycle.external`.

**Tech Stack:** Python mixins, EvoCore GA ask/tell state, lifecycle snapshot helpers, pytest, ruff.

---

## Dependency

Complete `docs/superpowers/plans/2026-06-17-phase-1-shared-external-state-api.md` first.

## File Structure

- Create: `evocore/optimizers/ga/external.py`
  - `GeneticAlgorithmExternalStateMixin`.
  - GA-specific warm-start and injection semantics.
  - GA-specific snapshot scopes.
- Modify: `evocore/optimizers/ga/engine.py`
  - Import and add the mixin to `GeneticAlgorithmOptimizer`.
- Modify: `evocore/optimizers/ga/__init__.py`
  - Re-export `GeneticAlgorithmExternalStateMixin`.
- Create: `tests/unit/test_ga_external_state.py`
  - GA warm-start, injection, top-k, duplicate, and checkpoint tests.

## GA Semantics

- `warm_start(mode="state")` creates current-run candidates, applies cached or trusted records, records telemetry, and appends them to `_trusted_population_vnext`.
- `warm_start(mode="tracked")` creates current-run scored candidates and events, but does not append them to `_trusted_population_vnext`.
- `inject_candidates(mode="proposed")` creates pending GA candidates in a new batch. Their scores are not trusted until the caller later uses `tell(...)`.
- `inject_candidates(mode="tracked")` creates non-pending candidates for archive/reporting visibility only.
- Duplicates are detected with `self.gene_space.value_hash(values)`.
- Fresh candidate IDs are generated with `_core.candidate_id(self.seed, event_index, candidate_index)`.
- Each external operation consumes exactly one event index when it accepts at least one candidate.

## Task 1: Write GA External State Tests

**Files:**
- Create: `tests/unit/test_ga_external_state.py`

- [ ] **Step 1: Add shared test fixtures**

```python
import pytest

from evocore import (
    EvaluationRecord,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)
from evocore.core import ConfigurationError


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("n", "int", 1, 10),
            Gene("enabled", "bool"),
        ]
    )


def _optimizer() -> GeneticAlgorithmOptimizer:
    return GeneticAlgorithmOptimizer(_space(), population_size=4, max_generations=3, seed=123)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "n": 3, "enabled": True}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "n": 4, "enabled": False}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "n": 5, "enabled": True}, score=5.0),
    ]
```

- [ ] **Step 2: Test runtime protocol and capabilities**

```python
def test_ga_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is True
    assert capabilities.proposed_candidate_injection is True
    assert capabilities.population_snapshots is True
```

- [ ] **Step 3: Test state warm start fills trusted population and influences ask**

```python
def test_ga_warm_start_state_populates_trusted_snapshot_and_next_ask() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records())

    assert result.accepted_count == 3
    assert result.cached_count == 3
    assert result.state_accepted_count == 3
    assert result.best_score == 20.0

    trusted = optimizer.candidate_snapshot(scope="trusted")
    assert trusted.trusted_count == 3
    assert [item.score for item in optimizer.top_candidates(2)] == [20.0, 10.0]

    next_batch = optimizer.ask(4)
    assert {candidate.origin for candidate in next_batch} == {"mutation"}
```

- [ ] **Step 4: Test tracked warm start records scores without changing GA state**

```python
def test_ga_warm_start_tracked_does_not_seed_reproduction_state() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.accepted_count == 3
    assert result.state_accepted_count == 0
    assert optimizer.state_summary().trusted_count == 0
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 3
    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()

    batch = optimizer.ask(4)
    assert {candidate.origin for candidate in batch} == {"random"}
```

- [ ] **Step 5: Test duplicate reporting**

```python
def test_ga_warm_start_skips_duplicate_values_by_default() -> None:
    optimizer = _optimizer()
    duplicate = WarmStartRecord(params={"x": 1.0, "n": 3, "enabled": True}, score=12.0)

    result = optimizer.warm_start([_records()[0], duplicate])
    snapshot = optimizer.candidate_snapshot(scope="trusted")

    assert result.accepted_count == 1
    assert snapshot.trusted_count == 1
    assert len(snapshot.candidates) == 1
```

- [ ] **Step 6: Test proposed injection and later tell**

```python
def test_ga_inject_candidates_creates_pending_batch_for_later_tell() -> None:
    optimizer = _optimizer()

    injected = optimizer.inject_candidates(
        [WarmStartRecord(params={"x": 0.5, "n": 2, "enabled": True}, score=0.0)],
        metadata={"candidate_source": "domain_seed"},
    )

    assert len(injected.accepted) == 1
    candidate_id = injected.accepted[0].candidate_id
    assert optimizer.state_summary().pending_batch_ids == (injected.accepted[0].batch_id,)

    result = optimizer.tell(
        [
            EvaluationRecord(
                candidate_id=candidate_id,
                batch_id=injected.accepted[0].batch_id,
                score=7.0,
                confidence="trusted_full",
                stage="full",
            )
        ]
    )

    assert result.state_accepted_count == 1
    assert optimizer.top_candidates(1)[0].candidate_id == candidate_id
```

- [ ] **Step 7: Test checkpoint round trip after warm start**

```python
def test_ga_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records())
    checkpoint_path = tmp_path / "ga-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.values for item in restored.top_candidates(2)] == [
        (2.0, 4, False),
        (1.0, 3, True),
    ]
    assert restored.state_summary().trusted_count == 3
```

- [ ] **Step 8: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_external_state.py -v
```

Expected: fails because GA does not yet implement the external-state methods.

## Task 2: Implement `GeneticAlgorithmExternalStateMixin`

**Files:**
- Create: `evocore/optimizers/ga/external.py`

- [ ] **Step 1: Create the mixin and imports**

The mixin must import:

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
    build_candidate_snapshot,
    build_population_snapshot,
    resolve_warm_start_values,
    top_candidate_snapshots,
)
from evocore.search_space import Solution
```

- [ ] **Step 2: Add capability method**

Implement:

```python
class GeneticAlgorithmExternalStateMixin:
    """External-state integration API for GeneticAlgorithmOptimizer."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        return ExternalStateCapabilities(
            warm_start_before_ask=True,
            warm_start_after_ask=True,
            proposed_candidate_injection=True,
            state_candidate_injection=False,
            tracked_only_injection=True,
            population_snapshots=True,
            top_candidate_snapshots=True,
            cached_record_helpers=True,
        )
```

- [ ] **Step 3: Add helper methods**

Add helpers with these responsibilities:

- `_external_known_candidates()` returns `list(self._candidates_by_id.values())`.
- `_external_trusted_candidates()` returns `list(self._trusted_population_vnext)`.
- `_external_pending_candidates()` returns candidates whose batch id is in `self._pending_batch_ids()`.
- `_external_scored_candidates()` returns candidates with at least one `candidate.scores` item.
- `_external_existing_hashes()` returns value hashes for all known candidates.
- `_external_candidate_from_record(...)` resolves values, creates a `Candidate` using `Solution(list(values))` and `solution_to_candidate`, applies merged metadata, and assigns `origin="memory_seed"` unless overridden.

Use `_core.candidate_id(self.seed, event_index, candidate_index)` for candidate ids and `batch_id_from_seed(self.seed, event_index)` for the batch id.

- [ ] **Step 4: Implement `candidate_snapshot` and `top_candidates`**

Rules:

- `scope="trusted"` uses `_trusted_population_vnext`.
- `scope="known"` uses every candidate in `_candidates_by_id`.
- `scope="pending"` uses pending batches.
- `scope="scored"` uses candidates with scores.
- Reject unknown scope with `ConfigurationError`.
- `candidate_snapshot` returns `build_population_snapshot(...)`.
- `top_candidates` returns `top_candidate_snapshots(...)`.

- [ ] **Step 5: Implement `warm_start`**

Rules:

- Reject `cma_mean_strategy` values other than `"best"` and `"top_k_centroid"` only if the value is not one of those strings; GA ignores the valid CMA-specific option.
- Resolve and validate values through `resolve_warm_start_values`.
- Deduplicate against known candidate hashes and accepted-in-call hashes when `deduplicate=True`.
- Create a new `CandidateBatch` for accepted records.
- For each accepted candidate:
  - Add it to `_candidates_by_id`.
  - Record proposed telemetry with `self.vnext_telemetry.record_proposed_candidates(...)`.
  - Append an ask event using `self._append_ask_events`.
  - Create an `EvaluationRecord` from `WarmStartRecord`.
  - Apply the record to the candidate and batch.
  - Append a tell event with `self._append_tell_event`.
  - Record telemetry with `record_evaluation_telemetry`.
  - When `mode="state"`, call `self._record_state_candidate(candidate)` and add an `AcceptanceDecision(reason="warm_start_state_accepted")`.
- Sort and trim `_trusted_population_vnext` to `self.population_size`.
- Return `UpdateResult` with counts, best candidate, pending ids, telemetry, and acceptance decisions.

- [ ] **Step 6: Implement `inject_candidates`**

Rules:

- Reject `mode` outside `"proposed"` and `"tracked"`.
- Use `origin` as the public `Candidate.origin`.
- Merge method-level metadata with each record metadata. Record metadata wins.
- Do not apply scores from `WarmStartRecord` during injection.
- For `mode="proposed"`, create a non-consumed `CandidateBatch` and leave it pending.
- For `mode="tracked"`, create a consumed `CandidateBatch` so snapshots and checkpoints contain the candidates but `_pending_batch_ids()` does not include the batch.
- Append ask events for accepted injected candidates.
- Return `InjectionResult` with accepted and skipped duplicate snapshots.

- [ ] **Step 7: Run focused GA tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_external_state.py -v
```

Expected: all GA external-state tests pass.

## Task 3: Wire the Mixin Into GA

**Files:**
- Modify: `evocore/optimizers/ga/engine.py`
- Modify: `evocore/optimizers/ga/__init__.py`

- [ ] **Step 1: Import the mixin in `engine.py`**

Add:

```python
from evocore.optimizers.ga.external import GeneticAlgorithmExternalStateMixin
```

- [ ] **Step 2: Add the mixin to class inheritance**

Change the class header to put external API methods first:

```python
class GeneticAlgorithmOptimizer(
    GeneticAlgorithmExternalStateMixin,
    GeneticAlgorithmAskTellMixin,
    GeneticAlgorithmGenerationLoopMixin,
    GeneticAlgorithmCheckpointingMixin,
    GeneticAlgorithmMultiRunMixin,
    GeneticAlgorithmReproductionMixin,
):
```

- [ ] **Step 3: Export the mixin from `ga/__init__.py`**

Add import and `__all__` entry:

```python
from evocore.optimizers.ga.external import GeneticAlgorithmExternalStateMixin
```

- [ ] **Step 4: Run GA regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_external_state.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py -v
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
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_external_state.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ask_tell_checkpointing.py -v
```

Expected: all commands pass.

- [ ] **Step 2: Commit GA implementation**

Run:

```powershell
git status --short
git add evocore/optimizers/ga/external.py evocore/optimizers/ga/engine.py evocore/optimizers/ga/__init__.py tests/unit/test_ga_external_state.py
git commit -m "feat(ga): support external state integration"
```

Expected: commit succeeds and includes only GA external-state changes.

## Compatibility Notes

- Existing GA runs are unchanged when the new methods are not called.
- Warm start and injection consume event indexes only when called, so deterministic behavior remains reproducible for the same external interaction schedule.
- Checkpoint schema does not change because existing candidate, batch, telemetry, and event payloads already store the required data.
- Candidate IDs are fresh current-run IDs. Historical IDs belong in metadata such as `"source_candidate_id"`.

## Self-Review Notes

- Spec coverage: GA warm start, top-k snapshots, candidate injection, cached confidence accounting, metadata, duplicates, and checkpoint visibility are covered.
- Scope boundary: archive policy, family quotas, and diversity pressure remain out of this phase.
