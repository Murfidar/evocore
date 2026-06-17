# Phase 1 CMA-ES External State API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the shared external-state API on `CMAESOptimizer` with safe pre-start mean construction and tracked post-start behavior.

**Architecture:** Add a CMA-ES external-state mixin that can derive `initial_mean` from warm-start records before the Rust state exists. After CMA-ES state is created, state mutation by warm start is rejected and callers can still track scored external candidates for reporting, archives, and checkpoints.

**Tech Stack:** Python mixins, EvoCore CMA-ES Rust-backed state boundary, lifecycle snapshot helpers, pytest, ruff.

---

## Dependency

Complete `docs/superpowers/plans/2026-06-17-phase-1-shared-external-state-api.md` first.

## File Structure

- Create: `evocore/optimizers/cmaes/external.py`
  - `CMAESExternalStateMixin`.
  - Pre-start mean derivation.
  - Tracked post-start warm-start records.
  - Snapshot scopes.
- Modify: `evocore/optimizers/cmaes/engine.py`
  - Import and add the mixin to `CMAESOptimizer`.
- Modify: `evocore/optimizers/cmaes/__init__.py`
  - Re-export `CMAESExternalStateMixin`.
- Create: `tests/unit/test_cmaes_external_state.py`
  - CMA-ES warm-start, mean strategy, post-start rejection, tracked mode, and checkpoint tests.

## CMA-ES Semantics

- `warm_start(mode="state")` is allowed only while `self._state is None`.
- State warm start records are stored as current-run scored candidates and also set `self.initial_mean`.
- `cma_mean_strategy="best"` uses the best state-eligible record after direction normalization.
- `cma_mean_strategy="top_k_centroid"` uses the decoded centroid of the top `top_k` records, then encodes it through `encode_gene_values`.
- `warm_start(mode="tracked")` is allowed before or after start and never mutates `initial_mean` or `_state`.
- `inject_candidates(mode="proposed")` is not supported for CMA-ES in Phase 1 because arbitrary samples cannot be safely inserted into the Rust covariance batch.
- `inject_candidates(mode="tracked")` is supported for reporting/checkpoint visibility.

## Task 1: Write CMA-ES External State Tests

**Files:**
- Create: `tests/unit/test_cmaes_external_state.py`

- [ ] **Step 1: Add fixtures**

```python
import pytest

from evocore import (
    CMAESOptimizer,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    WarmStartRecord,
)
from evocore.core import ConfigurationError
from evocore.search_space import encode_gene_values


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("y", "float", -5.0, 5.0),
        ]
    )


def _optimizer() -> CMAESOptimizer:
    return CMAESOptimizer(_space(), population_size=4, max_generations=3, seed=789)


def _records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0),
        WarmStartRecord(params={"x": -1.0, "y": -1.0}, score=5.0),
    ]
```

- [ ] **Step 2: Test runtime protocol and capabilities**

```python
def test_cmaes_implements_external_state_protocol() -> None:
    optimizer = _optimizer()

    assert isinstance(optimizer, ExternalStateOptimizer)
    capabilities = optimizer.external_state_capabilities()
    assert capabilities.warm_start_before_ask is True
    assert capabilities.warm_start_after_ask is False
    assert capabilities.proposed_candidate_injection is False
    assert capabilities.tracked_only_injection is True
```

- [ ] **Step 3: Test best-record mean warm start before first ask**

```python
def test_cmaes_warm_start_state_sets_initial_mean_from_best_record() -> None:
    optimizer = _optimizer()

    result = optimizer.warm_start(_records(), cma_mean_strategy="best")

    assert result.accepted_count == 3
    assert result.state_accepted_count == 3
    assert result.best_score == 20.0
    assert optimizer.initial_mean == encode_gene_values(_space(), [2.0, 2.0])
    assert [item.score for item in optimizer.top_candidates(2)] == [20.0, 10.0]
```

- [ ] **Step 4: Test top-k centroid mean warm start**

```python
def test_cmaes_warm_start_state_sets_initial_mean_from_top_k_centroid() -> None:
    optimizer = _optimizer()

    optimizer.warm_start(_records(), cma_mean_strategy="top_k_centroid", top_k=2)

    assert optimizer.initial_mean == encode_gene_values(_space(), [1.5, 1.5])
```

- [ ] **Step 5: Test post-start state warm start rejects**

```python
def test_cmaes_state_warm_start_rejects_after_state_exists() -> None:
    optimizer = _optimizer()
    optimizer.ask()

    with pytest.raises(ConfigurationError, match="before the first CMA-ES ask"):
        optimizer.warm_start(_records())
```

- [ ] **Step 6: Test tracked warm start after start**

```python
def test_cmaes_tracked_warm_start_after_start_records_scores_only() -> None:
    optimizer = _optimizer()
    optimizer.ask()

    result = optimizer.warm_start(_records(), mode="tracked")

    assert result.state_accepted_count == 0
    assert optimizer.initial_mean is None
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 7
```

- [ ] **Step 7: Test proposed injection rejection and tracked injection acceptance**

```python
def test_cmaes_injection_supports_tracked_only() -> None:
    optimizer = _optimizer()

    with pytest.raises(ConfigurationError, match="tracked"):
        optimizer.inject_candidates(_records(), mode="proposed")

    result = optimizer.inject_candidates(_records(), mode="tracked")

    assert len(result.accepted) == 3
    assert optimizer.state_summary().pending_batch_ids == ()
```

- [ ] **Step 8: Test checkpoint round trip after CMA warm start**

```python
def test_cmaes_external_state_checkpoint_round_trip(tmp_path) -> None:
    source = _optimizer()
    source.warm_start(_records(), cma_mean_strategy="best")
    checkpoint_path = tmp_path / "cmaes-external-state.evocore-checkpoint.json"
    source.save_checkpoint(checkpoint_path, source.ask_tell_checkpoint(metadata={"phase": "warm"}))

    restored = _optimizer()
    restored.resume_ask_tell_checkpoint(checkpoint_path)

    assert [item.score for item in restored.top_candidates(2)] == [20.0, 10.0]
    assert restored.state_summary().trusted_count == 3
```

- [ ] **Step 9: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_external_state.py -v
```

Expected: fails because CMA-ES does not yet implement the external-state methods.

## Task 2: Implement `CMAESExternalStateMixin`

**Files:**
- Create: `evocore/optimizers/cmaes/external.py`

- [ ] **Step 1: Create mixin with imports**

Use these imports:

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
from evocore.search_space import encode_gene_values
```

- [ ] **Step 2: Add capability method**

```python
class CMAESExternalStateMixin:
    """External-state integration API for CMAESOptimizer."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        return ExternalStateCapabilities(
            warm_start_before_ask=True,
            warm_start_after_ask=False,
            proposed_candidate_injection=False,
            state_candidate_injection=False,
            tracked_only_injection=True,
            population_snapshots=True,
            top_candidate_snapshots=True,
            cached_record_helpers=True,
        )
```

- [ ] **Step 3: Add CMA candidate scope helpers**

Implement:

- `_external_known_candidates()` returns all candidates.
- `_external_pending_candidates()` returns candidates from `_pending_batch_ids()`.
- `_external_scored_candidates()` returns candidates with scores.
- `_external_trusted_candidates()` returns scored candidates with at least one state-eligible score.
- `_external_existing_hashes()` hashes known candidates.

- [ ] **Step 4: Implement snapshots and top-k**

Rules:

- `scope="trusted"` maps to `_external_trusted_candidates()`.
- `scope="known"`, `"pending"`, and `"scored"` match the shared scope names.
- Snapshot `trusted_count` uses `self._trusted_count()`.
- `top_candidates` delegates to `top_candidate_snapshots`.

- [ ] **Step 5: Implement state warm start and mean construction**

Rules:

- Reject `mode="state"` when `self._state is not None`.
- Resolve all accepted `WarmStartRecord` values.
- Apply records to current-run candidates and record telemetry.
- Keep these candidates in `_candidates_by_id` and checkpoint batches.
- Build sorted accepted candidates by `state_comparison_score(self.direction)`.
- For `"best"`, set `self.initial_mean = encode_gene_values(self.gene_space, best_candidate.genes)`.
- For `"top_k_centroid"`, require `top_k` to be positive when provided; default to all accepted records when `top_k is None`.
- For centroid, average each decoded numeric gene over selected records, validate the centroid with `self.gene_space.validate_genes`, then set `self.initial_mean = encode_gene_values(self.gene_space, centroid)`.
- Return `UpdateResult` with `reason="warm_start_initial_mean"` acceptance decisions.

- [ ] **Step 6: Implement tracked warm start**

Rules:

- Create scored candidates and consumed batches.
- Apply records and append ask/tell events.
- Record telemetry.
- Do not mutate `initial_mean`.
- Do not call `_ensure_state`.
- Return `UpdateResult(state_accepted_count=0)`.

- [ ] **Step 7: Implement injection**

Rules:

- Reject `mode="proposed"` with `ConfigurationError("CMAESOptimizer supports inject_candidates(mode='tracked') in Phase 1.")`.
- `mode="tracked"` creates current-run candidates, consumed batches, and ask events.
- Do not apply warm-start scores during candidate injection.
- Return `InjectionResult`.

- [ ] **Step 8: Run focused CMA-ES tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_external_state.py -v
```

Expected: all CMA-ES external-state tests pass.

## Task 3: Wire the Mixin Into CMA-ES

**Files:**
- Modify: `evocore/optimizers/cmaes/engine.py`
- Modify: `evocore/optimizers/cmaes/__init__.py`

- [ ] **Step 1: Import the mixin in `engine.py`**

Add:

```python
from evocore.optimizers.cmaes.external import CMAESExternalStateMixin
```

- [ ] **Step 2: Add the mixin to class inheritance**

Change the class header:

```python
class CMAESOptimizer(CMAESExternalStateMixin, CMAESCheckpointingMixin, CMAESAskTellMixin):
```

- [ ] **Step 3: Export the mixin**

Modify `evocore/optimizers/cmaes/__init__.py`:

```python
from evocore.optimizers.cmaes.external import CMAESExternalStateMixin
```

Add `"CMAESExternalStateMixin"` to `__all__`.

- [ ] **Step 4: Run CMA-ES regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_external_state.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
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
.\.venv\Scripts\python.exe -m pytest tests/unit/test_cmaes_external_state.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
```

Expected: all commands pass.

- [ ] **Step 2: Commit CMA-ES implementation**

Run:

```powershell
git status --short
git add evocore/optimizers/cmaes/external.py evocore/optimizers/cmaes/engine.py evocore/optimizers/cmaes/__init__.py tests/unit/test_cmaes_external_state.py
git commit -m "feat(cmaes): support external state warm starts"
```

Expected: commit succeeds and contains only CMA-ES external-state changes.

## Compatibility Notes

- Existing CMA-ES behavior is unchanged when the new methods are not called.
- `warm_start(mode="state")` changes `initial_mean`; this is visible in `config_signature()` and checkpoint identity after use.
- Calling `ask_tell_checkpoint()` after warm start still creates Rust state through `_ensure_state()`, now using the warm-start mean.
- Post-start state injection is explicitly unsupported to avoid corrupting covariance adaptation.

## Self-Review Notes

- Spec coverage: CMA-ES prior-best initialization, top-k centroid mean construction, tracked external records, snapshots, and checkpoint visibility are covered.
- Scope boundary: restart strategies, covariance warm-start, and template-specific inner optimization helpers remain Phase 3 work.
