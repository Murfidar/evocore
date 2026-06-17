# Phase 1 Shared External State API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the shared public lifecycle API for warm starts, snapshots, top-k access, candidate injection results, and cached evaluation record helpers.

**Architecture:** Put the optimizer-neutral contract in `evocore.lifecycle.external`, then re-export it from `evocore.lifecycle` and `evocore`. Optimizer-specific plans will use these dataclasses and helpers while keeping optimizer state mutation inside GA, DE, and CMA-ES mixins.

**Tech Stack:** Python dataclasses, `typing.Protocol`, EvoCore lifecycle primitives, pytest, ruff.

---

## Context

Source design: `docs/superpowers/specs/2026-06-17-evocore-phase-1-external-integration-api-design.md`.

Graphify and `rg` inspection point to these existing surfaces:

- `evocore/lifecycle/records.py`: `Candidate`, `EvaluationRecord`, confidence semantics, value hashing.
- `evocore/lifecycle/telemetry.py`: `OptimizationTelemetry`, `UpdateResult`, `AcceptanceDecision`.
- `evocore/lifecycle/ask_tell_helpers.py`: event append helpers and `record_evaluation_telemetry`.
- `evocore/search_space/genes.py`: `GeneSpace.validate_genes`, `params_for`, and `value_hash`.
- `evocore/lifecycle/protocols.py`: structural optimizer protocol.
- `evocore/__init__.py` and `evocore/lifecycle/__init__.py`: public export surfaces.

## File Structure

- Create: `evocore/lifecycle/external.py`
  - Public frozen dataclasses.
  - Warm-start value resolution.
  - Read-only candidate and population snapshot builders.
  - Top-k selection helpers.
  - Cached evaluation record conversion helper.
  - JSON-safe metadata validation helper.
- Modify: `evocore/lifecycle/protocols.py`
  - Add `ExternalStateOptimizer` structural protocol.
- Modify: `evocore/lifecycle/__init__.py`
  - Re-export the new lifecycle API names.
- Modify: `evocore/__init__.py`
  - Re-export the public convenience names.
- Create: `tests/unit/test_external_state_core.py`
  - Unit tests for shared dataclasses, helpers, protocol runtime checks, and cached record conversion.
- Modify: `tests/unit/test_protocols.py`
  - Add a protocol-focused smoke test if the new protocol is not fully covered by `test_external_state_core.py`.

## Public API Names

Export these names from `evocore.lifecycle` and top-level `evocore`:

- `WarmStartRecord`
- `CandidateSnapshot`
- `PopulationSnapshot`
- `ExternalStateCapabilities`
- `InjectionResult`
- `ExternalStateOptimizer`
- `cached_records`

Keep helper functions such as `resolve_warm_start_values`, `build_candidate_snapshot`, and `build_population_snapshot` importable from `evocore.lifecycle.external`, but do not add them to top-level `evocore.__all__` unless they become documented public API.

## Task 1: Write Shared Core Tests

**Files:**
- Create: `tests/unit/test_external_state_core.py`

- [ ] **Step 1: Add tests for warm-start record validation and value resolution**

```python
import pytest

from evocore import Candidate, EvaluationRecord, Gene, GeneSpace, WarmStartRecord
from evocore.core import ConfigurationError, FitnessError
from evocore.lifecycle.external import resolve_warm_start_values


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("fast", "int", 1, 10),
            Gene("slow", "int", 5, 50),
            Gene("enabled", "bool"),
        ]
    )


def test_warm_start_record_resolves_params_in_gene_order() -> None:
    record = WarmStartRecord(
        params={"slow": 21, "enabled": True, "fast": 3},
        score=12.5,
        metadata={"source": "search_memory"},
    )

    assert resolve_warm_start_values(record, _space()) == (3, 21, True)


def test_warm_start_record_rejects_missing_values_and_params() -> None:
    with pytest.raises(ConfigurationError, match="values or params"):
        WarmStartRecord(score=1.0)


def test_warm_start_record_rejects_values_and_params_together() -> None:
    with pytest.raises(ConfigurationError, match="not both"):
        WarmStartRecord(values=(1, 5, False), params={"fast": 1}, score=1.0)


def test_warm_start_record_rejects_non_state_confidence() -> None:
    with pytest.raises(ConfigurationError, match="trusted_full or cached"):
        WarmStartRecord(values=(1, 5, False), score=1.0, confidence="partial")


def test_resolve_warm_start_values_rejects_unknown_param() -> None:
    record = WarmStartRecord(params={"fast": 3, "slow": 21, "enabled": True, "extra": 1}, score=1.0)

    with pytest.raises(ConfigurationError, match="unknown parameter"):
        resolve_warm_start_values(record, _space())
```

- [ ] **Step 2: Add tests for cached record conversion**

Append this test to `tests/unit/test_external_state_core.py`:

```python
from evocore import cached_records


def test_cached_records_converts_hash_mapping_to_evaluation_records() -> None:
    space = _space()
    candidate = Candidate(
        candidate_id="c-1",
        genes=[3, 21, True],
        params=space.params_for([3, 21, True]),
        batch_id="b-1",
        event_index=0,
        metadata={"candidate_source": "ask"},
    )
    cache_key = space.value_hash(candidate.genes)

    records = cached_records(
        [candidate],
        gene_space=space,
        cache={
            cache_key: {
                "score": 99.0,
                "metrics": {"fold": 2},
                "metadata": {"cache_reason": "exact_hash"},
            }
        },
        stage="search_memory",
        cost=0.0,
        metadata={"cache_table": "trusted_elites"},
    )

    assert records == (
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=99.0,
            confidence="cached",
            stage="search_memory",
            cost=0.0,
            metrics={"fold": 2},
            metadata={
                "cache_key": cache_key,
                "cache_table": "trusted_elites",
                "cache_reason": "exact_hash",
            },
        ),
    )


def test_cached_records_rejects_non_finite_cached_score() -> None:
    space = _space()
    candidate = Candidate(candidate_id="c-1", genes=[3, 21, True], batch_id="b-1")
    cache_key = space.value_hash(candidate.genes)

    with pytest.raises(FitnessError, match="finite score"):
        cached_records(
            [candidate],
            gene_space=space,
            cache={cache_key: {"score": float("nan")}},
            stage="search_memory",
        )
```

- [ ] **Step 3: Add tests for immutable snapshots and top-k sorting**

Append this test code:

```python
from evocore.lifecycle.external import (
    build_candidate_snapshot,
    build_population_snapshot,
    top_candidate_snapshots,
)
from evocore.lifecycle.telemetry import OptimizationTelemetry


def _scored_candidate(candidate_id: str, values: list[object], score: float) -> Candidate:
    candidate = Candidate(candidate_id=candidate_id, genes=list(values), batch_id="b-1")
    candidate.apply_record(
        EvaluationRecord(
            candidate_id=candidate_id,
            batch_id="b-1",
            score=score,
            confidence="cached",
            stage="warm_start",
            metadata={"fold": 1},
        )
    )
    return candidate


def test_candidate_snapshot_is_detached_from_candidate_mutation() -> None:
    space = _space()
    candidate = _scored_candidate("c-1", [3, 21, True], 10.0)

    snapshot = build_candidate_snapshot(candidate, gene_space=space, direction="maximize")
    candidate.metadata["record_metadata"]["fold"] = 9

    assert snapshot.candidate_id == "c-1"
    assert snapshot.candidate_hash == space.value_hash([3, 21, True])
    assert snapshot.metadata["record_metadata"]["fold"] == 1
    assert snapshot.score == 10.0


def test_population_snapshot_copies_telemetry_and_pending_batches() -> None:
    space = _space()
    telemetry = OptimizationTelemetry()
    telemetry.record_cached(1, stage="warm_start", cost=0.0)
    candidate = _scored_candidate("c-1", [3, 21, True], 10.0)

    snapshot = build_population_snapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        direction="maximize",
        event_index=4,
        pending_batch_ids=("b-open",),
        trusted_count=1,
        candidates=[candidate],
        gene_space=space,
        telemetry=telemetry,
    )
    telemetry.record_cached(1, stage="after_snapshot", cost=0.0)

    assert snapshot.optimizer_type == "GeneticAlgorithmOptimizer"
    assert snapshot.pending_batch_ids == ("b-open",)
    assert snapshot.telemetry.candidates_cached == 1


def test_top_candidate_snapshots_respects_direction_and_confidence() -> None:
    space = _space()
    low = _scored_candidate("c-low", [3, 21, True], 1.0)
    high = _scored_candidate("c-high", [4, 22, False], 9.0)

    selected = top_candidate_snapshots(
        [low, high],
        k=1,
        gene_space=space,
        direction="maximize",
        confidence=("trusted_full", "cached"),
    )

    assert [item.candidate_id for item in selected] == ["c-high"]
```

- [ ] **Step 4: Run the tests and verify the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py -v
```

Expected: fails with import errors for `WarmStartRecord`, `cached_records`, or `evocore.lifecycle.external`.

## Task 2: Implement `evocore.lifecycle.external`

**Files:**
- Create: `evocore/lifecycle/external.py`

- [ ] **Step 1: Add dataclasses and public literals**

Implement these names and signatures:

```python
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.records import (
    Candidate,
    CandidateOrigin,
    CandidateStatus,
    Direction,
    EvaluationConfidence,
    EvaluationRecord,
    ScoreObservation,
    is_state_update_confidence,
)
from evocore.lifecycle.telemetry import AcceptanceDecision, OptimizationTelemetry
from evocore.search_space import GeneSpace, GeneValue

WarmStartMode = Literal["state", "tracked"]
InjectionMode = Literal["proposed", "tracked"]
SnapshotScope = Literal["trusted", "known", "pending", "scored"]
CmaMeanStrategy = Literal["best", "top_k_centroid"]


@dataclass(frozen=True)
class WarmStartRecord:
    values: tuple[GeneValue, ...] | None = None
    params: Mapping[str, GeneValue] | None = None
    score: float = 0.0
    confidence: EvaluationConfidence = "cached"
    stage: str = "warm_start"
    cost: float = 0.0
    metrics: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSnapshot:
    candidate_id: str
    candidate_hash: str
    values: tuple[GeneValue, ...]
    params: Mapping[str, GeneValue] | None
    origin: CandidateOrigin
    batch_id: str
    event_index: int
    generation: int | None
    status: CandidateStatus
    stage: str | None
    confidence: EvaluationConfidence | None
    score: float | None
    scores: Mapping[str, ScoreObservation]
    cost: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class PopulationSnapshot:
    optimizer_type: str
    direction: Direction
    event_index: int
    pending_batch_ids: tuple[str, ...]
    trusted_count: int
    candidates: tuple[CandidateSnapshot, ...]
    telemetry: OptimizationTelemetry
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalStateCapabilities:
    warm_start_before_ask: bool
    warm_start_after_ask: bool
    proposed_candidate_injection: bool
    state_candidate_injection: bool
    tracked_only_injection: bool
    population_snapshots: bool
    top_candidate_snapshots: bool
    cached_record_helpers: bool


@dataclass(frozen=True)
class InjectionResult:
    accepted: tuple[CandidateSnapshot, ...]
    skipped_duplicates: tuple[CandidateSnapshot, ...]
    rejected: tuple[Mapping[str, object], ...]
    acceptance_decisions: tuple[AcceptanceDecision, ...] = ()
```

- [ ] **Step 2: Add dataclass validation**

Add `WarmStartRecord.__post_init__` with these exact rules:

- Exactly one of `values` or `params` must be provided.
- `score` must be finite.
- `confidence` must be `trusted_full` or `cached`.
- `stage` must be a non-empty string.
- `cost` must be finite and greater than or equal to zero.
- `metadata` and `metrics` must pass through `json_safe`.

Use `ConfigurationError` for invalid warm-start inputs.

- [ ] **Step 3: Add value and metadata helpers**

Implement:

```python
def json_safe_mapping(value: Mapping[str, object] | None, *, field_name: str) -> dict[str, object]:
    payload = json_safe(dict(value or {}))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


def resolve_warm_start_values(record: WarmStartRecord, gene_space: GeneSpace) -> tuple[GeneValue, ...]:
    if record.values is not None:
        values = tuple(record.values)
    else:
        params = dict(record.params or {})
        expected = set(gene_space.names)
        provided = set(params)
        unknown = sorted(provided - expected)
        missing = sorted(expected - provided)
        if unknown:
            raise ConfigurationError(f"WarmStartRecord contains unknown parameter(s): {unknown!r}.")
        if missing:
            raise ConfigurationError(f"WarmStartRecord missing parameter(s): {missing!r}.")
        values = tuple(params[name] for name in gene_space.names)
    gene_space.validate_genes(values)
    return values
```

- [ ] **Step 4: Add snapshot helpers**

Implement:

- `build_candidate_snapshot(candidate, *, gene_space, direction) -> CandidateSnapshot`
- `build_population_snapshot(...)-> PopulationSnapshot`
- `top_candidate_snapshots(candidates, *, k, gene_space, direction, confidence)-> tuple[CandidateSnapshot, ...]`

Rules:

- Never return the live `Candidate` object.
- Copy `params`, `scores`, `metadata`, and telemetry with `copy.deepcopy`.
- Use `candidate.candidate_hash(gene_space)` for duplicate identity.
- `score` is the best state-eligible raw score for the requested direction when finite; otherwise use the best observed finite raw score; otherwise `None`.
- `top_candidate_snapshots` rejects `k < 0` with `ConfigurationError` and returns an empty tuple for `k == 0`.

- [ ] **Step 5: Add cached record helper**

Implement:

```python
CacheLookup = Mapping[str, float | Mapping[str, object]] | Callable[[CandidateSnapshot], float | Mapping[str, object] | None]


def cached_records(
    candidates: Sequence[Candidate],
    *,
    gene_space: GeneSpace,
    cache: CacheLookup,
    stage: str = "cached",
    cost: float = 0.0,
    metadata: Mapping[str, object] | None = None,
) -> tuple[EvaluationRecord, ...]:
    if not stage:
        raise FitnessError("cached_records stage must be non-empty.")
    if not math.isfinite(float(cost)) or float(cost) < 0.0:
        raise FitnessError("cached_records cost must be finite and >= 0.")

    helper_metadata = json_safe_mapping(metadata, field_name="metadata")
    output: list[EvaluationRecord] = []
    for candidate in candidates:
        snapshot = build_candidate_snapshot(candidate, gene_space=gene_space, direction="maximize")
        cache_key = snapshot.candidate_hash
        raw = cache(snapshot) if callable(cache) else cache.get(cache_key)
        if raw is None:
            continue

        metrics: dict[str, object] = {}
        entry_metadata: dict[str, object] = {}
        if isinstance(raw, Mapping):
            if "score" not in raw:
                raise FitnessError("cached record mapping requires a score.")
            score_value = raw["score"]
            raw_metrics = raw.get("metrics", {})
            raw_metadata = raw.get("metadata", {})
            if not isinstance(raw_metrics, Mapping):
                raise FitnessError("cached record metrics must be a mapping.")
            if not isinstance(raw_metadata, Mapping):
                raise FitnessError("cached record metadata must be a mapping.")
            metrics = dict(raw_metrics)
            entry_metadata = dict(raw_metadata)
        else:
            score_value = raw

        score = float(score_value)
        if not math.isfinite(score):
            raise FitnessError("cached record score must be finite.")

        record_metadata = dict(helper_metadata)
        record_metadata["cache_key"] = cache_key
        record_metadata.update(entry_metadata)
        output.append(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=score,
                confidence="cached",
                stage=stage,
                cost=float(cost),
                metrics=metrics,
                metadata=record_metadata,
            )
        )
    return tuple(output)
```

Implementation rules:

- Reject an empty `stage`.
- Reject negative or non-finite `cost`.
- For mapping caches, use `candidate.candidate_hash(gene_space)` as the lookup key.
- For callable caches, pass a detached `CandidateSnapshot`.
- Accept cache values as either a raw finite score or a mapping with `score`, optional `metrics`, and optional `metadata`.
- Merge helper-level metadata first, then cache-entry metadata. Cache-entry keys win.
- Always include `"cache_key": candidate_hash` in the record metadata.
- Return an `EvaluationRecord` with `confidence="cached"` and `batch_id=candidate.batch_id`.
- Skip candidates not found in the cache instead of creating rejected records.

- [ ] **Step 6: Run shared core tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py -v
```

Expected: all tests in `test_external_state_core.py` pass.

## Task 3: Add the Structural Protocol and Exports

**Files:**
- Modify: `evocore/lifecycle/protocols.py`
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_protocols.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Extend `evocore/lifecycle/protocols.py`**

Add imports from `typing`:

```python
from typing import Literal, Protocol, runtime_checkable
```

Import the new shared API:

```python
from evocore.lifecycle.external import (
    CmaMeanStrategy,
    ExternalStateCapabilities,
    InjectionMode,
    InjectionResult,
    PopulationSnapshot,
    SnapshotScope,
    WarmStartMode,
    WarmStartRecord,
)
from evocore.lifecycle.records import CandidateOrigin, EvaluationConfidence
```

Add this protocol below `Optimizer`:

```python
@runtime_checkable
class ExternalStateOptimizer(Optimizer, Protocol):
    """Structural protocol for optimizers with external-state integration APIs."""

    def external_state_capabilities(self) -> ExternalStateCapabilities:
        raise NotImplementedError

    def warm_start(
        self,
        records: Sequence[WarmStartRecord],
        *,
        deduplicate: bool = True,
        mode: WarmStartMode = "state",
        cma_mean_strategy: CmaMeanStrategy = "best",
        top_k: int | None = None,
    ) -> UpdateResult:
        raise NotImplementedError

    def inject_candidates(
        self,
        records: Sequence[WarmStartRecord],
        *,
        origin: CandidateOrigin = "memory_seed",
        mode: InjectionMode = "proposed",
        deduplicate: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> InjectionResult:
        raise NotImplementedError

    def candidate_snapshot(
        self,
        *,
        scope: SnapshotScope = "trusted",
    ) -> PopulationSnapshot:
        raise NotImplementedError

    def top_candidates(
        self,
        k: int = 10,
        *,
        scope: SnapshotScope = "trusted",
        confidence: tuple[EvaluationConfidence, ...] = ("trusted_full", "cached"),
    ) -> tuple:
        raise NotImplementedError
```

Make sure `Mapping` is imported from `collections.abc`.

- [ ] **Step 2: Export public names from lifecycle and top-level packages**

Add the new names to `evocore/lifecycle/__init__.py` imports and `__all__`.

Add the user-facing names to `evocore/__init__.py` imports and `__all__`:

- `CandidateSnapshot`
- `ExternalStateCapabilities`
- `ExternalStateOptimizer`
- `InjectionResult`
- `PopulationSnapshot`
- `WarmStartRecord`
- `cached_records`

- [ ] **Step 3: Add protocol and package export tests**

Append to `tests/unit/test_protocols.py`:

```python
from evocore import ExternalStateOptimizer


def test_external_state_optimizer_protocol_is_runtime_checkable() -> None:
    assert isinstance(ExternalStateOptimizer, type)
```

Append to `tests/unit/test_package_init.py`:

```python
def test_external_state_public_exports() -> None:
    from evocore import (
        CandidateSnapshot,
        ExternalStateCapabilities,
        ExternalStateOptimizer,
        InjectionResult,
        PopulationSnapshot,
        WarmStartRecord,
        cached_records,
    )

    assert WarmStartRecord is not None
    assert CandidateSnapshot is not None
    assert PopulationSnapshot is not None
    assert ExternalStateCapabilities is not None
    assert ExternalStateOptimizer is not None
    assert InjectionResult is not None
    assert cached_records is not None
```

- [ ] **Step 4: Run protocol and package tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_protocols.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

## Task 4: Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run formatting and lint checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both commands pass.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_protocols.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Commit shared API work**

Run:

```powershell
git status --short
git add evocore/lifecycle/external.py evocore/lifecycle/protocols.py evocore/lifecycle/__init__.py evocore/__init__.py tests/unit/test_external_state_core.py tests/unit/test_protocols.py tests/unit/test_package_init.py
git commit -m "feat(lifecycle): add external state API contract"
```

Expected: commit succeeds and contains only shared lifecycle API changes.

## Compatibility Notes

- This is additive public API.
- No existing method signatures change.
- Existing checkpoints remain readable because this plan does not alter checkpoint payload schemas.
- Snapshot objects expose copied candidate data; users must not rely on mutating snapshots to affect optimizer state.
- `WarmStartRecord` rejects non-JSON-safe metadata after `json_safe` conversion if the result is not a mapping.

## Self-Review Notes

- Spec coverage: shared public API shape, cached evaluation helper, metadata preservation, top-k helper, and protocol contract are covered.
- Type consistency: the plan uses `stage`, `events`, and `optimizer_type`, matching repository vocabulary.
- Scope boundary: optimizer state mutation is intentionally left to the GA, DE, and CMA-ES implementation plans.
