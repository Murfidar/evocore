# Phase 2A Archive Selection Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable archive/search-memory and survivor-selection utilities over Phase 1 public snapshots.

**Architecture:** Implement archive and selection as lifecycle utility modules that consume `CandidateSnapshot` and `PopulationSnapshot`, then emit `WarmStartRecord` objects or explicit selection decisions. Optimizers remain the only owners of ask/tell state; this phase adds no optimizer-private state access.

**Tech Stack:** Python dataclasses, EvoCore lifecycle snapshots, `json_safe` serialization helpers, pytest, ruff, MkDocs Markdown.

---

## Dependency

Complete and merge the Phase 1 external-state API before starting this plan:

- `docs/superpowers/plans/2026-06-17-phase-1-shared-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-ga-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-de-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-cmaes-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-docs-compatibility-and-recipes.md`

Source design:

- `docs/superpowers/specs/2026-06-17-evocore-phase-2-expensive-optimization-toolkit-design.md`

## File Structure

- Create: `evocore/lifecycle/archives.py`
  - Archive dataclasses.
  - Duplicate policy handling.
  - Snapshot ingestion.
  - Export to `WarmStartRecord`.
  - Schema-versioned JSON-safe archive export/import.
- Create: `evocore/lifecycle/selection.py`
  - Selection quota and cap dataclasses.
  - Deterministic survivor selection.
  - Duplicate suppression.
  - Selection reasons and summaries.
- Modify: `evocore/lifecycle/__init__.py`
  - Re-export lifecycle archive and selection names.
- Modify: `evocore/__init__.py`
  - Re-export only the core convenience names.
- Create: `tests/unit/test_lifecycle_archives.py`
  - Archive policy, export, and JSON round-trip tests.
- Create: `tests/unit/test_lifecycle_selection.py`
  - Top-k, duplicate, quota, cap, and missing metadata tests.
- Create: `tests/unit/test_phase2a_external_optimizer_integration.py`
  - GA, DE, and CMA-ES snapshot ingestion smoke tests.
- Modify: `tests/unit/test_package_init.py`
  - Top-level export smoke tests.
- Modify: `docs/site/api.md`
  - API reference for archive and selection modules.
- Modify: `docs/site/expensive-external-evaluations.md`
  - Search memory archive and survivor selection recipes.
- Modify: `CHANGELOG.md`
  - Add Phase 2A public API entry.

## Public API Names

Export these names from `evocore.lifecycle`:

- `ARCHIVE_SCHEMA_VERSION`
- `ArchiveEntry`
- `ArchiveExport`
- `ArchivePolicy`
- `CandidateArchive`
- `DuplicatePolicy`
- `DuplicateSelectionPolicy`
- `FamilyQuota`
- `MissingMetadataPolicy`
- `SelectionDecision`
- `SelectionResult`
- `SpecialistCap`
- `select_candidates`

Export these names from top-level `evocore`:

- `CandidateArchive`
- `FamilyQuota`
- `SelectionResult`
- `SpecialistCap`
- `select_candidates`

Keep lower-level policy literals importable from `evocore.lifecycle`, not top-level `evocore`, unless later docs show they are common enough.

## Task 1: Write Archive Tests

**Files:**
- Create: `tests/unit/test_lifecycle_archives.py`

- [ ] **Step 1: Add archive fixtures and duplicate policy tests**

Create `tests/unit/test_lifecycle_archives.py`:

```python
import pytest

from evocore import (
    CandidateArchive,
    CandidateSnapshot,
    PopulationSnapshot,
    WarmStartRecord,
)
from evocore.lifecycle import OptimizationTelemetry


def _snapshot(
    candidate_id: str,
    candidate_hash: str,
    score: float,
    *,
    family: str = "baseline",
    confidence: str = "cached",
) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        values=(float(score),),
        params={"x": float(score)},
        origin="memory_seed",
        batch_id="batch-1",
        event_index=1,
        generation=None,
        status="trusted",
        stage="archive",
        confidence=confidence,
        score=score,
        scores={},
        cost=0.0,
        metadata={"family": family, "record_metadata": {"fold": 1}},
    )


def test_candidate_archive_keep_first_duplicate_policy() -> None:
    archive = CandidateArchive(duplicate_policy="keep_first", score_direction="maximize")

    archive.add_candidates(
        [
            _snapshot("c-1", "same", 1.0),
            _snapshot("c-2", "same", 9.0),
        ],
        source="stage1",
    )

    assert [entry.candidate_id for entry in archive.entries] == ["c-1"]
    assert archive.entries[0].score == 1.0


def test_candidate_archive_keep_latest_duplicate_policy() -> None:
    archive = CandidateArchive(duplicate_policy="keep_latest", score_direction="maximize")

    archive.add_candidates([_snapshot("c-1", "same", 1.0)], source="stage1")
    archive.add_candidates([_snapshot("c-2", "same", 9.0)], source="stage2")

    assert [entry.candidate_id for entry in archive.entries] == ["c-2"]
    assert archive.entries[0].source == "stage2"


def test_candidate_archive_keep_best_duplicate_policy_respects_direction() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="minimize")

    archive.add_candidates(
        [
            _snapshot("c-1", "same", 9.0),
            _snapshot("c-2", "same", 1.0),
        ],
        source="stage1",
    )

    assert [entry.candidate_id for entry in archive.entries] == ["c-2"]
    assert archive.entries[0].score == 1.0
```

- [ ] **Step 2: Add warm-start export and population ingestion tests**

Append:

```python
def test_candidate_archive_exports_warm_start_records() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
    archive.add_candidates(
        [
            _snapshot("c-1", "hash-a", 1.0, family="a"),
            _snapshot("c-2", "hash-b", 9.0, family="b"),
        ],
        source="stage1",
    )

    records = archive.to_warm_start_records(k=1, stage="refine", confidence="cached")

    assert records == (
        WarmStartRecord(
            params={"x": 9.0},
            score=9.0,
            confidence="cached",
            stage="refine",
            cost=0.0,
            metrics={},
            metadata={
                "archive_candidate_id": "c-2",
                "archive_candidate_hash": "hash-b",
                "archive_source": "stage1",
                "family": "b",
                "record_metadata": {"fold": 1},
            },
        ),
    )


def test_candidate_archive_add_population_inherits_direction() -> None:
    population = PopulationSnapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        direction="minimize",
        event_index=4,
        pending_batch_ids=(),
        trusted_count=2,
        candidates=(
            _snapshot("c-1", "hash-a", 5.0),
            _snapshot("c-2", "hash-b", 1.0),
        ),
        telemetry=OptimizationTelemetry(),
    )
    archive = CandidateArchive()

    archive.add_population(population, source="trusted")
    records = archive.to_warm_start_records(k=1)

    assert records[0].score == 1.0
```

- [ ] **Step 3: Add JSON round-trip and invalid snapshot tests**

Append:

```python
from evocore.core import ConfigurationError
from evocore.lifecycle.archives import ARCHIVE_SCHEMA_VERSION


def test_candidate_archive_round_trips_json_safe_dict() -> None:
    archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
    archive.add_candidates([_snapshot("c-1", "hash-a", 2.0)], source="stage1")

    payload = archive.to_dict()
    restored = CandidateArchive.from_dict(payload)

    assert payload["schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert restored.duplicate_policy == "keep_best"
    assert restored.score_direction == "maximize"
    assert restored.entries == archive.entries


def test_candidate_archive_rejects_snapshot_without_score() -> None:
    snapshot = _snapshot("c-1", "hash-a", 1.0)
    snapshot = CandidateSnapshot(
        candidate_id=snapshot.candidate_id,
        candidate_hash=snapshot.candidate_hash,
        values=snapshot.values,
        params=snapshot.params,
        origin=snapshot.origin,
        batch_id=snapshot.batch_id,
        event_index=snapshot.event_index,
        generation=snapshot.generation,
        status=snapshot.status,
        stage=snapshot.stage,
        confidence=snapshot.confidence,
        score=None,
        scores=snapshot.scores,
        cost=snapshot.cost,
        metadata=snapshot.metadata,
    )
    archive = CandidateArchive()

    with pytest.raises(ConfigurationError, match="finite score"):
        archive.add_candidates([snapshot], source="stage1")
```

- [ ] **Step 4: Run archive tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_archives.py -v
```

Expected: fails with import errors for `CandidateArchive` and `evocore.lifecycle.archives`.

## Task 2: Implement Archive Utilities

**Files:**
- Create: `evocore/lifecycle/archives.py`

- [ ] **Step 1: Add dataclasses, literals, and validation helpers**

Create `evocore/lifecycle/archives.py`:

```python
"""Archive utilities for expensive external optimization workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe, stable_json_dumps
from evocore.lifecycle.external import CandidateSnapshot, PopulationSnapshot, WarmStartRecord
from evocore.lifecycle.records import Direction, EvaluationConfidence, score_for_direction
from evocore.search_space import GeneValue

ARCHIVE_SCHEMA_VERSION = 1
DuplicatePolicy = Literal["keep_first", "keep_latest", "keep_best"]


def _validate_direction(direction: Direction) -> Direction:
    if direction not in ("maximize", "minimize"):
        raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
    return direction


def _validate_duplicate_policy(policy: DuplicatePolicy) -> DuplicatePolicy:
    if policy not in ("keep_first", "keep_latest", "keep_best"):
        raise ConfigurationError(
            "duplicate_policy must be 'keep_first', 'keep_latest', or 'keep_best'."
        )
    return policy


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    payload = json_safe(value)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload
```

- [ ] **Step 2: Add archive dataclasses**

Append:

```python
@dataclass(frozen=True)
class ArchiveEntry:
    candidate_id: str
    candidate_hash: str
    values: tuple[GeneValue, ...]
    params: dict[str, GeneValue] | None
    score: float
    confidence: EvaluationConfidence
    stage: str
    cost: float
    metrics: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    source: str = "archive"
    inserted_index: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ConfigurationError("ArchiveEntry candidate_id must be non-empty.")
        if not self.candidate_hash:
            raise ConfigurationError("ArchiveEntry candidate_hash must be non-empty.")
        if self.confidence not in ("trusted_full", "cached"):
            raise ConfigurationError("ArchiveEntry confidence must be trusted_full or cached.")
        if not self.stage:
            raise ConfigurationError("ArchiveEntry stage must be non-empty.")
        if not math.isfinite(float(self.score)):
            raise ConfigurationError("ArchiveEntry score must be finite.")
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0.0:
            raise ConfigurationError("ArchiveEntry cost must be finite and >= 0.")
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "metrics", _json_mapping(self.metrics, field_name="metrics"))
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, field_name="metadata"),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CandidateSnapshot,
        *,
        source: str,
        inserted_index: int,
    ) -> ArchiveEntry:
        if snapshot.score is None or not math.isfinite(float(snapshot.score)):
            raise ConfigurationError("ArchiveEntry requires a finite score.")
        if snapshot.confidence not in ("trusted_full", "cached"):
            raise ConfigurationError("ArchiveEntry confidence must be trusted_full or cached.")
        if not snapshot.stage:
            raise ConfigurationError("ArchiveEntry stage must be non-empty.")
        return cls(
            candidate_id=snapshot.candidate_id,
            candidate_hash=snapshot.candidate_hash,
            values=tuple(snapshot.values),
            params=None if snapshot.params is None else dict(snapshot.params),
            score=float(snapshot.score),
            confidence=snapshot.confidence,
            stage=snapshot.stage,
            cost=float(snapshot.cost),
            metrics=dict(snapshot.metadata.get("metrics", {})),
            metadata=dict(snapshot.metadata),
            source=source,
            inserted_index=inserted_index,
        )

    def to_warm_start_record(
        self,
        *,
        stage: str | None = None,
        confidence: EvaluationConfidence | None = None,
    ) -> WarmStartRecord:
        metadata = dict(self.metadata)
        metadata.update(
            {
                "archive_candidate_id": self.candidate_id,
                "archive_candidate_hash": self.candidate_hash,
                "archive_source": self.source,
            }
        )
        return WarmStartRecord(
            values=None if self.params is not None else self.values,
            params=self.params,
            score=self.score,
            confidence=confidence or self.confidence,
            stage=stage or self.stage,
            cost=self.cost,
            metrics=dict(self.metrics),
            metadata=metadata,
        )
```

- [ ] **Step 3: Add `CandidateArchive`**

Append:

```python
@dataclass(frozen=True)
class ArchiveExport:
    records: tuple[WarmStartRecord, ...]
    selected_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ArchivePolicy:
    duplicate_policy: DuplicatePolicy = "keep_best"
    score_direction: Direction = "maximize"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "duplicate_policy",
            _validate_duplicate_policy(self.duplicate_policy),
        )
        object.__setattr__(self, "score_direction", _validate_direction(self.score_direction))


class CandidateArchive:
    """User-owned durable archive for scored candidate snapshots."""

    def __init__(
        self,
        *,
        duplicate_policy: DuplicatePolicy = "keep_best",
        score_direction: Direction = "maximize",
    ) -> None:
        self.duplicate_policy = _validate_duplicate_policy(duplicate_policy)
        self.score_direction = _validate_direction(score_direction)
        self._entries_by_hash: dict[str, ArchiveEntry] = {}
        self._next_inserted_index = 0

    @property
    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(sorted(self._entries_by_hash.values(), key=lambda entry: entry.inserted_index))

    def add_population(self, snapshot: PopulationSnapshot, *, source: str) -> tuple[ArchiveEntry, ...]:
        self.score_direction = snapshot.direction
        return self.add_candidates(snapshot.candidates, source=source)

    def add_candidates(
        self,
        candidates: tuple[CandidateSnapshot, ...] | list[CandidateSnapshot],
        *,
        source: str,
    ) -> tuple[ArchiveEntry, ...]:
        accepted: list[ArchiveEntry] = []
        for snapshot in candidates:
            entry = ArchiveEntry.from_snapshot(
                snapshot,
                source=source,
                inserted_index=self._next_inserted_index,
            )
            self._next_inserted_index += 1
            existing = self._entries_by_hash.get(entry.candidate_hash)
            if existing is None or self._should_replace(existing, entry):
                self._entries_by_hash[entry.candidate_hash] = entry
                accepted.append(entry)
        return tuple(accepted)

    def _should_replace(self, existing: ArchiveEntry, incoming: ArchiveEntry) -> bool:
        if self.duplicate_policy == "keep_first":
            return False
        if self.duplicate_policy == "keep_latest":
            return True
        existing_score = score_for_direction(existing.score, self.score_direction)
        incoming_score = score_for_direction(incoming.score, self.score_direction)
        if incoming_score == existing_score:
            return incoming.inserted_index > existing.inserted_index
        return incoming_score > existing_score

    def ranked_entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    score_for_direction(entry.score, self.score_direction),
                    -entry.inserted_index,
                ),
                reverse=True,
            )
        )

    def to_warm_start_records(
        self,
        *,
        k: int | None = None,
        stage: str | None = None,
        confidence: EvaluationConfidence | None = None,
    ) -> tuple[WarmStartRecord, ...]:
        if k is not None and int(k) < 0:
            raise ConfigurationError("k must be >= 0.")
        entries = self.ranked_entries() if k is None else self.ranked_entries()[: int(k)]
        return tuple(
            entry.to_warm_start_record(stage=stage, confidence=confidence)
            for entry in entries
        )
```

- [ ] **Step 4: Add archive serialization**

Append:

```python
    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "duplicate_policy": self.duplicate_policy,
            "score_direction": self.score_direction,
            "next_inserted_index": self._next_inserted_index,
            "entries": [
                {
                    "candidate_id": entry.candidate_id,
                    "candidate_hash": entry.candidate_hash,
                    "values": list(entry.values),
                    "params": entry.params,
                    "score": entry.score,
                    "confidence": entry.confidence,
                    "stage": entry.stage,
                    "cost": entry.cost,
                    "metrics": entry.metrics,
                    "metadata": entry.metadata,
                    "source": entry.source,
                    "inserted_index": entry.inserted_index,
                }
                for entry in self.entries
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return stable_json_dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CandidateArchive:
        if payload.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
            raise ConfigurationError("Unsupported CandidateArchive schema_version.")
        archive = cls(
            duplicate_policy=payload.get("duplicate_policy", "keep_best"),
            score_direction=payload.get("score_direction", "maximize"),
        )
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise ConfigurationError("CandidateArchive entries must be a list.")
        for raw in entries:
            if not isinstance(raw, dict):
                raise ConfigurationError("CandidateArchive entry must be a mapping.")
            entry = ArchiveEntry(
                candidate_id=str(raw["candidate_id"]),
                candidate_hash=str(raw["candidate_hash"]),
                values=tuple(raw["values"]),
                params=raw.get("params"),
                score=float(raw["score"]),
                confidence=raw["confidence"],
                stage=str(raw["stage"]),
                cost=float(raw.get("cost", 0.0)),
                metrics=dict(raw.get("metrics", {})),
                metadata=dict(raw.get("metadata", {})),
                source=str(raw.get("source", "archive")),
                inserted_index=int(raw.get("inserted_index", len(archive._entries_by_hash))),
            )
            archive._entries_by_hash[entry.candidate_hash] = entry
            archive._next_inserted_index = max(archive._next_inserted_index, entry.inserted_index + 1)
        return archive
```

Then add:

```python
__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "ArchiveEntry",
    "ArchiveExport",
    "ArchivePolicy",
    "CandidateArchive",
    "DuplicatePolicy",
]
```

- [ ] **Step 5: Run archive tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_archives.py -v
```

Expected: all archive tests pass.

## Task 3: Write Selection Tests

**Files:**
- Create: `tests/unit/test_lifecycle_selection.py`

- [ ] **Step 1: Add selection fixtures and deterministic top-k tests**

Create `tests/unit/test_lifecycle_selection.py`:

```python
import pytest

from evocore import CandidateSnapshot, FamilyQuota, SpecialistCap, select_candidates
from evocore.core import ConfigurationError


def _candidate(
    candidate_id: str,
    candidate_hash: str,
    score: float | None,
    *,
    family: str | None = "core",
    specialist: str | None = None,
) -> CandidateSnapshot:
    metadata = {}
    if family is not None:
        metadata["family"] = family
    if specialist is not None:
        metadata["specialist"] = specialist
    return CandidateSnapshot(
        candidate_id=candidate_id,
        candidate_hash=candidate_hash,
        values=(score or 0.0,),
        params=None,
        origin="memory_seed",
        batch_id="batch-1",
        event_index=1,
        generation=None,
        status="trusted",
        stage="full",
        confidence="trusted_full",
        score=score,
        scores={},
        cost=0.0,
        metadata=metadata,
    )


def test_select_candidates_top_k_is_direction_aware_and_deterministic() -> None:
    result = select_candidates(
        [
            _candidate("c-low", "h-low", 1.0),
            _candidate("c-high", "h-high", 9.0),
            _candidate("c-tie", "h-tie", 9.0),
        ],
        k=2,
        score_direction="maximize",
    )

    assert [item.candidate_id for item in result.selected] == ["c-high", "c-tie"]
    assert result.summary["selected"] == 2
    assert [decision.reason for decision in result.decisions if not decision.selected] == [
        "overflow"
    ]


def test_select_candidates_minimize_prefers_lower_scores() -> None:
    result = select_candidates(
        [
            _candidate("c-low", "h-low", 1.0),
            _candidate("c-high", "h-high", 9.0),
        ],
        k=1,
        score_direction="minimize",
    )

    assert [item.candidate_id for item in result.selected] == ["c-low"]
```

- [ ] **Step 2: Add duplicate, family quota, and specialist cap tests**

Append:

```python
def test_select_candidates_suppresses_duplicate_hashes() -> None:
    result = select_candidates(
        [
            _candidate("c-first", "same", 8.0),
            _candidate("c-second", "same", 7.0),
            _candidate("c-third", "other", 6.0),
        ],
        k=3,
        score_direction="maximize",
        duplicate_policy="suppress",
    )

    assert [item.candidate_id for item in result.selected] == ["c-first", "c-third"]
    assert result.rejected[0].candidate_id == "c-second"
    assert result.decisions[1].reason == "duplicate"


def test_select_candidates_enforces_family_quota() -> None:
    result = select_candidates(
        [
            _candidate("c-a1", "h-a1", 9.0, family="a"),
            _candidate("c-a2", "h-a2", 8.0, family="a"),
            _candidate("c-b1", "h-b1", 7.0, family="b"),
        ],
        k=3,
        score_direction="maximize",
        quotas=[FamilyQuota(metadata_key="family", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-a1", "c-b1"]
    assert result.rejected[0].candidate_id == "c-a2"
    assert result.decisions[1].reason == "quota:family"


def test_select_candidates_enforces_specialist_cap() -> None:
    result = select_candidates(
        [
            _candidate("c-s1", "h-s1", 9.0, specialist="fast"),
            _candidate("c-s2", "h-s2", 8.0, specialist="fast"),
            _candidate("c-s3", "h-s3", 7.0, specialist="slow"),
        ],
        k=3,
        score_direction="maximize",
        caps=[SpecialistCap(metadata_key="specialist", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-s1", "c-s3"]
    assert result.rejected[0].candidate_id == "c-s2"
    assert result.decisions[1].reason == "cap:specialist"
```

- [ ] **Step 3: Add missing metadata and no-score tests**

Append:

```python
def test_select_candidates_missing_metadata_defaults_to_unknown_bucket() -> None:
    result = select_candidates(
        [
            _candidate("c-1", "h-1", 9.0, family=None),
            _candidate("c-2", "h-2", 8.0, family=None),
        ],
        k=2,
        score_direction="maximize",
        quotas=[FamilyQuota(metadata_key="family", max_count=1)],
    )

    assert [item.candidate_id for item in result.selected] == ["c-1"]
    assert result.rejected[0].candidate_id == "c-2"


def test_select_candidates_strict_missing_metadata_raises() -> None:
    with pytest.raises(ConfigurationError, match="missing metadata key"):
        select_candidates(
            [_candidate("c-1", "h-1", 9.0, family=None)],
            k=1,
            score_direction="maximize",
            quotas=[FamilyQuota(metadata_key="family", max_count=1)],
            missing_metadata="error",
        )


def test_select_candidates_rejects_candidates_without_score() -> None:
    result = select_candidates(
        [_candidate("c-1", "h-1", None)],
        k=1,
        score_direction="maximize",
    )

    assert result.selected == ()
    assert result.rejected[0].candidate_id == "c-1"
    assert result.decisions[0].reason == "no_score"
```

- [ ] **Step 4: Run selection tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_selection.py -v
```

Expected: fails with import errors for `FamilyQuota`, `SpecialistCap`, and `select_candidates`.

## Task 4: Implement Selection Utilities

**Files:**
- Create: `evocore/lifecycle/selection.py`

- [ ] **Step 1: Add dataclasses and policy literals**

Create `evocore/lifecycle/selection.py`:

```python
"""Selection utilities for public candidate snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import CandidateSnapshot, WarmStartRecord
from evocore.lifecycle.records import Direction, score_for_direction

DuplicateSelectionPolicy = Literal["allow", "suppress"]
MissingMetadataPolicy = Literal["unknown", "error"]


@dataclass(frozen=True)
class FamilyQuota:
    metadata_key: str
    max_count: int
    unknown_value: str = "unknown"

    def __post_init__(self) -> None:
        if not self.metadata_key:
            raise ConfigurationError("FamilyQuota metadata_key must be non-empty.")
        if int(self.max_count) <= 0:
            raise ConfigurationError("FamilyQuota max_count must be positive.")


@dataclass(frozen=True)
class SpecialistCap:
    metadata_key: str
    max_count: int
    unknown_value: str = "unknown"

    def __post_init__(self) -> None:
        if not self.metadata_key:
            raise ConfigurationError("SpecialistCap metadata_key must be non-empty.")
        if int(self.max_count) <= 0:
            raise ConfigurationError("SpecialistCap max_count must be positive.")


@dataclass(frozen=True)
class SelectionDecision:
    candidate_id: str
    candidate_hash: str
    selected: bool
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[CandidateSnapshot, ...]
    rejected: tuple[CandidateSnapshot, ...]
    skipped: tuple[CandidateSnapshot, ...]
    decisions: tuple[SelectionDecision, ...]
    summary: dict[str, object]

    def to_warm_start_records(
        self,
        *,
        stage: str,
        confidence: str = "cached",
    ) -> tuple[WarmStartRecord, ...]:
        return tuple(
            WarmStartRecord(
                values=None if candidate.params is not None else candidate.values,
                params=candidate.params,
                score=float(candidate.score),
                confidence=confidence,
                stage=stage,
                cost=float(candidate.cost),
                metadata=dict(candidate.metadata),
            )
            for candidate in self.selected
        )
```

- [ ] **Step 2: Add helper functions**

Append:

```python
def _metadata_bucket(
    candidate: CandidateSnapshot,
    key: str,
    *,
    missing_metadata: MissingMetadataPolicy,
    unknown_value: str,
) -> str:
    if key not in candidate.metadata:
        if missing_metadata == "error":
            raise ConfigurationError(f"Candidate {candidate.candidate_id!r} missing metadata key {key!r}.")
        return unknown_value
    return str(candidate.metadata[key])


def _ranked_candidates(
    candidates: list[CandidateSnapshot],
    *,
    score_direction: Direction,
) -> list[CandidateSnapshot]:
    if score_direction not in ("maximize", "minimize"):
        raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
    return sorted(
        candidates,
        key=lambda item: (
            -score_for_direction(float(item.score), score_direction),
            int(item.event_index),
            item.candidate_id,
        ),
    )


def _summary(selected: list[CandidateSnapshot], rejected: list[CandidateSnapshot]) -> dict[str, object]:
    families: dict[str, int] = {}
    for candidate in selected:
        family = str(candidate.metadata.get("family", "unknown"))
        families[family] = families.get(family, 0) + 1
    return {
        "selected": len(selected),
        "rejected": len(rejected),
        "selected_by_family": families,
    }
```

- [ ] **Step 3: Add `select_candidates`**

Append:

```python
def select_candidates(
    candidates: list[CandidateSnapshot] | tuple[CandidateSnapshot, ...],
    *,
    k: int,
    score_direction: Direction,
    duplicate_policy: DuplicateSelectionPolicy = "suppress",
    quotas: list[FamilyQuota] | tuple[FamilyQuota, ...] = (),
    caps: list[SpecialistCap] | tuple[SpecialistCap, ...] = (),
    missing_metadata: MissingMetadataPolicy = "unknown",
) -> SelectionResult:
    if int(k) < 0:
        raise ConfigurationError("k must be >= 0.")
    if duplicate_policy not in ("allow", "suppress"):
        raise ConfigurationError("duplicate_policy must be 'allow' or 'suppress'.")
    if missing_metadata not in ("unknown", "error"):
        raise ConfigurationError("missing_metadata must be 'unknown' or 'error'.")

    decisions: list[SelectionDecision] = []
    rejected: list[CandidateSnapshot] = []
    skipped: list[CandidateSnapshot] = []
    selected: list[CandidateSnapshot] = []
    seen_hashes: set[str] = set()
    quota_counts: dict[tuple[str, str], int] = {}
    cap_counts: dict[tuple[str, str], int] = {}

    scored: list[CandidateSnapshot] = []
    for candidate in candidates:
        if candidate.score is None:
            rejected.append(candidate)
            decisions.append(
                SelectionDecision(candidate.candidate_id, candidate.candidate_hash, False, "no_score")
            )
            continue
        scored.append(candidate)

    for candidate in _ranked_candidates(scored, score_direction=score_direction):
        decision_metadata = json_safe(dict(candidate.metadata))
        if duplicate_policy == "suppress" and candidate.candidate_hash in seen_hashes:
            rejected.append(candidate)
            decisions.append(
                SelectionDecision(
                    candidate.candidate_id,
                    candidate.candidate_hash,
                    False,
                    "duplicate",
                    decision_metadata,
                )
            )
            continue

        blocked_reason: str | None = None
        for quota in quotas:
            bucket = _metadata_bucket(
                candidate,
                quota.metadata_key,
                missing_metadata=missing_metadata,
                unknown_value=quota.unknown_value,
            )
            key = (quota.metadata_key, bucket)
            if quota_counts.get(key, 0) >= quota.max_count:
                blocked_reason = f"quota:{quota.metadata_key}"
                break
        if blocked_reason is None:
            for cap in caps:
                bucket = _metadata_bucket(
                    candidate,
                    cap.metadata_key,
                    missing_metadata=missing_metadata,
                    unknown_value=cap.unknown_value,
                )
                key = (cap.metadata_key, bucket)
                if cap_counts.get(key, 0) >= cap.max_count:
                    blocked_reason = f"cap:{cap.metadata_key}"
                    break

        if blocked_reason is not None:
            rejected.append(candidate)
            decisions.append(
                SelectionDecision(
                    candidate.candidate_id,
                    candidate.candidate_hash,
                    False,
                    blocked_reason,
                    decision_metadata,
                )
            )
            continue

        if len(selected) >= int(k):
            skipped.append(candidate)
            decisions.append(
                SelectionDecision(
                    candidate.candidate_id,
                    candidate.candidate_hash,
                    False,
                    "overflow",
                    decision_metadata,
                )
            )
            continue

        selected.append(candidate)
        seen_hashes.add(candidate.candidate_hash)
        for quota in quotas:
            bucket = _metadata_bucket(
                candidate,
                quota.metadata_key,
                missing_metadata=missing_metadata,
                unknown_value=quota.unknown_value,
            )
            key = (quota.metadata_key, bucket)
            quota_counts[key] = quota_counts.get(key, 0) + 1
        for cap in caps:
            bucket = _metadata_bucket(
                candidate,
                cap.metadata_key,
                missing_metadata=missing_metadata,
                unknown_value=cap.unknown_value,
            )
            key = (cap.metadata_key, bucket)
            cap_counts[key] = cap_counts.get(key, 0) + 1
        decisions.append(
            SelectionDecision(
                candidate.candidate_id,
                candidate.candidate_hash,
                True,
                "selected",
                decision_metadata,
            )
        )

    return SelectionResult(
        selected=tuple(selected),
        rejected=tuple(rejected),
        skipped=tuple(skipped),
        decisions=tuple(decisions),
        summary=_summary(selected, rejected),
    )
```

Then add:

```python
__all__ = [
    "DuplicateSelectionPolicy",
    "FamilyQuota",
    "MissingMetadataPolicy",
    "SelectionDecision",
    "SelectionResult",
    "SpecialistCap",
    "select_candidates",
]
```

- [ ] **Step 4: Run selection tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_selection.py -v
```

Expected: all selection tests pass.

## Task 5: Add Exports, Integration Tests, and Docs

**Files:**
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`
- Create: `tests/unit/test_phase2a_external_optimizer_integration.py`
- Modify: `docs/site/api.md`
- Modify: `docs/site/expensive-external-evaluations.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Re-export lifecycle and top-level names**

In `evocore/lifecycle/__init__.py`, import from `archives` and `selection`, then add all names listed in the Public API Names section to `__all__`.

In `evocore/__init__.py`, import and add these names to `__all__`:

```python
CandidateArchive
FamilyQuota
SelectionResult
SpecialistCap
select_candidates
```

- [ ] **Step 2: Add package export smoke test**

Append to `tests/unit/test_package_init.py`:

```python
def test_phase2a_public_exports():
    from evocore import (
        CandidateArchive,
        FamilyQuota,
        SelectionResult,
        SpecialistCap,
        select_candidates,
    )

    assert CandidateArchive is not None
    assert FamilyQuota is not None
    assert SelectionResult is not None
    assert SpecialistCap is not None
    assert select_candidates is not None
```

- [ ] **Step 3: Add cross-optimizer archive ingestion smoke tests**

Create `tests/unit/test_phase2a_external_optimizer_integration.py`:

```python
import pytest

from evocore import (
    CMAESOptimizer,
    CandidateArchive,
    DifferentialEvolutionOptimizer,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)


@pytest.fixture(params=["ga", "de", "cmaes"])
def optimizer(request):
    space = GeneSpace.uniform(-5.0, 5.0, 2)
    if request.param == "ga":
        return GeneticAlgorithmOptimizer(space, population_size=4, seed=10)
    if request.param == "de":
        return DifferentialEvolutionOptimizer(space, population_size=4, seed=10)
    return CMAESOptimizer(space, population_size=4, seed=10)


def test_archive_accepts_phase1_snapshots_from_all_optimizers(optimizer) -> None:
    optimizer.warm_start(
        [
            WarmStartRecord(values=(1.0, 1.0), score=1.0, metadata={"family": "a"}),
            WarmStartRecord(values=(2.0, 2.0), score=2.0, metadata={"family": "b"}),
        ]
    )
    archive = CandidateArchive()

    archive.add_population(optimizer.candidate_snapshot(scope="trusted"), source="trusted")

    records = archive.to_warm_start_records(k=1)
    assert len(records) == 1
    assert records[0].score == 2.0
```

- [ ] **Step 4: Update API docs**

Add this section to `docs/site/api.md` after the Phase 1 external-state entries:

```markdown
::: evocore.lifecycle.CandidateArchive

::: evocore.lifecycle.ArchiveEntry

::: evocore.lifecycle.ArchiveExport

::: evocore.lifecycle.select_candidates

::: evocore.lifecycle.SelectionResult

::: evocore.lifecycle.SelectionDecision

::: evocore.lifecycle.FamilyQuota

::: evocore.lifecycle.SpecialistCap
```

- [ ] **Step 5: Update expensive external evaluation docs**

Add a section to `docs/site/expensive-external-evaluations.md` after "Promote Survivors Without Private State":

````markdown
## Archive Search Memory And Select Survivors

Use `CandidateArchive` to keep scored candidates outside optimizer checkpoints, then export the best records back into `warm_start(...)`.

```python
from evocore import CandidateArchive, FamilyQuota, select_candidates

archive = CandidateArchive(duplicate_policy="keep_best", score_direction="maximize")
archive.add_population(optimizer.candidate_snapshot(scope="trusted"), source="stage1")

selection = select_candidates(
    optimizer.candidate_snapshot(scope="trusted").candidates,
    k=8,
    score_direction="maximize",
    quotas=[FamilyQuota(metadata_key="family", max_count=3)],
)

archive_records = archive.to_warm_start_records(k=8)
```

For selection over archive contents, use `archive.ranked_entries()` to inspect stored candidates or select directly from optimizer snapshots before archiving.
````

- [ ] **Step 6: Add changelog entry**

Under `CHANGELOG.md` `## [Unreleased]` `### Added`, add:

```markdown
- Added archive and survivor-selection utilities for expensive external optimization workflows, including `CandidateArchive`, duplicate suppression, family quotas, specialist caps, and deterministic snapshot selection.
```

## Task 6: Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_archives.py tests/unit/test_lifecycle_selection.py tests/unit/test_phase2a_external_optimizer_integration.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run existing Phase 1 external-state regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_external_state_optimizer_contract.py tests/unit/test_ga_external_state.py tests/unit/test_de_external_state.py tests/unit/test_cmaes_external_state.py -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Run formatting and linting**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both commands pass.

- [ ] **Step 4: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: MkDocs builds successfully with no strict-mode warnings.

- [ ] **Step 5: Commit Phase 2A**

Run:

```powershell
git status --short
git add evocore/lifecycle/archives.py evocore/lifecycle/selection.py evocore/lifecycle/__init__.py evocore/__init__.py tests/unit/test_lifecycle_archives.py tests/unit/test_lifecycle_selection.py tests/unit/test_phase2a_external_optimizer_integration.py tests/unit/test_package_init.py docs/site/api.md docs/site/expensive-external-evaluations.md CHANGELOG.md
git commit -m "feat(lifecycle): add archive and selection utilities"
```

Expected: commit succeeds and contains only Phase 2A archive, selection, docs, tests, and changelog changes.

## Compatibility Notes

- This plan is additive public API.
- Optimizer checkpoints remain unchanged.
- Archive JSON export starts at `ARCHIVE_SCHEMA_VERSION = 1`.
- Archive duplicate identity uses `candidate_hash`.
- Selection utilities operate on snapshots and must not mutate optimizer state.
- Top-level exports are intentionally limited to common user-facing names.

## Self-Review Notes

- Spec coverage: archive/search memory, duplicate suppression, warm-start export, survivor selection, family quotas, specialist caps, docs, and tests are covered.
- Type consistency: archive and selection consume Phase 1 `CandidateSnapshot` and `PopulationSnapshot` and export existing `WarmStartRecord`.
- Scope boundary: novelty pressure and richer diversity metrics are left to later policy work; Phase 2A ships deterministic duplicate, family, and specialist diversity controls first.
