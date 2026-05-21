# Result History Telemetry Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize EvoCore result, history, telemetry, JSON export, pandas export, and reproducibility metadata contracts for single-objective optimizer runs.

**Architecture:** Add a small private JSON export helper, keep generation summaries in `Logbook`, add append-only `EventHistory` in `evocore.stats`, and extend `RunResult`/`MultiRunResult` as the public export envelopes. GA and CMA engines record ask/tell events during lifecycle use, attach deterministic reproducibility metadata to results, and keep runtime timing opt-in during export.

**Tech Stack:** Python 3.11+ dataclasses, `typing.Literal`, JSON serialization, SHA-256 canonical hashes, optional pandas exports, pytest, Hypothesis, Ruff, maturin, Cargo.

---

## Scope Check

This plan implements the approved result/history/telemetry contract only.

In scope:

- Deterministic `to_dict()` and `to_json()` exports for `RunResult`, `MultiRunResult`, `Logbook`, `LogEntry`, and `OptimizationTelemetry`.
- Append-only `EventRecord` and `EventHistory` for ask, tell, and generation observations.
- Raw score and direction-aware comparison score exports.
- Optional pandas exports with clear install errors.
- Reproducibility metadata for version, engine type, seed, direction, gene-space signature/hash, and public optimizer config.
- GA and CMA ask/tell event recording.
- GA and CMA result metadata attachment.
- Public docs, API reference, and changelog updates.

Out of scope:

- `from_dict()` and `from_json()`.
- Resume-from-result or checkpoint migration.
- Trading-specific metrics, objective logic, or external benchmark comparisons.
- Multi-objective or Pareto result contracts.
- Making pandas a required dependency.
- Version bumps in `pyproject.toml` or `Cargo.toml`.

## File Structure Map

- Create `evocore/exporting.py`: private JSON-safe conversion, stable JSON dumps, canonical hash, and package-version helper.
- Modify `evocore/stats.py`: `LogEntry.to_dict()`, `Logbook.to_dict()`, `Logbook.to_json()`, `EventRecord`, `EventHistory`, `ReproducibilityMetadata`, and gene-space signature/hash helpers.
- Modify `evocore/evaluation.py`: `OptimizationTelemetry.to_dict()` and `OptimizationTelemetry.to_json()`.
- Modify `evocore/ga.py`: extend `RunResult` and `MultiRunResult`, record GA ask/tell events, attach reproducibility metadata, and export run envelopes.
- Modify `evocore/cmaes.py`: record CMA ask/tell events and attach result metadata for ask/tell and generation-loop paths.
- Modify `evocore/__init__.py`: export `EventRecord`, `EventHistory`, and `ReproducibilityMetadata`.
- Modify `tests/unit/test_stats.py`: logbook, event history, and reproducibility metadata tests.
- Modify `tests/unit/test_vnext_evaluation.py`: telemetry export tests.
- Modify `tests/unit/test_ga_engine.py`: result export, multi-run export, positional compatibility, and legacy generation-history tests.
- Modify `tests/unit/test_ga_ask_tell_vnext.py`: GA ask/tell event-history and vNext result metadata tests.
- Modify `tests/unit/test_cmaes_engine.py`: CMA generation-loop result metadata tests.
- Modify `tests/unit/test_cmaes_ask_tell_vnext.py`: CMA ask/tell event-history tests.
- Modify `tests/unit/test_package_init.py`: top-level export tests.
- Create `tests/property/test_result_export_properties.py`: JSON round-trip property coverage for event rows.
- Modify `docs/site/api.md`: API reference for event history and reproducibility metadata.
- Modify `docs/site/ask-tell-engines.md`: event history contract for manual lifecycle users.
- Modify `docs/site/optimizer-telemetry.md`: stable telemetry export fields.
- Modify `docs/site/ga.md`: result export examples.
- Modify `docs/site/cmaes.md`: result export examples.
- Modify `CHANGELOG.md`: public contract note.

## Preconditions

- [ ] **Step 1: Confirm branch and worktree**

Run:

```powershell
git status --short --branch
```

Expected: branch is `feature/general-optimizer-framework`; inspect uncommitted files before editing and do not overwrite unrelated user work.

- [ ] **Step 2: Confirm this plan builds on lifecycle protocol work**

Run:

```powershell
rg -n "class TellResult|class EvaluationContext|class CandidateBatch|def score_for_direction" evocore
```

Expected: matches in `evocore/evaluation.py` and `evocore/batches.py`. If these names are missing, stop and complete `docs/superpowers/plans/2026-05-14-optimizer-lifecycle-protocols.md` first.

---

### Task 1: Add JSON-Safe Export Helpers, Telemetry Exports, And Logbook Exports

**Files:**
- Create: `evocore/exporting.py`
- Modify: `evocore/evaluation.py`
- Modify: `evocore/stats.py`
- Modify: `tests/unit/test_vnext_evaluation.py`
- Modify: `tests/unit/test_stats.py`

- [ ] **Step 1: Add failing telemetry export tests**

Append these tests to `tests/unit/test_vnext_evaluation.py`:

```python
def test_telemetry_to_dict_exports_sorted_hashes_and_unique_count() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.total_candidates_proposed = 3
    telemetry.unique_candidate_hashes.update({"hash-b", "hash-a"})
    telemetry.candidates_screened = 1
    telemetry.candidates_partial_evaluated = 2
    telemetry.candidates_full_evaluated = 3
    telemetry.promoted_by_rung = {"cheap": 2}
    telemetry.eliminated_by_rung = {"cheap": 1}
    telemetry.cost_by_rung = {"full": 2.0, "cheap": 0.5}

    assert telemetry.to_dict() == {
        "total_candidates_proposed": 3,
        "unique_candidate_hashes": ["hash-a", "hash-b"],
        "unique_candidate_count": 2,
        "candidates_screened": 1,
        "candidates_partial_evaluated": 2,
        "candidates_full_evaluated": 3,
        "promoted_by_rung": {"cheap": 2},
        "eliminated_by_rung": {"cheap": 1},
        "cost_by_rung": {"cheap": 0.5, "full": 2.0},
    }


def test_telemetry_to_json_is_deterministic() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.unique_candidate_hashes.update({"z", "a"})
    telemetry.cost_by_rung = {"full": 1.0, "cheap": 0.25}

    first = telemetry.to_json()
    second = telemetry.to_json()

    assert first == second
    assert '"unique_candidate_hashes": ["a", "z"]' in first
```

- [ ] **Step 2: Add failing logbook export tests**

Append these tests to `tests/unit/test_stats.py`:

```python
def test_log_entry_to_dict_is_json_safe_and_preserves_custom_metrics():
    entry = LogEntry(
        gen=2,
        best_fitness=1.5,
        mean_fitness=1.0,
        std_fitness=0.25,
        wall_time_ms=12.0,
        n_evaluations=8,
        nan_fitness_count=0,
        cached_count=1,
        diversity=[0.1, 0.2],
        custom={"loss": 0.4, "tags": {"b", "a"}},
    )

    assert entry.to_dict() == {
        "gen": 2,
        "best_fitness": 1.5,
        "mean_fitness": 1.0,
        "std_fitness": 0.25,
        "wall_time_ms": 12.0,
        "n_evaluations": 8,
        "nan_fitness_count": 0,
        "cached_count": 1,
        "diversity": [0.1, 0.2],
        "loss": 0.4,
        "tags": ["a", "b"],
    }


def test_logbook_to_dict_and_json_are_stable():
    book = Logbook()
    book.append(LogEntry(0, 1.0, 0.5, 0.1, 12.0, 10, 0, 0, [], {"z": 2, "a": 1}))

    assert book.to_dict() == [
        {
            "gen": 0,
            "best_fitness": 1.0,
            "mean_fitness": 0.5,
            "std_fitness": 0.1,
            "wall_time_ms": 12.0,
            "n_evaluations": 10,
            "nan_fitness_count": 0,
            "cached_count": 0,
            "diversity": [],
            "a": 1,
            "z": 2,
        }
    ]
    assert book.to_json() == book.to_json()
```

- [ ] **Step 3: Run the export tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py::test_telemetry_to_dict_exports_sorted_hashes_and_unique_count tests/unit/test_vnext_evaluation.py::test_telemetry_to_json_is_deterministic tests/unit/test_stats.py::test_log_entry_to_dict_is_json_safe_and_preserves_custom_metrics tests/unit/test_stats.py::test_logbook_to_dict_and_json_are_stable -v
```

Expected: FAIL with `AttributeError` for missing `to_dict()` or `to_json()`.

- [ ] **Step 4: Create `evocore/exporting.py`**

Create `evocore/exporting.py` with this content:

```python
"""Private helpers for stable public export payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-safe representation with deterministic container ordering."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, set | frozenset):
        return sorted((json_safe(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    return repr(value)


def stable_json_dumps(payload: Any, *, indent: int | None = None) -> str:
    """Dump a JSON-safe payload with deterministic key ordering."""
    return json.dumps(json_safe(payload), sort_keys=True, indent=indent, allow_nan=False)


def canonical_json_hash(payload: Any) -> str:
    """Return a SHA-256 hash over canonical compact JSON."""
    text = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_version() -> str:
    """Return the installed EvoCore version or the local source fallback."""
    try:
        return importlib_metadata.version("evocore")
    except importlib_metadata.PackageNotFoundError:
        return "0.7.0"
```

- [ ] **Step 5: Add telemetry export helpers**

In `evocore/evaluation.py`, add this import:

```python
from evocore.exporting import stable_json_dumps
```

Add these methods to `OptimizationTelemetry` after `record_eliminated()`:

```python
    def to_dict(self) -> dict[str, Any]:
        """Export stable JSON-safe telemetry fields."""
        return {
            "total_candidates_proposed": self.total_candidates_proposed,
            "unique_candidate_hashes": sorted(self.unique_candidate_hashes),
            "unique_candidate_count": len(self.unique_candidate_hashes),
            "candidates_screened": self.candidates_screened,
            "candidates_partial_evaluated": self.candidates_partial_evaluated,
            "candidates_full_evaluated": self.candidates_full_evaluated,
            "promoted_by_rung": {
                key: self.promoted_by_rung[key] for key in sorted(self.promoted_by_rung)
            },
            "eliminated_by_rung": {
                key: self.eliminated_by_rung[key] for key in sorted(self.eliminated_by_rung)
            },
            "cost_by_rung": {key: self.cost_by_rung[key] for key in sorted(self.cost_by_rung)},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Export telemetry as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)
```

- [ ] **Step 6: Add logbook export helpers**

In `evocore/stats.py`, add these imports:

```python
import json
from typing import Any, Literal

from evocore.exporting import json_safe, stable_json_dumps
```

Add this method to `LogEntry`:

```python
    def to_dict(self) -> dict[str, Any]:
        """Export this generation summary as a JSON-safe dictionary."""
        row: dict[str, Any] = {
            "gen": self.gen,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "wall_time_ms": self.wall_time_ms,
            "n_evaluations": self.n_evaluations,
            "nan_fitness_count": self.nan_fitness_count,
            "cached_count": self.cached_count,
            "diversity": list(self.diversity),
        }
        row.update(self.custom)
        return json_safe(row)
```

Replace `Logbook.to_rows()` with:

```python
    def to_rows(self) -> list[dict[str, Any]]:
        """Convert log entries into JSON-serializable row dictionaries."""
        return [entry.to_dict() for entry in self._entries]
```

Add these methods after `to_rows()`:

```python
    def to_dict(self) -> list[dict[str, Any]]:
        """Export stable generation-summary dictionaries."""
        return self.to_rows()

    def to_json(self, *, indent: int | None = None) -> str:
        """Export generation summaries as deterministic JSON."""
        return stable_json_dumps(self.to_dict(), indent=indent)
```

Remove the unused `json` import if Ruff reports it.

- [ ] **Step 7: Run tests and Ruff for this slice**

Run:

```powershell
python -m pytest tests/unit/test_vnext_evaluation.py tests/unit/test_stats.py -v
python -m ruff format evocore/exporting.py evocore/evaluation.py evocore/stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_stats.py
python -m ruff check evocore/exporting.py evocore/evaluation.py evocore/stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_stats.py
```

Expected: PASS.

- [ ] **Step 8: Commit export helpers**

Run:

```powershell
git add evocore/exporting.py evocore/evaluation.py evocore/stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_stats.py
git commit -m "feat: add stable telemetry and logbook exports"
```

Expected: commit succeeds with only the files listed above staged.

---

### Task 2: Add EventRecord And EventHistory

**Files:**
- Modify: `evocore/stats.py`
- Modify: `tests/unit/test_stats.py`

- [ ] **Step 1: Add failing event-history tests**

Append these imports to `tests/unit/test_stats.py`:

```python
from evocore.exceptions import ConfigurationError
from evocore.stats import EventHistory, EventRecord
```

If `tests/unit/test_stats.py` already imports from `evocore.stats`, merge the import into one block:

```python
from evocore.stats import EventHistory, EventRecord, Logbook, LogEntry
```

Append these tests:

```python
def test_event_history_to_rows_preserves_append_order():
    history = EventHistory()
    history.append(
        EventRecord(
            event_index=0,
            event_type="ask",
            batch_id="b-1",
            candidate_id="c-1",
            candidate_hash="hash-1",
            origin="random",
            genes=(1.0, 2),
            params={"x": 1.0, "period": 2},
        )
    )
    history.append(
        EventRecord(
            event_index=1,
            event_type="tell",
            batch_id="b-1",
            candidate_id="c-1",
            candidate_hash="hash-1",
            confidence="trusted_full",
            raw_score=4.0,
            comparison_score=4.0,
            cost=1.0,
            status="trusted",
            origin="random",
            genes=(1.0, 2),
            params={"x": 1.0, "period": 2},
            metrics={"loss": 0.2},
            metadata={"source": "unit"},
        )
    )

    assert len(history) == 2
    assert history[0].event_type == "ask"
    assert [event.event_type for event in history] == ["ask", "tell"]
    assert history.to_rows() == [
        {
            "event_index": 0,
            "event_type": "ask",
            "batch_id": "b-1",
            "candidate_id": "c-1",
            "candidate_hash": "hash-1",
            "generation": None,
            "rung": None,
            "confidence": None,
            "raw_score": None,
            "comparison_score": None,
            "cost": 0.0,
            "status": None,
            "origin": "random",
            "parents": [],
            "genes": [1.0, 2],
            "params": {"period": 2, "x": 1.0},
            "metrics": {},
            "metadata": {},
        },
        {
            "event_index": 1,
            "event_type": "tell",
            "batch_id": "b-1",
            "candidate_id": "c-1",
            "candidate_hash": "hash-1",
            "generation": None,
            "rung": None,
            "confidence": "trusted_full",
            "raw_score": 4.0,
            "comparison_score": 4.0,
            "cost": 1.0,
            "status": "trusted",
            "origin": "random",
            "parents": [],
            "genes": [1.0, 2],
            "params": {"period": 2, "x": 1.0},
            "metrics": {"loss": 0.2},
            "metadata": {"source": "unit"},
        },
    ]


def test_event_history_rejects_non_append_event_index():
    history = EventHistory()

    with pytest.raises(ConfigurationError, match="append-only"):
        history.append(EventRecord(event_index=2, event_type="ask"))


def test_event_history_to_dataframe_missing_pandas_message(monkeypatch):
    history = EventHistory()
    history.append(EventRecord(event_index=0, event_type="generation", generation=0))
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="pip install pandas"):
        history.to_dataframe()
```

- [ ] **Step 2: Run event-history tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_event_history_to_rows_preserves_append_order tests/unit/test_stats.py::test_event_history_rejects_non_append_event_index tests/unit/test_stats.py::test_event_history_to_dataframe_missing_pandas_message -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `EventRecord` and `EventHistory` do not exist.

- [ ] **Step 3: Add event imports to `evocore/stats.py`**

Add these imports near the top of `evocore/stats.py`:

```python
from evocore.evaluation import (
    CandidateOrigin,
    CandidateStatus,
    EvaluationConfidence,
)
from evocore.exceptions import ConfigurationError
from evocore.individual import GeneValue
```

- [ ] **Step 4: Add `EventRecord` and `EventHistory`**

Append these definitions after `Logbook` in `evocore/stats.py`:

```python
@dataclass(frozen=True)
class EventRecord:
    """Represent one append-only optimizer lifecycle observation."""

    event_index: int
    event_type: Literal["ask", "tell", "generation"]
    batch_id: str | None = None
    candidate_id: str | None = None
    candidate_hash: str | None = None
    generation: int | None = None
    rung: str | None = None
    confidence: EvaluationConfidence | None = None
    raw_score: float | None = None
    comparison_score: float | None = None
    cost: float = 0.0
    status: CandidateStatus | None = None
    origin: CandidateOrigin | None = None
    parents: tuple[str, ...] = ()
    genes: tuple[GeneValue, ...] = ()
    params: dict[str, GeneValue] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export this event as one JSON-safe row."""
        return json_safe(
            {
                "event_index": self.event_index,
                "event_type": self.event_type,
                "batch_id": self.batch_id,
                "candidate_id": self.candidate_id,
                "candidate_hash": self.candidate_hash,
                "generation": self.generation,
                "rung": self.rung,
                "confidence": self.confidence,
                "raw_score": self.raw_score,
                "comparison_score": self.comparison_score,
                "cost": self.cost,
                "status": self.status,
                "origin": self.origin,
                "parents": list(self.parents),
                "genes": list(self.genes),
                "params": self.params,
                "metrics": self.metrics,
                "metadata": self.metadata,
            }
        )


class EventHistory:
    """Store append-only optimizer lifecycle events."""

    def __init__(self) -> None:
        self._events: list[EventRecord] = []

    def append(self, event: EventRecord) -> None:
        """Append one event in strict sequence order."""
        if event.event_index != len(self._events):
            raise ConfigurationError(
                "EventHistory is append-only; event_index must match the next row index."
            )
        self._events.append(event)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self._events)

    def __getitem__(self, index: int) -> EventRecord:
        return self._events[index]

    def to_rows(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return [event.to_dict() for event in self._events]

    def to_dict(self) -> list[dict[str, Any]]:
        """Export lifecycle events as JSON-safe row dictionaries."""
        return self.to_rows()

    def to_dataframe(self):
        """Convert event rows into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "EventHistory.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        return pd.DataFrame(self.to_rows())
```

- [ ] **Step 5: Run event-history tests and Ruff**

Run:

```powershell
python -m pytest tests/unit/test_stats.py -v
python -m ruff format evocore/stats.py tests/unit/test_stats.py
python -m ruff check evocore/stats.py tests/unit/test_stats.py
```

Expected: PASS.

- [ ] **Step 6: Commit event history**

Run:

```powershell
git add evocore/stats.py tests/unit/test_stats.py
git commit -m "feat: add optimizer event history"
```

Expected: commit succeeds with only `evocore/stats.py` and `tests/unit/test_stats.py` staged.

---

### Task 3: Add Reproducibility Metadata And Gene-Space Hashing

**Files:**
- Modify: `evocore/stats.py`
- Modify: `tests/unit/test_stats.py`

- [ ] **Step 1: Add failing reproducibility metadata tests**

Append this import to `tests/unit/test_stats.py`:

```python
from evocore import GeneDef, GeneSpace
from evocore.stats import ReproducibilityMetadata, gene_space_hash, gene_space_signature
```

If `tests/unit/test_stats.py` already imports from `evocore.stats`, merge the new names into that import.

Append these tests:

```python
def test_gene_space_signature_preserves_gene_order_and_fields():
    space = GeneSpace(
        [
            GeneDef("x", "float", -1.0, 1.0, sigma=0.2),
            GeneDef("period", "int", 2, 20),
            GeneDef("enabled", "bool"),
        ]
    )

    assert gene_space_signature(space) == {
        "genes": [
            {"name": "x", "kind": "float", "low": -1.0, "high": 1.0, "sigma": 0.2},
            {"name": "period", "kind": "int", "low": 2, "high": 20, "sigma": None},
            {"name": "enabled", "kind": "bool", "low": None, "high": None, "sigma": None},
        ],
        "has_names": True,
        "length": 3,
    }


def test_gene_space_hash_is_stable_for_equivalent_spaces():
    left = GeneSpace([GeneDef("x", "float", -1.0, 1.0)])
    right = GeneSpace([GeneDef("x", "float", -1.0, 1.0)])

    assert gene_space_hash(gene_space_signature(left)) == gene_space_hash(
        gene_space_signature(right)
    )


def test_reproducibility_metadata_to_dict_is_json_safe():
    metadata = ReproducibilityMetadata(
        evocore_version="0.7.0",
        engine_type="GAEngine",
        seed=42,
        direction="maximize",
        gene_space_signature={"genes": [{"name": "x", "kind": "float"}]},
        gene_space_hash="abc123",
        optimizer_config={"population_size": 8, "callbacks": {"not", "serialized"}},
    )

    assert metadata.to_dict() == {
        "evocore_version": "0.7.0",
        "engine_type": "GAEngine",
        "seed": 42,
        "direction": "maximize",
        "gene_space_signature": {"genes": [{"kind": "float", "name": "x"}]},
        "gene_space_hash": "abc123",
        "optimizer_config": {"callbacks": ["not", "serialized"], "population_size": 8},
        "extension": {},
    }
```

- [ ] **Step 2: Run reproducibility metadata tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_stats.py::test_gene_space_signature_preserves_gene_order_and_fields tests/unit/test_stats.py::test_gene_space_hash_is_stable_for_equivalent_spaces tests/unit/test_stats.py::test_reproducibility_metadata_to_dict_is_json_safe -v
```

Expected: FAIL because `ReproducibilityMetadata`, `gene_space_signature`, and `gene_space_hash` do not exist.

- [ ] **Step 3: Add reproducibility metadata helpers**

In `evocore/stats.py`, add these imports:

```python
from evocore.evaluation import Direction
from evocore.exporting import canonical_json_hash
from evocore.gene_space import GeneSpace
```

Append these definitions after `EventHistory`:

```python
@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Capture deterministic optimizer and environment identity for a result."""

    evocore_version: str
    engine_type: str
    seed: int
    direction: Direction
    gene_space_signature: dict[str, Any]
    gene_space_hash: str
    optimizer_config: dict[str, Any]
    extension: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export reproducibility metadata as JSON-safe stable fields."""
        return json_safe(
            {
                "evocore_version": self.evocore_version,
                "engine_type": self.engine_type,
                "seed": self.seed,
                "direction": self.direction,
                "gene_space_signature": self.gene_space_signature,
                "gene_space_hash": self.gene_space_hash,
                "optimizer_config": self.optimizer_config,
                "extension": self.extension,
            }
        )


def gene_space_signature(gene_space: GeneSpace) -> dict[str, Any]:
    """Return a deterministic JSON-safe signature for a gene space."""
    return {
        "genes": [
            {
                "name": gene.name,
                "kind": gene.kind,
                "low": gene.low,
                "high": gene.high,
                "sigma": gene.sigma,
            }
            for gene in gene_space.genes
        ],
        "has_names": gene_space.has_names,
        "length": gene_space.length,
    }


def gene_space_hash(signature: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a gene-space signature."""
    return canonical_json_hash(signature)
```

- [ ] **Step 4: Run reproducibility tests and Ruff**

Run:

```powershell
python -m pytest tests/unit/test_stats.py -v
python -m ruff format evocore/stats.py tests/unit/test_stats.py
python -m ruff check evocore/stats.py tests/unit/test_stats.py
```

Expected: PASS.

- [ ] **Step 5: Commit reproducibility metadata**

Run:

```powershell
git add evocore/stats.py tests/unit/test_stats.py
git commit -m "feat: add reproducibility metadata exports"
```

Expected: commit succeeds with only `evocore/stats.py` and `tests/unit/test_stats.py` staged.

---

### Task 4: Extend RunResult And MultiRunResult Export Contracts

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing `RunResult` export tests**

In `tests/unit/test_ga_engine.py`, update the stats import:

```python
from evocore.stats import EventHistory, EventRecord, Logbook
```

Append these tests:

```python
def test_run_result_preserves_existing_positional_construction():
    result = make_result(7, 1.25)

    assert result.best_fitness == pytest.approx(1.25)
    assert result.direction == "maximize"
    assert result.engine_type == ""
    assert result.best_candidate_id is None
    assert result.best_score is None
    assert len(result.history) == 0
    assert result.metadata == {}


def test_run_result_to_dict_excludes_runtime_by_default():
    result = make_result(7, 1.25)

    payload = result.to_dict()

    assert payload["schema_version"] == 1
    assert payload["seed"] == 7
    assert payload["best"]["fitness"] == pytest.approx(1.25)
    assert payload["best"]["score"] == pytest.approx(1.25)
    assert payload["n_evaluations"] == 1
    assert "runtime" not in payload


def test_run_result_to_dict_includes_runtime_when_requested():
    result = make_result(7, 1.25)

    payload = result.to_dict(include_runtime=True)

    assert payload["runtime"]["wall_time_seconds"] == pytest.approx(0.01)


def test_run_result_to_json_is_deterministic():
    result = make_result(7, 1.25)

    assert result.to_json() == result.to_json()


def test_run_result_to_dataframe_uses_history_when_present(monkeypatch):
    result = make_result(7, 1.25)
    result.history.append(EventRecord(event_index=0, event_type="generation", generation=0))
    captured_rows = {}

    class FakeDataFrame:
        def __init__(self, rows):
            captured_rows["rows"] = rows

    class FakePandas:
        DataFrame = FakeDataFrame

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            return FakePandas
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result.to_dataframe()

    assert captured_rows["rows"][0]["event_type"] == "generation"
```

- [ ] **Step 2: Add failing `MultiRunResult` export tests**

Append these tests to `tests/unit/test_ga_engine.py`:

```python
def test_multi_run_result_to_dict_preserves_run_order_and_excludes_runtime_by_default():
    r1 = make_result(1, 1.0)
    r2 = make_result(2, 3.0)
    multi = MultiRunResult(
        best=r2,
        all_runs=[r2, r1],
        n_runs=2,
        wall_time_seconds=0.05,
        direction="maximize",
    )

    payload = multi.to_dict()

    assert payload["schema_version"] == 1
    assert payload["direction"] == "maximize"
    assert [run["seed"] for run in payload["runs"]] == [2, 1]
    assert payload["best"]["seed"] == 2
    assert "runtime" not in payload


def test_multi_run_result_to_json_and_dataframe_are_stable(monkeypatch):
    r1 = make_result(1, 1.0)
    r2 = make_result(2, 3.0)
    multi = MultiRunResult(best=r2, all_runs=[r2, r1], n_runs=2, wall_time_seconds=0.05)
    captured_rows = {}

    class FakeDataFrame:
        def __init__(self, rows):
            captured_rows["rows"] = rows

    class FakePandas:
        DataFrame = FakeDataFrame

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            return FakePandas
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert multi.to_json() == multi.to_json()
    multi.to_dataframe()

    assert captured_rows["rows"] == [
        {"run_index": 0, "seed": 2, "best_fitness": 3.0, "best_score": 3.0, "n_evaluations": 1},
        {"run_index": 1, "seed": 1, "best_fitness": 1.0, "best_score": 1.0, "n_evaluations": 1},
    ]
```

- [ ] **Step 3: Run result export tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py::test_run_result_preserves_existing_positional_construction tests/unit/test_ga_engine.py::test_run_result_to_dict_excludes_runtime_by_default tests/unit/test_ga_engine.py::test_run_result_to_dict_includes_runtime_when_requested tests/unit/test_ga_engine.py::test_run_result_to_json_is_deterministic tests/unit/test_ga_engine.py::test_run_result_to_dataframe_uses_history_when_present tests/unit/test_ga_engine.py::test_multi_run_result_to_dict_preserves_run_order_and_excludes_runtime_by_default tests/unit/test_ga_engine.py::test_multi_run_result_to_json_and_dataframe_are_stable -v
```

Expected: FAIL because `RunResult` and `MultiRunResult` do not have the new fields or export helpers.

- [ ] **Step 4: Extend imports in `evocore/ga.py`**

In `evocore/ga.py`, change:

```python
from typing import Literal
```

to:

```python
from typing import Any, Literal
```

Add these imports:

```python
from evocore.exporting import json_safe, stable_json_dumps
from evocore.stats import (
    EventHistory,
    Logbook,
    LogEntry,
    ReproducibilityMetadata,
)
```

Remove the old `from evocore.stats import Logbook, LogEntry` line after adding the combined import.

- [ ] **Step 5: Extend `RunResult`**

In `evocore/ga.py`, add these fields after `telemetry` in `RunResult`:

```python
    direction: Direction = "maximize"
    engine_type: str = ""
    best_candidate_id: str | None = None
    best_score: float | None = None
    history: EventHistory = field(default_factory=EventHistory)
    reproducibility: ReproducibilityMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Add these methods to `RunResult`:

```python
    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export this run result as a stable JSON-safe dictionary."""
        best_score = self.best_score if self.best_score is not None else self.best_fitness
        payload: dict[str, Any] = {
            "schema_version": 1,
            "engine_type": self.engine_type,
            "direction": self.direction,
            "seed": self.seed,
            "best": {
                "fitness": self.best_fitness,
                "score": best_score,
                "candidate_id": self.best_candidate_id,
                "genes": list(self.best_individual.genes),
                "params": self.best_individual.metadata.get("params"),
            },
            "stop": {
                "stopped_early": self.stopped_early,
                "reason": self.stop_reason,
            },
            "budget": {
                "max_evaluations": self.max_evaluations,
                "budget_reached": self.budget_reached,
            },
            "n_evaluations": self.n_evaluations,
            "reproducibility": (
                self.reproducibility.to_dict() if self.reproducibility is not None else None
            ),
            "telemetry": self.telemetry.to_dict(),
            "history": self.history.to_dict(),
            "logbook": self.logbook.to_dict(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export this run result as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return event history as a DataFrame, falling back to generation logbook rows."""
        if len(self.history):
            return self.history.to_dataframe()
        return self.logbook.to_dataframe()
```

- [ ] **Step 6: Extend `MultiRunResult`**

In `evocore/ga.py`, add these fields after `wall_time_seconds` in `MultiRunResult`:

```python
    direction: Direction = "maximize"
    metadata: dict[str, Any] = field(default_factory=dict)
```

Add these methods to `MultiRunResult`:

```python
    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        """Export aggregate run results as a stable JSON-safe dictionary."""
        payload: dict[str, Any] = {
            "schema_version": 1,
            "direction": self.direction,
            "n_runs": self.n_runs,
            "best": self.best.to_dict(include_runtime=include_runtime),
            "runs": [run.to_dict(include_runtime=include_runtime) for run in self.all_runs],
            "fitness_summary": self.fitness_summary(),
            "metadata": self.metadata,
        }
        if include_runtime:
            payload["runtime"] = {"wall_time_seconds": self.wall_time_seconds}
        return json_safe(payload)

    def to_json(self, *, include_runtime: bool = False, indent: int | None = None) -> str:
        """Export aggregate run results as deterministic JSON."""
        return stable_json_dumps(self.to_dict(include_runtime=include_runtime), indent=indent)

    def to_dataframe(self):
        """Return one pandas DataFrame row per child run."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "MultiRunResult.to_dataframe() requires pandas. Install with: pip install pandas"
            ) from exc

        rows = [
            {
                "run_index": index,
                "seed": run.seed,
                "best_fitness": run.best_fitness,
                "best_score": run.best_score if run.best_score is not None else run.best_fitness,
                "n_evaluations": run.n_evaluations,
            }
            for index, run in enumerate(self.all_runs)
        ]
        return pd.DataFrame(rows)
```

- [ ] **Step 7: Set `direction` on multi-run aggregates**

In `GAEngine.run_multiple()`, update the `MultiRunResult(...)` construction to include:

```python
            direction=self.direction,
```

- [ ] **Step 8: Run result export tests and Ruff**

Run:

```powershell
python -m pytest tests/unit/test_ga_engine.py -v
python -m ruff format evocore/ga.py tests/unit/test_ga_engine.py
python -m ruff check evocore/ga.py tests/unit/test_ga_engine.py
```

Expected: PASS.

- [ ] **Step 9: Commit result export envelopes**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_engine.py
git commit -m "feat: add stable result export envelopes"
```

Expected: commit succeeds with only `evocore/ga.py` and `tests/unit/test_ga_engine.py` staged.

---

### Task 5: Record GA Event History And Attach GA Result Metadata

**Files:**
- Modify: `evocore/ga.py`
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `tests/unit/test_ga_engine.py`

- [ ] **Step 1: Add failing GA ask/tell event-history tests**

Append these tests to `tests/unit/test_ga_ask_tell_vnext.py`:

```python
def test_ga_ask_records_append_only_ask_events() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123)

    candidates = engine.ask(2)

    assert len(engine.history) == 2
    rows = engine.history.to_rows()
    assert [row["event_index"] for row in rows] == [0, 1]
    assert all(row["event_type"] == "ask" for row in rows)
    assert rows[0]["batch_id"] == candidates[0].batch_id
    assert rows[0]["candidate_id"] == candidates[0].candidate_id
    assert rows[0]["candidate_hash"] == candidates[0].candidate_hash()
    assert rows[0]["genes"] == list(candidates[0].genes)
    assert rows[0]["params"] == candidates[0].params


def test_ga_tell_records_raw_and_comparison_scores_for_minimize() -> None:
    engine = GAEngine(_space(), population_size=4, generations=5, seed=123, direction="minimize")
    candidates = engine.ask(1)

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=2.5,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
                metrics={"loss": 0.25},
                metadata={"source": "unit"},
            )
        ]
    )

    row = engine.history.to_rows()[-1]
    assert row["event_type"] == "tell"
    assert row["raw_score"] == pytest.approx(2.5)
    assert row["comparison_score"] == pytest.approx(-2.5)
    assert row["status"] == "trusted"
    assert row["metrics"] == {"loss": 0.25}
    assert row["metadata"] == {"source": "unit"}
```

- [ ] **Step 2: Add failing GA result metadata tests**

Append these tests to `tests/unit/test_ga_engine.py`:

```python
def test_ga_vnext_run_attaches_history_and_reproducibility_metadata():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, generations=2, seed=42)

    result = engine.run(
        CallableEvaluator(lambda genes: -sum(float(v) ** 2 for v in genes)),
        policy=full_policy(4, batch_size=4),
    )

    assert result.engine_type == "GAEngine"
    assert result.direction == "maximize"
    assert result.best_candidate_id is not None
    assert result.best_score == pytest.approx(result.best_fitness)
    assert len(result.history) >= 8
    assert result.reproducibility is not None
    assert result.reproducibility.engine_type == "GAEngine"
    assert result.reproducibility.seed == 42
    assert result.reproducibility.direction == "maximize"
    assert result.reproducibility.gene_space_signature["length"] == 2
    assert result.reproducibility.gene_space_hash
    assert result.reproducibility.optimizer_config["population_size"] == 4


def test_ga_generation_loop_result_includes_generation_history():
    engine = GAEngine(GeneSpace.uniform(-1.0, 1.0, 2), population_size=6, generations=2, seed=42)

    result = engine._run_from_population(
        engine._initial_population(),
        lambda ind: -sum(float(v) ** 2 for v in ind.genes),
        start_generation=0,
    )

    assert result.engine_type == "GAEngine"
    assert result.best_score == pytest.approx(result.best_fitness)
    assert [event.event_type for event in result.history] == ["generation", "generation"]
    assert result.history.to_rows()[0]["generation"] == 0
```

- [ ] **Step 3: Run GA event/history tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_ask_records_append_only_ask_events tests/unit/test_ga_ask_tell_vnext.py::test_ga_tell_records_raw_and_comparison_scores_for_minimize tests/unit/test_ga_engine.py::test_ga_vnext_run_attaches_history_and_reproducibility_metadata tests/unit/test_ga_engine.py::test_ga_generation_loop_result_includes_generation_history -v
```

Expected: FAIL because GA does not record `history` or attach reproducibility metadata yet.

- [ ] **Step 4: Add GA history and reproducibility imports**

In `evocore/ga.py`, extend the stats import from Task 4 to include:

```python
    EventRecord,
    gene_space_hash,
    gene_space_signature,
```

Extend the exporting import to include:

```python
from evocore.exporting import json_safe, package_version, stable_json_dumps
```

- [ ] **Step 5: Reset GA history with vNext state**

In `GAEngine._reset_vnext_state()`, add:

```python
        self.history = EventHistory()
```

- [ ] **Step 6: Add GA event helper methods**

Add these methods before `ask()` in `GAEngine`:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed candidates."""
        for candidate in candidates:
            self.history.append(
                EventRecord(
                    event_index=len(self.history),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(),
                    generation=candidate.generation,
                    origin=candidate.origin,
                    parents=tuple(candidate.parents),
                    genes=tuple(candidate.genes),
                    params=dict(candidate.params) if candidate.params is not None else None,
                    metadata=dict(candidate.metadata),
                )
            )

    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        raw_score = float(record.score) if record.score is not None else None
        comparison_score = (
            score_for_direction(raw_score, self.direction)
            if raw_score is not None and math.isfinite(raw_score)
            else None
        )
        self.history.append(
            EventRecord(
                event_index=len(self.history),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(),
                generation=candidate.generation,
                rung=record.rung,
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
                metadata=dict(record.metadata),
            )
        )
```

- [ ] **Step 7: Wire GA ask/tell events**

In `GAEngine.ask()`, after:

```python
        self.vnext_telemetry.record_proposed_candidates(candidates)
```

add:

```python
        self._append_ask_events(candidates)
```

In `GAEngine.tell()`, immediately after:

```python
            candidate.apply_record(record)
```

add:

```python
            self._append_tell_event(candidate, record)
```

- [ ] **Step 8: Add GA reproducibility helper methods**

Add these methods before `run()` in `GAEngine`:

```python
    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable GA constructor configuration."""
        return json_safe(
            {
                "population_size": self.population_size,
                "generations": self.generations,
                "crossover": self.crossover,
                "crossover_prob": self.crossover_prob,
                "crossover_eta": self.crossover_eta,
                "crossover_alpha": self.crossover_alpha,
                "mutation": self.mutation,
                "mutation_prob": self.mutation_prob,
                "mutation_individual_prob": self.mutation_individual_prob,
                "mutation_sigma": self.mutation_sigma,
                "mutation_sigma_schedule": self.mutation_sigma_schedule,
                "mutation_sigma_end": self.mutation_sigma_end,
                "selection": self.selection,
                "tournament_size": self.tournament_size,
                "elitism": self.elitism,
                "parallel": self.parallel,
                "n_workers": self.n_workers,
                "max_evaluations": self.max_evaluations,
                "track_diversity": self.track_diversity,
            }
        )

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = gene_space_signature(self.gene_space)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            engine_type="GAEngine",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=gene_space_hash(signature),
            optimizer_config=self._optimizer_config(),
        )

    def _generation_history(self, logbook: Logbook) -> EventHistory:
        """Convert generation logbook entries into generation events."""
        history = EventHistory()
        for entry in logbook:
            history.append(
                EventRecord(
                    event_index=len(history),
                    event_type="generation",
                    generation=entry.gen,
                    raw_score=entry.best_fitness,
                    comparison_score=score_for_direction(entry.best_fitness, self.direction),
                    metrics=entry.to_dict(),
                )
            )
        return history
```

- [ ] **Step 9: Attach GA metadata to legacy generation-loop results**

In `GAEngine._run_from_population()`, update the `RunResult(...)` construction near the end to include:

```python
            direction=self.direction,
            engine_type="GAEngine",
            best_score=float(best.fitness),
            history=self._generation_history(logbook),
            reproducibility=self._reproducibility_metadata(),
```

Keep existing positional and keyword arguments unchanged.

- [ ] **Step 10: Attach GA metadata to defensive empty vNext result**

In `GAEngine.run()`, inside the `if self.best_candidate is None:` `RunResult(...)`, add:

```python
                direction=self.direction,
                engine_type="GAEngine",
                best_score=float("-inf"),
                history=self.history,
                reproducibility=self._reproducibility_metadata(),
```

- [ ] **Step 11: Attach GA metadata to normal vNext result**

In the final `RunResult(...)` in `GAEngine.run()`, add:

```python
            direction=self.direction,
            engine_type="GAEngine",
            best_candidate_id=self.best_candidate.candidate_id,
            best_score=float(best.fitness),
            history=self.history,
            reproducibility=self._reproducibility_metadata(),
```

- [ ] **Step 12: Run GA event/history tests and Ruff**

Run:

```powershell
python -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py -v
python -m ruff format evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py
python -m ruff check evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py
```

Expected: PASS.

- [ ] **Step 13: Commit GA event history**

Run:

```powershell
git add evocore/ga.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_ga_engine.py
git commit -m "feat: record ga result event history"
```

Expected: commit succeeds with only GA implementation and tests staged.

---

### Task 6: Record CMA Event History And Attach CMA Result Metadata

**Files:**
- Modify: `evocore/cmaes.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `tests/unit/test_cmaes_engine.py`

- [ ] **Step 1: Add failing CMA ask/tell event-history tests**

Append these tests to `tests/unit/test_cmaes_ask_tell_vnext.py`:

```python
def test_cma_ask_records_append_only_ask_events() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7)

    candidates = engine.ask()

    assert len(engine.history) == 4
    rows = engine.history.to_rows()
    assert [row["event_index"] for row in rows] == [0, 1, 2, 3]
    assert all(row["event_type"] == "ask" for row in rows)
    assert rows[0]["batch_id"] == candidates[0].batch_id
    assert rows[0]["candidate_id"] == candidates[0].candidate_id
    assert rows[0]["candidate_hash"] == candidates[0].candidate_hash()


def test_cma_tell_records_raw_and_comparison_scores_for_minimize() -> None:
    engine = CMAESEngine(_space(), population_size=4, seed=7, direction="minimize")
    candidates = engine.ask()

    engine.tell(
        [
            EvaluationRecord(
                candidate_id=candidates[0].candidate_id,
                batch_id=candidates[0].batch_id,
                score=3.0,
                confidence="trusted_full",
                rung="full",
                cost=1.0,
            )
        ]
    )

    row = engine.history.to_rows()[-1]
    assert row["event_type"] == "tell"
    assert row["raw_score"] == pytest.approx(3.0)
    assert row["comparison_score"] == pytest.approx(-3.0)
    assert row["status"] == "trusted"
```

- [ ] **Step 2: Add failing CMA generation-result metadata test**

Append this test to `tests/unit/test_cmaes_engine.py`:

```python
def test_cma_generation_loop_result_attaches_history_and_reproducibility():
    engine = CMAESEngine(GeneSpace.uniform(-2.0, 2.0, 3), population_size=6, generations=2, seed=42)

    result = engine.run(lambda ind: -sum(float(v) ** 2 for v in ind.genes))

    assert result.engine_type == "CMAESEngine"
    assert result.direction == "maximize"
    assert result.best_score == pytest.approx(result.best_fitness)
    assert [event.event_type for event in result.history] == ["generation", "generation"]
    assert result.reproducibility is not None
    assert result.reproducibility.engine_type == "CMAESEngine"
    assert result.reproducibility.optimizer_config["population_size"] == 6
```

- [ ] **Step 3: Run CMA event/history tests and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py::test_cma_ask_records_append_only_ask_events tests/unit/test_cmaes_ask_tell_vnext.py::test_cma_tell_records_raw_and_comparison_scores_for_minimize tests/unit/test_cmaes_engine.py::test_cma_generation_loop_result_attaches_history_and_reproducibility -v
```

Expected: FAIL because CMA does not record `history` or attach result metadata yet.

- [ ] **Step 4: Add CMA history and reproducibility imports**

In `evocore/cmaes.py`, add:

```python
from typing import Any

from evocore.exporting import json_safe, package_version
from evocore.stats import (
    EventHistory,
    EventRecord,
    ReproducibilityMetadata,
    gene_space_hash,
    gene_space_signature,
)
```

If adding `from typing import Any` creates a separate typing import, keep a single typing import block.

- [ ] **Step 5: Initialize CMA history**

In `CMAESEngine.__init__()`, after:

```python
        self._candidates_by_id: dict[str, Candidate] = {}
```

add:

```python
        self.history = EventHistory()
```

- [ ] **Step 6: Add CMA event helper methods**

Add these methods before `ask()` in `CMAESEngine`:

```python
    def _append_ask_events(self, candidates: Sequence[Candidate]) -> None:
        """Record ask events for proposed CMA candidates."""
        for candidate in candidates:
            self.history.append(
                EventRecord(
                    event_index=len(self.history),
                    event_type="ask",
                    batch_id=candidate.batch_id,
                    candidate_id=candidate.candidate_id,
                    candidate_hash=candidate.candidate_hash(),
                    generation=candidate.generation,
                    origin=candidate.origin,
                    parents=tuple(candidate.parents),
                    genes=tuple(candidate.genes),
                    params=dict(candidate.params) if candidate.params is not None else None,
                    metadata=dict(candidate.metadata),
                )
            )

    def _append_tell_event(self, candidate: Candidate, record: EvaluationRecord) -> None:
        """Record a tell event after candidate state has been updated."""
        raw_score = float(record.score) if record.score is not None else None
        comparison_score = (
            score_for_direction(raw_score, self.direction)
            if raw_score is not None and math.isfinite(raw_score)
            else None
        )
        self.history.append(
            EventRecord(
                event_index=len(self.history),
                event_type="tell",
                batch_id=candidate.batch_id,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.candidate_hash(),
                generation=candidate.generation,
                rung=record.rung,
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
                metadata=dict(record.metadata),
            )
        )
```

- [ ] **Step 7: Wire CMA ask/tell events**

In `CMAESEngine.ask()`, after:

```python
        self.vnext_telemetry.record_proposed_candidates(candidates)
```

add:

```python
        self._append_ask_events(candidates)
```

In `CMAESEngine.tell()`, immediately after:

```python
            candidate.apply_record(record)
```

add:

```python
            self._append_tell_event(candidate, record)
```

- [ ] **Step 8: Add CMA reproducibility helper methods**

Add these methods before `run()` in `CMAESEngine`:

```python
    def _optimizer_config(self) -> dict[str, Any]:
        """Return public serializable CMA constructor configuration."""
        return json_safe(
            {
                "population_size": self.population_size,
                "initial_mean": self.initial_mean,
                "initial_sigma": self.initial_sigma,
                "generations": self.generations,
                "parallel": self.parallel,
                "n_workers": self.n_workers,
                "track_diversity": self.track_diversity,
            }
        )

    def _reproducibility_metadata(self) -> ReproducibilityMetadata:
        """Return deterministic reproducibility metadata for this engine."""
        signature = gene_space_signature(self.gene_space)
        return ReproducibilityMetadata(
            evocore_version=package_version(),
            engine_type="CMAESEngine",
            seed=self.seed,
            direction=self.direction,
            gene_space_signature=signature,
            gene_space_hash=gene_space_hash(signature),
            optimizer_config=self._optimizer_config(),
        )

    def _generation_history(self, logbook: Logbook) -> EventHistory:
        """Convert generation logbook entries into generation events."""
        history = EventHistory()
        for entry in logbook:
            history.append(
                EventRecord(
                    event_index=len(history),
                    event_type="generation",
                    generation=entry.gen,
                    raw_score=entry.best_fitness,
                    comparison_score=score_for_direction(entry.best_fitness, self.direction),
                    metrics=entry.to_dict(),
                )
            )
        return history
```

- [ ] **Step 9: Attach CMA metadata to generation-loop results**

In `CMAESEngine.run()`, update the final `RunResult(...)` construction to include:

```python
            direction=self.direction,
            engine_type="CMAESEngine",
            best_score=best_fitness,
            history=self._generation_history(logbook),
            reproducibility=self._reproducibility_metadata(),
```

- [ ] **Step 10: Run CMA event/history tests and Ruff**

Run:

```powershell
python -m pytest tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_engine.py -v
python -m ruff format evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_engine.py
python -m ruff check evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_engine.py
```

Expected: PASS.

- [ ] **Step 11: Commit CMA event history**

Run:

```powershell
git add evocore/cmaes.py tests/unit/test_cmaes_ask_tell_vnext.py tests/unit/test_cmaes_engine.py
git commit -m "feat: record cma result event history"
```

Expected: commit succeeds with only CMA implementation and tests staged.

---

### Task 7: Export Public Names And Update Documentation

**Files:**
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`
- Modify: `docs/site/api.md`
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add failing top-level export tests**

Append these assertions to `test_vnext_public_exports_are_available()` in `tests/unit/test_package_init.py`:

```python
    assert evocore.EventRecord.__name__ == "EventRecord"
    assert evocore.EventHistory.__name__ == "EventHistory"
    assert evocore.ReproducibilityMetadata.__name__ == "ReproducibilityMetadata"
```

- [ ] **Step 2: Run package export test and confirm failure**

Run:

```powershell
python -m pytest tests/unit/test_package_init.py::test_vnext_public_exports_are_available -v
```

Expected: FAIL because the new stats names are not exported from `evocore`.

- [ ] **Step 3: Export new stats names from `evocore/__init__.py`**

Change:

```python
from evocore.stats import Logbook, LogEntry
```

to:

```python
from evocore.stats import EventHistory, EventRecord, Logbook, LogEntry, ReproducibilityMetadata
```

Add these names to `__all__` in the CamelCase section:

```python
    "EventHistory",
    "EventRecord",
```

Add this name near the other CamelCase result/export records:

```python
    "ReproducibilityMetadata",
```

- [ ] **Step 4: Update API reference**

In `docs/site/api.md`, under the optimizer lifecycle section, insert these autodoc entries after `::: evocore.evaluation.OptimizationTelemetry`:

```markdown
::: evocore.stats.EventRecord

::: evocore.stats.EventHistory

::: evocore.stats.ReproducibilityMetadata
```

- [ ] **Step 5: Update ask/tell docs**

Append this section to `docs/site/ask-tell-engines.md`:

```markdown
## Event History

Ask/tell engines record append-only lifecycle events. Every proposed candidate receives
an `ask` event with its batch ID, candidate ID, genome hash, origin, genes, params, and
metadata. Every accepted evaluation record receives a `tell` event with the raw score,
direction-aware comparison score, confidence, rung, cost, resulting status, metrics, and
record metadata.

```python
candidates = engine.ask(4)
records = [
    EvaluationRecord(
        candidate_id=candidate.candidate_id,
        batch_id=candidate.batch_id,
        score=score_candidate(candidate),
        confidence="trusted_full",
        rung="full",
        cost=1.0,
    )
    for candidate in candidates
]
engine.tell(records)

rows = engine.history.to_rows()
```

Raw user scores are stored under `raw_score`. EvoCore stores the value used for
direction-aware comparisons separately under `comparison_score`.
```

- [ ] **Step 6: Update telemetry docs**

Replace `docs/site/optimizer-telemetry.md` with:

```markdown
# Optimizer Telemetry

`OptimizationTelemetry` tracks the true breadth and cost of optimizer search.

Stable export fields are:

- `total_candidates_proposed`
- `unique_candidate_hashes`
- `unique_candidate_count`
- `candidates_screened`
- `candidates_partial_evaluated`
- `candidates_full_evaluated`
- `promoted_by_rung`
- `eliminated_by_rung`
- `cost_by_rung`

`unique_candidate_hashes` is exported as a sorted list and
`unique_candidate_count` is derived from that set. Use `to_dict()` for a JSON-safe
payload or `to_json()` for deterministic JSON with sorted keys.

Cached evaluation records are state-eligible and count toward full-budget accounting,
while remaining visible through `TellResult.cached_count` and event history rows with
`confidence="cached"`.
```

- [ ] **Step 7: Update GA docs with export example**

In `docs/site/ga.md`, after the evaluator example and before the autodoc block, insert:

```markdown
## Result Export

`RunResult` is the stable envelope for a completed run:

```python
result = engine.run(Objective())
payload = result.to_dict()
json_text = result.to_json(indent=2)
events = result.history.to_rows()
```

Runtime timing is excluded from deterministic exports by default. Pass
`include_runtime=True` to include `wall_time_seconds` under the `runtime` key.
```

- [ ] **Step 8: Update CMA docs with export example**

In `docs/site/cmaes.md`, before the autodoc block, insert:

```markdown
## Result Export

Generation-oriented CMA runs attach generation events to `RunResult.history` and keep
generation summaries in `RunResult.logbook`.

```python
result = engine.run(fitness_fn)
payload = result.to_dict()
events = result.history.to_rows()
```

Ask/tell CMA usage records `ask` and `tell` events on `engine.history`.
```

- [ ] **Step 9: Update changelog**

In `CHANGELOG.md`, under `[Unreleased]` `### Added`, add:

```markdown
- Stable `RunResult`, `MultiRunResult`, `Logbook`, and `OptimizationTelemetry` export
  helpers with deterministic JSON output by default.
- Append-only `EventRecord` and `EventHistory` APIs for ask/tell audit rows and
  generation-level observations.
- `ReproducibilityMetadata` on run results with version, engine, seed, direction,
  gene-space signature/hash, and serializable optimizer configuration.
```

Under `[Unreleased]` `### Changed`, add:

```markdown
- Runtime timing in result exports now lives under `runtime` and is included only when
  callers pass `include_runtime=True`.
```

- [ ] **Step 10: Run docs-adjacent checks**

Run:

```powershell
python -m pytest tests/unit/test_package_init.py -v
python -m mkdocs build --strict --site-dir (Join-Path $env:TEMP ('evocore-mkdocs-site-' + [guid]::NewGuid().ToString('N')))
python -m ruff format evocore/__init__.py tests/unit/test_package_init.py
python -m ruff check evocore/__init__.py tests/unit/test_package_init.py
```

Expected: PASS. MkDocs may print the existing Material advisory.

- [ ] **Step 11: Commit public exports and docs**

Run:

```powershell
git add evocore/__init__.py tests/unit/test_package_init.py docs/site/api.md docs/site/ask-tell-engines.md docs/site/optimizer-telemetry.md docs/site/ga.md docs/site/cmaes.md CHANGELOG.md
git commit -m "docs: document result history telemetry contract"
```

Expected: commit succeeds with only package exports, docs, tests, and changelog staged.

---

### Task 8: Add Property Tests And Final Verification

**Files:**
- Create: `tests/property/test_result_export_properties.py`
- Verify all changed files.

- [ ] **Step 1: Create JSON round-trip property tests**

Create `tests/property/test_result_export_properties.py` with this content:

```python
import json

from hypothesis import given
from hypothesis import strategies as st

from evocore.stats import EventHistory, EventRecord

json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-1000, max_value=1000)
    | st.floats(
        min_value=-1000.0,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    | st.text(max_size=20)
)
json_values = st.recursive(
    json_scalars,
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=3),
    max_leaves=8,
)


@given(st.lists(st.dictionaries(st.text(min_size=1, max_size=12), json_values, max_size=3), max_size=8))
def test_event_history_rows_are_json_round_trippable(metadata_rows):
    history = EventHistory()
    for index, metadata in enumerate(metadata_rows):
        history.append(
            EventRecord(
                event_index=index,
                event_type="tell",
                batch_id=f"b-{index}",
                candidate_id=f"c-{index}",
                candidate_hash=f"hash-{index}",
                confidence="trusted_full",
                raw_score=float(index),
                comparison_score=float(index),
                cost=1.0,
                status="trusted",
                origin="random",
                genes=(float(index), index),
                params={"x": float(index), "period": index},
                metadata=metadata,
            )
        )

    rows = history.to_rows()
    assert json.loads(json.dumps(rows, sort_keys=True, allow_nan=False)) == rows
```

- [ ] **Step 2: Run property tests**

Run:

```powershell
python -m pytest tests/property/test_result_export_properties.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit property tests**

Run:

```powershell
git add tests/property/test_result_export_properties.py
git commit -m "test: cover result event export properties"
```

Expected: commit succeeds with only the new property test staged.

- [ ] **Step 4: Run targeted result/history/telemetry tests**

Run:

```powershell
python -m pytest tests/unit/test_stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/property/test_result_export_properties.py -v
```

Expected: PASS.

- [ ] **Step 5: Run Python formatting check**

Run:

```powershell
python -m ruff format --check
```

Expected: PASS.

- [ ] **Step 6: Run Python lint**

Run:

```powershell
python -m ruff check
```

Expected: PASS.

- [ ] **Step 7: Run docs build**

Run:

```powershell
python -m mkdocs build --strict --site-dir (Join-Path $env:TEMP ('evocore-mkdocs-site-' + [guid]::NewGuid().ToString('N')))
```

Expected: PASS. MkDocs may print the existing Material advisory.

- [ ] **Step 8: Rebuild the Python extension**

Run:

```powershell
python -m maturin develop --release
```

Expected: PASS and installs the local EvoCore extension.

- [ ] **Step 9: Run Python unit and integration tests**

Run:

```powershell
python -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 10: Run property suite**

Run:

```powershell
python -m pytest tests/property/ -v
```

Expected: PASS.

- [ ] **Step 11: Run Rust checks only if Rust files changed**

If `git diff --name-only origin/feature/general-optimizer-framework...HEAD` or `git diff --name-only HEAD` includes files under `src/`, run:

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Expected: PASS. If no Rust files changed, record `not run; no Rust files changed`.

- [ ] **Step 12: Check diff hygiene**

Run:

```powershell
git diff --check
git status --short --branch
```

Expected: no whitespace errors. Working tree is clean after the task commits, except for unrelated pre-existing user files.

- [ ] **Step 13: Stop if verification fails**

If any verification command fails, do not push or open a PR. Report the failing command, the relevant error summary, and the likely files involved.

- [ ] **Step 14: Prepare completion summary**

Use this summary template:

```markdown
Implemented the result/history/telemetry export contract:
- Added deterministic JSON-safe exports for results, multi-run aggregates, logbooks, and telemetry.
- Added append-only event history for ask, tell, and generation observations.
- Added reproducibility metadata with version, engine, seed, direction, gene-space signature/hash, and public optimizer config.
- Wired GA and CMA to record lifecycle events and attach result metadata.
- Updated docs, API reference, changelog, and property coverage.

Verification:
- `python -m pytest tests/unit/test_stats.py tests/unit/test_vnext_evaluation.py tests/unit/test_ga_engine.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_engine.py tests/unit/test_cmaes_ask_tell_vnext.py tests/property/test_result_export_properties.py -v`: result observed during Task 8 Step 4.
- `python -m ruff format --check`: result observed during Task 8 Step 5.
- `python -m ruff check`: result observed during Task 8 Step 6.
- `python -m mkdocs build --strict --site-dir <temp>`: result observed during Task 8 Step 7.
- `python -m maturin develop --release`: result observed during Task 8 Step 8.
- `python -m pytest tests/unit/ tests/integration/ -v`: result observed during Task 8 Step 9.
- `python -m pytest tests/property/ -v`: result observed during Task 8 Step 10.
- Rust checks: result observed during Task 8 Step 11.
```

## Self-Review

- Spec coverage: Tasks cover result envelopes, multi-run envelopes, logbook exports, telemetry exports, append-only event history, deterministic JSON defaults, optional runtime exports, optional pandas exports, reproducibility metadata, GA integration, CMA integration, docs, changelog, property tests, and final verification.
- Scope control: The plan does not add `from_dict()`, `from_json()`, resume-from-result, checkpoint migration, multi-objective contracts, domain metrics, benchmark comparisons, or pandas as a required dependency.
- Placeholder scan: The plan contains no banned placeholder phrases.
- Type consistency: `Direction`, `CandidateOrigin`, `CandidateStatus`, `EvaluationConfidence`, `GeneValue`, `EventRecord`, `EventHistory`, `ReproducibilityMetadata`, `RunResult`, and `MultiRunResult` names match the definitions used across tasks.
- Compatibility: Existing positional `RunResult(...)`, `MultiRunResult(...)`, and `LogEntry(...)` construction remains valid because new dataclass fields are keyword-default fields appended after existing fields.
