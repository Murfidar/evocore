# Checkpoint Resume Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first stable checkpoint/resume contract for EvoCore with a JSON-safe checkpoint envelope, GA generation-loop resume support, callback writing, validation, docs, and changelog coverage.

**Architecture:** Create a shared checkpoint envelope in `evocore.results.checkpointing` and keep optimizer-specific continuation state in `evocore.optimizers.ga.checkpointing`. The v1 implementation supports GA generation-loop continuation from stable checkpoint files and preserves legacy pickle loading as explicit compatibility behavior. GA ask/tell snapshots and CMA-ES Rust state snapshots are separate follow-on plans because they require additional state-ledger and Rust/PyO3 export work.

**Tech Stack:** Python 3.11+, dataclasses, JSON-safe serialization helpers, pytest, MkDocs, existing EvoCore GA/callback/result modules.

---

## Scope Check

The design spec covers three subsystems:

- Shared checkpoint envelope and compatibility validation.
- GA checkpoint/resume implementation.
- Future GA ask/tell and CMA-ES state snapshots.

This plan implements the first complete v1 slice: shared envelope plus GA generation-loop checkpoint/resume. It explicitly documents that GA ask/tell checkpointing and CMA-ES checkpointing are unsupported in v1 so users do not mistake event exports or result exports for resumable state.

## File Structure

- Create `evocore/results/checkpointing.py`
  - Owns the stable checkpoint envelope dataclass, JSON save/load helpers, schema validation, and identity validation.
- Modify `evocore/results/__init__.py`
  - Re-exports checkpoint contract helpers from the results package.
- Modify `evocore/__init__.py`
  - Adds top-level convenience exports for the checkpoint contract.
- Modify `tests/unit/test_checkpointing.py`
  - New focused tests for checkpoint envelope validation and GA stable resume.
- Modify `tests/unit/test_domain_imports.py`
  - Verifies focused module ownership for checkpoint helpers.
- Modify `tests/unit/test_package_init.py`
  - Verifies top-level checkpoint exports.
- Modify `evocore/optimizers/ga/checkpointing.py`
  - Adds stable GA generation-loop checkpoint creation and resume.
  - Keeps legacy pickle loading isolated.
- Modify `evocore/optimizers/ga/generation_loop.py`
  - Binds the GA checkpoint factory into callbacks.
- Modify `evocore/callbacks/checkpointing.py`
  - Adds `format="stable"` and `format="legacy_pickle"` output modes.
- Modify `tests/unit/test_callbacks.py`
  - Covers stable callback output and explicit legacy pickle output.
- Modify `docs/site/callbacks-checkpointing.md`
  - Documents stable checkpoints, legacy pickle checkpoints, and non-checkpoint exports.
- Modify `docs/site/ga.md`
  - Shows GA stable checkpoint/resume usage.
- Modify `docs/site/cmaes.md`
  - States that CMA-ES checkpoint/resume is unsupported until Rust state serialization exists.
- Modify `docs/site/optimizer-telemetry.md`
  - Clarifies telemetry is accounting/audit data, not resumable optimizer state.
- Modify `CHANGELOG.md`
  - Records the new stable checkpoint contract and legacy pickle behavior.

## Task 1: Shared Checkpoint Envelope

**Files:**
- Create: `tests/unit/test_checkpointing.py`
- Create: `evocore/results/checkpointing.py`

- [ ] **Step 1: Write failing shared checkpoint tests**

Create `tests/unit/test_checkpointing.py` with this content:

```python
import json

import pytest

from evocore import CheckpointError, GeneSpace, GeneticAlgorithmOptimizer
from evocore.results import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSnapshot,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_identity,
)


def _engine():
    return GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-1.0, 1.0, 2),
        population_size=4,
        max_generations=3,
        seed=42,
    )


def _snapshot() -> CheckpointSnapshot:
    engine = _engine()
    return CheckpointSnapshot(
        optimizer_type="GeneticAlgorithmOptimizer",
        optimizer_config=engine.config_signature(),
        optimizer_config_hash=engine.config_hash(),
        gene_space_signature=engine.gene_space.signature(),
        gene_space_hash=engine.gene_space.hash(),
        direction=engine.direction,
        seed=engine.seed,
        position={"generation": 1, "event_index": 0, "n_evaluations": 8},
        state={
            "optimizer_type": "GeneticAlgorithmOptimizer",
            "schema_version": 1,
            "payload": {"state_kind": "ga_generation_loop", "population": []},
        },
        audit={"events": [], "telemetry": {}, "best": None},
        metadata={"source": "unit"},
    )


def test_checkpoint_snapshot_exports_required_envelope() -> None:
    payload = _snapshot().to_dict()

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["checkpoint_kind"] == CHECKPOINT_KIND
    assert payload["created_by"]["evocore_version"]
    assert payload["optimizer"]["optimizer_type"] == "GeneticAlgorithmOptimizer"
    assert payload["optimizer"]["seed_derivation"] == {
        "algorithm": "py_derive_seed",
        "version": 1,
    }
    assert payload["position"] == {
        "event_index": 0,
        "generation": 1,
        "n_evaluations": 8,
    }
    assert payload["state"]["payload"]["state_kind"] == "ga_generation_loop"
    assert payload["metadata"] == {"source": "unit"}


def test_checkpoint_save_and_load_round_trip_json(tmp_path) -> None:
    path = tmp_path / "checkpoint_gen_1.evocore-checkpoint.json"

    save_checkpoint(path, _snapshot())
    loaded = load_checkpoint(path)

    assert loaded == _snapshot().to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == loaded


def test_checkpoint_load_missing_file_lists_available(tmp_path) -> None:
    (tmp_path / "checkpoint_gen_1.evocore-checkpoint.json").write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="Available checkpoints"):
        load_checkpoint(tmp_path / "checkpoint_gen_9.evocore-checkpoint.json")


def test_checkpoint_load_rejects_wrong_kind(tmp_path) -> None:
    payload = _snapshot().to_dict()
    payload["checkpoint_kind"] = "result_export"
    path = tmp_path / "bad.evocore-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint_kind"):
        load_checkpoint(path)


def test_checkpoint_load_rejects_unsupported_schema(tmp_path) -> None:
    payload = _snapshot().to_dict()
    payload["checkpoint_schema_version"] = 999
    path = tmp_path / "bad.evocore-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint_schema_version"):
        load_checkpoint(path)


def test_checkpoint_identity_validation_rejects_gene_space_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="gene_space_hash"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash="different",
            optimizer_config_hash=engine.config_hash(),
            seed=engine.seed,
            direction=engine.direction,
        )


def test_checkpoint_identity_validation_rejects_config_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash="different",
            seed=engine.seed,
            direction=engine.direction,
        )


def test_checkpoint_identity_validation_rejects_seed_and_direction_mismatch() -> None:
    engine = _engine()
    payload = _snapshot().to_dict()

    with pytest.raises(CheckpointError, match="seed"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash=engine.config_hash(),
            seed=999,
            direction=engine.direction,
        )

    with pytest.raises(CheckpointError, match="direction"):
        validate_checkpoint_identity(
            payload,
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=engine.gene_space.hash(),
            optimizer_config_hash=engine.config_hash(),
            seed=engine.seed,
            direction="minimize",
        )
```

- [ ] **Step 2: Run shared checkpoint tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py -v
```

Expected: FAIL because `evocore.results.CheckpointSnapshot`, constants, and helper functions do not exist.

- [ ] **Step 3: Implement shared checkpoint envelope helpers**

Create `evocore/results/checkpointing.py`:

```python
"""Stable optimizer checkpoint envelope helpers."""

from __future__ import annotations

import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.core.serialization import json_safe, package_version, stable_json_dumps
from evocore.lifecycle import Direction

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "optimizer_state"
SEED_DERIVATION_ALGORITHM = "py_derive_seed"
SEED_DERIVATION_VERSION = 1


def _created_by() -> dict[str, str]:
    """Return stable writer metadata for checkpoint diagnostics."""
    return {
        "evocore_version": package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _available_checkpoints(path: Path) -> list[str]:
    """Return nearby checkpoint-like files for missing-file diagnostics."""
    directory = path.parent if path.parent != Path("") else Path(".")
    if not directory.is_dir():
        return []
    return sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name.endswith(".evocore-checkpoint.json")
        or entry.name.startswith("checkpoint_gen_")
    )


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Represent one stable optimizer checkpoint envelope."""

    optimizer_type: str
    optimizer_config: Mapping[str, Any]
    optimizer_config_hash: str
    gene_space_signature: Mapping[str, Any]
    gene_space_hash: str
    direction: Direction
    seed: int
    position: Mapping[str, Any]
    state: Mapping[str, Any]
    audit: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_by: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Export this checkpoint as a JSON-safe stable dictionary."""
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": CHECKPOINT_KIND,
            "created_by": dict(self.created_by or _created_by()),
            "optimizer": {
                "optimizer_type": self.optimizer_type,
                "optimizer_config": dict(self.optimizer_config),
                "optimizer_config_hash": self.optimizer_config_hash,
                "gene_space_signature": dict(self.gene_space_signature),
                "gene_space_hash": self.gene_space_hash,
                "direction": self.direction,
                "seed": int(self.seed),
                "seed_derivation": {
                    "algorithm": SEED_DERIVATION_ALGORITHM,
                    "version": SEED_DERIVATION_VERSION,
                },
            },
            "position": dict(self.position),
            "state": dict(self.state),
            "audit": dict(self.audit),
            "metadata": dict(self.metadata),
        }
        return json_safe(payload)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint field {key!r} must be an object.")
    return value


def validate_checkpoint_envelope(payload: object) -> dict[str, Any]:
    """Validate the shared checkpoint envelope and return it as a dict."""
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint payload must be a JSON object.")
    data = dict(payload)
    version = data.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            "unsupported checkpoint_schema_version "
            f"{version!r}; supported version is {CHECKPOINT_SCHEMA_VERSION}."
        )
    if data.get("checkpoint_kind") != CHECKPOINT_KIND:
        raise CheckpointError(
            f"checkpoint_kind must be {CHECKPOINT_KIND!r}, got {data.get('checkpoint_kind')!r}."
        )
    created_by = _require_mapping(data, "created_by")
    if not created_by.get("evocore_version"):
        raise CheckpointError("checkpoint created_by.evocore_version is required.")
    optimizer = _require_mapping(data, "optimizer")
    _require_mapping(data, "position")
    state = _require_mapping(data, "state")
    if not optimizer.get("optimizer_type"):
        raise CheckpointError("checkpoint optimizer.optimizer_type is required.")
    if not optimizer.get("optimizer_config_hash"):
        raise CheckpointError("checkpoint optimizer.optimizer_config_hash is required.")
    if not optimizer.get("gene_space_hash"):
        raise CheckpointError("checkpoint optimizer.gene_space_hash is required.")
    if optimizer.get("direction") not in ("maximize", "minimize"):
        raise CheckpointError("checkpoint optimizer.direction must be 'maximize' or 'minimize'.")
    if "seed" not in optimizer:
        raise CheckpointError("checkpoint optimizer.seed is required.")
    seed_derivation = optimizer.get("seed_derivation")
    if not isinstance(seed_derivation, Mapping):
        raise CheckpointError("checkpoint optimizer.seed_derivation must be an object.")
    if seed_derivation.get("algorithm") != SEED_DERIVATION_ALGORITHM:
        raise CheckpointError(
            "checkpoint seed_derivation.algorithm "
            f"{seed_derivation.get('algorithm')!r} is unsupported."
        )
    if seed_derivation.get("version") != SEED_DERIVATION_VERSION:
        raise CheckpointError(
            "checkpoint seed_derivation.version "
            f"{seed_derivation.get('version')!r} is unsupported."
        )
    if state.get("optimizer_type") != optimizer.get("optimizer_type"):
        raise CheckpointError("checkpoint state.optimizer_type must match optimizer.optimizer_type.")
    if state.get("schema_version") != 1:
        raise CheckpointError("checkpoint state.schema_version must be 1.")
    if not isinstance(state.get("payload"), Mapping):
        raise CheckpointError("checkpoint state.payload must be an object.")
    return data


def validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    optimizer_type: str,
    gene_space_hash: str,
    optimizer_config_hash: str,
    seed: int,
    direction: Direction,
) -> None:
    """Raise when checkpoint identity does not match the receiving optimizer."""
    data = validate_checkpoint_envelope(payload)
    optimizer = _require_mapping(data, "optimizer")
    state = _require_mapping(data, "state")
    if optimizer.get("optimizer_type") != optimizer_type:
        raise CheckpointError(
            "checkpoint optimizer_type "
            f"{optimizer.get('optimizer_type')!r} does not match {optimizer_type!r}."
        )
    if state.get("optimizer_type") != optimizer_type:
        raise CheckpointError(
            "checkpoint state.optimizer_type "
            f"{state.get('optimizer_type')!r} does not match {optimizer_type!r}."
        )
    if optimizer.get("gene_space_hash") != gene_space_hash:
        raise CheckpointError(
            "checkpoint gene_space_hash "
            f"{optimizer.get('gene_space_hash')!r} does not match {gene_space_hash!r}."
        )
    if optimizer.get("optimizer_config_hash") != optimizer_config_hash:
        raise CheckpointError(
            "checkpoint optimizer_config_hash "
            f"{optimizer.get('optimizer_config_hash')!r} does not match {optimizer_config_hash!r}."
        )
    if int(optimizer.get("seed")) != int(seed):
        raise CheckpointError(
            f"checkpoint seed {optimizer.get('seed')!r} does not match optimizer seed {seed!r}."
        )
    if optimizer.get("direction") != direction:
        raise CheckpointError(
            "checkpoint direction "
            f"{optimizer.get('direction')!r} does not match optimizer direction {direction!r}."
        )


def save_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: CheckpointSnapshot | Mapping[str, Any],
) -> None:
    """Write a checkpoint envelope as deterministic UTF-8 JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.to_dict() if isinstance(checkpoint, CheckpointSnapshot) else checkpoint
    data = validate_checkpoint_envelope(payload)
    target.write_text(stable_json_dumps(data, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a stable checkpoint JSON file."""
    source = Path(path)
    if not source.exists():
        available = _available_checkpoints(source)
        raise CheckpointError(
            f"checkpoint file {str(source)!r} not found. Available checkpoints: "
            f"{', '.join(available) or 'none'}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint file {str(source)!r} is corrupt or incompatible: {exc}"
        ) from exc
    return validate_checkpoint_envelope(payload)


__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "SEED_DERIVATION_ALGORITHM",
    "SEED_DERIVATION_VERSION",
    "CheckpointSnapshot",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_envelope",
    "validate_checkpoint_identity",
]
```

- [ ] **Step 4: Run shared checkpoint tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py -v
```

Expected: PASS for the shared envelope tests.

- [ ] **Step 5: Commit shared checkpoint envelope**

```powershell
git add evocore/results/checkpointing.py tests/unit/test_checkpointing.py
git commit -m "feat(results): add checkpoint envelope"
```

## Task 2: Public Checkpoint Exports

**Files:**
- Modify: `evocore/results/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_domain_imports.py`
- Modify: `tests/unit/test_package_init.py`

- [ ] **Step 1: Add failing export tests**

In `tests/unit/test_domain_imports.py`, update `test_new_domain_imports_are_available` by adding:

```python
        "evocore.results.checkpointing",
```

In `test_new_domain_symbols_are_importable`, extend the imports and assertions:

```python
    from evocore import CheckpointSnapshot as TopLevelCheckpointSnapshot
    from evocore.results import CheckpointSnapshot

    assert TopLevelCheckpointSnapshot is CheckpointSnapshot
    assert CheckpointSnapshot is not None
```

In `test_domain_packages_export_symbols_owned_by_focused_modules`, extend the result imports:

```python
    from evocore.results.checkpointing import (
        CheckpointSnapshot,
        load_checkpoint,
        save_checkpoint,
        validate_checkpoint_identity,
    )
```

Add these assertions near the other result ownership assertions:

```python
    assert CheckpointSnapshot.__module__ == "evocore.results.checkpointing"
    assert load_checkpoint.__module__ == "evocore.results.checkpointing"
    assert save_checkpoint.__module__ == "evocore.results.checkpointing"
    assert validate_checkpoint_identity.__module__ == "evocore.results.checkpointing"
```

In `tests/unit/test_package_init.py`, update `test_result_exports_accessible_from_top_level`:

```python
    from evocore import (
        CheckpointSnapshot,
        GenerationHistory,
        GenerationRecord,
        OptimizationBatchResult,
        OptimizationResult,
    )

    assert CheckpointSnapshot is not None
```

- [ ] **Step 2: Run export tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: FAIL because checkpoint helpers are not re-exported yet.

- [ ] **Step 3: Export checkpoint helpers from results package**

Modify `evocore/results/__init__.py` by adding this import:

```python
from evocore.results.checkpointing import (
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    SEED_DERIVATION_ALGORITHM,
    SEED_DERIVATION_VERSION,
    CheckpointSnapshot,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_envelope,
    validate_checkpoint_identity,
)
```

Add these names to `__all__`:

```python
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointSnapshot",
    "SEED_DERIVATION_ALGORITHM",
    "SEED_DERIVATION_VERSION",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_envelope",
    "validate_checkpoint_identity",
```

- [ ] **Step 4: Export checkpoint helpers from top-level package**

Modify the `from evocore.results import (...)` block in `evocore/__init__.py` to include:

```python
    CHECKPOINT_KIND,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSnapshot,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_envelope,
    validate_checkpoint_identity,
```

Add these names to top-level `__all__`:

```python
    "CHECKPOINT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointSnapshot",
    "load_checkpoint",
    "save_checkpoint",
    "validate_checkpoint_envelope",
    "validate_checkpoint_identity",
```

- [ ] **Step 5: Run export tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py tests/unit/test_package_init.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit public checkpoint exports**

```powershell
git add evocore/results/__init__.py evocore/__init__.py tests/unit/test_domain_imports.py tests/unit/test_package_init.py
git commit -m "feat(results): export checkpoint contract"
```

## Task 3: GA Stable Generation-Loop Checkpoints

**Files:**
- Modify: `tests/unit/test_checkpointing.py`
- Modify: `evocore/optimizers/ga/checkpointing.py`

- [ ] **Step 1: Add failing GA stable resume tests**

Append this code to `tests/unit/test_checkpointing.py`:

```python
from evocore.results import GenerationHistory
from evocore.search_space import Solution


def _sphere(solution: Solution) -> float:
    return -sum(float(value) ** 2 for value in solution.values)


def _run_generation_loop(engine: GeneticAlgorithmOptimizer):
    return engine._run_from_population(
        engine._initial_population(),
        _sphere,
        start_generation=0,
    )


def _population_after_generation_zero(engine: GeneticAlgorithmOptimizer) -> list[Solution]:
    working_population, fitnesses, evaluated_now, _ = engine._evaluate_with_budget(
        engine._initial_population(),
        _sphere,
        gen=-1,
        n_evaluations=0,
    )
    generation_history = GenerationHistory()
    working_population, _, _, stopped, _ = engine._run_generation(
        working_population=working_population,
        fitnesses=fitnesses,
        objective_fn=_sphere,
        gen=0,
        n_evaluations=evaluated_now,
        elite_history=[],
        diversity_history=[],
        generation_history=generation_history,
    )
    assert stopped is False
    return working_population


def test_ga_checkpoint_generation_snapshot_contains_population_state() -> None:
    engine = _engine()
    population = [
        Solution([0.25, -0.5], score=-0.3125, score_valid=True, metadata={"rank": 1}),
        Solution([0.5, -0.25], score=-0.3125, score_valid=True, metadata={"rank": 2}),
    ]

    snapshot = engine.checkpoint(generation=2, population=population)
    payload = snapshot.to_dict()
    state_payload = payload["state"]["payload"]

    assert state_payload["state_kind"] == "ga_generation_loop"
    assert payload["position"]["generation"] == 2
    assert payload["position"]["event_index"] == engine.state_summary().event_index
    assert state_payload["population"] == [
        {
            "values": [0.25, -0.5],
            "score": -0.3125,
            "score_valid": True,
            "metadata": {"rank": 1},
        },
        {
            "values": [0.5, -0.25],
            "score": -0.3125,
            "score_valid": True,
            "metadata": {"rank": 2},
        },
    ]


def test_ga_resume_from_stable_checkpoint_matches_uninterrupted_generation_loop(tmp_path) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    checkpoint_path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"

    partial = GeneticAlgorithmOptimizer(
        space,
        population_size=6,
        max_generations=3,
        seed=123,
    )
    generation_zero_population = _population_after_generation_zero(partial)
    partial.save_checkpoint(
        checkpoint_path,
        partial.checkpoint(generation=0, population=generation_zero_population),
    )

    resumed = GeneticAlgorithmOptimizer(
        space,
        population_size=6,
        max_generations=3,
        seed=123,
    ).resume_from_checkpoint(_sphere, checkpoint_path)
    uninterrupted = _run_generation_loop(
        GeneticAlgorithmOptimizer(
            space,
            population_size=6,
            max_generations=3,
            seed=123,
        )
    )

    assert resumed.best_score == pytest.approx(uninterrupted.best_score)
    assert [solution.values for solution in resumed.final_solutions] == [
        solution.values for solution in uninterrupted.final_solutions
    ]
    assert resumed.seed == 123
    assert resumed.stop_reason == uninterrupted.stop_reason


def test_ga_resume_from_stable_checkpoint_rejects_config_mismatch(tmp_path) -> None:
    space = GeneSpace.uniform(-1.0, 1.0, 3)
    source = GeneticAlgorithmOptimizer(space, population_size=6, max_generations=3, seed=123)
    generation_zero_population = _population_after_generation_zero(source)
    checkpoint_path = tmp_path / "checkpoint_gen_0.evocore-checkpoint.json"
    source.save_checkpoint(
        checkpoint_path,
        source.checkpoint(generation=0, population=generation_zero_population),
    )

    mismatched = GeneticAlgorithmOptimizer(space, population_size=8, max_generations=3, seed=123)

    with pytest.raises(CheckpointError, match="optimizer_config_hash"):
        mismatched.resume_from_checkpoint(_sphere, checkpoint_path)


def test_ga_resume_keeps_legacy_pickle_path(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint_gen_0.pkl"
    population = [
        Solution([0.0, 0.0], score=0.0, score_valid=True),
        Solution([0.1, 0.0], score=-0.01, score_valid=True),
        Solution([0.0, 0.1], score=-0.01, score_valid=True),
        Solution([0.1, 0.1], score=-0.02, score_valid=True),
    ]

    import pickle

    checkpoint_path.write_bytes(
        pickle.dumps({"population": population, "generation": 0, "seed": 42})
    )

    result = _engine().resume(_sphere, str(checkpoint_path))

    assert result.seed == 42
    assert result.best_solution.score_valid
```

- [ ] **Step 2: Run GA stable resume tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py -v
```

Expected: FAIL because `GeneticAlgorithmOptimizer.checkpoint`, `save_checkpoint`, and `resume_from_checkpoint` do not exist.

- [ ] **Step 3: Implement GA stable checkpoint serialization and resume**

Replace `evocore/optimizers/ga/checkpointing.py` with this content, preserving the module name:

```python
from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evocore.core.errors import CheckpointError
from evocore.results import (
    CheckpointSnapshot,
    OptimizationResult,
    load_checkpoint as load_checkpoint_payload,
    save_checkpoint as save_checkpoint_payload,
    validate_checkpoint_identity,
)
from evocore.search_space import Solution


def _solution_to_checkpoint(solution: Solution) -> dict[str, Any]:
    """Export one solution into the GA checkpoint payload."""
    return {
        "values": list(solution.values),
        "score": solution.score,
        "score_valid": bool(solution.score_valid),
        "metadata": dict(solution.metadata),
    }


def _solution_from_checkpoint(payload: Mapping[str, Any]) -> Solution:
    """Restore one Solution from a GA checkpoint payload row."""
    values = payload.get("values")
    if not isinstance(values, list):
        raise CheckpointError("checkpoint solution.values must be a list.")
    return Solution(
        values,
        score=payload.get("score"),
        score_valid=bool(payload.get("score_valid", False)),
        metadata=dict(payload.get("metadata") or {}),
    )


class GeneticAlgorithmCheckpointingMixin:
    """Checkpoint loading and resume helpers for GA."""

    def checkpoint(
        self,
        *,
        generation: int | None = None,
        population: Sequence[Solution] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointSnapshot:
        """Return a stable GA generation-loop checkpoint snapshot."""
        if generation is None or population is None:
            raise CheckpointError(
                "GA stable checkpoint v1 requires generation and population. "
                "Use CheckpointCallback during generation-loop runs or pass both arguments."
            )
        population_payload = [_solution_to_checkpoint(solution) for solution in population]
        state_payload = {
            "state_kind": "ga_generation_loop",
            "generation": int(generation),
            "population": population_payload,
        }
        best_payload = None
        scored = [
            solution
            for solution in population
            if solution.score is not None and solution.score_valid
        ]
        if scored:
            best = max(scored, key=lambda solution: float(solution.score))
            best_payload = _solution_to_checkpoint(best)
        return CheckpointSnapshot(
            optimizer_type="GeneticAlgorithmOptimizer",
            optimizer_config=self.config_signature(),
            optimizer_config_hash=self.config_hash(),
            gene_space_signature=self.gene_space.signature(),
            gene_space_hash=self.gene_space.hash(),
            direction=self.direction,
            seed=self.seed,
            position={
                "generation": int(generation),
                "event_index": self.state_summary().event_index,
                "n_evaluations": None,
            },
            state={
                "optimizer_type": "GeneticAlgorithmOptimizer",
                "schema_version": 1,
                "payload": state_payload,
            },
            audit={
                "events": self.events.to_dict(),
                "telemetry": self.vnext_telemetry.to_dict(),
                "best": best_payload,
            },
            metadata=dict(metadata or {}),
        )

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
            optimizer_type="GeneticAlgorithmOptimizer",
            gene_space_hash=self.gene_space.hash(),
            optimizer_config_hash=self.config_hash(),
            seed=self.seed,
            direction=self.direction,
        )

    def _population_from_stable_checkpoint(self, payload: Mapping[str, Any]) -> tuple[list[Solution], int]:
        state = payload["state"]
        state_payload = state["payload"]
        if state_payload.get("state_kind") != "ga_generation_loop":
            raise CheckpointError(
                "checkpoint state_kind "
                f"{state_payload.get('state_kind')!r} is not supported by GA generation-loop resume."
            )
        raw_population = state_payload.get("population")
        if not isinstance(raw_population, list):
            raise CheckpointError("checkpoint state.payload.population must be a list.")
        population = []
        for row in raw_population:
            if not isinstance(row, Mapping):
                raise CheckpointError("checkpoint population entries must be objects.")
            population.append(_solution_from_checkpoint(row))
        generation = int(state_payload.get("generation", payload["position"]["generation"]))
        return population, generation

    def resume_from_checkpoint(
        self,
        objective_fn: Callable,
        checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    ) -> OptimizationResult:
        """Resume a GA generation-loop run from a stable checkpoint."""
        payload = (
            load_checkpoint_payload(checkpoint)
            if isinstance(checkpoint, str | os.PathLike)
            else dict(checkpoint)
        )
        self._validate_stable_checkpoint_identity(payload)
        population, saved_generation = self._population_from_stable_checkpoint(payload)
        return self._run_from_population(
            population,
            objective_fn,
            start_generation=saved_generation + 1,
        )

    def _resume_legacy_pickle(self, objective_fn: Callable, checkpoint: str) -> OptimizationResult:
        """Resume from the legacy GA pickle checkpoint format."""
        if not os.path.exists(checkpoint):
            directory = os.path.dirname(checkpoint) or "."
            available = []
            if os.path.isdir(directory):
                available = sorted(
                    name for name in os.listdir(directory) if name.startswith("checkpoint_gen_")
                )
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} not found. Available checkpoints: "
                f"{', '.join(available) or 'none'}"
            )

        try:
            with open(checkpoint, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError(
                f"checkpoint file {checkpoint!r} is corrupt or incompatible: {exc}"
            ) from exc

        solutions = payload.get("population")
        if solutions is None:
            solutions = payload.get("SolutionSet")
        if not isinstance(solutions, list) or not all(
            isinstance(solution, Solution) for solution in solutions
        ):
            raise CheckpointError(
                "checkpoint payload must contain a list[Solution] under key 'population'."
            )

        saved_generation = int(payload.get("generation", -1))
        saved_seed = payload.get("seed")
        if saved_seed is not None and int(saved_seed) != self.seed:
            raise CheckpointError(
                f"checkpoint seed {saved_seed} does not match engine seed {self.seed}."
            )

        return self._run_from_population(
            solutions,
            objective_fn,
            start_generation=saved_generation + 1,
        )

    def resume(self, objective_fn: Callable, checkpoint: str) -> OptimizationResult:
        """Resume a GA run from a stable JSON checkpoint or legacy pickle checkpoint."""
        if str(checkpoint).endswith(".json"):
            return self.resume_from_checkpoint(objective_fn, checkpoint)
        return self._resume_legacy_pickle(objective_fn, checkpoint)


__all__ = ["GeneticAlgorithmCheckpointingMixin"]
```

- [ ] **Step 4: Run GA stable resume tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_ga_engine.py::test_resume_missing_checkpoint_lists_available -v
```

Expected: PASS.

- [ ] **Step 5: Commit GA stable checkpoint resume**

```powershell
git add evocore/optimizers/ga/checkpointing.py tests/unit/test_checkpointing.py
git commit -m "feat(ga): add stable checkpoint resume"
```

## Task 4: CheckpointCallback Stable And Legacy Modes

**Files:**
- Modify: `tests/unit/test_callbacks.py`
- Modify: `evocore/callbacks/checkpointing.py`
- Modify: `evocore/optimizers/ga/generation_loop.py`

- [ ] **Step 1: Add failing callback mode tests**

Modify `tests/unit/test_callbacks.py`.

Update the existing legacy pickle test to pass an explicit format:

```python
def test_checkpoint_callback_writes_legacy_pickle(tmp_path):
    cb = CheckpointCallback(path=str(tmp_path), every=1, format="legacy_pickle")
    cb.bind_context(seed=42)
    pop = SolutionSet([Solution([1.0], score=2.0)])
    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))
    payload = pickle.loads((tmp_path / "checkpoint_gen_3.pkl").read_bytes())
    assert payload["generation"] == 3
    assert payload["seed"] == 42
```

Add this new test:

```python
def test_checkpoint_callback_writes_stable_checkpoint(tmp_path):
    pop = SolutionSet([Solution([1.0], score=2.0, score_valid=True)])

    def factory(*, generation, population, metadata):
        assert generation == 3
        assert [solution.values for solution in population] == [[1.0]]
        assert metadata["callback"]["generation_info"] == {
            "generation": 3,
            "nan_score_count": 0,
            "cached_count": 0,
        }
        engine = GeneticAlgorithmOptimizer(
            GeneSpace.uniform(-1.0, 1.0, 1),
            population_size=4,
            max_generations=4,
            seed=42,
        )
        return engine.checkpoint(
            generation=generation,
            population=population,
            metadata=metadata,
        )

    cb = CheckpointCallback(path=str(tmp_path), every=1, format="stable")
    cb.bind_context(seed=42, checkpoint_factory=factory)

    cb.on_generation_end(3, pop, GenerationInfo(3, 0, 0))

    payload = json.loads(
        (tmp_path / "checkpoint_gen_3.evocore-checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["checkpoint_kind"] == "optimizer_state"
    assert payload["position"]["generation"] == 3
```

Also add missing imports at the top:

```python
from evocore import GeneSpace, GeneticAlgorithmOptimizer
```

- [ ] **Step 2: Run callback tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_callbacks.py -v
```

Expected: FAIL because `CheckpointCallback` does not accept `format` and does not write stable JSON checkpoints.

- [ ] **Step 3: Implement callback modes**

Replace `evocore/callbacks/checkpointing.py` with:

```python
"""Checkpoint-writing callback."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from typing import Literal

from evocore.callbacks.base import Callback, GenerationInfo
from evocore.core.errors import CheckpointError
from evocore.results import CheckpointSnapshot, save_checkpoint
from evocore.search_space import Solution, SolutionSet

CheckpointFormat = Literal["stable", "legacy_pickle"]


class CheckpointCallback(Callback):
    """Write optimizer checkpoints at a fixed generation interval."""

    def __init__(
        self,
        path: str = "./checkpoints",
        every: int = 10,
        format: CheckpointFormat = "legacy_pickle",
    ) -> None:
        if format not in ("stable", "legacy_pickle"):
            raise CheckpointError("CheckpointCallback format must be 'stable' or 'legacy_pickle'.")
        self.path = path
        self.every = every
        self.format = format
        self._seed: int | None = None
        self._checkpoint_factory: Callable[..., CheckpointSnapshot] | None = None

    def bind_context(self, **kwargs) -> None:
        """Capture engine checkpoint context."""
        self._seed = kwargs.get("seed")
        factory = kwargs.get("checkpoint_factory")
        self._checkpoint_factory = factory if callable(factory) else None

    def _generation_metadata(self, info: GenerationInfo) -> dict[str, object]:
        return {
            "callback": {
                "generation_info": {
                    "generation": info.generation,
                    "nan_score_count": info.nan_score_count,
                    "cached_count": info.cached_count,
                }
            }
        }

    def _write_legacy_pickle(self, gen: int, pop: SolutionSet) -> None:
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.pkl")
        with open(filename, "wb") as handle:
            pickle.dump({"population": list(pop), "generation": gen, "seed": self._seed}, handle)

    def _write_stable(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        if self._checkpoint_factory is None:
            raise CheckpointError(
                "CheckpointCallback(format='stable') requires optimizer checkpoint support. "
                "Use format='legacy_pickle' for the legacy population pickle format."
            )
        filename = os.path.join(self.path, f"checkpoint_gen_{gen}.evocore-checkpoint.json")
        snapshot = self._checkpoint_factory(
            generation=gen,
            population=[solution.clone() if isinstance(solution, Solution) else solution for solution in pop],
            metadata=self._generation_metadata(info),
        )
        save_checkpoint(filename, snapshot)

    def on_generation_end(self, gen: int, pop: SolutionSet, info: GenerationInfo) -> None:
        """Persist a checkpoint when the current generation matches the cadence."""
        if self.every <= 0 or gen % self.every != 0:
            return

        os.makedirs(self.path, exist_ok=True)
        if self.format == "legacy_pickle":
            self._write_legacy_pickle(gen, pop)
        else:
            self._write_stable(gen, pop, info)
```

- [ ] **Step 4: Bind GA checkpoint factory into callbacks**

Modify `_bind_callbacks` in `evocore/optimizers/ga/generation_loop.py`:

```python
    def _bind_callbacks(self) -> None:
        for callback in self.callbacks:
            callback.should_stop = False
            callback.bind_context(
                seed=self.seed,
                max_generations=self.max_generations,
                checkpoint_factory=self.checkpoint,
            )
```

- [ ] **Step 5: Run callback and GA checkpoint tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_callbacks.py tests/unit/test_checkpointing.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit callback checkpoint modes**

```powershell
git add evocore/callbacks/checkpointing.py evocore/optimizers/ga/generation_loop.py tests/unit/test_callbacks.py
git commit -m "feat(callbacks): write stable checkpoints"
```

## Task 5: Documentation And Changelog

**Files:**
- Modify: `docs/site/callbacks-checkpointing.md`
- Modify: `docs/site/ga.md`
- Modify: `docs/site/cmaes.md`
- Modify: `docs/site/optimizer-telemetry.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update checkpoint docs**

Replace `docs/site/callbacks-checkpointing.md` with:

```markdown
# Callbacks And Checkpointing

Callbacks observe or influence optimization runs.

::: evocore.callbacks.Callback

::: evocore.callbacks.EarlyStopping

::: evocore.callbacks.ProgressBar

::: evocore.callbacks.CheckpointCallback

::: evocore.callbacks.MetricsLogger

## Stable Checkpoints

`CheckpointCallback` writes stable JSON checkpoint files when `format="stable"`:

```python
from evocore import CheckpointCallback, GeneSpace, GeneticAlgorithmOptimizer

optimizer = GeneticAlgorithmOptimizer(
    GeneSpace.uniform(-1.0, 1.0, 3),
    population_size=20,
    max_generations=10,
    seed=42,
    callbacks=[CheckpointCallback(path="./checkpoints", every=1, format="stable")],
)
```

Stable checkpoint files are named
`checkpoint_gen_{generation}.evocore-checkpoint.json`. They are optimizer state
snapshots for continuation. They are separate from `OptimizationResult.to_dict()`
and `EventHistory.to_rows()`, which are analysis and audit exports.

GA generation-loop checkpoints validate optimizer type, seed, direction,
gene-space hash, optimizer config hash, and seed derivation version before
resuming. Resume fails with `CheckpointError` when the receiving optimizer does
not match the checkpoint identity.

## Legacy Pickle Checkpoints

The old pickle format remains the checkpoint v1 default for compatibility and is
also available explicitly:

```python
CheckpointCallback(path="./checkpoints", every=1, format="legacy_pickle")
```

Legacy pickle files are named `checkpoint_gen_{generation}.pkl` and contain the
population, generation, and seed. They are retained for GA compatibility, but the
stable JSON checkpoint format is the forward contract.

## Unsupported Checkpoint Surfaces

GA ask/tell checkpointing is not part of checkpoint v1. `EventHistory` remains
audit data and is not replayed to rebuild optimizer state.

`CMAESOptimizer` checkpoint/resume is unsupported until the Rust-backed CMA-ES
state exposes a stable export/import contract. CMA-ES result export and event
audit history remain available.
```

- [ ] **Step 2: Update GA docs**

In `docs/site/ga.md`, add this section before `## Result Export`:

```markdown
## Checkpoint Resume

GA generation-loop callback hooks can write stable JSON checkpoints with
`CheckpointCallback(format="stable")`:

```python
from evocore import CheckpointCallback, GeneSpace, GeneticAlgorithmOptimizer

space = GeneSpace.uniform(-1.0, 1.0, 3)
optimizer = GeneticAlgorithmOptimizer(
    space,
    population_size=20,
    max_generations=10,
    seed=42,
    callbacks=[CheckpointCallback(path="./checkpoints", every=1, format="stable")],
)
```

Resume with a matching optimizer:

```python
resumed = GeneticAlgorithmOptimizer(
    space,
    population_size=20,
    max_generations=10,
    seed=42,
).resume_from_checkpoint(
    lambda solution: -sum(float(value) ** 2 for value in solution.values),
    "./checkpoints/checkpoint_gen_3.evocore-checkpoint.json",
)
```

The receiving optimizer must match the checkpoint seed, direction, gene space,
and optimizer configuration. Policy-driven `run(evaluator)` and manual ask/tell
checkpointing are not part of checkpoint v1. Result JSON and event rows are not
checkpoint files.
```

Also update the API member list to include:

```yaml
        - resume_from_checkpoint
        - checkpoint
        - save_checkpoint
        - load_checkpoint
```

- [ ] **Step 3: Update CMA-ES docs**

In `docs/site/cmaes.md`, add this section before `## Result Export`:

```markdown
## Checkpoint Resume

`CMAESOptimizer` checkpoint/resume is unsupported in checkpoint v1. The Rust-backed
CMA-ES state must expose stable serializable fields before EvoCore can continue
the same covariance trajectory from a checkpoint.

Use `OptimizationResult.to_dict()` for completed-run export and `engine.events`
for ask/tell audit rows. Those exports are not checkpoint files and are not
replayed to rebuild CMA-ES state.
```

- [ ] **Step 4: Update telemetry docs**

Append this section to `docs/site/optimizer-telemetry.md`:

```markdown
## Telemetry And Checkpoints

Checkpoint files may include telemetry for audit continuity, but telemetry is not
the source of optimizer state. Resume uses the checkpoint state payload and
validates optimizer identity before mutating the optimizer.

`OptimizationResult.to_dict()`, `OptimizationTelemetry.to_dict()`, and
`EventHistory.to_rows()` are export surfaces for analysis and inspection. They are
not accepted as checkpoint resume files.
```

- [ ] **Step 5: Update changelog**

In `CHANGELOG.md`, under `[Unreleased]` / `### Added`, add:

```markdown
- Stable JSON checkpoint envelope helpers and GA generation-loop checkpoint/resume
  support with optimizer, gene-space, config, seed, direction, and seed-derivation
  validation.
```

Under `[Unreleased]` / `### Changed`, add:

```markdown
- `CheckpointCallback` now supports `format="stable"` for JSON checkpoint files
  while keeping the legacy pickle population format as the checkpoint v1 default.
```

- [ ] **Step 6: Run documentation-focused checks**

Run:

```powershell
git diff --check
```

Expected: PASS.

- [ ] **Step 7: Commit docs and changelog**

```powershell
git add docs/site/callbacks-checkpointing.md docs/site/ga.md docs/site/cmaes.md docs/site/optimizer-telemetry.md CHANGELOG.md
git commit -m "docs: document checkpoint resume contract"
```

## Task 6: Final Verification

**Files:**
- Verify all touched Python, docs, and tests.

- [ ] **Step 1: Run targeted checkpoint and callback tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpointing.py tests/unit/test_callbacks.py -v
```

Expected: PASS.

- [ ] **Step 2: Run public import and GA regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_domain_imports.py tests/unit/test_package_init.py tests/unit/test_ga_engine.py -v
```

Expected: PASS.

- [ ] **Step 3: Run formatter and linter checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
git diff --check
```

Expected: PASS for all commands.

- [ ] **Step 4: Run broader Python suite if targeted checks pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
```

Expected: PASS.

- [ ] **Step 5: Confirm no verification-only edits remain**

Run:

```powershell
git status --short
```

Expected: no uncommitted files after the verification commands.

## Task 7: Open Follow-On Plans

**Files:**
- No source edits in this task.

- [ ] **Step 1: Record follow-on scope in the final implementation summary**

When implementation finishes, include these follow-on items in the final response:

```text
Follow-on plan candidates:
- GA ask/tell checkpoint snapshots: candidate ledger, batch ledger, trusted population, telemetry, best candidate, event index, and pending batches.
- CMA-ES checkpoint snapshots: Rust/PyO3 state export/import for mean, sigma, covariance, evolution paths, generation, and pending continuous samples.
```

- [ ] **Step 2: Confirm PR status**

Run:

```powershell
gh pr view --json url,isDraft,state,number,title,headRefName,baseRefName
```

Expected: Existing draft PR `#13` for `feature/general-optimizer-framework` is open.

- [ ] **Step 3: Push the branch**

Run:

```powershell
git status --short --branch
git push origin feature/general-optimizer-framework
```

Expected: branch pushes cleanly and working tree is clean.
