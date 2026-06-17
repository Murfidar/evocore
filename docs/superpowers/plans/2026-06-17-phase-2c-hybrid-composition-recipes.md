# Phase 2C Hybrid Composition Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lightweight helper APIs and docs for outer optimizer plus inner optimizer workflows, especially outer GA and inner CMA-ES refinement.

**Architecture:** Implement generic lifecycle composition helpers for deterministic nested seeds, lineage metadata, and converting inner optimization results into outer `EvaluationRecord` objects. Keep this recipe-first and do not add a formal `HybridOptimizer` abstraction in this phase.

**Tech Stack:** Python helper functions, EvoCore lifecycle records, stable hashing, pytest, ruff, MkDocs Markdown.

---

## Dependency

Complete Phase 1 first. Phase 2A is recommended because archive-backed inner warm starts are part of the docs recipe, but this helper module does not require archive internals.

Source design:

- `docs/superpowers/specs/2026-06-17-evocore-phase-2-expensive-optimization-toolkit-design.md`

## File Structure

- Create: `evocore/lifecycle/composition.py`
  - Child seed helper.
  - Candidate identity helper.
  - Lineage metadata helper.
  - Inner-result-to-outer-record helper.
- Modify: `evocore/lifecycle/__init__.py`
  - Re-export composition names.
- Modify: `evocore/__init__.py`
  - Re-export common helper names.
- Create: `tests/unit/test_lifecycle_composition.py`
  - Unit tests for deterministic seeds, lineage metadata, and record conversion.
- Create: `tests/unit/test_phase2c_hybrid_recipe.py`
  - Small outer GA / inner CMA smoke test using helpers.
- Modify: `tests/unit/test_package_init.py`
  - Export smoke tests.
- Modify: `docs/site/api.md`
  - API reference entries.
- Modify: `docs/site/expensive-external-evaluations.md`
  - Replace the current hand-written hybrid example with helper-based recipe.
- Modify: `CHANGELOG.md`
  - Public API entry.

## Public API Names

Export these names from `evocore.lifecycle` and top-level `evocore`:

- `derive_child_seed`
- `inner_result_record`
- `lineage_metadata`

Keep private identity helpers in `evocore.lifecycle.composition` unless they become documented public API later.

## Task 1: Write Composition Unit Tests

**Files:**
- Create: `tests/unit/test_lifecycle_composition.py`

- [ ] **Step 1: Add fixtures and seed derivation tests**

Create `tests/unit/test_lifecycle_composition.py`:

```python
from evocore import (
    Candidate,
    CandidateSnapshot,
    EvaluationRecord,
    GeneSpace,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)


def _snapshot() -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id="outer-1",
        candidate_hash="hash-outer",
        values=(1.0, 2.0),
        params={"x": 1.0, "y": 2.0},
        origin="memory_seed",
        batch_id="outer-batch",
        event_index=3,
        generation=None,
        status="trusted",
        stage="template",
        confidence="cached",
        score=10.0,
        scores={},
        cost=0.0,
        metadata={"family": "template-a"},
    )


def test_derive_child_seed_is_deterministic() -> None:
    first = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")
    second = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")

    assert first == second
    assert isinstance(first, int)
    assert 0 <= first < 2**32


def test_derive_child_seed_changes_with_hash_or_stage() -> None:
    base = derive_child_seed(parent_seed=42, candidate_hash="abc", stage="inner_cma")

    assert derive_child_seed(parent_seed=42, candidate_hash="def", stage="inner_cma") != base
    assert derive_child_seed(parent_seed=42, candidate_hash="abc", stage="audit") != base
```

- [ ] **Step 2: Add lineage metadata tests**

Append:

```python
def test_lineage_metadata_from_candidate_snapshot_is_json_safe() -> None:
    metadata = lineage_metadata(
        outer_candidate=_snapshot(),
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=123,
        stage="inner_cma",
        checkpoint_path="runs/inner-1.json",
        metadata={"archive_id": "template-a"},
    )

    assert metadata == {
        "outer_candidate_id": "outer-1",
        "outer_candidate_hash": "hash-outer",
        "outer_batch_id": "outer-batch",
        "inner_optimizer_type": "CMAESOptimizer",
        "inner_seed": 123,
        "composition_stage": "inner_cma",
        "inner_checkpoint_path": "runs/inner-1.json",
        "archive_id": "template-a",
    }


def test_lineage_metadata_from_candidate_requires_gene_space_for_hash() -> None:
    space = GeneSpace.uniform(-5.0, 5.0, 2)
    candidate = Candidate(
        candidate_id="outer-raw",
        genes=[1.0, 2.0],
        batch_id="outer-batch",
    )

    metadata = lineage_metadata(
        outer_candidate=candidate,
        gene_space=space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=123,
        stage="inner_cma",
    )

    assert metadata["outer_candidate_id"] == "outer-raw"
    assert metadata["outer_candidate_hash"] == space.value_hash([1.0, 2.0])
```

- [ ] **Step 3: Add inner result record tests**

Append:

```python
def test_inner_result_record_targets_outer_candidate() -> None:
    snapshot = _snapshot()
    record = inner_result_record(
        outer_candidate=snapshot,
        score=17.5,
        confidence="trusted_full",
        stage="inner_cma",
        cost=32.0,
        metrics={"inner_generations": 4},
        metadata={"inner_seed": 123},
    )

    assert record == EvaluationRecord(
        candidate_id="outer-1",
        batch_id="outer-batch",
        score=17.5,
        confidence="trusted_full",
        stage="inner_cma",
        cost=32.0,
        metrics={"inner_generations": 4},
        metadata={"inner_seed": 123},
    )
```

- [ ] **Step 4: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_composition.py -v
```

Expected: fails with import errors for composition helper names.

## Task 2: Implement Composition Helpers

**Files:**
- Create: `evocore/lifecycle/composition.py`

- [ ] **Step 1: Add imports and candidate identity helper**

Create `evocore/lifecycle/composition.py`:

```python
"""Composition helpers for nested expensive optimization workflows."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from evocore import _core
from evocore.core.errors import ConfigurationError, FitnessError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import CandidateSnapshot
from evocore.lifecycle.records import Candidate, EvaluationConfidence, EvaluationRecord
from evocore.search_space import GeneSpace


def _json_metadata(value: Mapping[str, object] | None, *, field_name: str) -> dict[str, object]:
    payload = json_safe(dict(value or {}))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"{field_name} must be a JSON-safe mapping.")
    return payload


def _outer_identity(
    outer_candidate: Candidate | CandidateSnapshot,
    *,
    gene_space: GeneSpace | None = None,
) -> tuple[str, str, str | None]:
    if isinstance(outer_candidate, CandidateSnapshot):
        return (
            outer_candidate.candidate_id,
            outer_candidate.candidate_hash,
            outer_candidate.batch_id,
        )
    if isinstance(outer_candidate, Candidate):
        if gene_space is None:
            raise ConfigurationError("gene_space is required when outer_candidate is Candidate.")
        return (
            outer_candidate.candidate_id,
            outer_candidate.candidate_hash(gene_space),
            outer_candidate.batch_id or None,
        )
    raise ConfigurationError("outer_candidate must be Candidate or CandidateSnapshot.")
```

- [ ] **Step 2: Add deterministic child seed helper**

Append:

```python
def derive_child_seed(
    *,
    parent_seed: int,
    candidate_hash: str,
    stage: str,
) -> int:
    """Derive a deterministic nested optimizer seed from parent seed, hash, and stage."""
    if not candidate_hash:
        raise ConfigurationError("candidate_hash must be non-empty.")
    if not stage:
        raise ConfigurationError("stage must be non-empty.")
    digest = hashlib.sha256(f"{candidate_hash}:{stage}".encode("utf-8")).digest()
    child_index = int.from_bytes(digest[:4], "big", signed=False)
    return int(_core.py_derive_seed(int(parent_seed), 0, child_index, _core.OP_MULTI_RUN)) % (2**32)
```

- [ ] **Step 3: Add lineage metadata helper**

Append:

```python
def lineage_metadata(
    *,
    outer_candidate: Candidate | CandidateSnapshot,
    inner_optimizer_type: str,
    inner_seed: int,
    stage: str,
    gene_space: GeneSpace | None = None,
    checkpoint_path: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build JSON-safe lineage metadata for a nested optimizer run."""
    if not inner_optimizer_type:
        raise ConfigurationError("inner_optimizer_type must be non-empty.")
    if not stage:
        raise ConfigurationError("stage must be non-empty.")
    candidate_id, candidate_hash, batch_id = _outer_identity(
        outer_candidate,
        gene_space=gene_space,
    )
    payload = {
        "outer_candidate_id": candidate_id,
        "outer_candidate_hash": candidate_hash,
        "inner_optimizer_type": inner_optimizer_type,
        "inner_seed": int(inner_seed),
        "composition_stage": stage,
    }
    if batch_id is not None:
        payload["outer_batch_id"] = batch_id
    if checkpoint_path is not None:
        payload["inner_checkpoint_path"] = str(checkpoint_path)
    payload.update(_json_metadata(metadata, field_name="metadata"))
    return _json_metadata(payload, field_name="lineage metadata")
```

- [ ] **Step 4: Add inner-result-to-outer-record helper**

Append:

```python
def inner_result_record(
    *,
    outer_candidate: Candidate | CandidateSnapshot,
    score: float,
    confidence: EvaluationConfidence,
    stage: str,
    cost: float = 0.0,
    metrics: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    gene_space: GeneSpace | None = None,
) -> EvaluationRecord:
    """Convert an inner optimizer result into an outer candidate evaluation record."""
    if not math.isfinite(float(score)):
        raise FitnessError("inner_result_record score must be finite.")
    candidate_id, _, batch_id = _outer_identity(outer_candidate, gene_space=gene_space)
    return EvaluationRecord(
        candidate_id=candidate_id,
        batch_id=batch_id,
        score=float(score),
        confidence=confidence,
        stage=stage,
        cost=float(cost),
        metrics=_json_metadata(metrics, field_name="metrics"),
        metadata=_json_metadata(metadata, field_name="metadata"),
    )


__all__ = [
    "derive_child_seed",
    "inner_result_record",
    "lineage_metadata",
]
```

- [ ] **Step 5: Run unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_composition.py -v
```

Expected: all composition unit tests pass.

## Task 3: Add Hybrid Recipe Smoke Test and Exports

**Files:**
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`
- Create: `tests/unit/test_phase2c_hybrid_recipe.py`

- [ ] **Step 1: Re-export helper names**

In `evocore/lifecycle/__init__.py` and `evocore/__init__.py`, import and add:

```python
derive_child_seed
inner_result_record
lineage_metadata
```

- [ ] **Step 2: Add top-level export test**

Append to `tests/unit/test_package_init.py`:

```python
def test_phase2c_composition_public_exports():
    from evocore import derive_child_seed, inner_result_record, lineage_metadata

    assert derive_child_seed is not None
    assert inner_result_record is not None
    assert lineage_metadata is not None
```

- [ ] **Step 3: Add hybrid helper smoke test**

Create `tests/unit/test_phase2c_hybrid_recipe.py`:

```python
from evocore import (
    CMAESOptimizer,
    EvaluationRecord,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)


def test_outer_ga_inner_cma_helper_flow() -> None:
    outer_space = GeneSpace.uniform(-5.0, 5.0, 2)
    outer = GeneticAlgorithmOptimizer(outer_space, population_size=4, seed=100)
    outer_candidate = outer.ask(1)[0]
    outer_hash = outer_candidate.candidate_hash(outer_space)

    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=outer_hash,
        stage="inner_cma",
    )
    inner = CMAESOptimizer(GeneSpace.uniform(-1.0, 1.0, 2), population_size=4, seed=inner_seed)

    inner_candidates = inner.ask(4)
    inner_update = inner.tell(
        [
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                batch_id=candidate.batch_id,
                score=float(index),
                confidence="trusted_full",
                stage="inner_full",
            )
            for index, candidate in enumerate(inner_candidates)
        ]
    )
    metadata = lineage_metadata(
        outer_candidate=outer_candidate,
        gene_space=outer_space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=inner_seed,
        stage="inner_cma",
    )
    record = inner_result_record(
        outer_candidate=outer_candidate,
        gene_space=outer_space,
        score=inner_update.best_score,
        confidence="trusted_full",
        stage="inner_cma",
        metadata=metadata,
    )

    update = outer.tell([record])

    assert update.trusted_count == 1
    assert update.best_candidate_id == outer_candidate.candidate_id
```

- [ ] **Step 4: Run composition and integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_composition.py tests/unit/test_phase2c_hybrid_recipe.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

## Task 4: Add Docs and Changelog

**Files:**
- Modify: `docs/site/api.md`
- Modify: `docs/site/expensive-external-evaluations.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add API docs**

Add to `docs/site/api.md` near lifecycle helpers:

```markdown
::: evocore.lifecycle.derive_child_seed

::: evocore.lifecycle.lineage_metadata

::: evocore.lifecycle.inner_result_record
```

- [ ] **Step 2: Replace hybrid recipe with helper-based example**

In `docs/site/expensive-external-evaluations.md`, update the "Hybrid Outer GA And Inner CMA-ES" code block to use the helpers:

````markdown
## Hybrid Outer GA And Inner CMA-ES

A common expensive workflow is an outer optimizer over structures or templates and an inner optimizer over active continuous parameters.

```python
from evocore import (
    CMAESOptimizer,
    GeneticAlgorithmOptimizer,
    derive_child_seed,
    inner_result_record,
    lineage_metadata,
)

outer = GeneticAlgorithmOptimizer(template_space, population_size=24, seed=100)

for template_candidate in outer.ask(4):
    template_hash = template_candidate.candidate_hash(template_space)
    inner_seed = derive_child_seed(
        parent_seed=100,
        candidate_hash=template_hash,
        stage="inner_cma",
    )
    template = decode_template(template_candidate.params)
    inner_space = template.active_parameter_space()
    inner = CMAESOptimizer(inner_space, population_size=16, seed=inner_seed)

    prior_records = lookup_template_archive(template.name)
    if prior_records:
        inner.warm_start(prior_records, mode="state", cma_mean_strategy="top_k_centroid")

    tuned = run_inner_backtests(inner, template)
    metadata = lineage_metadata(
        outer_candidate=template_candidate,
        gene_space=template_space,
        inner_optimizer_type="CMAESOptimizer",
        inner_seed=inner_seed,
        stage="inner_cma",
        metadata={"template_name": template.name},
    )
    outer.tell(
        [
            inner_result_record(
                outer_candidate=template_candidate,
                gene_space=template_space,
                score=tuned.best_score,
                confidence="trusted_full",
                stage="inner_cma",
                metadata=metadata,
            )
        ]
    )
```
````

- [ ] **Step 3: Add changelog entry**

Under `CHANGELOG.md` `## [Unreleased]` `### Added`, add:

```markdown
- Added hybrid composition helpers for nested expensive optimization workflows, including deterministic child seeds, lineage metadata, and inner-result evaluation records.
```

## Task 5: Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_composition.py tests/unit/test_phase2c_hybrid_recipe.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Phase 1 external-state regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_external_state_optimizer_contract.py tests/unit/test_ga_external_state.py tests/unit/test_cmaes_external_state.py -v
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

- [ ] **Step 5: Commit Phase 2C**

Run:

```powershell
git status --short
git add evocore/lifecycle/composition.py evocore/lifecycle/__init__.py evocore/__init__.py tests/unit/test_lifecycle_composition.py tests/unit/test_phase2c_hybrid_recipe.py tests/unit/test_package_init.py docs/site/api.md docs/site/expensive-external-evaluations.md CHANGELOG.md
git commit -m "feat(lifecycle): add hybrid composition helpers"
```

Expected: commit succeeds and contains only Phase 2C composition, docs, tests, and changelog changes.

## Compatibility Notes

- This plan is additive public API.
- No optimizer method signatures change.
- No checkpoint schema changes are required.
- Seed derivation is deterministic but new; docs should not claim it matches previous hand-written downstream seed formulas.
- Helper metadata is JSON-safe and generic. Trading-specific metadata belongs in the caller-provided `metadata` mapping.

## Self-Review Notes

- Spec coverage: outer/inner composition helpers, deterministic nested seeds, lineage metadata, inner-result records, docs, and tests are covered.
- Type consistency: helpers accept `Candidate` or `CandidateSnapshot` and emit existing `EvaluationRecord`.
- Scope boundary: formal `HybridOptimizer`, conditional search spaces, active subspaces, and CMA restart strategies remain Phase 3 work.
