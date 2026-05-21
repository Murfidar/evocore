# Candidate/Solution Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize EvoCore's `Candidate`/`Solution` boundary with explicit conversion helpers and GeneSpace-owned candidate value hashing.

**Architecture:** `GeneSpace` owns schema-aware value signatures and hashes. `Candidate` keeps decoded proposal values for evaluator ergonomics, but delegates preferred hash semantics to `GeneSpace`. A new `evocore.lifecycle.conversion` module owns `Candidate`/`Solution` conversions so GA and CMA-ES stop hand-rolling subtly different result materialization code.

**Tech Stack:** Python dataclasses and type hints, existing `canonical_json_hash` serialization helper, pytest and Hypothesis, MkDocs markdown docs, repository-local `.venv` commands.

---

## Scope Check

This plan implements the approved spec at:

```text
docs/superpowers/specs/2026-05-19-candidate-solution-boundary-design.md
```

The spec is one coherent stabilization slice. It does not add islands, archives,
migration, surrogate memory, checkpoint reload, or public `Individual` aliases.

## File Structure

Create or modify these files:

```text
evocore/search_space/genes.py
```

Owns `GeneSpace.value_signature(values)` and `GeneSpace.value_hash(values)`.

```text
evocore/lifecycle/records.py
```

Keeps `Candidate` lifecycle-facing and adds the optional `gene_space` argument to
`Candidate.candidate_hash(...)`.

```text
evocore/lifecycle/telemetry.py
```

Records proposed candidate hashes with an optional `GeneSpace`, preserving old behavior
when no schema is supplied.

```text
evocore/lifecycle/conversion.py
```

New focused module for `candidate_to_solution(...)` and `solution_to_candidate(...)`.

```text
evocore/lifecycle/__init__.py
```

Exports the conversion helpers from the lifecycle package.

```text
evocore/optimizers/ga/ask_tell.py
evocore/optimizers/cmaes/ask_tell.py
```

Use GeneSpace-backed candidate hashes for events and telemetry. GA result construction
uses shared conversion helpers where lifecycle candidates become public `Solution`
records.

```text
tests/unit/test_gene_space.py
tests/property/test_gene_space_properties.py
tests/unit/test_vnext_evaluation.py
tests/unit/test_lifecycle_conversion.py
tests/unit/test_domain_imports.py
tests/unit/test_ga_ask_tell_vnext.py
tests/unit/test_cmaes_ask_tell_vnext.py
```

Focused contract tests for value hashing, candidate hash semantics, conversions, exports,
and engine event/telemetry integration.

```text
docs/site/ask-tell-engines.md
docs/site/api.md
CHANGELOG.md
```

Public docs and changelog updates for the stabilized boundary.

## Task 1: GeneSpace Value Identity

**Files:**
- Modify: `tests/unit/test_gene_space.py`
- Modify: `tests/property/test_gene_space_properties.py`
- Modify: `evocore/search_space/genes.py`

- [ ] **Step 1: Add failing unit tests for value signatures and hashes**

Append these tests to `tests/unit/test_gene_space.py`:

```python
def test_gene_space_value_signature_uses_declared_gene_kinds():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )

    signature = space.value_signature([1, 3, True])

    assert signature == {
        "schema_version": 1,
        "gene_space_hash": space.hash(),
        "values": [
            {"name": "x", "kind": "float", "value": float(1).hex()},
            {"name": "period", "kind": "int", "value": 3},
            {"name": "enabled", "kind": "bool", "value": True},
        ],
    }


def test_gene_space_value_hash_is_deterministic():
    space = GeneSpace(
        [
            Gene("x", "float", -1.0, 1.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )

    first = space.value_hash([0.25, 10, False])
    second = space.value_hash([0.25, 10, False])

    assert first == second
    assert len(first) == 64


def test_gene_space_value_hash_includes_gene_space_identity():
    left = GeneSpace([Gene("x", "float", -1.0, 1.0)])
    right = GeneSpace([Gene("renamed_x", "float", -1.0, 1.0)])

    assert left.value_hash([0.0]) != right.value_hash([0.0])


def test_gene_space_value_signature_reuses_validate_genes_errors():
    space = GeneSpace([Gene("enabled", "bool")])

    with pytest.raises(ConfigurationError, match="expects bool"):
        space.value_signature([1])
```

- [ ] **Step 2: Run the focused unit tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_gene_space.py::test_gene_space_value_signature_uses_declared_gene_kinds tests/unit/test_gene_space.py::test_gene_space_value_hash_is_deterministic tests/unit/test_gene_space.py::test_gene_space_value_hash_includes_gene_space_identity tests/unit/test_gene_space.py::test_gene_space_value_signature_reuses_validate_genes_errors -v
```

Expected: FAIL with `AttributeError: 'GeneSpace' object has no attribute 'value_signature'`.

- [ ] **Step 3: Implement GeneSpace value identity helpers**

In `evocore/search_space/genes.py`, add these methods inside `class GeneSpace` after
`hash()` and before `to_json(...)`:

```python
    def value_signature(self, values: Sequence[float | int | bool]) -> dict[str, Any]:
        """Return a stable schema-aware signature for decoded gene values."""
        self.validate_genes(values)

        encoded_values: list[dict[str, Any]] = []
        for value, gene in zip(values, self._genes, strict=False):
            if gene.kind == "bool":
                encoded_value: str | int | bool = bool(value)
            elif gene.kind == "int":
                encoded_value = int(value)
            else:
                encoded_value = float(value).hex()
            encoded_values.append(
                {
                    "name": gene.name,
                    "kind": gene.kind,
                    "value": encoded_value,
                }
            )

        return {
            "schema_version": 1,
            "gene_space_hash": self.hash(),
            "values": encoded_values,
        }

    def value_hash(self, values: Sequence[float | int | bool]) -> str:
        """Return a stable SHA-256 hash for decoded values in this gene space."""
        return canonical_json_hash(self.value_signature(values))
```

- [ ] **Step 4: Run the focused unit tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_gene_space.py::test_gene_space_value_signature_uses_declared_gene_kinds tests/unit/test_gene_space.py::test_gene_space_value_hash_is_deterministic tests/unit/test_gene_space.py::test_gene_space_value_hash_includes_gene_space_identity tests/unit/test_gene_space.py::test_gene_space_value_signature_reuses_validate_genes_errors -v
```

Expected: PASS.

- [ ] **Step 5: Add failing property tests for value hash stability**

Append this composite strategy and tests to `tests/property/test_gene_space_properties.py`:

```python
@st.composite
def valid_flat_gene_spaces_with_values(draw):
    kinds = draw(st.lists(st.sampled_from(["float", "int", "bool"]), min_size=1, max_size=8))
    genes = []
    values = []
    for index, kind in enumerate(kinds):
        name = f"gene_{index}"
        if kind == "float":
            low = draw(
                st.floats(
                    min_value=-1000.0,
                    max_value=999.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            span = draw(
                st.floats(
                    min_value=1e-6,
                    max_value=1000.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            high = low + span
            value = draw(
                st.floats(
                    min_value=low,
                    max_value=high,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            genes.append(Gene(name, "float", low, high))
            values.append(value)
        elif kind == "int":
            low = draw(st.integers(min_value=-1000, max_value=999))
            high = draw(st.integers(min_value=low + 1, max_value=low + 1000))
            value = draw(st.integers(min_value=low, max_value=high))
            genes.append(Gene(name, "int", low, high))
            values.append(value)
        else:
            value = draw(st.booleans())
            genes.append(Gene(name, "bool"))
            values.append(value)
    space = GeneSpace(genes)
    return space, values


@given(valid_flat_gene_spaces_with_values())
def test_gene_space_value_hash_is_stable_for_equivalent_values(case):
    space, values = case
    equivalent = GeneSpace(list(space.genes), has_names=space.has_names)

    assert equivalent.value_signature(values) == space.value_signature(values)
    assert equivalent.value_hash(values) == space.value_hash(values)


@given(valid_flat_gene_spaces_with_values())
def test_gene_space_value_signature_json_round_trips(case):
    space, values = case

    assert json.loads(stable_json_dumps(space.value_signature(values))) == space.value_signature(
        values
    )
```

Also update the imports at the top of `tests/property/test_gene_space_properties.py`:

```python
from evocore.core.serialization import stable_json_dumps
```

- [ ] **Step 6: Run the property tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/property/test_gene_space_properties.py::test_gene_space_value_hash_is_stable_for_equivalent_values tests/property/test_gene_space_properties.py::test_gene_space_value_signature_json_round_trips -v
```

Expected: PASS.

- [ ] **Step 7: Commit GeneSpace value identity**

Run:

```powershell
git add evocore/search_space/genes.py tests/unit/test_gene_space.py tests/property/test_gene_space_properties.py
git commit -m "feat(search-space): add schema-aware value hashes"
```

## Task 2: Candidate Hash And Telemetry Semantics

**Files:**
- Modify: `tests/unit/test_vnext_evaluation.py`
- Modify: `evocore/lifecycle/records.py`
- Modify: `evocore/lifecycle/telemetry.py`

- [ ] **Step 1: Add failing tests for schema-aware candidate hashes**

Append these tests to `tests/unit/test_vnext_evaluation.py`:

```python
def test_candidate_hash_uses_gene_space_value_hash_when_supplied() -> None:
    space = GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )
    left = Candidate(candidate_id="c-left", genes=[1.0, 5, True], event_index=0)
    right = Candidate(candidate_id="c-right", genes=[1, 5, True], event_index=0)

    assert left.candidate_id != right.candidate_id
    assert left.candidate_hash(space) == space.value_hash(left.genes)
    assert left.candidate_hash(space) == right.candidate_hash(space)


def test_candidate_hash_without_gene_space_keeps_legacy_fallback() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[1.0, 5, True], event_index=0)

    assert candidate.candidate_hash() == candidate.candidate_hash()
    assert len(candidate.candidate_hash()) == 64


def test_telemetry_records_schema_aware_candidate_hashes() -> None:
    space = GeneSpace([Gene("x", "float", -5.0, 5.0)])
    telemetry = OptimizationTelemetry()
    candidates = [
        Candidate(candidate_id="c-1", genes=[1.0], origin="random", event_index=0),
        Candidate(candidate_id="c-2", genes=[1], origin="random", event_index=0),
    ]

    telemetry.record_proposed_candidates(candidates, gene_space=space)

    assert telemetry.unique_candidate_hashes == {space.value_hash([1.0])}
```

Update the imports near the top of `tests/unit/test_vnext_evaluation.py`:

```python
from evocore.search_space import Gene, GeneSpace
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vnext_evaluation.py::test_candidate_hash_uses_gene_space_value_hash_when_supplied tests/unit/test_vnext_evaluation.py::test_candidate_hash_without_gene_space_keeps_legacy_fallback tests/unit/test_vnext_evaluation.py::test_telemetry_records_schema_aware_candidate_hashes -v
```

Expected: FAIL because `Candidate.candidate_hash(...)` and
`OptimizationTelemetry.record_proposed_candidates(...)` do not accept `gene_space`.

- [ ] **Step 3: Update Candidate.candidate_hash to accept an optional GeneSpace**

In `evocore/lifecycle/records.py`, add `TYPE_CHECKING` to the typing import:

```python
from typing import TYPE_CHECKING, Any, Literal
```

Add this block after the `GeneValue` import:

```python
if TYPE_CHECKING:
    from evocore.search_space import GeneSpace
```

Replace `Candidate.candidate_hash` with:

```python
    def candidate_hash(self, gene_space: GeneSpace | None = None) -> str:
        """Return a stable hash for this candidate's decoded genes."""
        if gene_space is not None:
            return gene_space.value_hash(self.genes)

        encoded: list[list[Any]] = []
        for value in self.genes:
            if isinstance(value, bool):
                encoded.append(["bool", value])
            elif isinstance(value, int):
                encoded.append(["int", value])
            elif isinstance(value, float):
                encoded.append(["float", value.hex()])
            else:
                encoded.append([type(value).__name__, repr(value)])
        payload = json.dumps(encoded, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Update telemetry to accept an optional GeneSpace**

In `evocore/lifecycle/telemetry.py`, update imports:

```python
from typing import TYPE_CHECKING, Any
```

Add this block after the `Candidate` import:

```python
if TYPE_CHECKING:
    from evocore.search_space import GeneSpace
```

Replace `record_proposed_candidates` with:

```python
    def record_proposed_candidates(
        self,
        candidates: Sequence[Candidate],
        *,
        gene_space: GeneSpace | None = None,
    ) -> None:
        """Record newly proposed candidates and their unique genome hashes."""
        proposed = list(candidates)
        self.record_proposed(len(proposed))
        self.unique_candidate_hashes.update(
            candidate.candidate_hash(gene_space) for candidate in proposed
        )
```

- [ ] **Step 5: Run the focused tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vnext_evaluation.py::test_candidate_hash_uses_gene_space_value_hash_when_supplied tests/unit/test_vnext_evaluation.py::test_candidate_hash_without_gene_space_keeps_legacy_fallback tests/unit/test_vnext_evaluation.py::test_telemetry_records_schema_aware_candidate_hashes -v
```

Expected: PASS.

- [ ] **Step 6: Commit candidate hash semantics**

Run:

```powershell
git add evocore/lifecycle/records.py evocore/lifecycle/telemetry.py tests/unit/test_vnext_evaluation.py
git commit -m "feat(lifecycle): align candidate hashes with gene spaces"
```

## Task 3: Candidate/Solution Conversion Helpers

**Files:**
- Create: `tests/unit/test_lifecycle_conversion.py`
- Create: `evocore/lifecycle/conversion.py`
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `tests/unit/test_domain_imports.py`

- [ ] **Step 1: Add failing conversion tests**

Create `tests/unit/test_lifecycle_conversion.py`:

```python
import pytest

from evocore.lifecycle import (
    Candidate,
    EvaluationRecord,
    candidate_to_solution,
    solution_to_candidate,
)
from evocore.search_space import Gene, GeneSpace, Solution


def _space() -> GeneSpace:
    return GeneSpace(
        [
            Gene("x", "float", -5.0, 5.0),
            Gene("period", "int", 2, 20),
            Gene("enabled", "bool"),
        ]
    )


def test_candidate_to_solution_copies_state_eligible_score_and_provenance() -> None:
    space = _space()
    candidate = Candidate(
        candidate_id="c-1",
        batch_id="b-1",
        genes=[1.0, 5, True],
        params={"x": 1.0, "period": 5, "enabled": True},
        origin="random",
        event_index=3,
        generation=2,
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            batch_id="b-1",
            score=10.0,
            confidence="trusted_full",
            stage="full",
            cost=1.0,
        )
    )

    solution = candidate_to_solution(candidate, direction="maximize", gene_space=space)

    assert solution.values == [1.0, 5, True]
    assert solution.score == pytest.approx(10.0)
    assert solution.score_valid is True
    assert solution.metadata["params"] == {"x": 1.0, "period": 5, "enabled": True}
    assert solution.metadata["candidate_id"] == "c-1"
    assert solution.metadata["candidate_hash"] == space.value_hash([1.0, 5, True])
    assert solution.metadata["batch_id"] == "b-1"
    assert solution.metadata["origin"] == "random"
    assert solution.metadata["generation"] == 2
    assert "stage" not in solution.metadata
    assert "status" not in solution.metadata
    assert "scores" not in solution.metadata
    assert not hasattr(solution, "stage")
    assert not hasattr(solution, "status")


def test_candidate_to_solution_uses_raw_minimize_state_score() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[0.0], event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=5.0,
            confidence="cached",
            stage="full",
            cost=0.0,
        )
    )
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=2.0,
            confidence="trusted_full",
            stage="rerun",
            cost=1.0,
        )
    )

    solution = candidate_to_solution(candidate, direction="minimize")

    assert solution.score == pytest.approx(2.0)
    assert solution.score_valid is True


def test_candidate_to_solution_leaves_non_state_observation_invalid() -> None:
    candidate = Candidate(candidate_id="c-1", genes=[0.0], event_index=0)
    candidate.apply_record(
        EvaluationRecord(
            candidate_id="c-1",
            score=99.0,
            confidence="partial",
            stage="cheap",
            cost=0.1,
        )
    )

    solution = candidate_to_solution(candidate, direction="maximize")

    assert solution.values == [0.0]
    assert solution.score is None
    assert solution.score_valid is False
    assert solution.metadata["candidate_id"] == "c-1"


def test_candidate_to_solution_can_omit_provenance() -> None:
    candidate = Candidate(candidate_id="c-1", batch_id="b-1", genes=[0.0], event_index=0)

    solution = candidate_to_solution(
        candidate,
        direction="maximize",
        include_provenance=False,
    )

    assert solution.metadata == {}


def test_solution_to_candidate_recomputes_params_and_does_not_copy_score() -> None:
    space = _space()
    solution = Solution(
        [1.0, 5, True],
        score=123.0,
        score_valid=True,
        metadata={"params": {"x": "stale"}},
    )

    candidate = solution_to_candidate(
        solution,
        gene_space=space,
        candidate_id="c-2",
        batch_id="b-2",
        origin="memory_seed",
        event_index=4,
        parents=("c-parent",),
        generation=3,
        metadata={"source": "unit"},
    )

    assert candidate.candidate_id == "c-2"
    assert candidate.batch_id == "b-2"
    assert candidate.genes == [1.0, 5, True]
    assert candidate.params == {"x": 1.0, "period": 5, "enabled": True}
    assert candidate.origin == "memory_seed"
    assert candidate.parents == ("c-parent",)
    assert candidate.event_index == 4
    assert candidate.generation == 3
    assert candidate.metadata == {"source": "unit"}
    assert candidate.scores == {}
    assert candidate.confidence is None
    assert candidate.status == "proposed"
```

- [ ] **Step 2: Add failing lifecycle export assertions**

In `tests/unit/test_domain_imports.py`, add these imports inside
`test_domain_packages_export_symbols_owned_by_focused_modules`:

```python
    from evocore.lifecycle import candidate_to_solution, solution_to_candidate
```

Add these assertions near the other lifecycle module assertions:

```python
    assert candidate_to_solution.__module__ == "evocore.lifecycle.conversion"
    assert solution_to_candidate.__module__ == "evocore.lifecycle.conversion"
```

- [ ] **Step 3: Run conversion tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_conversion.py tests/unit/test_domain_imports.py::test_domain_packages_export_symbols_owned_by_focused_modules -v
```

Expected: FAIL because `evocore.lifecycle.conversion` and the exported helpers do not
exist.

- [ ] **Step 4: Implement conversion helpers**

Create `evocore/lifecycle/conversion.py`:

```python
"""Conversions between lifecycle candidates and result solutions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evocore.lifecycle.records import (
    STATE_UPDATE_CONFIDENCES,
    Candidate,
    CandidateOrigin,
    Direction,
)
from evocore.search_space import GeneSpace, GeneValue, Solution


def _has_state_observation(candidate: Candidate) -> bool:
    return any(
        observation.score is not None and observation.confidence in STATE_UPDATE_CONFIDENCES
        for observation in candidate.scores.values()
    )


def candidate_to_solution(
    candidate: Candidate,
    *,
    direction: Direction,
    gene_space: GeneSpace | None = None,
    include_provenance: bool = True,
) -> Solution:
    """Convert a lifecycle candidate into a population/result solution."""
    score_valid = _has_state_observation(candidate)
    score = candidate.best_state_score(direction) if score_valid else None
    metadata: dict[str, Any] = {}

    if candidate.params is not None:
        metadata["params"] = dict(candidate.params)

    if include_provenance:
        metadata["candidate_id"] = candidate.candidate_id
        if gene_space is not None:
            metadata["candidate_hash"] = gene_space.value_hash(candidate.genes)
        if candidate.batch_id:
            metadata["batch_id"] = candidate.batch_id
        metadata["origin"] = candidate.origin
        if candidate.generation is not None:
            metadata["generation"] = candidate.generation

    return Solution(
        list(candidate.genes),
        score=score,
        score_valid=score_valid,
        metadata=metadata,
    )


def solution_to_candidate(
    solution: Solution,
    *,
    gene_space: GeneSpace,
    candidate_id: str,
    batch_id: str,
    origin: CandidateOrigin,
    event_index: int,
    parents: Sequence[str] = (),
    generation: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Candidate:
    """Convert a population/result solution into a fresh lifecycle candidate."""
    values: list[GeneValue] = list(solution.values)
    gene_space.validate_genes(values)
    return Candidate(
        candidate_id=candidate_id,
        genes=values,
        batch_id=batch_id,
        params=gene_space.params_for(values),
        origin=origin,
        parents=tuple(parents),
        event_index=event_index,
        generation=generation,
        metadata=dict(metadata or {}),
    )


__all__ = ["candidate_to_solution", "solution_to_candidate"]
```

- [ ] **Step 5: Export conversion helpers from lifecycle package**

In `evocore/lifecycle/__init__.py`, add:

```python
from evocore.lifecycle.conversion import candidate_to_solution, solution_to_candidate
```

Add these names to `__all__`:

```python
    "candidate_to_solution",
    "solution_to_candidate",
```

- [ ] **Step 6: Run conversion tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_conversion.py tests/unit/test_domain_imports.py::test_domain_packages_export_symbols_owned_by_focused_modules -v
```

Expected: PASS.

- [ ] **Step 7: Commit conversion helpers**

Run:

```powershell
git add evocore/lifecycle/conversion.py evocore/lifecycle/__init__.py tests/unit/test_lifecycle_conversion.py tests/unit/test_domain_imports.py
git commit -m "feat(lifecycle): add candidate solution conversions"
```

## Task 4: GA And CMA-ES Integration

**Files:**
- Modify: `tests/unit/test_ga_ask_tell_vnext.py`
- Modify: `tests/unit/test_cmaes_ask_tell_vnext.py`
- Modify: `evocore/optimizers/ga/ask_tell.py`
- Modify: `evocore/optimizers/cmaes/ask_tell.py`

- [ ] **Step 1: Update GA ask/tell tests for GeneSpace-backed hashes and result provenance**

In `tests/unit/test_ga_ask_tell_vnext.py`, update
`test_ga_ask_populates_unique_candidate_hash_telemetry`:

```python
def test_ga_ask_populates_unique_candidate_hash_telemetry() -> None:
    space = _space()
    engine = GeneticAlgorithmOptimizer(space, population_size=6, max_generations=5, seed=123)

    candidates = engine.ask(4)

    assert engine.vnext_telemetry.total_candidates_proposed == 4
    assert engine.vnext_telemetry.unique_candidate_hashes == {
        space.value_hash(candidate.genes) for candidate in candidates
    }
```

In `test_ga_ask_records_append_only_ask_events`, replace the hash assertion with:

```python
    assert rows[0]["candidate_hash"] == engine.gene_space.value_hash(candidates[0].genes)
```

Append this result provenance test:

```python
def test_ga_run_final_solutions_use_candidate_conversion_provenance() -> None:
    space = _space()
    engine = GeneticAlgorithmOptimizer(space, population_size=4, max_generations=20, seed=123)

    result = engine.run(
        SphereEvaluator(),
        policy=BudgetPolicy.single_full(max_evaluations=4, batch_size=4),
    )

    assert result.best_solution.metadata["candidate_id"] == result.best_candidate_id
    assert result.best_solution.metadata["candidate_hash"] == space.value_hash(
        result.best_solution.values
    )
    assert result.best_solution.metadata["origin"] == "random"
    assert result.best_solution.metadata["params"] == space.params_for(
        result.best_solution.values
    )
    assert "stage" not in result.best_solution.metadata
    assert all("candidate_hash" in solution.metadata for solution in result.final_solutions)
    assert all("status" not in solution.metadata for solution in result.final_solutions)
```

- [ ] **Step 2: Update CMA ask/tell tests for GeneSpace-backed hashes**

In `tests/unit/test_cmaes_ask_tell_vnext.py`, update
`test_cma_ask_records_append_only_ask_events`:

```python
def test_cma_ask_records_append_only_ask_events() -> None:
    space = _space()
    engine = CMAESOptimizer(space, population_size=4, seed=7)

    candidates = engine.ask()

    assert len(engine.events) == 4
    rows = engine.events.to_rows()
    assert [row["event_index"] for row in rows] == [0, 1, 2, 3]
    assert all(row["event_type"] == "ask" for row in rows)
    assert rows[0]["batch_id"] == candidates[0].batch_id
    assert rows[0]["candidate_id"] == candidates[0].candidate_id
    assert rows[0]["candidate_hash"] == space.value_hash(candidates[0].genes)
    assert engine.vnext_telemetry.unique_candidate_hashes == {
        space.value_hash(candidate.genes) for candidate in candidates
    }
```

- [ ] **Step 3: Run focused engine tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_ask_populates_unique_candidate_hash_telemetry tests/unit/test_ga_ask_tell_vnext.py::test_ga_ask_records_append_only_ask_events tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_final_solutions_use_candidate_conversion_provenance tests/unit/test_cmaes_ask_tell_vnext.py::test_cma_ask_records_append_only_ask_events -v
```

Expected: FAIL because engines still use legacy candidate hashes and GA result
construction still hand-builds `Solution` metadata.

- [ ] **Step 4: Update GA ask/tell integration**

In `evocore/optimizers/ga/ask_tell.py`, update the lifecycle import block to include the
conversion helpers:

```python
from evocore.lifecycle import (
    BudgetPolicy,
    BudgetScheduler,
    Candidate,
    CandidateBatch,
    EvaluationContext,
    EvaluationRecord,
    Evaluator,
    UpdateResult,
    batch_id_from_seed,
    candidate_to_solution,
    is_state_update_confidence,
    score_for_direction,
    solution_to_candidate,
)
```

Replace `_candidate_from_genes(...)` with:

```python
    def _candidate_from_genes(
        self,
        genes: list[float | int | bool],
        *,
        batch_id: str,
        origin: str,
        event_index: int,
        candidate_index: int,
        parents: Sequence[str] = (),
    ) -> Candidate:
        candidate_id = _core.candidate_id(self.seed, event_index, candidate_index)
        return solution_to_candidate(
            Solution(genes),
            gene_space=self.gene_space,
            candidate_id=candidate_id,
            batch_id=batch_id,
            origin=origin,
            event_index=event_index,
            parents=parents,
        )
```

In `ask(...)`, update telemetry recording:

```python
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
```

In `_append_ask_events(...)`, update:

```python
                    candidate_hash=candidate.candidate_hash(self.gene_space),
```

In `_append_tell_event(...)`, update:

```python
                candidate_hash=candidate.candidate_hash(self.gene_space),
```

In `run(...)`, replace the best solution construction with:

```python
        best = candidate_to_solution(
            self.best_candidate,
            direction=self.direction,
            gene_space=self.gene_space,
        )
```

Replace the `final_solutions = SolutionSet([...])` block with:

```python
        final_solutions = SolutionSet(
            [
                candidate_to_solution(
                    candidate,
                    direction=self.direction,
                    gene_space=self.gene_space,
                )
                for candidate in final_candidates
            ]
        )
```

Keep the existing trusted-population conversion for reproduction as a local block because
it intentionally uses `state_comparison_score(...)` as the selection score.

- [ ] **Step 5: Update CMA ask/tell integration**

In `evocore/optimizers/cmaes/ask_tell.py`, update telemetry recording in `ask(...)`:

```python
        self.vnext_telemetry.record_proposed_candidates(candidates, gene_space=self.gene_space)
```

In `_append_ask_events(...)`, update:

```python
                    candidate_hash=candidate.candidate_hash(self.gene_space),
```

In `_append_tell_event(...)`, update:

```python
                candidate_hash=candidate.candidate_hash(self.gene_space),
```

- [ ] **Step 6: Run focused engine tests and verify pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py::test_ga_ask_populates_unique_candidate_hash_telemetry tests/unit/test_ga_ask_tell_vnext.py::test_ga_ask_records_append_only_ask_events tests/unit/test_ga_ask_tell_vnext.py::test_ga_run_final_solutions_use_candidate_conversion_provenance tests/unit/test_cmaes_ask_tell_vnext.py::test_cma_ask_records_append_only_ask_events -v
```

Expected: PASS.

- [ ] **Step 7: Run broader ask/tell tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit engine integration**

Run:

```powershell
git add evocore/optimizers/ga/ask_tell.py evocore/optimizers/cmaes/ask_tell.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py
git commit -m "feat(optimizers): use schema-aware candidate hashes"
```

## Task 5: Docs And Changelog

**Files:**
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/api.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update ask/tell docs**

In `docs/site/ask-tell-engines.md`, add this section after the introductory paragraph
that explains `ask()` and `tell()`:

```markdown
## Candidate And Solution Boundary

`Candidate` is the lifecycle-facing record returned by `ask()`. It carries lifecycle
identity and scheduling state such as `candidate_id`, `batch_id`, `origin`, `parents`,
`stage`, `status`, decoded `genes`, optional `params`, and evaluation observations.
Evaluators receive candidates because they need proposal identity as well as decoded
values.

`Solution` is the population/result-facing record exposed by completed optimizer runs. It
stores decoded `values`, `score`, `score_valid`, and result metadata. Result metadata may
include provenance such as `candidate_id`, `candidate_hash`, `batch_id`, `origin`, and
`generation`, but scheduler state and observation history stay on lifecycle records.

Use `candidate_id` to refer to one lifecycle proposal. Use
`candidate.candidate_hash(gene_space)` or `gene_space.value_hash(candidate.genes)` to
compare search-space values. The hash is schema-aware and includes the `GeneSpace` hash,
so identical raw values in incompatible spaces do not collapse into the same search
point.
```

In the existing Event History section, change:

```markdown
an `ask` event with its batch ID, candidate ID, genome hash, origin, genes, params, and
```

to:

```markdown
an `ask` event with its batch ID, candidate ID, schema-aware candidate hash, origin,
genes, params, and
```

- [ ] **Step 2: Update API docs if lifecycle module reference is missing**

Open `docs/site/api.md`. If it already contains `::: evocore.lifecycle`, no change is
needed for API discovery. If it does not, add this block near the search-space and result
API references:

```markdown
::: evocore.lifecycle
```

- [ ] **Step 3: Update changelog**

In `CHANGELOG.md`, under `## [Unreleased]` / `### Added`, add:

```markdown
- Schema-aware `GeneSpace.value_signature(...)` and `GeneSpace.value_hash(...)`
  helpers for stable search-point identity.
- Lifecycle conversion helpers for explicit `Candidate` to `Solution` and `Solution` to
  `Candidate` transitions.
```

Under `## [Unreleased]` / `### Changed`, add:

```markdown
- Ask/tell event history and telemetry now use GeneSpace-backed candidate hashes in
  optimizer internals while preserving the zero-argument `Candidate.candidate_hash()`
  compatibility fallback.
```

- [ ] **Step 4: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS.

- [ ] **Step 5: Commit docs**

Run:

```powershell
git add docs/site/ask-tell-engines.md docs/site/api.md CHANGELOG.md
git commit -m "docs: clarify candidate solution boundary"
```

## Task 6: Focused Regression And Formatting

**Files:**
- Modify only files with failing imports, formatting, or lint issues from this task.

- [ ] **Step 1: Run Python formatting check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
```

Expected: PASS. If it fails, run:

```powershell
.\.venv\Scripts\python.exe -m ruff format
.\.venv\Scripts\python.exe -m ruff format --check
```

Then include formatted files in the verification fix commit.

- [ ] **Step 2: Run Python lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check
```

Expected: PASS. Fix only issues introduced by this plan.

- [ ] **Step 3: Run focused unit and property tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_gene_space.py tests/property/test_gene_space_properties.py tests/unit/test_vnext_evaluation.py tests/unit/test_lifecycle_conversion.py tests/unit/test_domain_imports.py tests/unit/test_ga_ask_tell_vnext.py tests/unit/test_cmaes_ask_tell_vnext.py -v
```

Expected: PASS.

- [ ] **Step 4: Run optimizer regression tests touched by conversion and hash changes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ga_engine.py tests/unit/test_cmaes_engine.py tests/integration/test_mixed_gene_space.py tests/integration/test_sphere_function.py tests/integration/test_cmaes_rosenbrock.py -v
```

Expected: PASS.

- [ ] **Step 5: Run docs build**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build
```

Expected: PASS.

- [ ] **Step 6: Commit verification fixes if needed**

If formatting, lint, or regression fixes changed files, run:

```powershell
git add evocore tests docs/site CHANGELOG.md
git commit -m "fix: complete candidate solution boundary stabilization"
```

If no files changed, do not create an empty commit.

## Final Verification

Before reporting completion, run the repository-relevant verification set:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
.\.venv\Scripts\python.exe -m maturin develop --release
.\.venv\Scripts\python.exe -m pytest tests/unit/ tests/integration/ -v
.\.venv\Scripts\python.exe -m pytest tests/property/ -v
.\.venv\Scripts\python.exe -m mkdocs build
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

If `.venv` is missing, broken, or points to an unavailable interpreter, stop and report
that before using another Python.

If any command fails, stop and report the failing command, the relevant error summary,
and the likely files involved. Do not push or open a PR after failed verification.

If all commands pass, push the task branch and open or update the draft PR according to
the repository workflow.
