# Phase 1 Docs Compatibility and Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the Phase 1 external-state API and verify cross-optimizer compatibility for expensive external optimization workflows.

**Architecture:** Add user-facing docs and examples after the shared API and all three optimizer implementations land. Use one cross-optimizer contract test module to make the public API shape consistent while preserving optimizer-specific capability flags.

**Tech Stack:** MkDocs Markdown, pytest, ruff, EvoCore public API examples.

---

## Dependency

Complete these plans first:

- `docs/superpowers/plans/2026-06-17-phase-1-shared-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-ga-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-de-external-state-api.md`
- `docs/superpowers/plans/2026-06-17-phase-1-cmaes-external-state-api.md`

## File Structure

- Create: `tests/unit/test_external_state_optimizer_contract.py`
  - Cross-optimizer public API tests.
- Modify: `docs/site/ask-tell-engines.md`
  - Add warm-start, injection, snapshot, and cached-record usage notes.
- Modify: `docs/site/budget-aware-optimization.md`
  - Clarify cached records and budget accounting for external systems.
- Modify: `docs/site/examples.md`
  - Extend expensive external queue example or link to the new recipe.
- Create: `docs/site/expensive-external-evaluations.md`
  - Recipe page covering cached backtests, async workers, warm starts, top-k survivor selection, and deterministic checkpoints.
- Modify: `docs/site/api.md`
  - Add API reference entries for the new lifecycle external-state types and helpers.
- Modify: `mkdocs.yml`
  - Add the new recipe page to navigation.
- Modify: `CHANGELOG.md`
  - Add an unreleased entry for the additive public API.

## Task 1: Add Cross-Optimizer Contract Tests

**Files:**
- Create: `tests/unit/test_external_state_optimizer_contract.py`

- [ ] **Step 1: Add fixtures and optimizer factory cases**

```python
import pytest

from evocore import (
    CMAESOptimizer,
    DifferentialEvolutionOptimizer,
    ExternalStateOptimizer,
    Gene,
    GeneSpace,
    GeneticAlgorithmOptimizer,
    WarmStartRecord,
)


def _space() -> GeneSpace:
    return GeneSpace([Gene("x", "float", -5.0, 5.0), Gene("y", "float", -5.0, 5.0)])


def _warm_records() -> list[WarmStartRecord]:
    return [
        WarmStartRecord(params={"x": 1.0, "y": 1.0}, score=10.0, metadata={"source": "elite_a"}),
        WarmStartRecord(params={"x": 2.0, "y": 2.0}, score=20.0, metadata={"source": "elite_b"}),
    ]


@pytest.fixture(params=["ga", "de", "cmaes"])
def optimizer(request):
    space = _space()
    if request.param == "ga":
        return GeneticAlgorithmOptimizer(space, population_size=4, seed=11)
    if request.param == "de":
        return DifferentialEvolutionOptimizer(space, population_size=4, seed=11)
    return CMAESOptimizer(space, population_size=4, seed=11)
```

- [ ] **Step 2: Test shared public API shape**

```python
def test_all_phase_1_optimizers_support_external_state_protocol(optimizer) -> None:
    assert isinstance(optimizer, ExternalStateOptimizer)

    capabilities = optimizer.external_state_capabilities()
    assert capabilities.population_snapshots is True
    assert capabilities.top_candidate_snapshots is True
    assert capabilities.cached_record_helpers is True
```

- [ ] **Step 3: Test warm-start snapshots preserve metadata and top-k**

```python
def test_warm_start_top_candidates_preserve_metadata(optimizer) -> None:
    result = optimizer.warm_start(_warm_records())

    assert result.accepted_count == 2
    top = optimizer.top_candidates(1)

    assert len(top) == 1
    assert top[0].score == 20.0
    assert top[0].metadata["record_metadata"]["source"] == "elite_b"
```

- [ ] **Step 4: Test tracked mode leaves trusted scope empty on fresh optimizers**

```python
def test_tracked_warm_start_is_visible_as_scored_not_trusted(optimizer) -> None:
    optimizer.warm_start(_warm_records(), mode="tracked")

    assert optimizer.candidate_snapshot(scope="trusted").candidates == ()
    assert len(optimizer.candidate_snapshot(scope="scored").candidates) == 2
```

- [ ] **Step 5: Test duplicate handling consistency**

```python
def test_duplicate_warm_start_values_are_skipped_consistently(optimizer) -> None:
    records = _warm_records()
    result = optimizer.warm_start([records[0], records[0]])

    assert result.accepted_count == 1
```

- [ ] **Step 6: Run tests and verify expected pass**

Run after optimizer implementation plans are complete:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_optimizer_contract.py -v
```

Expected: all cross-optimizer contract tests pass.

## Task 2: Update API Reference

**Files:**
- Modify: `docs/site/api.md`

- [ ] **Step 1: Add lifecycle external-state section**

Add this section near the lifecycle API content:

```markdown
## External State Integration

::: evocore.lifecycle.external.WarmStartRecord

::: evocore.lifecycle.external.CandidateSnapshot

::: evocore.lifecycle.external.PopulationSnapshot

::: evocore.lifecycle.external.ExternalStateCapabilities

::: evocore.lifecycle.external.InjectionResult

::: evocore.lifecycle.external.cached_records
```

- [ ] **Step 2: Run docs API smoke check**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check docs/site/api.md
```

Expected: command exits successfully or reports that Markdown is not handled by ruff. If ruff does not check Markdown, continue to the MkDocs verification in Task 6.

## Task 3: Update Ask/Tell and Budget Docs

**Files:**
- Modify: `docs/site/ask-tell-engines.md`
- Modify: `docs/site/budget-aware-optimization.md`

- [ ] **Step 1: Add ask/tell external-state section**

Add this section to `docs/site/ask-tell-engines.md` after the confidence semantics section:

````markdown
## External State Integration

Ask/tell optimizers can be initialized from trusted external knowledge without reaching into private state.

```python
from evocore import GeneticAlgorithmOptimizer, GeneSpace, WarmStartRecord

optimizer = GeneticAlgorithmOptimizer(GeneSpace.uniform(-5.0, 5.0, 3), population_size=8, seed=42)
optimizer.warm_start(
    [
        WarmStartRecord(values=(1.0, 0.0, -1.0), score=0.82, confidence="cached"),
        WarmStartRecord(values=(0.5, 0.5, -0.5), score=0.79, confidence="trusted_full"),
    ]
)

survivors = optimizer.top_candidates(4)
```

Use `candidate_snapshot(scope="trusted")` for promotion decisions and reports. Use `scope="scored"` when you need scored tracked records that are not part of optimizer state.
````

- [ ] **Step 2: Add cached record budget note**

Add this paragraph to `docs/site/budget-aware-optimization.md` near the cached confidence discussion:

```markdown
`cached_records(...)` converts cache hits for current candidates into `EvaluationRecord(confidence="cached")` objects. The helper does not call `tell(...)` and does not spend fresh evaluation budget by itself. Budget accounting is updated only when those records are passed to the optimizer through `tell(...)` or through `warm_start(...)`.
```

## Task 4: Add Expensive External Evaluation Recipe

**Files:**
- Create: `docs/site/expensive-external-evaluations.md`
- Modify: `docs/site/examples.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Create recipe page**

Create `docs/site/expensive-external-evaluations.md` with this structure:

````markdown
# Expensive External Evaluations

Use this pattern when objective evaluations happen outside EvoCore: backtests, simulations, cloud jobs, lab measurements, or asynchronous worker queues.

## Warm Start From Search Memory

```python
from evocore import GeneticAlgorithmOptimizer, GeneSpace, WarmStartRecord

space = GeneSpace.uniform(-5.0, 5.0, 3)
optimizer = GeneticAlgorithmOptimizer(space, population_size=16, seed=42)

optimizer.warm_start(
    [
        WarmStartRecord(values=(1.0, 0.0, -1.0), score=0.82, metadata={"source": "archive"}),
        WarmStartRecord(values=(0.5, 0.5, -0.5), score=0.79, metadata={"source": "search_memory"}),
    ]
)
```

## Cached Backtest Results

```python
from evocore import cached_records

candidates = optimizer.ask(16)
records = cached_records(
    candidates,
    gene_space=space,
    cache={
        space.value_hash(candidate.genes): {"score": 0.75, "metadata": {"cache_reason": "exact"}}
        for candidate in candidates[:2]
    },
    stage="cached_backtest",
)

if records:
    optimizer.tell(records)
```

## Promote Top Candidates

```python
survivors = optimizer.top_candidates(8)
for survivor in survivors:
    print(survivor.candidate_id, survivor.score, survivor.metadata)
```

## Checkpoint External Work

Save an ask/tell checkpoint after submitting work and after receiving partial results. Checkpoints include candidates, pending batches, telemetry, events, and external metadata stored on candidates and records.
````

- [ ] **Step 2: Link from examples page**

Add a short link near the external queue example in `docs/site/examples.md`:

```markdown
For a fuller warm-start, cached-evaluation, and survivor-selection workflow, see [Expensive External Evaluations](expensive-external-evaluations.md).
```

- [ ] **Step 3: Add MkDocs nav entry**

Add `Expensive External Evaluations: expensive-external-evaluations.md` to the examples or guides section in `mkdocs.yml`.

## Task 5: Update Changelog

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add public API entry**

Add an entry under the current unreleased section:

```markdown
- Added external-state integration APIs for GA, DE, and CMA-ES, including warm starts, candidate injection, read-only population snapshots, top-k candidate access, and cached evaluation record helpers.
```

If `CHANGELOG.md` has no unreleased section, add a new top section named `## Unreleased`.

## Task 6: Full Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run cross-optimizer and focused regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_external_state_core.py tests/unit/test_external_state_optimizer_contract.py tests/unit/test_ga_external_state.py tests/unit/test_de_external_state.py tests/unit/test_cmaes_external_state.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run existing checkpoint regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_ask_tell_checkpointing.py tests/unit/test_de_checkpointing.py tests/unit/test_cmaes_ask_tell_checkpointing.py -v
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

- [ ] **Step 5: Commit docs and compatibility work**

Run:

```powershell
git status --short
git add tests/unit/test_external_state_optimizer_contract.py docs/site/ask-tell-engines.md docs/site/budget-aware-optimization.md docs/site/examples.md docs/site/expensive-external-evaluations.md docs/site/api.md mkdocs.yml CHANGELOG.md
git commit -m "docs: add external state integration recipes"
```

Expected: commit succeeds and contains only docs, changelog, and cross-optimizer compatibility tests.

## Compatibility Notes

- API docs must describe optimizer-specific capability differences so users do not assume CMA-ES supports proposed candidate injection or DE supports arbitrary mid-run target replacement.
- The docs should avoid Trading-Algo-specific terms except as generic examples like cached backtests or external search memory.
- Changelog entry is required because this phase adds public API.

## Self-Review Notes

- Spec coverage: integration recipes, cached workflow, async external evaluation patterns, top-k survivor selection, and deterministic checkpoint guidance are covered.
- Scope boundary: Phase 2 archive/diversity utilities and Phase 3 hybrid composition helpers are referenced only as future capabilities, not documented as available APIs.
