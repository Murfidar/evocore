# Phase 2B Stop Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable public stop and stall policies for ask/tell loops and expensive external evaluations.

**Architecture:** Implement stop policies as lifecycle utilities that consume `UpdateResult`, `PopulationSnapshot`, or `OptimizationTelemetry` and return explicit `StopDecision` objects. Keep stop policies separate from `BudgetPolicy`; examples may compose them in user ask/tell loops.

**Tech Stack:** Python dataclasses, stateful policy objects, EvoCore lifecycle telemetry, pytest, ruff, MkDocs Markdown.

---

## Dependency

Complete Phase 1 first. Phase 2A is recommended before this plan because docs may refer to archive-backed workflows, but the stop-policy code does not depend on archive modules.

Source design:

- `docs/superpowers/specs/2026-06-17-evocore-phase-2-expensive-optimization-toolkit-design.md`

## File Structure

- Create: `evocore/lifecycle/stopping.py`
  - `StopDecision`.
  - `StopPolicy` protocol.
  - `EvaluationLimitPolicy`.
  - `NoImprovementPolicy`.
  - `ConvergencePolicy`.
  - `CompositeStopPolicy`.
- Modify: `evocore/lifecycle/__init__.py`
  - Re-export stopping names.
- Modify: `evocore/__init__.py`
  - Re-export common stopping names.
- Create: `tests/unit/test_lifecycle_stopping.py`
  - Unit tests for each policy.
- Create: `tests/unit/test_phase2b_stop_policy_integration.py`
  - Ask/tell usage smoke test against one optimizer.
- Modify: `tests/unit/test_package_init.py`
  - Top-level export smoke test.
- Modify: `docs/site/api.md`
  - API reference entries.
- Modify: `docs/site/expensive-external-evaluations.md`
  - Ask/tell stop-policy recipe.
- Modify: `CHANGELOG.md`
  - Public API entry.

## Public API Names

Export these names from `evocore.lifecycle` and top-level `evocore`:

- `CompositeStopPolicy`
- `ConvergencePolicy`
- `EvaluationLimitPolicy`
- `NoImprovementPolicy`
- `StopDecision`
- `StopPolicy`

## Task 1: Write Stop Policy Unit Tests

**Files:**
- Create: `tests/unit/test_lifecycle_stopping.py`

- [ ] **Step 1: Add fixtures and evaluation-limit tests**

Create `tests/unit/test_lifecycle_stopping.py`:

```python
from evocore import (
    CompositeStopPolicy,
    ConvergencePolicy,
    EvaluationLimitPolicy,
    NoImprovementPolicy,
    OptimizationTelemetry,
    UpdateResult,
)


def _update(
    *,
    best_score: float | None = None,
    trusted: int = 0,
    cached: int = 0,
    partial: int = 0,
) -> UpdateResult:
    return UpdateResult(
        accepted_count=trusted + cached + partial,
        trusted_count=trusted,
        partial_count=partial,
        surrogate_count=0,
        cached_count=cached,
        rejected_count=0,
        best_candidate_id="best" if best_score is not None else None,
        best_score=best_score,
    )


def test_evaluation_limit_policy_uses_update_counts() -> None:
    policy = EvaluationLimitPolicy(max_evaluations=3)

    first = policy.observe(_update(trusted=1, cached=1))
    second = policy.observe(_update(trusted=1))

    assert first.stop is False
    assert first.metadata["observed_evaluations"] == 2
    assert second.stop is True
    assert second.reason == "evaluation_limit"
    assert second.metadata["observed_evaluations"] == 3


def test_evaluation_limit_policy_can_ignore_cached_records() -> None:
    policy = EvaluationLimitPolicy(max_evaluations=2, include_cached=False)

    decision = policy.observe(_update(trusted=1, cached=10))

    assert decision.stop is False
    assert decision.metadata["observed_evaluations"] == 1
```

- [ ] **Step 2: Add telemetry-based evaluation-limit tests**

Append:

```python
def test_evaluation_limit_policy_prefers_explicit_telemetry_snapshot() -> None:
    telemetry = OptimizationTelemetry()
    telemetry.record_full(2, stage="full", cost=2.0)
    telemetry.record_cached(3, stage="cache", cost=0.0)
    policy = EvaluationLimitPolicy(max_evaluations=5)

    decision = policy.observe(telemetry=telemetry)

    assert decision.stop is True
    assert decision.metadata["observed_evaluations"] == 5
```

- [ ] **Step 3: Add no-improvement policy tests**

Append:

```python
def test_no_improvement_policy_stops_after_window_without_improvement() -> None:
    policy = NoImprovementPolicy(window=2, min_delta=0.5, score_direction="maximize")

    assert policy.observe(_update(best_score=10.0)).stop is False
    assert policy.observe(_update(best_score=10.2)).stop is False
    decision = policy.observe(_update(best_score=10.3))

    assert decision.stop is True
    assert decision.reason == "no_improvement"
    assert decision.metadata["best_score"] == 10.0


def test_no_improvement_policy_resets_on_improvement() -> None:
    policy = NoImprovementPolicy(window=2, min_delta=0.5, score_direction="maximize")

    policy.observe(_update(best_score=10.0))
    policy.observe(_update(best_score=10.2))
    decision = policy.observe(_update(best_score=10.8))

    assert decision.stop is False
    assert decision.metadata["stale_count"] == 0
```

- [ ] **Step 4: Add convergence and composite policy tests**

Append:

```python
def test_convergence_policy_stops_when_target_reached_for_maximize() -> None:
    policy = ConvergencePolicy(target_score=5.0, score_direction="maximize")

    assert policy.observe(_update(best_score=4.9)).stop is False
    decision = policy.observe(_update(best_score=5.0))

    assert decision.stop is True
    assert decision.reason == "convergence"


def test_convergence_policy_stops_when_target_reached_for_minimize() -> None:
    policy = ConvergencePolicy(target_score=1.0, score_direction="minimize")

    assert policy.observe(_update(best_score=1.1)).stop is False
    assert policy.observe(_update(best_score=1.0)).stop is True


def test_composite_stop_policy_returns_first_stop_decision() -> None:
    policy = CompositeStopPolicy(
        [
            EvaluationLimitPolicy(max_evaluations=10),
            ConvergencePolicy(target_score=5.0, score_direction="maximize"),
        ]
    )

    decision = policy.observe(_update(best_score=5.0, trusted=1))

    assert decision.stop is True
    assert decision.reason == "convergence"
```

- [ ] **Step 5: Run tests and verify expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_stopping.py -v
```

Expected: fails with import errors for stop-policy names.

## Task 2: Implement Stop Policies

**Files:**
- Create: `evocore/lifecycle/stopping.py`

- [ ] **Step 1: Add common dataclasses and helper functions**

Create `evocore/lifecycle/stopping.py`:

```python
"""Stop and stall policies for external ask/tell workflows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from evocore.core.errors import ConfigurationError
from evocore.core.serialization import json_safe
from evocore.lifecycle.external import PopulationSnapshot
from evocore.lifecycle.records import Direction, score_for_direction
from evocore.lifecycle.telemetry import OptimizationTelemetry, UpdateResult


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str | None = None
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop and not self.reason:
            raise ConfigurationError("StopDecision reason is required when stop=True.")
        payload = json_safe(dict(self.metadata))
        if not isinstance(payload, dict):
            raise ConfigurationError("StopDecision metadata must be JSON-safe.")
        object.__setattr__(self, "metadata", payload)


class StopPolicy(Protocol):
    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        ...

    def reset(self) -> None:
        ...


def _ok(metadata: dict[str, object] | None = None) -> StopDecision:
    return StopDecision(stop=False, metadata=metadata or {})
```

- [ ] **Step 2: Add evaluation-limit policy**

Append:

```python
class EvaluationLimitPolicy:
    """Stop after a configured number of counted evaluations."""

    def __init__(
        self,
        *,
        max_evaluations: int,
        include_cached: bool = True,
        include_partial: bool = False,
        include_surrogate: bool = False,
    ) -> None:
        if int(max_evaluations) <= 0:
            raise ConfigurationError("max_evaluations must be positive.")
        self.max_evaluations = int(max_evaluations)
        self.include_cached = bool(include_cached)
        self.include_partial = bool(include_partial)
        self.include_surrogate = bool(include_surrogate)
        self._observed_evaluations = 0

    def reset(self) -> None:
        self._observed_evaluations = 0

    def _count_from_telemetry(self, telemetry: OptimizationTelemetry) -> int:
        count = int(telemetry.candidates_full_evaluated)
        if self.include_cached:
            count += int(telemetry.candidates_cached)
        if self.include_partial:
            count += int(telemetry.candidates_partial_evaluated)
        if self.include_surrogate:
            count += int(telemetry.candidates_screened)
        return count

    def _count_from_update(self, update: UpdateResult) -> int:
        count = int(update.trusted_count)
        if self.include_cached:
            count += int(update.cached_count)
        if self.include_partial:
            count += int(update.partial_count)
        if self.include_surrogate:
            count += int(update.surrogate_count)
        return count

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        source = telemetry or (snapshot.telemetry if snapshot is not None else None)
        if source is not None:
            observed = self._count_from_telemetry(source)
            self._observed_evaluations = observed
        elif update is not None:
            self._observed_evaluations += self._count_from_update(update)
        observed = self._observed_evaluations
        metadata = {
            "observed_evaluations": observed,
            "max_evaluations": self.max_evaluations,
        }
        if observed >= self.max_evaluations:
            return StopDecision(
                stop=True,
                reason="evaluation_limit",
                message="Evaluation limit reached.",
                metadata=metadata,
            )
        return _ok(metadata)
```

- [ ] **Step 3: Add no-improvement and convergence policies**

Append:

```python
class NoImprovementPolicy:
    """Stop after a fixed window without meaningful best-score improvement."""

    def __init__(
        self,
        *,
        window: int,
        min_delta: float = 0.0,
        score_direction: Direction = "maximize",
    ) -> None:
        if int(window) <= 0:
            raise ConfigurationError("window must be positive.")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0.0:
            raise ConfigurationError("min_delta must be finite and >= 0.")
        if score_direction not in ("maximize", "minimize"):
            raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
        self.window = int(window)
        self.min_delta = float(min_delta)
        self.score_direction = score_direction
        self._best_comparison_score: float | None = None
        self._best_raw_score: float | None = None
        self._stale_count = 0

    def reset(self) -> None:
        self._best_comparison_score = None
        self._best_raw_score = None
        self._stale_count = 0

    def _best_score(self, update: UpdateResult | None, snapshot: PopulationSnapshot | None) -> float | None:
        if update is not None and update.best_score is not None:
            return float(update.best_score)
        if snapshot is not None:
            scored = [candidate.score for candidate in snapshot.candidates if candidate.score is not None]
            if scored:
                return max(scored) if snapshot.direction == "maximize" else min(scored)
        return None

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        raw_score = self._best_score(update, snapshot)
        if raw_score is None:
            return _ok({"stale_count": self._stale_count, "best_score": self._best_raw_score})
        comparison = score_for_direction(raw_score, self.score_direction)
        if self._best_comparison_score is None or comparison > self._best_comparison_score + self.min_delta:
            self._best_comparison_score = comparison
            self._best_raw_score = raw_score
            self._stale_count = 0
        else:
            self._stale_count += 1
        metadata = {"stale_count": self._stale_count, "best_score": self._best_raw_score}
        if self._stale_count >= self.window:
            return StopDecision(
                stop=True,
                reason="no_improvement",
                message="No improvement window reached.",
                metadata=metadata,
            )
        return _ok(metadata)


class ConvergencePolicy:
    """Stop when the best score reaches a target threshold."""

    def __init__(
        self,
        *,
        target_score: float,
        score_direction: Direction = "maximize",
        tolerance: float = 0.0,
    ) -> None:
        if not math.isfinite(float(target_score)):
            raise ConfigurationError("target_score must be finite.")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ConfigurationError("tolerance must be finite and >= 0.")
        if score_direction not in ("maximize", "minimize"):
            raise ConfigurationError("score_direction must be 'maximize' or 'minimize'.")
        self.target_score = float(target_score)
        self.score_direction = score_direction
        self.tolerance = float(tolerance)

    def reset(self) -> None:
        return None

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        best_score = update.best_score if update is not None else None
        if best_score is None and snapshot is not None:
            scored = [candidate.score for candidate in snapshot.candidates if candidate.score is not None]
            if scored:
                best_score = max(scored) if snapshot.direction == "maximize" else min(scored)
        if best_score is None:
            return _ok()
        best = float(best_score)
        target = self.target_score
        reached = best >= target - self.tolerance if self.score_direction == "maximize" else best <= target + self.tolerance
        metadata = {"best_score": best, "target_score": target, "tolerance": self.tolerance}
        if reached:
            return StopDecision(
                stop=True,
                reason="convergence",
                message="Convergence target reached.",
                metadata=metadata,
            )
        return _ok(metadata)
```

- [ ] **Step 4: Add composite policy and exports**

Append:

```python
class CompositeStopPolicy:
    """Evaluate stop policies in order and return the first stop decision."""

    def __init__(self, policies: Sequence[StopPolicy]) -> None:
        self.policies = tuple(policies)
        if not self.policies:
            raise ConfigurationError("CompositeStopPolicy requires at least one policy.")

    def reset(self) -> None:
        for policy in self.policies:
            policy.reset()

    def observe(
        self,
        update: UpdateResult | None = None,
        *,
        snapshot: PopulationSnapshot | None = None,
        telemetry: OptimizationTelemetry | None = None,
    ) -> StopDecision:
        last_metadata: dict[str, object] = {}
        for policy in self.policies:
            decision = policy.observe(update, snapshot=snapshot, telemetry=telemetry)
            if decision.stop:
                return decision
            last_metadata[policy.__class__.__name__] = decision.metadata
        return _ok(last_metadata)


__all__ = [
    "CompositeStopPolicy",
    "ConvergencePolicy",
    "EvaluationLimitPolicy",
    "NoImprovementPolicy",
    "StopDecision",
    "StopPolicy",
]
```

- [ ] **Step 5: Run stop policy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_stopping.py -v
```

Expected: all stop-policy tests pass.

## Task 3: Add Integration Test and Exports

**Files:**
- Modify: `evocore/lifecycle/__init__.py`
- Modify: `evocore/__init__.py`
- Modify: `tests/unit/test_package_init.py`
- Create: `tests/unit/test_phase2b_stop_policy_integration.py`

- [ ] **Step 1: Re-export stopping names**

In `evocore/lifecycle/__init__.py` and `evocore/__init__.py`, import and add:

```python
CompositeStopPolicy
ConvergencePolicy
EvaluationLimitPolicy
NoImprovementPolicy
StopDecision
StopPolicy
```

- [ ] **Step 2: Add top-level export test**

Append to `tests/unit/test_package_init.py`:

```python
def test_phase2b_stopping_public_exports():
    from evocore import (
        CompositeStopPolicy,
        ConvergencePolicy,
        EvaluationLimitPolicy,
        NoImprovementPolicy,
        StopDecision,
        StopPolicy,
    )

    assert CompositeStopPolicy is not None
    assert ConvergencePolicy is not None
    assert EvaluationLimitPolicy is not None
    assert NoImprovementPolicy is not None
    assert StopDecision is not None
    assert StopPolicy is not None
```

- [ ] **Step 3: Add ask/tell integration smoke test**

Create `tests/unit/test_phase2b_stop_policy_integration.py`:

```python
from evocore import EvaluationLimitPolicy, EvaluationRecord, GeneSpace, GeneticAlgorithmOptimizer


def test_stop_policy_can_drive_manual_ask_tell_loop() -> None:
    optimizer = GeneticAlgorithmOptimizer(
        GeneSpace.uniform(-5.0, 5.0, 2),
        population_size=4,
        seed=42,
    )
    stop_policy = EvaluationLimitPolicy(max_evaluations=4)
    candidates = optimizer.ask(4)
    records = [
        EvaluationRecord(
            candidate_id=candidate.candidate_id,
            batch_id=candidate.batch_id,
            score=float(index),
            confidence="trusted_full",
            stage="full",
        )
        for index, candidate in enumerate(candidates)
    ]

    update = optimizer.tell(records)
    decision = stop_policy.observe(update, snapshot=optimizer.candidate_snapshot(scope="trusted"))

    assert decision.stop is True
    assert decision.reason == "evaluation_limit"
```

- [ ] **Step 4: Run integration and export tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_stopping.py tests/unit/test_phase2b_stop_policy_integration.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

## Task 4: Add Docs and Changelog

**Files:**
- Modify: `docs/site/api.md`
- Modify: `docs/site/expensive-external-evaluations.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add API docs**

Add to `docs/site/api.md` near lifecycle API entries:

```markdown
::: evocore.lifecycle.StopDecision

::: evocore.lifecycle.StopPolicy

::: evocore.lifecycle.EvaluationLimitPolicy

::: evocore.lifecycle.NoImprovementPolicy

::: evocore.lifecycle.ConvergencePolicy

::: evocore.lifecycle.CompositeStopPolicy
```

- [ ] **Step 2: Add ask/tell stop-policy recipe**

Add this section to `docs/site/expensive-external-evaluations.md` before "Checkpoint Around External Work":

````markdown
## Stop Long-Running Ask/Tell Loops

Stop policies are reusable helpers for external loops. They do not spend budget and do not mutate optimizer state.

```python
from evocore import CompositeStopPolicy, EvaluationLimitPolicy, NoImprovementPolicy

stop_policy = CompositeStopPolicy(
    [
        EvaluationLimitPolicy(max_evaluations=500),
        NoImprovementPolicy(window=8, min_delta=0.001, score_direction="maximize"),
    ]
)

while True:
    candidates = optimizer.ask(16)
    records = expensive_evaluator(candidates)
    update = optimizer.tell(records)
    decision = stop_policy.observe(
        update,
        snapshot=optimizer.candidate_snapshot(scope="trusted"),
    )
    if decision.stop:
        break
```
````

- [ ] **Step 3: Add changelog entry**

Under `CHANGELOG.md` `## [Unreleased]` `### Added`, add:

```markdown
- Added lifecycle stop policies for external ask/tell workflows, including evaluation caps, no-improvement windows, convergence thresholds, and composite stop decisions.
```

## Task 5: Verification and Commit

**Files:**
- All files changed in this plan.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_lifecycle_stopping.py tests/unit/test_phase2b_stop_policy_integration.py tests/unit/test_package_init.py -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Run formatting and linting**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
```

Expected: both commands pass.

- [ ] **Step 3: Build docs**

Run:

```powershell
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

Expected: MkDocs builds successfully with no strict-mode warnings.

- [ ] **Step 4: Commit Phase 2B**

Run:

```powershell
git status --short
git add evocore/lifecycle/stopping.py evocore/lifecycle/__init__.py evocore/__init__.py tests/unit/test_lifecycle_stopping.py tests/unit/test_phase2b_stop_policy_integration.py tests/unit/test_package_init.py docs/site/api.md docs/site/expensive-external-evaluations.md CHANGELOG.md
git commit -m "feat(lifecycle): add external stop policies"
```

Expected: commit succeeds and contains only Phase 2B stopping, docs, tests, and changelog changes.

## Compatibility Notes

- This plan is additive public API.
- `BudgetPolicy` remains unchanged.
- Stop policies are user-owned objects and are not persisted in optimizer checkpoints.
- `EvaluationLimitPolicy` can accumulate update counts, so docs should encourage using either telemetry snapshots or one observe call per completed update.
- Stop reason strings are stable public strings: `evaluation_limit`, `no_improvement`, and `convergence`.

## Self-Review Notes

- Spec coverage: evaluation caps, no-improvement windows, convergence thresholds, custom stop decision metadata, docs, and tests are covered.
- Type consistency: policies consume `UpdateResult`, `PopulationSnapshot`, and `OptimizationTelemetry`.
- Scope boundary: callback integration is deferred; this phase makes policies usable in manual ask/tell loops first.
